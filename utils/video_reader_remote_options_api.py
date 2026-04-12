from __future__ import annotations

import asyncio
import hashlib
import math
import os
import struct
import threading
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

import folder_paths
import numpy as np

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - unavailable outside ComfyUI runtime
    web = None  # type: ignore[assignment]
    PromptServer = None  # type: ignore[assignment]

from ..nodes.video_reader import (
    _list_video_directories,
    _list_video_files,
    _resolve_video_path,
)

_ROUTES_REGISTERED = False
_PREVIEW_SUBDIR = "iis_video_reader_preview"
_PREVIEW_EXT = ".mp4"
_PREVIEW_CODEC_CANDIDATES = ("libx264", "h264", "mpeg4")
_PREVIEW_REFLECT_MAX_FRAMES = 360
_PREVIEW_REFLECT_MAX_PIXEL_FRAMES = 300_000_000
_PREVIEW_CACHE_VERSION = "v6"

_PREVIEW_LOCKS: dict[str, threading.Lock] = {}
_PREVIEW_LOCKS_GUARD = threading.Lock()


def _parse_int_query(raw: Any, *, key: str, minimum: int) -> int:
    try:
        value = int(str(raw))
    except Exception as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


def _parse_optional_int_query(raw: Any, *, key: str, minimum: int) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    try:
        value = int(text)
    except Exception as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


def _resolve_preview_cache_dir() -> Path:
    base_dir = Path(folder_paths.get_temp_directory()).resolve()
    cache_dir = (base_dir / _PREVIEW_SUBDIR).resolve()
    try:
        common = Path(os.path.commonpath((str(base_dir), str(cache_dir))))
    except ValueError as exc:
        raise ValueError("invalid preview cache directory") from exc
    if common != base_dir:
        raise ValueError("preview cache directory must be under ComfyUI temp directory")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _resolve_preview_fps_from_stream(video_stream: Any) -> float:
    for attr_name in ("average_rate", "base_rate", "guessed_rate"):
        rate = getattr(video_stream, attr_name, None)
        if rate is None:
            continue
        try:
            fps = float(rate)
        except Exception:
            continue
        if fps > 0:
            return fps
    return 24.0


def _resolve_preview_rate(value: float) -> Fraction:
    rate = Fraction(round(float(value) * 1000), 1000)
    if rate <= 0:
        raise ValueError("preview frame rate must be greater than 0")
    return rate


def _normalize_sample_aspect_ratio(value: Any) -> Fraction | None:
    if value is None:
        return None
    try:
        ratio = Fraction(value)
    except Exception:
        return None
    if ratio.numerator <= 0 or ratio.denominator <= 0:
        return None
    return ratio


def _normalize_rotation_degrees(value: float | int | None) -> int:
    if value is None:
        return 0
    degrees = float(value) % 360.0
    if degrees < 0:
        degrees += 360.0
    candidates = (0.0, 90.0, 180.0, 270.0)
    best = min(candidates, key=lambda c: abs(((degrees - c + 180.0) % 360.0) - 180.0))
    diff = abs(((degrees - best + 180.0) % 360.0) - 180.0)
    if diff <= 5.0:
        return int(best)
    return 0


def _parse_rotation_from_metadata(metadata: Mapping[str, Any] | None) -> int:
    if not isinstance(metadata, Mapping):
        return 0
    for key in ("rotate", "rotation"):
        raw = metadata.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        try:
            value = float(text)
        except Exception:
            continue
        normalized = _normalize_rotation_degrees(value)
        if normalized != 0:
            return normalized
    return 0


def _is_display_matrix_side_data(side_data: Any) -> bool:
    side_type = getattr(side_data, "type", None)
    if side_type is not None:
        side_type_name = str(getattr(side_type, "name", "") or "").upper()
        if "DISPLAYMATRIX" in side_type_name:
            return True
        side_type_text = str(side_type).upper()
        if "DISPLAYMATRIX" in side_type_text:
            return True
        try:
            if int(getattr(side_type, "value", side_type)) == 6:
                return True
        except Exception:
            pass

    class_name = type(side_data).__name__.upper()
    if "DISPLAYMATRIX" in class_name:
        return True
    return False


def _to_matrix_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None

    reshape = getattr(value, "reshape", None)
    if callable(reshape):
        try:
            flat = reshape(-1)
            if len(flat) >= 9:
                return tuple(int(flat[i]) for i in range(9))
        except Exception:
            pass

    try:
        seq = list(value)
    except Exception:
        seq = None
    if seq is not None and len(seq) >= 9:
        try:
            return tuple(int(seq[i]) for i in range(9))
        except Exception:
            pass
    return None


def _extract_matrix_from_side_data_entry(side_data: Any) -> tuple[int, ...] | None:
    for attr_name in ("matrix", "value", "data"):
        value = getattr(side_data, attr_name, None)
        if value is None:
            continue
        matrix = _to_matrix_tuple(value)
        if matrix is not None:
            return matrix

    to_ndarray = getattr(side_data, "to_ndarray", None)
    if callable(to_ndarray):
        try:
            matrix = _to_matrix_tuple(to_ndarray())
        except Exception:
            matrix = None
        if matrix is not None:
            return matrix

    raw_bytes: bytes | None = None
    try:
        raw_bytes = memoryview(side_data).tobytes()
    except Exception:
        raw_bytes = None
    if raw_bytes is None:
        try:
            raw_bytes = bytes(side_data)
        except Exception:
            raw_bytes = None

    if raw_bytes is not None and len(raw_bytes) >= 36:
        for endian in ("<", ">"):
            try:
                values = struct.unpack(f"{endian}9i", raw_bytes[:36])
            except Exception:
                continue
            if any(values[i] != 0 for i in (0, 1, 3, 4)):
                return tuple(int(v) for v in values)
    return None


def _rotation_from_display_matrix(matrix: tuple[int, ...] | None) -> int:
    if matrix is None or len(matrix) < 9:
        return 0
    try:
        a = float(matrix[0]) / 65536.0
        b = float(matrix[1]) / 65536.0
        c = float(matrix[3]) / 65536.0
        d = float(matrix[4]) / 65536.0
        sx = math.hypot(a, c)
        sy = math.hypot(b, d)
        if sx == 0.0 or sy == 0.0:
            return 0
        rotation = -math.degrees(math.atan2(b / sy, a / sx))
        return _normalize_rotation_degrees(rotation)
    except Exception:
        return 0


def _rotation_from_side_data(side_data_items: Any) -> int:
    if side_data_items is None:
        return 0
    try:
        iterable = list(side_data_items)
    except Exception:
        return 0
    for item in iterable:
        if not _is_display_matrix_side_data(item):
            continue
        for attr_name in ("rotation", "angle", "degrees"):
            raw_rotation = getattr(item, attr_name, None)
            if raw_rotation is None:
                continue
            try:
                normalized = _normalize_rotation_degrees(float(raw_rotation))
            except Exception:
                normalized = 0
            if normalized != 0:
                return normalized
        matrix = _extract_matrix_from_side_data_entry(item)
        rotation = _rotation_from_display_matrix(matrix)
        if rotation != 0:
            return rotation
    return 0


def _resolve_source_rotation_degrees(video_stream: Any) -> int:
    # 1) stream metadata tags (rotate / rotation)
    rotation = _parse_rotation_from_metadata(getattr(video_stream, "metadata", None))
    if rotation != 0:
        return rotation

    # 2) stream-level display matrix side data (if available in this PyAV build)
    stream_side_data = getattr(video_stream, "side_data", None)
    rotation = _rotation_from_side_data(stream_side_data)
    if rotation != 0:
        return rotation

    return 0


def _resolve_frame_rotation_degrees(frame: Any) -> int:
    rotation = _parse_rotation_from_metadata(getattr(frame, "metadata", None))
    if rotation != 0:
        return rotation
    return _rotation_from_side_data(getattr(frame, "side_data", None))


def _describe_side_data_types(side_data_items: Any) -> str:
    if side_data_items is None:
        return ""
    try:
        items = list(side_data_items)
    except Exception:
        return ""
    if len(items) == 0:
        return ""
    labels: list[str] = []
    for item in items:
        side_type = getattr(item, "type", None)
        label = str(getattr(side_type, "name", "") or str(side_type) or type(item).__name__)
        label = label.strip()
        if label:
            labels.append(label)
    if len(labels) == 0:
        return ""
    return ",".join(labels[:6])


def _rotate_frame_rgb(frame_np: Any, rotation_degrees: int):
    if rotation_degrees == 0:
        return frame_np
    if rotation_degrees == 90:
        return frame_np[:, ::-1, :].swapaxes(0, 1)
    if rotation_degrees == 180:
        return frame_np[::-1, ::-1, :]
    if rotation_degrees == 270:
        return frame_np[::-1, :, :].swapaxes(0, 1)
    return frame_np


def _normalize_display_aspect_ratio(video_stream: Any) -> Fraction | None:
    stream_dar = getattr(video_stream, "display_aspect_ratio", None)
    if stream_dar is not None:
        try:
            ratio = Fraction(stream_dar)
            if ratio.numerator > 0 and ratio.denominator > 0:
                return ratio
        except Exception:
            pass

    sar = _normalize_sample_aspect_ratio(getattr(video_stream, "sample_aspect_ratio", None))
    if sar is None:
        sar = _normalize_sample_aspect_ratio(getattr(video_stream.codec_context, "sample_aspect_ratio", None))
    width = int(getattr(video_stream.codec_context, "width", 0) or 0)
    height = int(getattr(video_stream.codec_context, "height", 0) or 0)
    if sar is not None and width > 0 and height > 0:
        return Fraction(width, height) * sar
    return None


def _compute_output_sample_aspect_ratio(
    *,
    source_display_aspect_ratio: Fraction | None,
    width: int,
    height: int,
) -> Fraction | None:
    if source_display_aspect_ratio is None or width <= 0 or height <= 0:
        return None
    sar = source_display_aspect_ratio / Fraction(width, height)
    try:
        sar = Fraction(sar).limit_denominator(10_000)
    except Exception:
        return None
    if sar.numerator <= 0 or sar.denominator <= 0:
        return None
    # Treat near-square as square to avoid introducing distortion.
    if abs(float(sar) - 1.0) < 0.0025:
        return None
    return sar


def _round_to_even_positive(value: float) -> int:
    n = int(round(float(value)))
    if n < 2:
        n = 2
    if (n % 2) != 0:
        n += 1
    return n


def _compute_rescaled_size_from_display_hint(
    *,
    source_width: int,
    source_height: int,
    display_width_hint: int | None,
    display_height_hint: int | None,
) -> tuple[int, int]:
    if source_width <= 0 or source_height <= 0:
        return source_width, source_height
    if display_width_hint is None or display_height_hint is None:
        return source_width, source_height
    if display_width_hint <= 0 or display_height_hint <= 0:
        return source_width, source_height

    target_ratio = float(display_width_hint) / float(display_height_hint)
    source_ratio = float(source_width) / float(source_height)
    if not math.isfinite(target_ratio) or target_ratio <= 0:
        return source_width, source_height
    if abs(target_ratio - source_ratio) < 0.01:
        return source_width, source_height

    # Use browser-reported intrinsic display size directly when available.
    # This is more reliable than preserving coded-frame area for anamorphic sources.
    out_w = _round_to_even_positive(float(display_width_hint))
    out_h = _round_to_even_positive(float(display_height_hint))
    if out_w <= 0 or out_h <= 0:
        return source_width, source_height
    return out_w, out_h


def _estimate_selected_frames(
    *,
    total_frames: int | None,
    skip_first_frames: int,
    select_every_nth: int,
    frame_load_cap: int,
) -> int | None:
    if total_frames is None:
        if frame_load_cap > 0:
            return frame_load_cap
        return None
    if total_frames <= 0 or skip_first_frames >= total_frames:
        return 0
    remaining = total_frames - skip_first_frames
    estimated = (remaining + select_every_nth - 1) // select_every_nth
    if frame_load_cap > 0:
        estimated = min(estimated, frame_load_cap)
    return max(0, estimated)


def _inspect_source_video(source_path: Path) -> dict[str, Any]:
    try:
        import av
    except Exception as exc:
        raise RuntimeError("PyAV is required for video preview. Install `av` first.") from exc

    with av.open(str(source_path), mode="r") as container:
        if not container.streams.video:
            raise ValueError("video stream not found")
        stream = container.streams.video[0]

        width = int(getattr(stream.codec_context, "width", 0) or 0)
        height = int(getattr(stream.codec_context, "height", 0) or 0)
        fps = _resolve_preview_fps_from_stream(stream)

        total_frames: int | None = None
        stream_frames = int(getattr(stream, "frames", 0) or 0)
        if stream_frames > 0:
            total_frames = stream_frames
        else:
            duration = getattr(stream, "duration", None)
            time_base = getattr(stream, "time_base", None)
            if duration is not None and time_base is not None:
                try:
                    seconds = float(duration * time_base)
                except Exception:
                    seconds = 0.0
                if seconds > 0 and fps > 0:
                    total_frames = max(1, int(round(seconds * fps)))
            elif getattr(container, "duration", None) is not None:
                try:
                    seconds = float(container.duration) / 1_000_000.0
                except Exception:
                    seconds = 0.0
                if seconds > 0 and fps > 0:
                    total_frames = max(1, int(round(seconds * fps)))

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
    }


def _make_preview_cache_key(
    *,
    video_path: Path,
    path_source: str,
    directory: str,
    file_name: str,
    frame_load_cap: int,
    skip_first_frames: int,
    select_every_nth: int,
    source_display_width_hint: int | None,
    source_display_height_hint: int | None,
) -> str:
    stat = video_path.stat()
    digest = hashlib.sha256()
    digest.update(f"IPT-VideoReader-Preview-{_PREVIEW_CACHE_VERSION}".encode("utf-8"))
    digest.update(str(path_source).encode("utf-8"))
    digest.update(str(directory).encode("utf-8"))
    digest.update(str(file_name).encode("utf-8"))
    digest.update(str(frame_load_cap).encode("utf-8"))
    digest.update(str(skip_first_frames).encode("utf-8"))
    digest.update(str(select_every_nth).encode("utf-8"))
    digest.update(str(source_display_width_hint).encode("utf-8"))
    digest.update(str(source_display_height_hint).encode("utf-8"))
    digest.update(str(video_path.resolve()).encode("utf-8"))
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _get_preview_lock(cache_key: str) -> threading.Lock:
    with _PREVIEW_LOCKS_GUARD:
        lock = _PREVIEW_LOCKS.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _PREVIEW_LOCKS[cache_key] = lock
        return lock


def _encode_preview_video(
    *,
    source_path: Path,
    target_path: Path,
    skip_first_frames: int,
    select_every_nth: int,
    frame_load_cap: int,
    source_display_width_hint: int | None,
    source_display_height_hint: int | None,
) -> dict[str, Any]:
    try:
        import av
    except Exception as exc:
        raise RuntimeError("PyAV is required for video preview. Install `av` first.") from exc

    selected_frames = 0
    source_fps = 24.0

    with av.open(str(source_path), mode="r") as input_container:
        if not input_container.streams.video:
            raise ValueError("video stream not found")

        video_stream = input_container.streams.video[0]
        source_rotation_degrees = _resolve_source_rotation_degrees(video_stream)
        source_display_aspect_ratio = _normalize_display_aspect_ratio(video_stream)
        stream_metadata = getattr(video_stream, "metadata", None)
        stream_rotate_raw = None
        if isinstance(stream_metadata, Mapping):
            stream_rotate_raw = stream_metadata.get("rotate")
            if stream_rotate_raw is None:
                stream_rotate_raw = stream_metadata.get("rotation")
        stream_side_data_types = _describe_side_data_types(getattr(video_stream, "side_data", None))
        source_fps = _resolve_preview_fps_from_stream(video_stream)
        preview_fps = max(1.0, source_fps / float(select_every_nth))
        preview_rate = _resolve_preview_rate(preview_fps)

        with av.open(str(target_path), mode="w") as output_container:
            output_stream = None
            output_codec = None
            last_error = None
            for codec in _PREVIEW_CODEC_CANDIDATES:
                try:
                    output_stream = output_container.add_stream(codec, rate=preview_rate)
                    output_codec = codec
                    break
                except Exception as exc:
                    last_error = exc
                    output_stream = None
            if output_stream is None:
                raise RuntimeError("no supported preview video codec found") from last_error

            output_stream.pix_fmt = "yuv420p"
            output_stream.bit_rate = 0
            output_sample_aspect_ratio: Fraction | None = None
            output_width = 0
            output_height = 0
            output_configured = False

            source_index = 0
            for decoded_frame in input_container.decode(video=video_stream.index):
                if source_rotation_degrees == 0:
                    frame_rotation = _resolve_frame_rotation_degrees(decoded_frame)
                    if frame_rotation != 0:
                        source_rotation_degrees = frame_rotation

                if source_index < skip_first_frames:
                    source_index += 1
                    continue
                if ((source_index - skip_first_frames) % select_every_nth) != 0:
                    source_index += 1
                    continue

                frame_np = decoded_frame.to_ndarray(format="rgb24")
                if frame_np.ndim != 3 or frame_np.shape[2] != 3:
                    source_index += 1
                    continue
                frame_sample_aspect_ratio = _normalize_sample_aspect_ratio(
                    getattr(decoded_frame, "sample_aspect_ratio", None)
                )
                frame_np = _rotate_frame_rgb(frame_np, source_rotation_degrees)
                height = int(frame_np.shape[0])
                width = int(frame_np.shape[1])
                target_height = height - (height % 2)
                target_width = width - (width % 2)
                if target_height <= 0 or target_width <= 0:
                    source_index += 1
                    continue
                if target_height != height or target_width != width:
                    frame_np = frame_np[:target_height, :target_width, :]
                frame_np = np.ascontiguousarray(frame_np)

                if source_display_aspect_ratio is None and frame_sample_aspect_ratio is not None:
                    source_display_aspect_ratio = Fraction(target_width, target_height) * frame_sample_aspect_ratio

                if not output_configured:
                    output_width, output_height = _compute_rescaled_size_from_display_hint(
                        source_width=target_width,
                        source_height=target_height,
                        display_width_hint=source_display_width_hint,
                        display_height_hint=source_display_height_hint,
                    )

                    output_stream.width = output_width
                    output_stream.height = output_height
                    output_sample_aspect_ratio = _compute_output_sample_aspect_ratio(
                        source_display_aspect_ratio=source_display_aspect_ratio,
                        width=output_width,
                        height=output_height,
                    )
                    # For 90/270-degree rotated sources, square pixels on the physically rotated frame
                    # avoid wrong DAR reconstruction in players that ignore/interpret rotation differently.
                    if source_rotation_degrees in (90, 270):
                        output_sample_aspect_ratio = None
                    # If browser-side display hint is used, output pixels are already at desired DAR.
                    if source_display_width_hint is not None and source_display_height_hint is not None:
                        output_sample_aspect_ratio = None
                    output_stream_sar = output_sample_aspect_ratio if output_sample_aspect_ratio is not None else Fraction(1, 1)
                    try:
                        output_stream.sample_aspect_ratio = output_stream_sar
                    except Exception:
                        pass
                    output_configured = True

                output_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                if output_width > 0 and output_height > 0 and (output_frame.width != output_width or output_frame.height != output_height):
                    output_frame = output_frame.reformat(width=output_width, height=output_height, format="yuv420p")
                output_frame_sar = output_sample_aspect_ratio if output_sample_aspect_ratio is not None else Fraction(1, 1)
                try:
                    output_frame.sample_aspect_ratio = output_frame_sar
                except Exception:
                    pass
                for packet in output_stream.encode(output_frame):
                    output_container.mux(packet)

                selected_frames += 1
                source_index += 1
                if frame_load_cap > 0 and selected_frames >= frame_load_cap:
                    break

            for packet in output_stream.encode(None):
                output_container.mux(packet)


    if selected_frames <= 0:
        raise ValueError("no frames selected by skip/select/cap options")

    return {
        "source_fps": source_fps,
        "preview_fps": max(1.0, source_fps / float(select_every_nth)),
        "selected_frames": selected_frames,
    }


def _build_source_view_url(*, path_source: str, directory: str, file_name: str) -> str:
    subfolder = directory.replace("\\", "/").strip()
    if subfolder in ("", "."):
        query = urlencode({"filename": file_name, "type": path_source})
    else:
        query = urlencode({"filename": file_name, "subfolder": subfolder, "type": path_source})
    return f"/view?{query}"


def _resolve_preview_mode(
    *,
    source_info: Mapping[str, Any],
    frame_load_cap: int,
    skip_first_frames: int,
    select_every_nth: int,
) -> tuple[str, str | None, int | None]:
    total_frames = source_info.get("total_frames")
    if total_frames is not None:
        total_frames = int(total_frames)
    estimated = _estimate_selected_frames(
        total_frames=total_frames,
        skip_first_frames=skip_first_frames,
        select_every_nth=select_every_nth,
        frame_load_cap=frame_load_cap,
    )
    if estimated is None:
        return "passthrough", "unknown_frame_count_without_cap", None
    if estimated <= 0:
        return "reflect", None, estimated
    if estimated > _PREVIEW_REFLECT_MAX_FRAMES:
        return "passthrough", "estimated_selected_frames_exceeded", estimated

    width = int(source_info.get("width", 0) or 0)
    height = int(source_info.get("height", 0) or 0)
    if width > 0 and height > 0:
        estimated_pixel_frames = int(estimated) * width * height
        if estimated_pixel_frames > _PREVIEW_REFLECT_MAX_PIXEL_FRAMES:
            return "passthrough", "estimated_pixel_frames_exceeded", estimated

    return "reflect", None, estimated


def _prepare_preview_file(
    *,
    path_source: str,
    directory: str,
    file_name: str,
    frame_load_cap: int,
    skip_first_frames: int,
    select_every_nth: int,
    source_display_width_hint: int | None,
    source_display_height_hint: int | None,
) -> dict[str, Any]:
    source_path = _resolve_video_path(directory, file_name, path_source)
    source_info = _inspect_source_video(source_path)
    mode, reason, estimated_selected_frames = _resolve_preview_mode(
        source_info=source_info,
        frame_load_cap=frame_load_cap,
        skip_first_frames=skip_first_frames,
        select_every_nth=select_every_nth,
    )
    if mode == "passthrough":
        return {
            "ok": True,
            "mode": "passthrough",
            "reason": reason,
            "cache_key": None,
            "url": _build_source_view_url(
                path_source=path_source,
                directory=directory,
                file_name=file_name,
            ),
            "source_fps": source_info.get("fps"),
            "preview_fps": source_info.get("fps"),
            "selected_frames": estimated_selected_frames,
            "threshold_frames": _PREVIEW_REFLECT_MAX_FRAMES,
            "threshold_pixel_frames": _PREVIEW_REFLECT_MAX_PIXEL_FRAMES,
        }

    cache_key = _make_preview_cache_key(
        video_path=source_path,
        path_source=path_source,
        directory=directory,
        file_name=file_name,
        frame_load_cap=frame_load_cap,
        skip_first_frames=skip_first_frames,
        select_every_nth=select_every_nth,
        source_display_width_hint=source_display_width_hint,
        source_display_height_hint=source_display_height_hint,
    )
    cache_dir = _resolve_preview_cache_dir()
    preview_path = (cache_dir / f"{cache_key}{_PREVIEW_EXT}").resolve()
    tmp_path = (cache_dir / f"{cache_key}.tmp{_PREVIEW_EXT}").resolve()

    if preview_path.exists() and preview_path.is_file() and preview_path.stat().st_size > 0:
        preview_info = {
            "source_fps": None,
            "preview_fps": None,
            "selected_frames": None,
        }
    else:
        lock = _get_preview_lock(cache_key)
        with lock:
            if not (preview_path.exists() and preview_path.is_file() and preview_path.stat().st_size > 0):
                try:
                    preview_info = _encode_preview_video(
                        source_path=source_path,
                        target_path=tmp_path,
                        skip_first_frames=skip_first_frames,
                        select_every_nth=select_every_nth,
                        frame_load_cap=frame_load_cap,
                        source_display_width_hint=source_display_width_hint,
                        source_display_height_hint=source_display_height_hint,
                    )
                    os.replace(tmp_path, preview_path)
                finally:
                    if tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except Exception:
                            pass
            else:
                preview_info = {
                    "source_fps": None,
                    "preview_fps": None,
                    "selected_frames": None,
                }

    view_query = urlencode(
        {
            "filename": preview_path.name,
            "subfolder": _PREVIEW_SUBDIR,
            "type": "temp",
        }
    )
    return {
        "ok": True,
        "mode": "reflect",
        "reason": None,
        "cache_key": cache_key,
        "url": f"/view?{view_query}",
        "source_fps": preview_info.get("source_fps"),
        "preview_fps": preview_info.get("preview_fps"),
        "selected_frames": preview_info.get("selected_frames"),
        "threshold_frames": _PREVIEW_REFLECT_MAX_FRAMES,
        "threshold_pixel_frames": _PREVIEW_REFLECT_MAX_PIXEL_FRAMES,
    }


def register_routes() -> None:
    global _ROUTES_REGISTERED

    if _ROUTES_REGISTERED:
        return
    if PromptServer is None or web is None:
        return
    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        return

    @PromptServer.instance.routes.get("/iis/video-reader/directories")
    async def iis_video_reader_directories(request):
        path_source = str(request.query.get("path_source", "input") or "input")
        try:
            directories = _list_video_directories(path_source)
        except Exception:
            directories = ["."]
        return web.json_response(directories)

    @PromptServer.instance.routes.get("/iis/video-reader/files")
    async def iis_video_reader_files(request):
        path_source = str(request.query.get("path_source", "input") or "input")
        directory = str(request.query.get("directory", ".") or ".")
        try:
            files = _list_video_files(directory=directory, path_source=path_source)
        except Exception:
            files = []
        return web.json_response(files)

    @PromptServer.instance.routes.get("/iis/video-reader/preview")
    async def iis_video_reader_preview(request):
        path_source = str(request.query.get("path_source", "input") or "input")
        directory = str(request.query.get("directory", ".") or ".")
        file_name = str(request.query.get("file", "") or "").strip()
        if not file_name:
            return web.json_response({"ok": False, "error": "file is required"}, status=400)

        try:
            frame_load_cap = _parse_int_query(
                request.query.get("frame_load_cap", 0),
                key="frame_load_cap",
                minimum=0,
            )
            skip_first_frames = _parse_int_query(
                request.query.get("skip_first_frames", 0),
                key="skip_first_frames",
                minimum=0,
            )
            select_every_nth = _parse_int_query(
                request.query.get("select_every_nth", 1),
                key="select_every_nth",
                minimum=1,
            )
            source_display_width_hint = _parse_optional_int_query(
                request.query.get("source_display_width_hint"),
                key="source_display_width_hint",
                minimum=1,
            )
            source_display_height_hint = _parse_optional_int_query(
                request.query.get("source_display_height_hint"),
                key="source_display_height_hint",
                minimum=1,
            )
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

        try:
            result = await asyncio.to_thread(
                _prepare_preview_file,
                path_source=path_source,
                directory=directory,
                file_name=file_name,
                frame_load_cap=frame_load_cap,
                skip_first_frames=skip_first_frames,
                select_every_nth=select_every_nth,
                source_display_width_hint=source_display_width_hint,
                source_display_height_hint=source_display_height_hint,
            )
            return web.json_response(result)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    _ROUTES_REGISTERED = True


register_routes()
