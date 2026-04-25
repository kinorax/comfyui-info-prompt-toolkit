# Copyright 2026 kinorax
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import folder_paths
import numpy as np
import torch
from comfy_api.latest import io as c_io
from comfy_api.latest import ui as c_ui

from .. import const as Const
from ..utils import cast as Cast
from ..utils.a1111_infotext import image_info_to_a1111_infotext
from ..utils.image_info_hash_extras import (
    add_civitai_hash_extras,
    clear_representative_hash_extras,
)
from ..utils.video_runtime_support import VIDEO_SAVER_PYAV_REQUIRED_MESSAGE

_OUTPUT_SUBDIR_OPTIONS = ("none", "year", "year_month", "iso_week", "year_month_day")
_CODEC_OPTIONS = ("av1", "h264")
_DEFAULT_CODEC = "av1"
_VIDEO_EXT = ".mp4"
_MISSING = object()
_SOURCE_IMAGEINFO_DISPLAY_NAME = "source_image_info"
_IMAGE_INFOTEXT_METADATA_KEY = "image_infotext"
_VIDEO_INFOTEXT_METADATA_KEY = "video_infotext"
_AV1_ENCODER = "libsvtav1"
_H264_ENCODER = "libx264"
_AV1_PRESET = 6
_H264_PRESET = "medium"
_YUV420P_PIXEL_FORMAT = "yuv420p"
_INVALID_FILE_STEM_CHARS = set('<>:"|?*')
_WINDOWS_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def _unwrap_input_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            return list(value[0])
        return list(value)
    return [value]


def _resolve_single_input(raw: Any, name: str) -> Any:
    values = _unwrap_input_list(raw)
    if len(values) == 0:
        return None
    if len(values) > 1:
        raise ValueError(f"{name} must be a single value")
    return values[0]


def _split_images_from_input(image_input: Any) -> tuple[list[torch.Tensor], torch.Tensor]:
    image_values = _unwrap_input_list(image_input)
    if len(image_values) == 0:
        raise ValueError("images is required")

    merged_tensors: list[torch.Tensor] = []
    for value in image_values:
        if not isinstance(value, torch.Tensor):
            raise ValueError("images input must be IMAGE")
        if value.ndim == 3:
            merged_tensors.append(value.unsqueeze(0))
            continue
        if value.ndim == 4:
            merged_tensors.append(value)
            continue
        raise ValueError(f"unsupported image tensor shape: {tuple(value.shape)}")

    if len(merged_tensors) == 1:
        merged_batch = merged_tensors[0]
    else:
        merged_batch = torch.cat(merged_tensors, dim=0)

    images: list[torch.Tensor] = [merged_batch[i] for i in range(merged_batch.shape[0])]
    return images, merged_batch


def _resolve_output_root() -> Path:
    return Path(folder_paths.get_output_directory()).resolve()


def _resolve_output_dir(output_dir: str | None, output_root: Path) -> Path:
    text = (output_dir or "").strip()
    base = Path(text) if text else output_root
    if not base.is_absolute():
        base = output_root / base
    resolved = base.resolve()

    try:
        common = Path(os.path.commonpath((str(output_root), str(resolved))))
    except ValueError as exc:
        raise ValueError("output_dir must be under ComfyUI output directory") from exc
    if common != output_root:
        raise ValueError("output_dir must be under ComfyUI output directory")
    return resolved


def _resolve_subdir(now: datetime, output_subdir: str) -> str:
    if output_subdir == "none":
        return ""
    if output_subdir == "year":
        return now.strftime("%Y")
    if output_subdir == "year_month":
        return now.strftime("%Y%m")
    if output_subdir == "year_month_day":
        return now.strftime("%Y%m%d")
    if output_subdir == "iso_week":
        iso = now.isocalendar()
        return f"{iso.year}W{iso.week:02d}"
    raise ValueError(f"unsupported output_subdir: {output_subdir}")


def _find_next_counter(folder: Path, date_prefix: str) -> int:
    current_max = 0
    expected_prefix = f"{date_prefix}-"
    for entry in os.scandir(folder):
        if not entry.is_file():
            continue

        name = entry.name
        if not name.startswith(expected_prefix):
            continue
        if not name.lower().endswith(_VIDEO_EXT):
            continue

        rest = name[len(expected_prefix):]
        if len(rest) < 6 or rest[5] != "-":
            continue
        counter_text = rest[:5]
        if not counter_text.isdigit():
            continue
        counter = int(counter_text)
        if counter > current_max:
            current_max = counter
    return current_max + 1


def _safe_filename_suffix(value: Any) -> str:
    suffix = str(value or "").strip()
    if "/" in suffix or "\\" in suffix:
        raise ValueError("filename_suffix must not contain path separators")
    return suffix


def _normalize_optional_file_stem_values(raw: Any) -> list[Any] | None:
    values = _unwrap_input_list(raw)
    if len(values) == 0:
        return None
    if len(values) == 1 and values[0] is None:
        return None
    return values


def _validate_file_stem(value: Any) -> str:
    stem = str(value or "").strip()
    if not stem:
        raise ValueError("file_stem must not be empty")
    if stem in (".", ".."):
        raise ValueError("file_stem must not be '.' or '..'")
    if "/" in stem or "\\" in stem:
        raise ValueError("file_stem must not contain path separators")
    if any(ord(ch) < 32 for ch in stem):
        raise ValueError("file_stem must not contain control characters")
    invalid_chars = sorted(ch for ch in set(stem) if ch in _INVALID_FILE_STEM_CHARS)
    if invalid_chars:
        raise ValueError(f"file_stem contains invalid characters: {''.join(invalid_chars)}")
    if stem.endswith(".") or stem.endswith(" "):
        raise ValueError("file_stem must not end with '.' or space")

    reserved_base = stem.split(".", 1)[0].upper()
    if reserved_base in _WINDOWS_RESERVED_BASENAMES:
        raise ValueError(f"file_stem uses a reserved Windows name: {reserved_base}")
    return stem


def _resolve_forced_file_stem(raw: Any) -> str | None:
    values = _normalize_optional_file_stem_values(raw)
    if values is None:
        return None
    if len(values) != 1:
        raise ValueError(f"file_stem must have length 1, got {len(values)}")
    return _validate_file_stem(values[0])


def _render_file_stem(date_prefix: str, counter: int, suffix: str) -> str:
    return f"{date_prefix}-{counter:05d}-{suffix}"


def _relative_to_output_root(path: Path, output_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(output_root)
    except Exception as exc:
        raise ValueError("saved file path is outside ComfyUI output directory") from exc
    return rel.as_posix()


def _build_frame_indices(image_count: int, pingpong: bool, loop_count: int) -> list[int]:
    if image_count <= 0:
        return []
    indices = list(range(image_count))
    if pingpong and image_count > 1:
        indices.extend(range(image_count - 2, 0, -1))
    if loop_count > 0:
        indices = indices * (loop_count + 1)
    return indices


def _serialize_metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _normalize_frame_rate_value(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame_rate must be a number") from exc
    if not math.isfinite(value):
        raise ValueError("frame_rate must be finite")

    normalized = round(value, 3)
    if normalized <= 0:
        raise ValueError("frame_rate must be greater than 0")
    return normalized


def _resolve_frame_rate_fraction(frame_rate: float) -> Fraction:
    rate = Fraction(round(float(frame_rate) * 1000), 1000)
    if rate <= 0:
        raise ValueError("frame_rate must be greater than 0")
    return rate


def _build_infotext(image_info: Any) -> str:
    if not isinstance(image_info, Mapping):
        return ""
    image_info_without_hashes = clear_representative_hash_extras(image_info)
    image_info_with_hashes = add_civitai_hash_extras(image_info_without_hashes)
    return image_info_to_a1111_infotext(image_info_with_hashes)


def _normalize_audio_for_mux(
    raw_audio: Any,
    target_samples: int,
) -> tuple[np.ndarray, int, str] | None:
    if not isinstance(raw_audio, Mapping):
        return None

    waveform = raw_audio.get("waveform")
    sample_rate_raw = raw_audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor):
        return None
    try:
        sample_rate = int(sample_rate_raw)
    except (TypeError, ValueError):
        return None
    if sample_rate <= 0:
        return None

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 3:
        return None
    if waveform.shape[0] < 1:
        return None

    waveform = waveform[0]
    if waveform.ndim != 2 or waveform.shape[1] <= 0:
        return None

    if target_samples > 0 and waveform.shape[1] > target_samples:
        waveform = waveform[:, :target_samples]
    if waveform.shape[1] <= 0:
        return None

    channels = int(waveform.shape[0])
    if channels <= 0:
        return None
    if channels == 1:
        layout = "mono"
    elif channels == 6:
        layout = "5.1"
    else:
        layout = "stereo"
        if channels >= 2:
            waveform = waveform[:2, :]
        else:
            waveform = waveform.repeat(2, 1)

    audio_np = waveform.float().cpu().contiguous().numpy()
    return audio_np, sample_rate, layout


def _encode_mp4(
    path: Path,
    frames: Sequence[torch.Tensor],
    frame_rate: float,
    *,
    codec_name: str,
    encoder_name: str,
    crf: int,
    preset: int | str,
    audio: Any,
    metadata: Mapping[str, Any],
) -> None:
    try:
        import av
    except Exception as exc:
        raise RuntimeError(VIDEO_SAVER_PYAV_REQUIRED_MESSAGE) from exc

    if len(frames) == 0:
        raise ValueError("images is required")

    first = frames[0]
    height = int(first.shape[0])
    width = int(first.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("images must have positive dimensions")
    if (width % 2) != 0 or (height % 2) != 0:
        raise ValueError(f"images width and height must be even for {codec_name} {_YUV420P_PIXEL_FORMAT} encoding")

    rate = _resolve_frame_rate_fraction(frame_rate)

    target_samples = 0

    try:
        with av.open(str(path), mode="w", options={"movflags": "use_metadata_tags"}) as container:
            for key, value in metadata.items():
                container.metadata[str(key)] = _serialize_metadata_value(value)

            try:
                video_stream = container.add_stream(encoder_name, rate=rate)
            except Exception as exc:
                raise RuntimeError(f"{codec_name} encoder `{encoder_name}` is unavailable in this PyAV/FFmpeg build") from exc

            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = _YUV420P_PIXEL_FORMAT
            video_stream.bit_rate = 0
            video_stream.options = {
                "crf": str(int(crf)),
                "preset": str(preset),
            }

            audio_payload: tuple[np.ndarray, int, str] | None = None
            if isinstance(audio, Mapping):
                sample_rate_raw = audio.get("sample_rate")
                try:
                    sample_rate = int(sample_rate_raw)
                except (TypeError, ValueError):
                    sample_rate = 0
                if sample_rate > 0:
                    target_samples = int(math.ceil((sample_rate / float(frame_rate)) * len(frames)))
                audio_payload = _normalize_audio_for_mux(audio, target_samples)

            audio_stream = None
            if audio_payload is not None:
                _, audio_sample_rate, audio_layout = audio_payload
                audio_stream = container.add_stream("aac", rate=audio_sample_rate, layout=audio_layout)

            for frame_tensor in frames:
                frame_np = np.clip(255.0 * frame_tensor[..., :3].cpu().numpy(), 0, 255).astype(np.uint8)
                video_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                for packet in video_stream.encode(video_frame):
                    container.mux(packet)

            for packet in video_stream.encode(None):
                container.mux(packet)

            if audio_stream is not None and audio_payload is not None:
                audio_np, audio_sample_rate, audio_layout = audio_payload
                audio_frame = av.AudioFrame.from_ndarray(audio_np, format="fltp", layout=audio_layout)
                audio_frame.sample_rate = audio_sample_rate
                audio_frame.pts = 0
                for packet in audio_stream.encode(audio_frame):
                    container.mux(packet)
                for packet in audio_stream.encode(None):
                    container.mux(packet)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"failed to save video: {path}") from exc


def _encode_av1_mp4(
    path: Path,
    frames: Sequence[torch.Tensor],
    frame_rate: float,
    crf: int,
    preset: int,
    audio: Any,
    metadata: Mapping[str, Any],
) -> None:
    _encode_mp4(
        path=path,
        frames=frames,
        frame_rate=frame_rate,
        codec_name="AV1",
        encoder_name=_AV1_ENCODER,
        crf=crf,
        preset=preset,
        audio=audio,
        metadata=metadata,
    )


def _encode_h264_mp4(
    path: Path,
    frames: Sequence[torch.Tensor],
    frame_rate: float,
    crf: int,
    preset: str,
    audio: Any,
    metadata: Mapping[str, Any],
) -> None:
    _encode_mp4(
        path=path,
        frames=frames,
        frame_rate=frame_rate,
        codec_name="H.264",
        encoder_name=_H264_ENCODER,
        crf=crf,
        preset=preset,
        audio=audio,
        metadata=metadata,
    )


class VideoSaver(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-VideoSaver",
            display_name="Video Saver",
            category=Const.CATEGORY_IMAGEINFO,
            is_output_node=True,
            is_input_list=True,
            inputs=[
                c_io.Image.Input(
                    "images",
                    display_name="image",
                    tooltip="Image or image batch to save as video",
                ),
                c_io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Optional audio to mux into output video",
                ),
                Const.IMAGEINFO_TYPE.Input(
                    Const.IMAGEINFO,
                    display_name=_SOURCE_IMAGEINFO_DISPLAY_NAME,
                    optional=True,
                    tooltip="Source image info",
                ),
                Const.IMAGEINFO_TYPE.Input(
                    Const.VIDEOINFO,
                    optional=True,
                    tooltip="Video generation info",
                ),
                c_io.String.Input(
                    "filename_suffix",
                    default="video",
                    tooltip="Suffix part of output filename",
                ),
                c_io.String.Input(
                    "file_stem",
                    optional=True,
                    force_input=True,
                    tooltip="Optional output filename stem; when connected, save as <file_stem>.mp4",
                ),
                c_io.String.Input(
                    "output_dir",
                    default="",
                    tooltip="Base output directory under ComfyUI output",
                ),
                c_io.Combo.Input(
                    "output_subdir",
                    options=list(_OUTPUT_SUBDIR_OPTIONS),
                    default="none",
                    tooltip="Output sub directory format",
                ),
                c_io.Combo.Input(
                    "codec",
                    display_name="format",
                    options=list(_CODEC_OPTIONS),
                    default=_DEFAULT_CODEC,
                    tooltip="Output video codec in MP4 container",
                ),
                c_io.Int.Input(
                    "av1_crf",
                    display_name="av1_crf",
                    default=23,
                    min=0,
                    max=63,
                    tooltip="AV1 CRF (0..63, lower is higher quality)",
                ),
                c_io.Int.Input(
                    "h264_crf",
                    display_name="h264_crf",
                    default=19,
                    min=0,
                    max=51,
                    tooltip="H.264 CRF (0..51, lower is higher quality)",
                ),
                c_io.Float.Input(
                    "frame_rate",
                    default=24.0,
                    min=0.001,
                    max=1000.0,
                    step=1.0,
                    tooltip="Output frame rate (supports up to 3 decimal places)",
                ),
                c_io.Int.Input(
                    "loop_count",
                    default=0,
                    min=0,
                    tooltip="Number of extra loops after first playback",
                ),
                c_io.Boolean.Input(
                    "pingpong",
                    default=False,
                    tooltip="Play forward then backward (excluding first and last duplicated endpoints)",
                ),
            ],
            outputs=[
                c_io.Image.Output(
                    Cast.out_id("images"),
                    display_name="image",
                    is_output_list=True,
                ),
                c_io.Audio.Output(
                    Cast.out_id("audio"),
                    display_name="audio",
                ),
                Const.IMAGEINFO_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO),
                    display_name=_SOURCE_IMAGEINFO_DISPLAY_NAME,
                ),
                Const.IMAGEINFO_TYPE.Output(
                    Cast.out_id(Const.VIDEOINFO),
                    display_name=Const.VIDEOINFO,
                ),
                c_io.String.Output(
                    Cast.out_id("file_path"),
                    display_name="file_path",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: Any,
        image_info: Any = None,
        audio: Any = None,
        video_info: Any = None,
        filename_suffix: Any = _MISSING,
        file_stem: Any = None,
        output_dir: Any = _MISSING,
        output_subdir: Any = _MISSING,
        codec: Any = _MISSING,
        av1_crf: Any = _MISSING,
        h264_crf: Any = _MISSING,
        frame_rate: Any = _MISSING,
        loop_count: Any = _MISSING,
        pingpong: Any = _MISSING,
    ) -> c_io.NodeOutput:
        image_batch_items, _ = _split_images_from_input(images)
        image_count = len(image_batch_items)
        if image_count == 0:
            raise ValueError("images is required")

        image_info_value = _resolve_single_input(image_info, Const.IMAGEINFO)
        has_image_info = image_info_value is not None
        image_info_output_value = image_info_value if has_image_info else {}
        video_info_value = _resolve_single_input(video_info, Const.VIDEOINFO)
        has_video_info = video_info_value is not None
        video_info_output_value = video_info_value if has_video_info else {}

        forced_file_stem = _resolve_forced_file_stem(file_stem)

        if filename_suffix is _MISSING and forced_file_stem is None:
            raise ValueError("filename_suffix is required")
        if output_dir is _MISSING:
            raise ValueError("output_dir is required")
        if output_subdir is _MISSING:
            raise ValueError("output_subdir is required")
        if codec is _MISSING:
            raise ValueError("codec is required")
        if av1_crf is _MISSING:
            raise ValueError("av1_crf is required")
        if h264_crf is _MISSING:
            raise ValueError("h264_crf is required")
        if frame_rate is _MISSING:
            raise ValueError("frame_rate is required")
        if loop_count is _MISSING:
            raise ValueError("loop_count is required")
        if pingpong is _MISSING:
            raise ValueError("pingpong is required")

        audio_value = _resolve_single_input(audio, "audio")
        output_dir_value = _resolve_single_input(output_dir, "output_dir")
        output_subdir_value = str(_resolve_single_input(output_subdir, "output_subdir"))
        codec_value = str(_resolve_single_input(codec, "codec") or "").strip().lower()
        av1_crf_value = int(_resolve_single_input(av1_crf, "av1_crf"))
        h264_crf_value = int(_resolve_single_input(h264_crf, "h264_crf"))
        frame_rate_value = _normalize_frame_rate_value(_resolve_single_input(frame_rate, "frame_rate"))
        loop_count_value = int(_resolve_single_input(loop_count, "loop_count"))
        pingpong_value = bool(_resolve_single_input(pingpong, "pingpong"))
        suffix = ""
        if forced_file_stem is None:
            suffix = _safe_filename_suffix(_resolve_single_input(filename_suffix, "filename_suffix"))

        if output_subdir_value not in _OUTPUT_SUBDIR_OPTIONS:
            raise ValueError(f"unsupported output_subdir: {output_subdir_value}")
        if codec_value not in _CODEC_OPTIONS:
            raise ValueError(f"unsupported codec: {codec_value}")
        if av1_crf_value < 0 or av1_crf_value > 63:
            raise ValueError("av1_crf must be in range 0..63")
        if h264_crf_value < 0 or h264_crf_value > 51:
            raise ValueError("h264_crf must be in range 0..51")
        if loop_count_value < 0:
            raise ValueError("loop_count must be 0 or greater")

        frame_indices = _build_frame_indices(image_count, pingpong_value, loop_count_value)
        if len(frame_indices) == 0:
            raise ValueError("images is required")
        render_frames = [image_batch_items[idx] for idx in frame_indices]

        now = datetime.now()
        output_root = _resolve_output_root()
        base_output_dir = _resolve_output_dir(None if output_dir_value is None else str(output_dir_value), output_root)
        subdir = _resolve_subdir(now, output_subdir_value)
        target_dir = (base_output_dir / subdir).resolve() if subdir else base_output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        date_prefix = now.strftime("%Y%m%d")
        if forced_file_stem is None:
            counter = _find_next_counter(target_dir, date_prefix)
            stem = _render_file_stem(date_prefix, counter, suffix)
        else:
            stem = forced_file_stem
        file_path = target_dir / f"{stem}{_VIDEO_EXT}"

        metadata: dict[str, Any] = {}
        if has_image_info:
            image_infotext = _build_infotext(image_info_value)
            if image_infotext:
                metadata[_IMAGE_INFOTEXT_METADATA_KEY] = image_infotext
        if has_video_info:
            video_infotext = _build_infotext(video_info_value)
            if video_infotext:
                metadata[_VIDEO_INFOTEXT_METADATA_KEY] = video_infotext
        metadata["video_saver"] = {
            "codec": codec_value,
            "frame_rate": frame_rate_value,
            "loop_count": loop_count_value,
            "pingpong": pingpong_value,
            "frame_count": len(frame_indices),
            "encoder": _AV1_ENCODER if codec_value == "av1" else _H264_ENCODER,
        }

        if codec_value == "av1":
            _encode_av1_mp4(
                path=file_path,
                frames=render_frames,
                frame_rate=frame_rate_value,
                crf=av1_crf_value,
                preset=_AV1_PRESET,
                audio=audio_value,
                metadata=metadata,
            )
        else:
            _encode_h264_mp4(
                path=file_path,
                frames=render_frames,
                frame_rate=frame_rate_value,
                crf=h264_crf_value,
                preset=_H264_PRESET,
                audio=audio_value,
                metadata=metadata,
            )

        rel_file_path = _relative_to_output_root(file_path, output_root)
        rel_parent = file_path.parent.resolve().relative_to(output_root).as_posix()
        # Preview the saved file itself instead of generating a preview-only asset.
        ui_result = c_ui.SavedResult(
            file_path.name,
            "" if rel_parent == "." else rel_parent,
            c_io.FolderType.output,
        )

        images_out_values = _unwrap_input_list(images)
        return c_io.NodeOutput(
            images_out_values,
            audio_value,
            image_info_output_value,
            video_info_output_value,
            rel_file_path,
            ui=c_ui.PreviewVideo([ui_result]),
        )
