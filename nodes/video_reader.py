# Copyright 2026 kinorax
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Mapping

import folder_paths
import torch
from comfy_api.latest import InputImpl
from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.metadata_encryption import image_info_from_metadata, unpack_video_encrypted_payloads
from ..utils.mp4_private_metadata import read_ipt_private_metadata
from ..utils.video_color import (
    COLOR_ENCODING_BT2020,
    COLOR_ENCODING_BT709,
    convert_rgb_tensor_color_encoding,
    describe_video_stream_color,
    video_color_diagnostic_warnings,
)
from ..utils.video_runtime_support import (
    VIDEO_READER_BACKEND_UNAVAILABLE_MESSAGE,
    get_video_from_file_loader,
    is_video_backend_unavailable_error,
)
from ..utils.webm_metadata import (
    extract_matroska_metadata,
    extract_matroska_private_metadata,
)

_PATH_SOURCE_OPTIONS = ("input", "output")
_VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv")
_MATROSKA_EXTENSIONS = (".webm", ".mkv")
_SOURCE_IMAGEINFO_DISPLAY_NAME = "source_image_info"
_IMAGE_INFOTEXT_METADATA_KEY = "image_infotext"
_VIDEO_INFOTEXT_METADATA_KEY = "video_infotext"
_LEGACY_INFOTEXT_METADATA_KEY = "infotext"
_IMAGE_INFOTEXT_RAW_OUTPUT_ID = "image_infotext_raw"
_VIDEO_INFOTEXT_RAW_OUTPUT_ID = "video_infotext_raw"
_LOGGER = logging.getLogger(__name__)


def _resolve_base_directory(path_source: str) -> tuple[Path, str]:
    normalized = str(path_source or "").strip().lower()
    if normalized == "input":
        return Path(folder_paths.get_input_directory()).resolve(), "ComfyUI input directory"
    if normalized == "output":
        return Path(folder_paths.get_output_directory()).resolve(), "ComfyUI output directory"
    raise ValueError("path_source must be one of: input, output")


def _is_target_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS


def _list_video_directories(path_source: str) -> list[str]:
    base_dir, _ = _resolve_base_directory(path_source)
    discovered: set[str] = set()

    for root, dirs, files in os.walk(base_dir, topdown=True, followlinks=False):
        root_path = Path(root).resolve()

        try:
            root_common = Path(os.path.commonpath((str(base_dir), str(root_path))))
        except ValueError:
            continue
        if root_common != base_dir:
            continue

        safe_dirs: list[str] = []
        for dir_name in dirs:
            candidate_path = (root_path / dir_name).resolve()
            try:
                common = Path(os.path.commonpath((str(base_dir), str(candidate_path))))
            except ValueError:
                continue
            if common != base_dir:
                continue
            safe_dirs.append(dir_name)
        dirs[:] = safe_dirs

        has_video = False
        for file_name in files:
            if (root_path / file_name).suffix.lower() in _VIDEO_EXTENSIONS:
                has_video = True
                break
        if not has_video:
            continue

        rel_root = root_path.relative_to(base_dir).as_posix()
        discovered.add("." if rel_root in ("", ".") else rel_root)

    if len(discovered) == 0:
        return ["."]
    return sorted(discovered, key=lambda item: (item != ".", item.casefold()))


def _resolve_target_directory(directory: str, path_source: str) -> Path:
    base_dir, base_label = _resolve_base_directory(path_source)
    normalized = (directory or "").strip()

    if normalized and Path(normalized).is_absolute():
        raise ValueError(f"directory must be relative to {base_label}")

    if normalized in ("", "."):
        target = base_dir
    else:
        target = (base_dir / normalized).resolve()

    try:
        common = Path(os.path.commonpath((str(base_dir), str(target))))
    except ValueError as exc:
        raise ValueError(f"directory must be under {base_label}") from exc
    if common != base_dir:
        raise ValueError(f"directory must be under {base_label}")

    if not target.exists():
        raise ValueError(f"directory not found: {normalized or '.'}")
    if not target.is_dir():
        raise ValueError(f"directory is not a folder: {normalized or '.'}")
    return target


def _list_video_files(directory: str, path_source: str) -> list[str]:
    try:
        target_dir = _resolve_target_directory(directory, path_source)
    except Exception:
        return []

    files: list[str] = []
    for entry in target_dir.iterdir():
        if _is_target_video_file(entry):
            files.append(entry.name)
    return sorted(files, key=str.casefold, reverse=True)


def _resolve_video_path(directory: str, file: str, path_source: str) -> Path:
    target_dir = _resolve_target_directory(directory, path_source)
    file_name = (file or "").strip()
    if not file_name:
        raise ValueError("file is required")
    if Path(file_name).is_absolute():
        raise ValueError("file must be a file name under selected directory")

    target = (target_dir / file_name).resolve()
    try:
        common = Path(os.path.commonpath((str(target_dir), str(target))))
    except ValueError as exc:
        raise ValueError("file must be under selected directory") from exc
    if common != target_dir:
        raise ValueError("file must be under selected directory")

    if not _is_target_video_file(target):
        raise ValueError(f"video file not found or not MP4/WebM/MKV: {file_name}")
    return target


def _read_metadata_text(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if value is None:
        return ""
    return str(value)


def _read_info_from_metadata(
    metadata: Mapping[str, Any],
    *,
    key: str,
    fallback_key: str | None = None,
    encrypted_payload: bytes | None = None,
    file_name: str | None = None,
) -> tuple[dict[str, Any], str]:
    infotext_raw = _read_metadata_text(metadata, key)
    if not infotext_raw and fallback_key:
        infotext_raw = _read_metadata_text(metadata, fallback_key)

    image_info = image_info_from_metadata(infotext_raw or None, encrypted_payload, file_name=file_name)
    return image_info, infotext_raw


def _read_container_metadata(
    video_path: Path,
    metadata: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, bytes | None]]:
    if video_path.suffix.lower() not in _MATROSKA_EXTENSIONS:
        return metadata, unpack_video_encrypted_payloads(read_ipt_private_metadata(video_path))

    logical_metadata = extract_matroska_metadata(metadata)
    try:
        packed_payload = extract_matroska_private_metadata(metadata)
    except ValueError:
        _log_invalid_matroska_private_metadata(video_path)
        return logical_metadata, {"image": None, "video": None}

    try:
        encrypted_payloads = unpack_video_encrypted_payloads(packed_payload, strict=True)
    except ValueError:
        _log_invalid_matroska_private_metadata(video_path)
        return logical_metadata, {"image": None, "video": None}
    return logical_metadata, encrypted_payloads


def _log_invalid_matroska_private_metadata(video_path: Path) -> None:
    _LOGGER.warning(
        '[IPT] Video Reader: invalid WebM/MKV private metadata for "%s"; '
        "using public metadata only.",
        video_path.name,
    )


def _probe_video_color_info(video_path: Path) -> dict[str, Any]:
    import av

    with av.open(str(video_path), mode="r") as container:
        if len(container.streams.video) == 0:
            raise ValueError("video has no video stream")
        return describe_video_stream_color(container.streams.video[0])


def _log_video_color_diagnostics(video_path: Path) -> dict[str, Any] | None:
    try:
        color_info = _probe_video_color_info(video_path)
    except Exception as exc:
        _LOGGER.warning(
            "[IPT] Video Reader: failed to inspect color metadata for %s: %s",
            video_path.name,
            exc,
        )
        return None

    for diagnostic in video_color_diagnostic_warnings(color_info):
        _LOGGER.warning(
            "[IPT] Video Reader: %s (%s)",
            diagnostic,
            video_path.name,
        )
    return color_info


def _is_ffv1_rgb_video(video_path: Path) -> bool:
    if video_path.suffix.lower() != ".mkv":
        return False

    try:
        import av

        with av.open(str(video_path), mode="r") as container:
            if len(container.streams.video) == 0:
                return False
            codec_context = container.streams.video[0].codec_context
            pixel_format = getattr(getattr(codec_context, "format", None), "name", None)
            return (
                codec_context.name == "ffv1"
                and pixel_format in {"bgr0", "gbrp16le"}
            )
    except Exception:
        return False


def _decode_ffv1_rgb_components(
    video_path: Path,
) -> tuple[torch.Tensor, Any, Mapping[str, Any]]:
    import av
    import numpy as np

    frames: list[torch.Tensor] = []
    audio_frames: list[Any] = []
    audio: Any = None

    with av.open(str(video_path), mode="r") as container:
        if len(container.streams.video) == 0:
            raise ValueError("video has no video stream")

        video_stream = container.streams.video[0]
        streams = [video_stream]
        audio_stream = next(
            (
                stream
                for stream in reversed(container.streams.audio)
                if stream.codec_context is not None
            ),
            None,
        )
        audio_resampler = None
        if audio_stream is not None:
            streams.append(audio_stream)
            audio_resampler = av.audio.resampler.AudioResampler(format="fltp")

        for packet in container.demux(*streams):
            if packet.stream.type == "video":
                try:
                    decoded_frames = packet.decode()
                except av.error.InvalidDataError:
                    _LOGGER.info(
                        "[IPT] Video Reader: skipped an invalid FFV1 packet in %s",
                        video_path.name,
                    )
                    continue

                for frame in decoded_frames:
                    image = np.ascontiguousarray(
                        frame.to_ndarray(format="gbrpf32le")
                    )
                    rotation = getattr(frame, "rotation", 0)
                    if rotation:
                        quarter_turns = int(round(rotation // 90))
                        image = np.rot90(
                            image,
                            k=quarter_turns,
                            axes=(0, 1),
                        ).copy()
                    frames.append(torch.from_numpy(image))
            elif audio_resampler is not None:
                try:
                    decoded_audio_frames = packet.decode()
                except av.error.InvalidDataError:
                    _LOGGER.info(
                        "[IPT] Video Reader: skipped an invalid audio packet in %s",
                        video_path.name,
                    )
                    continue

                for decoded_audio_frame in decoded_audio_frames:
                    for resampled_frame in audio_resampler.resample(decoded_audio_frame):
                        audio_frames.append(resampled_frame.to_ndarray())

        if audio_resampler is not None:
            for resampled_frame in audio_resampler.resample(None):
                audio_frames.append(resampled_frame.to_ndarray())

        if len(frames) == 0:
            raise ValueError("video has no frames")
        images = torch.stack(frames)

        if audio_stream is not None and len(audio_frames) > 0:
            audio_data = np.concatenate(audio_frames, axis=1)
            audio = {
                "waveform": torch.from_numpy(audio_data).unsqueeze(0),
                "sample_rate": (
                    int(audio_stream.sample_rate)
                    if audio_stream.sample_rate
                    else 1
                ),
            }

        metadata = dict(container.metadata)

    return images, audio, metadata


def _convert_decoded_images_to_bt709(
    images: torch.Tensor,
    color_info: Mapping[str, Any] | None,
) -> torch.Tensor:
    if not color_info:
        return images
    if color_info.get("color_encoding") != COLOR_ENCODING_BT2020:
        return images
    if color_info.get("color_matrix") != "bt2020nc":
        return images
    if color_info.get("color_transfer") != "bt2020-10":
        return images
    return convert_rgb_tensor_color_encoding(
        images,
        COLOR_ENCODING_BT2020,
        COLOR_ENCODING_BT709,
    )


def _read_video(
    video_path: Path,
    skip_first_frames: int,
    select_every_nth: int,
    frame_load_cap: int,
) -> tuple[torch.Tensor, int, dict[str, Any], dict[str, Any], str, str, Any]:
    use_ffv1_rgb_decoder = _is_ffv1_rgb_video(video_path)
    video_loader = (
        None
        if use_ffv1_rgb_decoder
        else get_video_from_file_loader(InputImpl)
    )

    try:
        if use_ffv1_rgb_decoder:
            images, audio, raw_metadata = _decode_ffv1_rgb_components(video_path)
        else:
            assert video_loader is not None
            video = video_loader(str(video_path))
            components = video.get_components()
            images = components.images
            audio = components.audio
            raw_metadata = (
                components.metadata
                if isinstance(components.metadata, Mapping)
                else {}
            )
    except Exception as exc:
        if is_video_backend_unavailable_error(exc):
            raise RuntimeError(VIDEO_READER_BACKEND_UNAVAILABLE_MESSAGE) from exc
        raise RuntimeError(f"failed to decode video: {video_path.name}") from exc

    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise RuntimeError("decoded video frames are invalid")

    total_frames = int(images.shape[0])
    if total_frames <= 0:
        raise ValueError("video has no frames")
    color_info = _log_video_color_diagnostics(video_path)
    if skip_first_frames >= total_frames:
        raise ValueError("skip_first_frames exceeds available frames")

    selected = images[skip_first_frames::select_every_nth]
    if frame_load_cap > 0:
        selected = selected[:frame_load_cap]
    if selected.shape[0] <= 0:
        raise ValueError("no frames selected by skip/select/cap options")
    selected = _convert_decoded_images_to_bt709(selected, color_info)
    frame_count = int(selected.shape[0])

    metadata, encrypted_payloads = _read_container_metadata(video_path, raw_metadata)
    image_info, image_infotext_raw = _read_info_from_metadata(
        metadata,
        key=_IMAGE_INFOTEXT_METADATA_KEY,
        fallback_key=_LEGACY_INFOTEXT_METADATA_KEY,
        encrypted_payload=encrypted_payloads.get("image"),
        file_name=video_path.name,
    )
    video_info, video_infotext_raw = _read_info_from_metadata(
        metadata,
        key=_VIDEO_INFOTEXT_METADATA_KEY,
        encrypted_payload=encrypted_payloads.get("video"),
        file_name=video_path.name,
    )

    return selected, frame_count, image_info, video_info, image_infotext_raw, video_infotext_raw, audio


def _split_images_to_list(images: torch.Tensor) -> list[torch.Tensor]:
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must be a rank-4 IMAGE batch tensor")
    # Keep the IMAGE batch dimension on each list item so downstream nodes that
    # expect [B, H, W, C] tensors continue to work during list expansion.
    return [images[index:index + 1] for index in range(int(images.shape[0]))]


class VideoReader(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        directory_options = _list_video_directories("input")
        default_directory = directory_options[0] if directory_options else "."
        file_options = _list_video_files(default_directory, "input")
        default_file = file_options[0] if file_options else ""

        return c_io.Schema(
            node_id="IPT-VideoReader",
            display_name="Video Reader",
            category=Const.CATEGORY_IMAGEINFO,
            description="Load MP4, WebM, or MKV video frames and metadata from selected directory/file.",
            search_aliases=[
                "video reader",
                "load video path",
                "video file reader",
                "mp4 reader",
                "webm reader",
            ],
            inputs=[
                c_io.Combo.Input(
                    "path_source",
                    options=list(_PATH_SOURCE_OPTIONS),
                    default="input",
                    tooltip="Base directory for directory/file selection",
                ),
                c_io.Combo.Input(
                    "directory",
                    options=directory_options,
                    default=default_directory,
                    tooltip="Directory that contains MP4, WebM, or MKV files (nested path)",
                ),
                c_io.Combo.Input(
                    "file",
                    options=file_options,
                    default=default_file,
                    tooltip="MP4, WebM, or MKV file name in selected directory",
                ),
                c_io.Int.Input(
                    "frame_load_cap",
                    default=0,
                    min=0,
                    tooltip="Maximum number of frames to load, 0 = unlimited",
                ),
                c_io.Int.Input(
                    "skip_first_frames",
                    default=0,
                    min=0,
                    tooltip="Number of frames to skip from the beginning",
                ),
                c_io.Int.Input(
                    "select_every_nth",
                    default=1,
                    min=1,
                    tooltip="Select one frame every N frames",
                ),
            ],
            outputs=[
                c_io.Image.Output(
                    Cast.out_id("images_list"),
                    display_name="image",
                    is_output_list=True,
                ),
                c_io.Audio.Output(
                    Cast.out_id("audio"),
                    display_name="audio",
                ),
                c_io.Int.Output(
                    Cast.out_id("frame_count"),
                    display_name="frame_count",
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
                    Cast.out_id(_IMAGE_INFOTEXT_RAW_OUTPUT_ID),
                    display_name="image_infotext(raw)",
                ),
                c_io.String.Output(
                    Cast.out_id(_VIDEO_INFOTEXT_RAW_OUTPUT_ID),
                    display_name="video_infotext(raw)",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        path_source: str,
        directory: str,
        file: str,
        frame_load_cap: int,
        skip_first_frames: int,
        select_every_nth: int,
    ) -> c_io.NodeOutput:
        video_path = _resolve_video_path(directory, file, path_source)
        images, frame_count, image_info, video_info, image_infotext_raw, video_infotext_raw, audio = _read_video(
            video_path=video_path,
            skip_first_frames=int(skip_first_frames),
            select_every_nth=int(select_every_nth),
            frame_load_cap=int(frame_load_cap),
        )
        return c_io.NodeOutput(
            _split_images_to_list(images),
            audio,
            frame_count,
            image_info,
            video_info,
            image_infotext_raw,
            video_infotext_raw,
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        path_source: str,
        directory: str,
        file: str,
        frame_load_cap: int,
        skip_first_frames: int,
        select_every_nth: int,
    ):
        video_path = _resolve_video_path(directory, file, path_source)
        stat = video_path.stat()
        digest = hashlib.sha256()
        digest.update(str(path_source).encode("utf-8"))
        digest.update(str(directory).encode("utf-8"))
        digest.update(str(file).encode("utf-8"))
        digest.update(str(frame_load_cap).encode("utf-8"))
        digest.update(str(skip_first_frames).encode("utf-8"))
        digest.update(str(select_every_nth).encode("utf-8"))
        digest.update(video_path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        return digest.hexdigest()

    @classmethod
    def validate_inputs(
        cls,
        path_source: str,
        directory: str,
        file: str,
        frame_load_cap: int,
        skip_first_frames: int,
        select_every_nth: int,
    ):
        try:
            _ = _resolve_video_path(directory, file, path_source)
            if int(frame_load_cap) < 0:
                return "frame_load_cap must be >= 0"
            if int(skip_first_frames) < 0:
                return "skip_first_frames must be >= 0"
            if int(select_every_nth) < 1:
                return "select_every_nth must be >= 1"
        except Exception as exc:
            return str(exc)
        return True
