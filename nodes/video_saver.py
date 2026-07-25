# Copyright 2026 kinorax
from __future__ import annotations

import json
import logging
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
from ..utils.metadata_encryption import pack_video_encrypted_payloads, prepare_image_info_metadata
from ..utils.mp4_private_metadata import append_ipt_private_metadata
from ..utils.video_color import (
    AVCOL_RANGE_JPEG,
    AVCOL_SPC_RGB,
    CHROMA_SUBSAMPLING_420,
    CHROMA_SUBSAMPLING_444,
    CHROMA_SUBSAMPLING_OPTIONS,
    COLOR_ENCODING_BT2020,
    COLOR_ENCODING_BT709,
    COLOR_ENCODING_BT709_10BIT,
    COLOR_ENCODING_OPTIONS,
    COLOR_RANGE_LIMITED,
    COLOR_RANGE_OPTIONS,
    FFV1_RGB16_BT709,
    FFV1_RGB8_BT709,
    SDR_BT709_LIMITED,
    VideoColorEncoding,
    convert_rgb_tensor_color_encoding,
    resolve_video_color_encoding,
)
from ..utils.video_runtime_support import VIDEO_SAVER_PYAV_REQUIRED_MESSAGE
from ..utils.webm_metadata import build_matroska_metadata

_OUTPUT_SUBDIR_OPTIONS = ("none", "year", "year_month", "iso_week", "year_month_day")
_CODEC_OPTIONS = ("av1", "h264")
_FFV1_RGB8_CODEC = "ffv1_v3_rgb8"
_FFV1_RGB16_CODEC = "ffv1_v3_rgb16"
_FFV1_COLOR_ENCODINGS = {
    _FFV1_RGB8_CODEC: FFV1_RGB8_BT709,
    _FFV1_RGB16_CODEC: FFV1_RGB16_BT709,
}
_FFV1_CODECS = frozenset(_FFV1_COLOR_ENCODINGS)
_V2_CODEC_OPTIONS = (
    "av1",
    "h264",
    "vp9",
    _FFV1_RGB8_CODEC,
    _FFV1_RGB16_CODEC,
)
_DEFAULT_CODEC = "av1"
_VIDEO_EXT = ".mp4"
_WEBM_EXT = ".webm"
_MKV_EXT = ".mkv"
_MISSING = object()
_SOURCE_IMAGEINFO_DISPLAY_NAME = "source_image_info"
_IMAGE_INFOTEXT_METADATA_KEY = "image_infotext"
_VIDEO_INFOTEXT_METADATA_KEY = "video_infotext"
_AV1_ENCODER = "libsvtav1"
_H264_ENCODER = "libx264"
_VP9_ENCODER = "libvpx-vp9"
_FFV1_ENCODER = "ffv1"
_WEBM_AUDIO_ENCODER = "libopus"
_MKV_AUDIO_ENCODER = "flac"
_AV1_PRESET = 6
_H264_PRESET = "medium"
_VP9_CPU_USED = 4
_WEBM_AUDIO_SAMPLE_RATE = 48000
_WEBM_AUDIO_BIT_RATE = 128000
_PYAV_RGB_SOURCE_PIXEL_FORMAT = "gbrpf32le"
_VIDEO_SAVER_NOTIFICATION_UI_KEY = "ipt_video_saver_notifications"
_ENCODER_BY_CODEC = {
    "av1": _AV1_ENCODER,
    "h264": _H264_ENCODER,
    "vp9": _VP9_ENCODER,
    _FFV1_RGB8_CODEC: _FFV1_ENCODER,
    _FFV1_RGB16_CODEC: _FFV1_ENCODER,
}
_CODEC_DISPLAY_NAMES = {
    "av1": "AV1 (MP4)",
    "h264": "H.264 (MP4)",
    "vp9": "VP9 (WebM)",
    _FFV1_RGB8_CODEC: "FFV1 v3 RGB 8-bit (MKV)",
    _FFV1_RGB16_CODEC: "FFV1 v3 RGB 16-bit (MKV)",
}
_DEFAULT_COLOR_ENCODING = COLOR_ENCODING_BT709
_DEFAULT_COLOR_RANGE = COLOR_RANGE_LIMITED
_DEFAULT_CHROMA_SUBSAMPLING = CHROMA_SUBSAMPLING_420
_LOGGER = logging.getLogger(__name__)
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


def _find_next_counter(folder: Path, date_prefix: str, extension: str = _VIDEO_EXT) -> int:
    current_max = 0
    expected_prefix = f"{date_prefix}-"
    for entry in os.scandir(folder):
        if not entry.is_file():
            continue

        name = entry.name
        if not name.startswith(expected_prefix):
            continue
        if not name.lower().endswith(extension.lower()):
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
    return prepare_image_info_metadata(image_info).infotext


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


def _set_color_properties(
    target: Any,
    color_encoding: VideoColorEncoding,
    *,
    rgb_source: bool = False,
    allow_legacy_frame_metadata: bool = False,
) -> None:
    properties = {
        "color_range": AVCOL_RANGE_JPEG if rgb_source else color_encoding.color_range,
        "color_primaries": color_encoding.color_primaries,
        "color_trc": color_encoding.color_transfer,
        "colorspace": AVCOL_SPC_RGB if rgb_source else color_encoding.color_matrix,
    }
    for name, value in properties.items():
        try:
            setattr(target, name, value)
        except (AttributeError, TypeError, ValueError) as exc:
            if allow_legacy_frame_metadata and name in {"color_primaries", "color_trc"}:
                continue
            raise RuntimeError(
                f"PyAV/FFmpeg build cannot apply required {color_encoding.color_encoding} color metadata"
            ) from exc


def _frame_tensor_to_rgb_array(
    frame_tensor: torch.Tensor,
    color_encoding: VideoColorEncoding,
) -> tuple[np.ndarray, str]:
    rgb_tensor = frame_tensor[..., :3].detach()
    if color_encoding.color_encoding == COLOR_ENCODING_BT2020:
        rgb_tensor = convert_rgb_tensor_color_encoding(
            rgb_tensor,
            COLOR_ENCODING_BT709,
            COLOR_ENCODING_BT2020,
        )

    frame_np = (
        rgb_tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .cpu()
        .contiguous()
        .numpy()
    )
    return frame_np, _PYAV_RGB_SOURCE_PIXEL_FORMAT


def _reformat_video_frame(video_frame: Any, color_encoding: VideoColorEncoding) -> Any:
    _set_color_properties(
        video_frame,
        color_encoding,
        rgb_source=True,
        allow_legacy_frame_metadata=True,
    )

    try:
        reformatted = video_frame.reformat(
            format=color_encoding.pixel_format,
            src_colorspace=color_encoding.color_matrix,
            dst_colorspace=color_encoding.color_matrix,
            src_color_range=AVCOL_RANGE_JPEG,
            dst_color_range=color_encoding.color_range,
            dst_color_trc=color_encoding.color_transfer,
            dst_color_primaries=color_encoding.color_primaries,
        )
    except TypeError:
        try:
            # PyAV 16.0 exposes explicit range conversion but not the
            # destination transfer/primaries keywords added in 16.1.
            reformatted = video_frame.reformat(
                format=color_encoding.pixel_format,
                src_colorspace=color_encoding.color_matrix,
                dst_colorspace=color_encoding.color_matrix,
                src_color_range=AVCOL_RANGE_JPEG,
                dst_color_range=color_encoding.color_range,
            )
        except TypeError:
            # Retain the matrix conversion on older optional PyAV installs.
            reformatted = video_frame.reformat(
                format=color_encoding.pixel_format,
                src_colorspace=color_encoding.color_matrix,
                dst_colorspace=color_encoding.color_matrix,
            )

    _set_color_properties(
        reformatted,
        color_encoding,
        allow_legacy_frame_metadata=True,
    )
    return reformatted


def _reformat_bt709_limited(video_frame: Any) -> Any:
    return _reformat_video_frame(video_frame, SDR_BT709_LIMITED)


def _reformat_ffv1_rgb_frame(
    video_frame: Any,
    color_encoding: VideoColorEncoding,
) -> Any:
    _set_color_properties(
        video_frame,
        color_encoding,
        rgb_source=True,
        allow_legacy_frame_metadata=True,
    )
    reformatted = video_frame.reformat(format=color_encoding.pixel_format)
    _set_color_properties(
        reformatted,
        color_encoding,
        allow_legacy_frame_metadata=True,
    )
    return reformatted


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
    color_encoding: VideoColorEncoding = SDR_BT709_LIMITED,
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
    if (
        color_encoding.chroma_subsampling == CHROMA_SUBSAMPLING_420
        and ((width % 2) != 0 or (height % 2) != 0)
    ):
        raise ValueError(
            f"images width and height must be even for {codec_name} {color_encoding.pixel_format} encoding"
        )

    rate = _resolve_frame_rate_fraction(frame_rate)

    target_samples = 0

    try:
        with av.open(
            str(path),
            mode="w",
            options={"movflags": "use_metadata_tags+write_colr"},
        ) as container:
            for key, value in metadata.items():
                container.metadata[str(key)] = _serialize_metadata_value(value)

            try:
                video_stream = container.add_stream(encoder_name, rate=rate)
            except Exception as exc:
                raise RuntimeError(f"{codec_name} encoder `{encoder_name}` is unavailable in this PyAV/FFmpeg build") from exc

            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = color_encoding.pixel_format
            video_stream.bit_rate = 0
            _set_color_properties(video_stream.codec_context, color_encoding)
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
                frame_np, rgb_format = _frame_tensor_to_rgb_array(frame_tensor, color_encoding)
                video_frame = av.VideoFrame.from_ndarray(frame_np, format=rgb_format)
                video_frame = _reformat_video_frame(video_frame, color_encoding)
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
    color_encoding: VideoColorEncoding = SDR_BT709_LIMITED,
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
        color_encoding=color_encoding,
    )


def _encode_h264_mp4(
    path: Path,
    frames: Sequence[torch.Tensor],
    frame_rate: float,
    crf: int,
    preset: str,
    audio: Any,
    metadata: Mapping[str, Any],
    color_encoding: VideoColorEncoding = SDR_BT709_LIMITED,
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
        color_encoding=color_encoding,
    )


def _encode_vp9_webm(
    path: Path,
    frames: Sequence[torch.Tensor],
    frame_rate: float,
    crf: int,
    cpu_used: int,
    audio: Any,
    metadata: Mapping[str, Any],
    color_encoding: VideoColorEncoding = SDR_BT709_LIMITED,
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
    if (
        color_encoding.chroma_subsampling == CHROMA_SUBSAMPLING_420
        and ((width % 2) != 0 or (height % 2) != 0)
    ):
        raise ValueError(
            f"images width and height must be even for VP9 {color_encoding.pixel_format} encoding"
        )

    rate = _resolve_frame_rate_fraction(frame_rate)
    target_samples = 0

    try:
        with av.open(str(path), mode="w") as container:
            for key, value in metadata.items():
                container.metadata[str(key)] = _serialize_metadata_value(value)

            try:
                video_stream = container.add_stream(_VP9_ENCODER, rate=rate)
            except Exception as exc:
                raise RuntimeError(
                    f"VP9 encoder `{_VP9_ENCODER}` is unavailable in this PyAV/FFmpeg build"
                ) from exc

            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = color_encoding.pixel_format
            video_stream.bit_rate = 0
            _set_color_properties(video_stream.codec_context, color_encoding)
            video_stream.options = {
                "crf": str(int(crf)),
                "deadline": "good",
                "cpu-used": str(int(cpu_used)),
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
                _, _, audio_layout = audio_payload
                try:
                    audio_stream = container.add_stream(
                        _WEBM_AUDIO_ENCODER,
                        rate=_WEBM_AUDIO_SAMPLE_RATE,
                        layout=audio_layout,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"WebM audio encoder `{_WEBM_AUDIO_ENCODER}` is unavailable "
                        "in this PyAV/FFmpeg build"
                    ) from exc
                audio_stream.bit_rate = _WEBM_AUDIO_BIT_RATE

            for frame_tensor in frames:
                frame_np, rgb_format = _frame_tensor_to_rgb_array(frame_tensor, color_encoding)
                video_frame = av.VideoFrame.from_ndarray(frame_np, format=rgb_format)
                video_frame = _reformat_video_frame(video_frame, color_encoding)
                for packet in video_stream.encode(video_frame):
                    container.mux(packet)

            for packet in video_stream.encode(None):
                container.mux(packet)

            if audio_stream is not None and audio_payload is not None:
                audio_np, audio_sample_rate, audio_layout = audio_payload
                audio_frame = av.AudioFrame.from_ndarray(audio_np, format="fltp", layout=audio_layout)
                audio_frame.sample_rate = audio_sample_rate
                audio_frame.pts = 0
                audio_frame.time_base = Fraction(1, audio_sample_rate)

                if audio_sample_rate == _WEBM_AUDIO_SAMPLE_RATE:
                    resampled_frames = [audio_frame]
                    audio_resampler = None
                else:
                    audio_resampler = av.audio.resampler.AudioResampler(
                        format="fltp",
                        layout=audio_layout,
                        rate=_WEBM_AUDIO_SAMPLE_RATE,
                    )
                    resampled_frames = audio_resampler.resample(audio_frame)

                for resampled_frame in resampled_frames:
                    for packet in audio_stream.encode(resampled_frame):
                        container.mux(packet)
                if audio_resampler is not None:
                    for resampled_frame in audio_resampler.resample(None):
                        for packet in audio_stream.encode(resampled_frame):
                            container.mux(packet)
                for packet in audio_stream.encode(None):
                    container.mux(packet)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"failed to save video: {path}") from exc


def _encode_ffv1_mkv(
    path: Path,
    frames: Sequence[torch.Tensor],
    frame_rate: float,
    audio: Any,
    metadata: Mapping[str, Any],
    color_encoding: VideoColorEncoding,
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

    rate = _resolve_frame_rate_fraction(frame_rate)
    target_samples = 0

    try:
        with av.open(str(path), mode="w", format="matroska") as container:
            for key, value in metadata.items():
                container.metadata[str(key)] = _serialize_metadata_value(value)

            try:
                video_stream = container.add_stream(_FFV1_ENCODER, rate=rate)
            except Exception as exc:
                raise RuntimeError(
                    f"FFV1 encoder `{_FFV1_ENCODER}` is unavailable in this PyAV/FFmpeg build"
                ) from exc

            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = color_encoding.pixel_format
            video_stream.bit_rate = 0
            video_stream.gop_size = 300
            _set_color_properties(video_stream.codec_context, color_encoding)
            video_stream.options = {
                "level": "3",
                "coder": "2",
                "context": "1",
                "slices": "4",
                "slicecrc": "1",
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
                try:
                    audio_stream = container.add_stream(
                        _MKV_AUDIO_ENCODER,
                        rate=audio_sample_rate,
                        layout=audio_layout,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"MKV audio encoder `{_MKV_AUDIO_ENCODER}` is unavailable "
                        "in this PyAV/FFmpeg build"
                    ) from exc

            for frame_tensor in frames:
                frame_np, rgb_format = _frame_tensor_to_rgb_array(
                    frame_tensor,
                    color_encoding,
                )
                video_frame = av.VideoFrame.from_ndarray(frame_np, format=rgb_format)
                video_frame = _reformat_ffv1_rgb_frame(
                    video_frame,
                    color_encoding,
                )
                for packet in video_stream.encode(video_frame):
                    container.mux(packet)

            for packet in video_stream.encode(None):
                container.mux(packet)

            if audio_stream is not None and audio_payload is not None:
                audio_np, audio_sample_rate, audio_layout = audio_payload
                audio_frame = av.AudioFrame.from_ndarray(
                    audio_np,
                    format="fltp",
                    layout=audio_layout,
                )
                audio_frame.sample_rate = audio_sample_rate
                audio_frame.pts = 0
                audio_frame.time_base = Fraction(1, audio_sample_rate)
                audio_resampler = av.audio.resampler.AudioResampler(
                    format="s32",
                    layout=audio_layout,
                    rate=audio_sample_rate,
                )
                for resampled_frame in audio_resampler.resample(audio_frame):
                    for packet in audio_stream.encode(resampled_frame):
                        container.mux(packet)
                for resampled_frame in audio_resampler.resample(None):
                    for packet in audio_stream.encode(resampled_frame):
                        container.mux(packet)
                for packet in audio_stream.encode(None):
                    container.mux(packet)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"failed to save video: {path}") from exc


def _get_encoder_supported_pixel_formats(encoder_name: str) -> set[str] | None:
    try:
        import av

        video_formats = av.Codec(encoder_name, "w").video_formats
    except Exception:
        return None

    if not video_formats:
        return None

    supported_formats = {
        str(format_name).strip().lower()
        for video_format in video_formats
        if (format_name := getattr(video_format, "name", None))
    }
    return supported_formats or None


def _video_color_fallback_candidates(
    requested: VideoColorEncoding,
) -> list[VideoColorEncoding]:
    range_name = requested.as_metadata()["color_range"]
    combinations: list[tuple[str, str]] = []
    requested_option = _video_color_encoding_option(requested)

    if requested.chroma_subsampling == CHROMA_SUBSAMPLING_444:
        combinations.append(
            (requested_option, CHROMA_SUBSAMPLING_420)
        )
    if requested.bit_depth > 8:
        combinations.append(
            (COLOR_ENCODING_BT709, requested.chroma_subsampling)
        )
        if requested.chroma_subsampling == CHROMA_SUBSAMPLING_444:
            combinations.append(
                (COLOR_ENCODING_BT709, CHROMA_SUBSAMPLING_420)
            )

    candidates: list[VideoColorEncoding] = []
    seen_pixel_formats = {requested.pixel_format}
    for color_encoding, chroma_subsampling in combinations:
        candidate = resolve_video_color_encoding(
            color_encoding,
            range_name,
            chroma_subsampling,
        )
        if candidate.pixel_format in seen_pixel_formats:
            continue
        seen_pixel_formats.add(candidate.pixel_format)
        candidates.append(candidate)
    return candidates


def _video_color_encoding_option(color_encoding: VideoColorEncoding) -> str:
    if (
        color_encoding.color_encoding == COLOR_ENCODING_BT709
        and color_encoding.bit_depth == 10
    ):
        return COLOR_ENCODING_BT709_10BIT
    return color_encoding.color_encoding


def _format_video_color_selection(color_encoding: VideoColorEncoding) -> str:
    range_label = (
        "Full"
        if color_encoding.as_metadata()["color_range"] == "full"
        else "Limited"
    )
    return (
        f"{_video_color_encoding_option(color_encoding)} / {range_label} / "
        f"{color_encoding.chroma_subsampling}"
    )


def _build_video_saver_fallback_notification(
    codec_value: str,
    requested: VideoColorEncoding,
    effective: VideoColorEncoding,
) -> dict[str, str]:
    encoder_name = _ENCODER_BY_CODEC[codec_value]
    codec_name = _CODEC_DISPLAY_NAMES[codec_value]
    return {
        "summary": "Video Saver: fallback applied",
        "detail": (
            f"{codec_name} encoder `{encoder_name}` does not support "
            f"{requested.pixel_format} in the current PyAV/FFmpeg build. "
            f"Requested: {_format_video_color_selection(requested)}. "
            f"Saved: {_format_video_color_selection(effective)}."
        ),
    }


def _resolve_supported_video_color_encoding(
    codec_value: str,
    requested: VideoColorEncoding,
) -> tuple[VideoColorEncoding, dict[str, str] | None]:
    encoder_name = _ENCODER_BY_CODEC[codec_value]
    supported_formats = _get_encoder_supported_pixel_formats(encoder_name)
    if supported_formats is None:
        # libsvtav1 is known not to accept 4:4:4. Preserve the established
        # compatibility fallback when codec capability introspection is absent.
        if (
            codec_value == "av1"
            and requested.chroma_subsampling == CHROMA_SUBSAMPLING_444
        ):
            effective = resolve_video_color_encoding(
                _video_color_encoding_option(requested),
                requested.as_metadata()["color_range"],
                CHROMA_SUBSAMPLING_420,
            )
            return (
                effective,
                _build_video_saver_fallback_notification(
                    codec_value,
                    requested,
                    effective,
                ),
            )
        return requested, None

    if requested.pixel_format.lower() in supported_formats:
        return requested, None

    for candidate in _video_color_fallback_candidates(requested):
        if candidate.pixel_format.lower() in supported_formats:
            return (
                candidate,
                _build_video_saver_fallback_notification(
                    codec_value,
                    requested,
                    candidate,
                ),
            )

    supported_list = ", ".join(sorted(supported_formats))
    raise RuntimeError(
        f"{_CODEC_DISPLAY_NAMES[codec_value]} encoder `{encoder_name}` does not "
        f"support requested pixel format {requested.pixel_format}, and no safe "
        f"fallback is available (supported: {supported_list})"
    )


class VideoSaver(c_io.ComfyNode):
    _include_loop_count_metadata = True
    _codec_options = _CODEC_OPTIONS

    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-VideoSaver",
            display_name="Video Saver (Deprecated)",
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
                    display_name="codec",
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
                    default=16.0,
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
        return cls._execute_common(
            images=images,
            image_info=image_info,
            audio=audio,
            video_info=video_info,
            filename_suffix=filename_suffix,
            file_stem=file_stem,
            output_dir=output_dir,
            output_subdir=output_subdir,
            codec=codec,
            av1_crf=av1_crf,
            h264_crf=h264_crf,
            vp9_crf=_MISSING,
            frame_rate=frame_rate,
            loop_count=loop_count,
            pingpong=pingpong,
            color_encoding=_DEFAULT_COLOR_ENCODING,
            color_range=_DEFAULT_COLOR_RANGE,
            chroma_subsampling=_DEFAULT_CHROMA_SUBSAMPLING,
        )

    @classmethod
    def _execute_common(
        cls,
        images: Any,
        image_info: Any,
        audio: Any,
        video_info: Any,
        filename_suffix: Any,
        file_stem: Any,
        output_dir: Any,
        output_subdir: Any,
        codec: Any,
        av1_crf: Any,
        h264_crf: Any,
        vp9_crf: Any,
        frame_rate: Any,
        loop_count: Any,
        pingpong: Any,
        color_encoding: Any,
        color_range: Any,
        chroma_subsampling: Any,
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
        if codec_value == "vp9":
            if vp9_crf is _MISSING:
                raise ValueError("vp9_crf is required")
            vp9_crf_value = int(_resolve_single_input(vp9_crf, "vp9_crf"))
        else:
            vp9_crf_value = None
        frame_rate_value = _normalize_frame_rate_value(_resolve_single_input(frame_rate, "frame_rate"))
        loop_count_value = int(_resolve_single_input(loop_count, "loop_count"))
        pingpong_value = bool(_resolve_single_input(pingpong, "pingpong"))
        if codec_value not in cls._codec_options:
            raise ValueError(f"unsupported codec: {codec_value}")
        if codec_value in _FFV1_CODECS:
            effective_color_encoding = _FFV1_COLOR_ENCODINGS[codec_value]
            fallback_notification = None
        else:
            requested_color_encoding = resolve_video_color_encoding(
                _resolve_single_input(color_encoding, "color_encoding"),
                _resolve_single_input(color_range, "color_range"),
                _resolve_single_input(chroma_subsampling, "chroma_subsampling"),
            )
            effective_color_encoding, fallback_notification = (
                _resolve_supported_video_color_encoding(
                    codec_value,
                    requested_color_encoding,
                )
            )
        suffix = ""
        if forced_file_stem is None:
            suffix = _safe_filename_suffix(_resolve_single_input(filename_suffix, "filename_suffix"))

        if output_subdir_value not in _OUTPUT_SUBDIR_OPTIONS:
            raise ValueError(f"unsupported output_subdir: {output_subdir_value}")
        if av1_crf_value < 0 or av1_crf_value > 63:
            raise ValueError("av1_crf must be in range 0..63")
        if h264_crf_value < 0 or h264_crf_value > 51:
            raise ValueError("h264_crf must be in range 0..51")
        if vp9_crf_value is not None and (vp9_crf_value < 0 or vp9_crf_value > 63):
            raise ValueError("vp9_crf must be in range 0..63")
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
        video_extension = {
            "vp9": _WEBM_EXT,
            _FFV1_RGB8_CODEC: _MKV_EXT,
            _FFV1_RGB16_CODEC: _MKV_EXT,
        }.get(codec_value, _VIDEO_EXT)
        if forced_file_stem is None:
            counter = _find_next_counter(target_dir, date_prefix, video_extension)
            stem = _render_file_stem(date_prefix, counter, suffix)
        else:
            stem = forced_file_stem
        file_path = target_dir / f"{stem}{video_extension}"

        metadata: dict[str, Any] = {}
        image_encrypted_payload: bytes | None = None
        video_encrypted_payload: bytes | None = None
        if has_image_info:
            image_prepared = prepare_image_info_metadata(image_info_value)
            image_infotext = image_prepared.infotext
            image_encrypted_payload = image_prepared.encrypted_payload
            if image_infotext:
                metadata[_IMAGE_INFOTEXT_METADATA_KEY] = image_infotext
        if has_video_info:
            video_prepared = prepare_image_info_metadata(video_info_value)
            video_infotext = video_prepared.infotext
            video_encrypted_payload = video_prepared.encrypted_payload
            if video_infotext:
                metadata[_VIDEO_INFOTEXT_METADATA_KEY] = video_infotext
        encrypted_metadata_payload = pack_video_encrypted_payloads(
            image_payload=image_encrypted_payload,
            video_payload=video_encrypted_payload,
        )
        loop_count_metadata = (
            {"loop_count": loop_count_value}
            if cls._include_loop_count_metadata
            else {}
        )
        video_saver_metadata = {
            "codec": codec_value,
            "frame_rate": frame_rate_value,
            **loop_count_metadata,
            "pingpong": pingpong_value,
            "frame_count": len(frame_indices),
            "encoder": _ENCODER_BY_CODEC[codec_value],
            **effective_color_encoding.as_metadata(),
        }
        metadata["video_saver"] = video_saver_metadata

        if codec_value == "av1":
            _encode_av1_mp4(
                path=file_path,
                frames=render_frames,
                frame_rate=frame_rate_value,
                crf=av1_crf_value,
                preset=_AV1_PRESET,
                audio=audio_value,
                metadata=metadata,
                color_encoding=effective_color_encoding,
            )
        elif codec_value == "h264":
            _encode_h264_mp4(
                path=file_path,
                frames=render_frames,
                frame_rate=frame_rate_value,
                crf=h264_crf_value,
                preset=_H264_PRESET,
                audio=audio_value,
                metadata=metadata,
                color_encoding=effective_color_encoding,
            )
        elif codec_value == "vp9":
            _encode_vp9_webm(
                path=file_path,
                frames=render_frames,
                frame_rate=frame_rate_value,
                crf=vp9_crf_value,
                cpu_used=_VP9_CPU_USED,
                audio=audio_value,
                metadata=build_matroska_metadata(metadata, encrypted_metadata_payload),
                color_encoding=effective_color_encoding,
            )
        else:
            _encode_ffv1_mkv(
                path=file_path,
                frames=render_frames,
                frame_rate=frame_rate_value,
                audio=audio_value,
                metadata=build_matroska_metadata(metadata, encrypted_metadata_payload),
                color_encoding=effective_color_encoding,
            )

        if (
            codec_value not in {"vp9", *_FFV1_CODECS}
            and encrypted_metadata_payload is not None
        ):
            append_ipt_private_metadata(file_path, encrypted_metadata_payload)

        rel_file_path = _relative_to_output_root(file_path, output_root)
        rel_parent = file_path.parent.resolve().relative_to(output_root).as_posix()
        # Preview the saved file itself instead of generating a preview-only asset.
        ui_result = c_ui.SavedResult(
            file_path.name,
            "" if rel_parent == "." else rel_parent,
            c_io.FolderType.output,
        )
        preview_video = c_ui.PreviewVideo([ui_result])
        if fallback_notification is None:
            ui_output: c_ui.PreviewVideo | dict[str, Any] = preview_video
        else:
            fallback_notification["detail"] += f" Saved as {rel_file_path}."
            _LOGGER.warning(
                "[IPT] Video Saver: %s",
                fallback_notification["detail"],
            )
            ui_output = preview_video.as_dict()
            ui_output[_VIDEO_SAVER_NOTIFICATION_UI_KEY] = [
                fallback_notification
            ]

        images_out_values = _unwrap_input_list(images)
        return c_io.NodeOutput(
            images_out_values,
            audio_value,
            image_info_output_value,
            video_info_output_value,
            rel_file_path,
            ui=ui_output,
        )


class VideoSaverV2(VideoSaver):
    _include_loop_count_metadata = False
    _codec_options = _V2_CODEC_OPTIONS

    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-VideoSaverV2",
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
                    "file_stem",
                    optional=True,
                    force_input=True,
                    tooltip=(
                        "Optional output filename stem; when connected, save with the "
                        "extension selected by codec"
                    ),
                ),
                c_io.String.Input(
                    "filename_suffix",
                    default="video",
                    tooltip="Suffix part of output filename",
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
                c_io.Float.Input(
                    "frame_rate",
                    default=16.0,
                    min=0.001,
                    max=1000.0,
                    step=1.0,
                    tooltip="Output frame rate (supports up to 3 decimal places)",
                ),
                c_io.Boolean.Input(
                    "pingpong",
                    default=False,
                    tooltip="Play forward then backward (excluding first and last duplicated endpoints)",
                ),
                c_io.Combo.Input(
                    "codec",
                    options=list(_V2_CODEC_OPTIONS),
                    default=_DEFAULT_CODEC,
                    tooltip="Output video codec and container",
                ),
                c_io.Combo.Input(
                    "color_encoding",
                    options=list(COLOR_ENCODING_OPTIONS),
                    default=_DEFAULT_COLOR_ENCODING,
                    tooltip=(
                        "SDR color encoding and bit depth; BT.709 uses 8-bit, "
                        "BT.709 (10-bit) and BT.2020 use 10-bit"
                    ),
                ),
                c_io.Combo.Input(
                    "color_range",
                    options=list(COLOR_RANGE_OPTIONS),
                    default=_DEFAULT_COLOR_RANGE,
                    tooltip="YUV signal range; limited is the compatibility-oriented default",
                ),
                c_io.Combo.Input(
                    "chroma_subsampling",
                    options=list(CHROMA_SUBSAMPLING_OPTIONS),
                    default=_DEFAULT_CHROMA_SUBSAMPLING,
                    tooltip="Chroma subsampling; unsupported encoder combinations use a compatible fallback",
                ),
                c_io.Int.Input(
                    "av1_crf",
                    display_name="crf",
                    default=23,
                    min=0,
                    max=63,
                    tooltip="AV1 CRF (0..63, lower is higher quality)",
                ),
                c_io.Int.Input(
                    "h264_crf",
                    display_name="crf",
                    default=19,
                    min=0,
                    max=51,
                    tooltip="H.264 CRF (0..51, lower is higher quality)",
                ),
                c_io.Int.Input(
                    "vp9_crf",
                    display_name="crf",
                    default=20,
                    min=0,
                    max=63,
                    tooltip="VP9 CRF (0..63, lower is higher quality)",
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
        file_stem: Any = None,
        filename_suffix: Any = _MISSING,
        output_dir: Any = _MISSING,
        output_subdir: Any = _MISSING,
        frame_rate: Any = _MISSING,
        pingpong: Any = _MISSING,
        codec: Any = _MISSING,
        color_encoding: Any = _DEFAULT_COLOR_ENCODING,
        color_range: Any = _DEFAULT_COLOR_RANGE,
        chroma_subsampling: Any = _DEFAULT_CHROMA_SUBSAMPLING,
        av1_crf: Any = _MISSING,
        h264_crf: Any = _MISSING,
        vp9_crf: Any = _MISSING,
    ) -> c_io.NodeOutput:
        return cls._execute_common(
            images=images,
            image_info=image_info,
            audio=audio,
            video_info=video_info,
            filename_suffix=filename_suffix,
            file_stem=file_stem,
            output_dir=output_dir,
            output_subdir=output_subdir,
            codec=codec,
            av1_crf=av1_crf,
            h264_crf=h264_crf,
            vp9_crf=vp9_crf,
            frame_rate=frame_rate,
            loop_count=0,
            pingpong=pingpong,
            color_encoding=color_encoding,
            color_range=color_range,
            chroma_subsampling=chroma_subsampling,
        )
