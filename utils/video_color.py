# Copyright 2026 kinorax
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# FFmpeg AVColor* enum values follow the codec-independent H.273 code points.
AVCOL_RANGE_UNSPECIFIED = 0
AVCOL_RANGE_MPEG = 1
AVCOL_RANGE_JPEG = 2

AVCOL_PRI_BT709 = 1
AVCOL_PRI_UNSPECIFIED = 2
AVCOL_PRI_BT2020 = 9

AVCOL_TRC_BT709 = 1
AVCOL_TRC_UNSPECIFIED = 2
AVCOL_TRC_BT2020_10 = 14

AVCOL_SPC_RGB = 0
AVCOL_SPC_BT709 = 1
AVCOL_SPC_UNSPECIFIED = 2
AVCOL_SPC_BT2020_NCL = 9

COLOR_ENCODING_BT709 = "BT.709"
COLOR_ENCODING_BT709_10BIT = "BT.709 (10-bit)"
COLOR_ENCODING_BT2020 = "BT.2020"
COLOR_ENCODING_UNKNOWN = "unknown"
COLOR_ENCODING_OPTIONS = (
    COLOR_ENCODING_BT709,
    COLOR_ENCODING_BT709_10BIT,
    COLOR_ENCODING_BT2020,
)

COLOR_RANGE_LIMITED = "limited"
COLOR_RANGE_FULL = "full"
COLOR_RANGE_OPTIONS = (COLOR_RANGE_LIMITED, COLOR_RANGE_FULL)

CHROMA_SUBSAMPLING_420 = "4:2:0"
CHROMA_SUBSAMPLING_444 = "4:4:4"
CHROMA_SUBSAMPLING_UNKNOWN = "unknown"
CHROMA_SUBSAMPLING_OPTIONS = (CHROMA_SUBSAMPLING_420, CHROMA_SUBSAMPLING_444)

_COLOR_RANGE_NAMES = {
    AVCOL_RANGE_UNSPECIFIED: "unspecified",
    AVCOL_RANGE_MPEG: "limited",
    AVCOL_RANGE_JPEG: "full",
}
_COLOR_PRIMARIES_NAMES = {
    0: "reserved",
    AVCOL_PRI_BT709: "bt709",
    AVCOL_PRI_UNSPECIFIED: "unspecified",
    4: "bt470m",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "film",
    9: "bt2020",
    10: "smpte428",
    11: "smpte431",
    12: "smpte432",
    22: "ebu3213",
}
_COLOR_TRANSFER_NAMES = {
    0: "reserved",
    AVCOL_TRC_BT709: "bt709",
    AVCOL_TRC_UNSPECIFIED: "unspecified",
    4: "gamma22",
    5: "gamma28",
    6: "smpte170m",
    7: "smpte240m",
    8: "linear",
    9: "log100",
    10: "log316",
    11: "iec61966-2-4",
    12: "bt1361",
    13: "srgb",
    14: "bt2020-10",
    15: "bt2020-12",
    16: "smpte2084",
    17: "smpte428",
    18: "arib-std-b67",
}
_COLORSPACE_NAMES = {
    AVCOL_SPC_RGB: "rgb",
    AVCOL_SPC_BT709: "bt709",
    AVCOL_SPC_UNSPECIFIED: "unspecified",
    4: "fcc",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "ycocg",
    9: "bt2020nc",
    10: "bt2020c",
    11: "smpte2085",
    12: "chroma-derived-nc",
    13: "chroma-derived-c",
    14: "ictcp",
}


@dataclass(frozen=True)
class VideoColorEncoding:
    color_encoding: str
    chroma_subsampling: str
    pixel_format: str
    bit_depth: int
    color_range: int
    color_primaries: int
    color_transfer: int
    color_matrix: int

    def as_metadata(self) -> dict[str, Any]:
        return {
            "color_encoding": self.color_encoding,
            "chroma_subsampling": self.chroma_subsampling,
            "pixel_format": self.pixel_format,
            "bit_depth": self.bit_depth,
            "color_range": color_range_name(self.color_range),
            "color_primaries": color_primaries_name(self.color_primaries),
            "color_transfer": color_transfer_name(self.color_transfer),
            "color_matrix": colorspace_name(self.color_matrix),
        }


SDR_BT709_LIMITED = VideoColorEncoding(
    color_encoding=COLOR_ENCODING_BT709,
    chroma_subsampling=CHROMA_SUBSAMPLING_420,
    pixel_format="yuv420p",
    bit_depth=8,
    color_range=AVCOL_RANGE_MPEG,
    color_primaries=AVCOL_PRI_BT709,
    color_transfer=AVCOL_TRC_BT709,
    color_matrix=AVCOL_SPC_BT709,
)

FFV1_RGB8_BT709 = VideoColorEncoding(
    color_encoding=COLOR_ENCODING_BT709,
    chroma_subsampling=CHROMA_SUBSAMPLING_444,
    pixel_format="bgr0",
    bit_depth=8,
    color_range=AVCOL_RANGE_JPEG,
    color_primaries=AVCOL_PRI_BT709,
    color_transfer=AVCOL_TRC_BT709,
    color_matrix=AVCOL_SPC_RGB,
)

FFV1_RGB16_BT709 = VideoColorEncoding(
    color_encoding=COLOR_ENCODING_BT709,
    chroma_subsampling=CHROMA_SUBSAMPLING_444,
    pixel_format="gbrp16le",
    bit_depth=16,
    color_range=AVCOL_RANGE_JPEG,
    color_primaries=AVCOL_PRI_BT709,
    color_transfer=AVCOL_TRC_BT709,
    color_matrix=AVCOL_SPC_RGB,
)


def normalize_color_encoding_option(value: Any) -> str:
    normalized = (
        str(value or "")
        .strip()
        .lower()
        .replace("_", "")
        .replace(".", "")
        .replace(" ", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
    )
    if normalized == "bt709":
        return COLOR_ENCODING_BT709
    if normalized in {"bt70910", "bt70910bit"}:
        return COLOR_ENCODING_BT709_10BIT
    if normalized == "bt2020":
        return COLOR_ENCODING_BT2020
    raise ValueError(f"unsupported color_encoding: {value}")


def normalize_color_encoding(value: Any) -> str:
    option_name = normalize_color_encoding_option(value)
    if option_name == COLOR_ENCODING_BT709_10BIT:
        return COLOR_ENCODING_BT709
    return option_name


def normalize_color_range(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in COLOR_RANGE_OPTIONS:
        return normalized
    raise ValueError(f"unsupported color_range: {value}")


def normalize_chroma_subsampling(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace(":", "")
    if normalized in {"420", "yuv420p"}:
        return CHROMA_SUBSAMPLING_420
    if normalized in {"444", "yuv444p"}:
        return CHROMA_SUBSAMPLING_444
    raise ValueError(f"unsupported chroma_subsampling: {value}")


def resolve_video_color_encoding(
    color_encoding: Any,
    color_range: Any,
    chroma_subsampling: Any,
) -> VideoColorEncoding:
    option_name = normalize_color_encoding_option(color_encoding)
    encoding_name = normalize_color_encoding(option_name)
    range_name = normalize_color_range(color_range)
    chroma_name = normalize_chroma_subsampling(chroma_subsampling)

    bit_depth = 8 if option_name == COLOR_ENCODING_BT709 else 10
    chroma_suffix = "420" if chroma_name == CHROMA_SUBSAMPLING_420 else "444"
    depth_suffix = "" if bit_depth == 8 else "10le"
    pixel_format = f"yuv{chroma_suffix}p{depth_suffix}"

    if encoding_name == COLOR_ENCODING_BT709:
        primaries = AVCOL_PRI_BT709
        transfer = AVCOL_TRC_BT709
        matrix = AVCOL_SPC_BT709
    else:
        primaries = AVCOL_PRI_BT2020
        transfer = AVCOL_TRC_BT2020_10
        matrix = AVCOL_SPC_BT2020_NCL

    return VideoColorEncoding(
        color_encoding=encoding_name,
        chroma_subsampling=chroma_name,
        pixel_format=pixel_format,
        bit_depth=bit_depth,
        color_range=AVCOL_RANGE_MPEG if range_name == COLOR_RANGE_LIMITED else AVCOL_RANGE_JPEG,
        color_primaries=primaries,
        color_transfer=transfer,
        color_matrix=matrix,
    )


def _enum_value(value: Any) -> int | None:
    if value is None:
        return None
    raw_value = getattr(value, "value", value)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _enum_name(value: Any, names: Mapping[int, str]) -> str:
    numeric_value = _enum_value(value)
    if numeric_value is None:
        return "unspecified"
    return names.get(numeric_value, f"unknown({numeric_value})")


def color_range_name(value: Any) -> str:
    return _enum_name(value, _COLOR_RANGE_NAMES)


def color_primaries_name(value: Any) -> str:
    return _enum_name(value, _COLOR_PRIMARIES_NAMES)


def color_transfer_name(value: Any) -> str:
    return _enum_name(value, _COLOR_TRANSFER_NAMES)


def colorspace_name(value: Any) -> str:
    return _enum_name(value, _COLORSPACE_NAMES)


def infer_color_encoding(color_primaries: Any, color_matrix: Any) -> str:
    primaries = str(color_primaries or "")
    matrix = str(color_matrix or "")
    if primaries == "bt2020" or matrix in {"bt2020nc", "bt2020c"}:
        return COLOR_ENCODING_BT2020
    if primaries == "bt709" or matrix == "bt709":
        return COLOR_ENCODING_BT709
    return COLOR_ENCODING_UNKNOWN


def infer_chroma_subsampling(pixel_format: Any) -> str:
    normalized = str(pixel_format or "").strip().lower()
    if "420" in normalized or normalized.startswith(("nv12", "nv21", "p010", "p012", "p016")):
        return CHROMA_SUBSAMPLING_420
    if "444" in normalized or normalized.startswith(("rgb", "bgr", "gbr")):
        return CHROMA_SUBSAMPLING_444
    return CHROMA_SUBSAMPLING_UNKNOWN


def _video_format_bit_depth(video_format: Any) -> int | None:
    components = getattr(video_format, "components", None)
    if components is None:
        return None

    bit_depths: list[int] = []
    for component in components:
        try:
            bits = int(getattr(component, "bits"))
        except (AttributeError, TypeError, ValueError):
            continue
        if bits > 0:
            bit_depths.append(bits)
    return max(bit_depths) if bit_depths else None


def describe_video_stream_color(stream: Any) -> dict[str, Any]:
    codec_context = getattr(stream, "codec_context", None)
    if codec_context is None:
        raise ValueError("video stream has no codec context")

    codec = getattr(codec_context, "codec", None)
    codec_name = getattr(codec, "name", None) or getattr(codec_context, "name", None) or "unknown"

    video_format = getattr(codec_context, "format", None)
    pixel_format = getattr(codec_context, "pix_fmt", None)
    if not pixel_format and video_format is not None:
        pixel_format = getattr(video_format, "name", None)

    color_primaries = color_primaries_name(getattr(codec_context, "color_primaries", None))
    color_matrix = colorspace_name(getattr(codec_context, "colorspace", None))

    return {
        "codec": str(codec_name),
        "pixel_format": str(pixel_format or "unknown"),
        "bit_depth": _video_format_bit_depth(video_format),
        "color_encoding": infer_color_encoding(color_primaries, color_matrix),
        "chroma_subsampling": infer_chroma_subsampling(pixel_format),
        "color_range": color_range_name(getattr(codec_context, "color_range", None)),
        "color_primaries": color_primaries,
        "color_transfer": color_transfer_name(getattr(codec_context, "color_trc", None)),
        "color_matrix": color_matrix,
    }


def convert_rgb_tensor_color_encoding(
    images: Any,
    source_encoding: str,
    target_encoding: str,
) -> Any:
    source = normalize_color_encoding(source_encoding)
    target = normalize_color_encoding(target_encoding)
    if source == target:
        return images

    import torch

    rgb = images[..., :3].float().clamp(0.0, 1.0)
    if source == COLOR_ENCODING_BT709:
        threshold = 0.081
        linear = torch.where(
            rgb < threshold,
            rgb / 4.5,
            torch.pow((rgb + 0.099) / 1.099, 1.0 / 0.45),
        )
        matrix_values = (
            (0.6274038959, 0.3292830384, 0.0433130657),
            (0.0690972894, 0.9195403951, 0.0113623156),
            (0.0163914389, 0.0880133079, 0.8955952532),
        )
    else:
        alpha = 1.0992968268
        beta = 0.0180539685
        threshold = 4.5 * beta
        linear = torch.where(
            rgb < threshold,
            rgb / 4.5,
            torch.pow((rgb + (alpha - 1.0)) / alpha, 1.0 / 0.45),
        )
        matrix_values = (
            (1.6604910021, -0.5876411388, -0.0728498633),
            (-0.1245504745, 1.1328998971, -0.0083494226),
            (-0.0181507634, -0.1005788980, 1.1187296614),
        )

    matrix = rgb.new_tensor(matrix_values)
    converted_linear = torch.matmul(linear, matrix.transpose(0, 1)).clamp(0.0, 1.0)

    if target == COLOR_ENCODING_BT709:
        converted = torch.where(
            converted_linear < 0.018,
            converted_linear * 4.5,
            1.099 * torch.pow(converted_linear, 0.45) - 0.099,
        )
    else:
        alpha = 1.0992968268
        beta = 0.0180539685
        converted = torch.where(
            converted_linear < beta,
            converted_linear * 4.5,
            alpha * torch.pow(converted_linear, 0.45) - (alpha - 1.0),
        )

    converted = converted.clamp(0.0, 1.0)
    if int(images.shape[-1]) <= 3:
        return converted
    return torch.cat((converted, images[..., 3:]), dim=-1)


def video_color_diagnostic_warnings(color_info: Mapping[str, Any]) -> list[str]:
    unspecified_fields = [
        field_name
        for field_name in ("color_range", "color_primaries", "color_transfer", "color_matrix")
        if color_info.get(field_name) == "unspecified"
    ]

    warnings: list[str] = []
    if unspecified_fields:
        warnings.append(f"color metadata is unspecified: {', '.join(unspecified_fields)}")

    transfer = str(color_info.get("color_transfer") or "")
    primaries = str(color_info.get("color_primaries") or "")
    matrix = str(color_info.get("color_matrix") or "")
    if transfer in {"smpte2084", "arib-std-b67"} or matrix == "ictcp":
        warnings.append("HDR color metadata is present; SDR IMAGE output may not preserve its appearance")
    elif matrix == "bt2020c":
        warnings.append("BT.2020 constant-luminance video is not color-converted by Video Reader")
    elif (
        color_info.get("color_encoding") == COLOR_ENCODING_BT2020
        and (matrix != "bt2020nc" or transfer != "bt2020-10")
    ):
        warnings.append(
            "this BT.2020 transfer/matrix combination is not color-converted by Video Reader"
        )
    elif primaries == "bt2020" and color_info.get("color_encoding") != COLOR_ENCODING_BT2020:
        warnings.append("wide-gamut color metadata is present but the color encoding could not be identified")

    bit_depth = color_info.get("bit_depth")
    if isinstance(bit_depth, int) and bit_depth > 8:
        warnings.append(f"{bit_depth}-bit video is decoded to ComfyUI IMAGE; the original bit depth is not preserved as metadata")

    return warnings
