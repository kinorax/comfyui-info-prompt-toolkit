# Copyright 2026 kinorax
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import hashlib
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from comfy_api.latest import io as c_io
from comfy_execution.graph_utils import GraphBuilder, is_link

from ... import const as Const
from ...utils import cast as Cast


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        output: list[Any] = []
        for item in value:
            output.extend(_flatten(item))
        return output
    return [value]


def _first_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return _first_value(value[0])
    return value


def _collect_image_frames(value: Any) -> list[torch.Tensor]:
    frames: list[torch.Tensor] = []
    for item in _flatten(value):
        if not isinstance(item, torch.Tensor):
            continue

        if item.ndim == 4:
            for i in range(item.shape[0]):
                frames.append(item[i])
            continue

        if item.ndim == 3:
            frames.append(item)
            continue

    return frames


def _frame_signature(frame: torch.Tensor) -> bytes:
    tensor = frame.detach().cpu()
    if tensor.ndim != 3:
        return b""

    h = int(tensor.shape[0])
    w = int(tensor.shape[1])
    step_h = max(1, h // 16)
    step_w = max(1, w // 16)
    sampled = tensor[::step_h, ::step_w, :]
    array = np.ascontiguousarray(sampled.numpy(), dtype=np.float32)
    return hashlib.sha1(array.tobytes()).digest()


def _unique_frame_count(frames: list[torch.Tensor]) -> int:
    signatures: set[bytes] = set()
    for frame in frames:
        signatures.add(_frame_signature(frame))
    return len(signatures)


def _normalize_frames_for_expected_count(
    frames: list[torch.Tensor],
    expected_cells: int,
    prefer_interleaved: bool,
) -> tuple[list[torch.Tensor], str]:
    if expected_cells <= 0:
        return frames, "none"

    total = len(frames)
    if total <= expected_cells:
        return frames, "none"

    if total % expected_cells != 0:
        return frames, "none"

    per_cell = total // expected_cells
    if per_cell <= 1:
        return frames, "none"

    grouped = [frames[index] for index in range(0, total, per_cell)]
    interleaved = frames[:expected_cells]

    grouped_unique = _unique_frame_count(grouped)
    interleaved_unique = _unique_frame_count(interleaved)

    if interleaved_unique > grouped_unique:
        return interleaved, "interleaved"
    if grouped_unique > interleaved_unique:
        return grouped, "grouped"
    if prefer_interleaved:
        return interleaved, "interleaved-tie"
    return grouped, "grouped-tie"


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        for item in value:
            if _has_value(item):
                return True
        return False
    return True


def _extract_raw_link(value: Any) -> list[Any] | None:
    if is_link(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            raw_link = _extract_raw_link(item)
            if raw_link is not None:
                return raw_link
    return None


def _resolve_open_node_id(dynprompt: Any, close_node_id: str) -> str | None:
    close_node = dynprompt.get_node(close_node_id)
    if not isinstance(close_node, dict):
        return None

    close_inputs = close_node.get("inputs", {})
    if not isinstance(close_inputs, dict):
        return None

    raw_link = _extract_raw_link(close_inputs.get("loop_control"))
    if raw_link is None:
        return None

    return str(raw_link[0])


def _xy_plot_info_dict(value: Any) -> dict[str, Any] | None:
    info = _first_value(value)
    if isinstance(info, dict):
        return info
    return None


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(_first_value(value))
    except Exception:
        return int(default)


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(_first_value(value))
    except Exception:
        return float(default)

def _hex_color_or_default(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    raw = _first_value(value)
    text = str(raw).strip() if raw is not None else ""
    if text.startswith("#"):
        text = text[1:]

    if len(text) != 6:
        return default

    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except Exception:
        return default


def _normalized_labels(value: Any, expected_count: int, prefix: str) -> list[str]:
    if expected_count <= 0:
        return []

    labels: list[str] = []
    if isinstance(value, (list, tuple)):
        labels = [str(v) for v in value]

    if len(labels) == expected_count:
        return labels

    if len(labels) == 1 and expected_count > 1:
        return [labels[0]] * expected_count

    out = labels[:expected_count]
    while len(out) < expected_count:
        out.append(f"{prefix}{len(out) + 1}")
    return out


_FONT_NAMES_JP_PREFERRED: tuple[str, ...] = (
    "YuGothM.ttc",
    "YuGothR.ttc",
    "meiryo.ttc",
    "msgothic.ttc",
    "Hiragino Sans W3.ttc",
    "Hiragino Sans W6.ttc",
    "Hiragino Kaku Gothic ProN W3.otf",
    "Hiragino Kaku Gothic ProN W6.otf",
    "IPAexGothic.ttf",
    "IPAGothic.ttf",
)

_FONT_NAMES_MULTILINGUAL: tuple[str, ...] = (
    "NotoSansCJKjp-Regular.otf",
    "NotoSansCJK-Regular.ttc",
    "NotoSansJP-Regular.otf",
    "NotoSans-Regular.ttf",
    "SourceHanSansJP-Regular.otf",
    "SourceHanSans-Regular.otf",
    "Arial Unicode.ttf",
    "Arial Unicode MS.ttf",
)

_FONT_NAMES_FALLBACK: tuple[str, ...] = (
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
    "FreeSans.ttf",
    "arial.ttf",
)

_FONT_SEARCH_DIRS: tuple[str, ...] = (
    "/System/Library/Fonts",
    "/Library/Fonts",
    "~/Library/Fonts",
    "/usr/share/fonts",
    "/usr/share/fonts/truetype",
    "/usr/share/fonts/truetype/noto",
    "/usr/share/fonts/opentype/noto",
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/truetype/liberation2",
    "/usr/local/share/fonts",
    "~/.local/share/fonts",
    "~/.fonts",
)

XY_PLOT_BG_COLOR: tuple[int, int, int] = (0xF8, 0xF9, 0xFB)
XY_PLOT_TEXT_COLOR: tuple[int, int, int] = (0x11, 0x18, 0x27)
XY_PLOT_GRID_COLOR: tuple[int, int, int] = (0x8A, 0x8F, 0x99)

Y_LABEL_WIDTH_RATIO_DEFAULT = 0.35
Y_LABEL_WIDTH_RATIO_MIN = 0.2
Y_LABEL_WIDTH_RATIO_MAX = 0.8
X_ONLY_MAX_COLUMNS_DEFAULT = 0
X_ONLY_MAX_COLUMNS_MIN = 0
X_ONLY_MAX_COLUMNS_MAX = 9999

XY_PLOT_BG_COLOR_DEFAULT_HEX = "F8F9FB"
XY_PLOT_TEXT_COLOR_DEFAULT_HEX = "111827"
XY_PLOT_GRID_COLOR_DEFAULT_HEX = "8A8F99"



def _iter_font_names_by_priority() -> tuple[str, ...]:
    ordered: list[str] = []
    ordered.extend(_FONT_NAMES_JP_PREFERRED)
    ordered.extend(_FONT_NAMES_MULTILINGUAL)
    ordered.extend(_FONT_NAMES_FALLBACK)
    return tuple(ordered)


@lru_cache(maxsize=1)
def _font_candidates() -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        text = str(value).strip()
        if len(text) == 0 or text in seen:
            return
        seen.add(text)
        ordered.append(text)

    env_font = os.environ.get("IPT_XY_PLOT_FONT")
    if env_font is not None:
        add(env_font)

    names = _iter_font_names_by_priority()

    search_dirs: list[Path] = []
    windir = os.environ.get("WINDIR")
    if windir is not None:
        search_dirs.append(Path(windir) / "Fonts")

    for raw_dir in _FONT_SEARCH_DIRS:
        candidate_dir = Path(raw_dir).expanduser()
        if candidate_dir.exists():
            search_dirs.append(candidate_dir)

    for base_dir in search_dirs:
        for name in names:
            full = base_dir / name
            if full.exists():
                add(str(full))

    for name in names:
        add(name)

    return tuple(ordered)


def _load_truetype_font(size: int) -> ImageFont.ImageFont | None:
    resolved_size = max(8, int(size))
    for font_name in _font_candidates():
        try:
            return ImageFont.truetype(font_name, resolved_size, encoding="unic")
        except Exception:
            continue
    return None


def _measure_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    box = draw.textbbox((0, 0), str(text), font=font)
    return max(0, box[2] - box[0]), max(0, box[3] - box[1])


def _line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    _, line_h = _measure_text(draw, "Ag", font)
    return max(1, line_h)


def _truncate_with_ellipsis(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    limit = max(1, int(max_width))
    source = str(text)
    if _measure_text(draw, source, font)[0] <= limit:
        return source

    ellipsis = "..."
    if _measure_text(draw, ellipsis, font)[0] > limit:
        trimmed = ellipsis
        while len(trimmed) > 0 and _measure_text(draw, trimmed, font)[0] > limit:
            trimmed = trimmed[:-1]
        return trimmed

    out = source
    while len(out) > 0 and _measure_text(draw, f"{out}{ellipsis}", font)[0] > limit:
        out = out[:-1]

    if len(out) == 0:
        return ellipsis
    return f"{out}{ellipsis}"


def _wrap_text_by_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    source = str(text)
    if len(source) == 0:
        return [""]

    limit = max(1, int(max_width))
    lines: list[str] = []
    paragraphs = source.splitlines() if len(source.splitlines()) > 0 else [source]

    for paragraph in paragraphs:
        if len(paragraph) == 0:
            if len(lines) == 0 or lines[-1] != "":
                lines.append("")
            continue

        current = ""
        char_index = 0
        while char_index < len(paragraph):
            ch = paragraph[char_index]
            candidate = f"{current}{ch}"
            candidate_w, _ = _measure_text(draw, candidate, font)
            if len(current) == 0 or candidate_w <= limit:
                current = candidate
                char_index += 1
                continue

            lines.append(current.rstrip() if len(current.rstrip()) > 0 else current)
            current = ""

        lines.append(current.rstrip() if len(current.rstrip()) > 0 else current)

    if len(lines) == 0:
        lines.append("")
    return lines

def _layout_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_text_w: int,
    max_text_h: int,
    allow_wrap: bool,
    force_fit: bool = False,
) -> tuple[list[str], int, int, int, bool]:
    label = str(text)
    limit_w = max(1, int(max_text_w))
    limit_h = max(1, int(max_text_h))

    if not allow_wrap:
        text_w, text_h = _measure_text(draw, label, font)
        overflow = text_w > limit_w or text_h > limit_h
        return [label], text_w, text_h, 0, overflow

    line_h = _line_height(draw, font)
    line_spacing = max(1, int(line_h * 0.20))
    max_lines = max(1, (limit_h + line_spacing) // (line_h + line_spacing))

    lines = _wrap_text_by_width(draw, label, font, limit_w)
    truncated = False
    if max_lines > 0 and len(lines) > max_lines:
        if force_fit:
            lines = lines[:max_lines]
            if len(lines) > 0:
                lines[-1] = _truncate_with_ellipsis(draw, lines[-1], font, limit_w)
            truncated = True
        else:
            truncated = True

    if len(lines) == 0:
        lines = [""]

    block_w = 0
    for line in lines:
        line_w, _ = _measure_text(draw, line, font)
        block_w = max(block_w, line_w)

    block_h = (line_h * len(lines)) + (line_spacing * max(0, len(lines) - 1))
    overflow = block_w > limit_w or block_h > limit_h

    if overflow and force_fit:
        fitted_lines: list[str] = []
        for line in lines:
            fitted_lines.append(_truncate_with_ellipsis(draw, line, font, limit_w))
        lines = fitted_lines

        block_w = 0
        for line in lines:
            line_w, _ = _measure_text(draw, line, font)
            block_w = max(block_w, line_w)
        block_h = (line_h * len(lines)) + (line_spacing * max(0, len(lines) - 1))
        overflow = block_w > limit_w or block_h > limit_h

    return lines, block_w, block_h, line_spacing, (overflow or truncated)

def _measure_labels(
    draw: ImageDraw.ImageDraw,
    labels: list[str],
    font: ImageFont.ImageFont,
    max_text_w: int,
    max_text_h: int,
    allow_wrap: bool,
) -> tuple[int, int, bool]:
    max_w = 0
    max_h = 0
    has_overflow = False
    for label in labels:
        _, block_w, block_h, _, overflow = _layout_label(
            draw,
            str(label),
            font,
            max_text_w,
            max_text_h,
            allow_wrap,
            force_fit=False,
        )
        max_w = max(max_w, block_w)
        max_h = max(max_h, block_h)
        if overflow:
            has_overflow = True
    return max_w, max_h, has_overflow

def _fit_axis_font(
    draw: ImageDraw.ImageDraw,
    labels: list[str],
    max_text_w: int,
    max_text_h: int,
    preferred_size: int,
    allow_wrap: bool = False,
    max_font_size: int | None = None,
) -> tuple[ImageFont.ImageFont, int, int, int]:
    normalized = [str(label) for label in labels] if len(labels) > 0 else [""]
    limit_w = max(1, int(max_text_w))
    limit_h = max(1, int(max_text_h))

    min_size = 8
    max_size = max(min_size, min(256, limit_h))
    if max_font_size is not None:
        max_size = max(min_size, min(max_size, int(max_font_size)))

    best: tuple[ImageFont.ImageFont, int, int, int] | None = None
    lo = min_size
    hi = max_size

    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = _load_truetype_font(mid)
        if candidate is None:
            break

        label_w, label_h, has_overflow = _measure_labels(
            draw,
            normalized,
            candidate,
            limit_w,
            limit_h,
            allow_wrap,
        )
        if not has_overflow and label_w <= limit_w and label_h <= limit_h:
            best = (candidate, label_w, label_h, mid)
            lo = mid + 1
        else:
            hi = mid - 1

    if best is not None:
        return best

    capped_preferred = int(preferred_size)
    if max_font_size is not None:
        capped_preferred = min(capped_preferred, int(max_font_size))
    fallback_size = min_size if allow_wrap else max(min_size, min(max_size, capped_preferred))
    fallback = _load_truetype_font(fallback_size)
    if fallback is not None:
        label_w, label_h, _ = _measure_labels(
            draw,
            normalized,
            fallback,
            limit_w,
            limit_h,
            allow_wrap,
        )
        return fallback, label_w, label_h, fallback_size

    default_font = ImageFont.load_default()
    label_w, label_h, _ = _measure_labels(
        draw,
        normalized,
        default_font,
        limit_w,
        limit_h,
        allow_wrap,
    )
    return default_font, label_w, label_h, 12

def _has_visible_labels(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(str(item).strip() for item in value)
    if value is None:
        return False
    return bool(str(value).strip())


def _to_pil_rgb(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().numpy()
    if array.ndim != 3:
        raise ValueError("image frame must be rank-3 tensor")

    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.shape[2] != 3:
        raise ValueError("image frame must have 3 channels")

    array_uint8 = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array_uint8, mode="RGB")


def _draw_center_text(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    allow_wrap: bool = False,
) -> None:
    left, top, right, bottom = rect
    box_w = max(0, right - left)
    box_h = max(0, bottom - top)

    if box_w <= 0 or box_h <= 0:
        return

    # Keep small horizontal safe margins to avoid glyph bleed over grid borders.
    content_padding_x = max(2, int(box_w * 0.02))
    content_left = left + content_padding_x
    content_w = max(1, box_w - (content_padding_x * 2))

    lines, _, text_h, line_spacing, _ = _layout_label(
        draw,
        text,
        font,
        content_w,
        box_h,
        allow_wrap,
        force_fit=True,
    )

    raw_text = str(text)
    has_explicit_newline = "\n" in raw_text or "\r" in raw_text
    left_align = has_explicit_newline or len(lines) > 1

    text_y = top + (box_h - text_h) / 2
    for line in lines:
        line_w, line_h = _measure_text(draw, line, font)
        if left_align:
            text_x = content_left
        else:
            text_x = content_left + (content_w - line_w) / 2
        draw.text((text_x, text_y), line, font=font, fill=fill)
        text_y += line_h + line_spacing

def _compose_xy_plot_image(
    collected_frames: list[torch.Tensor],
    xy_plot_info: dict[str, Any],
    y_label_width_ratio: float,
    x_only_max_columns: int,
    bg_color: Any,
    text_color: Any,
    border_color: Any,
) -> torch.Tensor:
    x_count = _int_or_default(xy_plot_info.get("x_count"), 0)
    y_count = _int_or_default(xy_plot_info.get("y_count"), 0)

    if x_count <= 0 or y_count <= 0:
        raise ValueError("xy_plot_info must include positive x_count and y_count")

    expected_cells = x_count * y_count
    if len(collected_frames) != expected_cells:
        raise ValueError(
            f"image count mismatch: expected {expected_cells}, got {len(collected_frames)}"
        )

    pil_frames = [_to_pil_rgb(frame) for frame in collected_frames]
    cell_w, cell_h = pil_frames[0].size
    for frame in pil_frames[1:]:
        if frame.size != (cell_w, cell_h):
            raise ValueError("all images must have same resolution")

    raw_y_labels = xy_plot_info.get("y_labels")
    x_labels = _normalized_labels(xy_plot_info.get("x_labels"), x_count, "X")
    y_labels = _normalized_labels(raw_y_labels, y_count, "Y")
    resolved_x_only_max_columns = _int_or_default(x_only_max_columns, X_ONLY_MAX_COLUMNS_DEFAULT)
    if resolved_x_only_max_columns < X_ONLY_MAX_COLUMNS_MIN:
        resolved_x_only_max_columns = X_ONLY_MAX_COLUMNS_MIN
    elif resolved_x_only_max_columns > X_ONLY_MAX_COLUMNS_MAX:
        resolved_x_only_max_columns = X_ONLY_MAX_COLUMNS_MAX

    is_x_only_wrapped = bool(y_count == 1 and resolved_x_only_max_columns >= 1)
    has_y_header = bool(y_count > 1 or _has_visible_labels(raw_y_labels))
    if is_x_only_wrapped:
        has_y_header = False

    y_width_ratio = float(y_label_width_ratio)
    if y_width_ratio < Y_LABEL_WIDTH_RATIO_MIN:
        y_width_ratio = Y_LABEL_WIDTH_RATIO_MIN
    elif y_width_ratio > Y_LABEL_WIDTH_RATIO_MAX:
        y_width_ratio = Y_LABEL_WIDTH_RATIO_MAX


    bg_color_rgb = _hex_color_or_default(bg_color, XY_PLOT_BG_COLOR)
    text_color_rgb = _hex_color_or_default(text_color, XY_PLOT_TEXT_COLOR)
    border_color_rgb = _hex_color_or_default(border_color, XY_PLOT_GRID_COLOR)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1), color=bg_color_rgb))
    scale_base = min(cell_w, cell_h)
    x_max_font_size = max(12, int(scale_base * 0.040))
    y_max_font_size = max(12, int(scale_base * 0.032))

    x_text_limit_w = max(32, int(cell_w * 0.92))
    x_text_limit_h = max(24, int(cell_h * 0.28))
    y_text_limit_w = max(64, int(cell_w * y_width_ratio))
    y_text_limit_h = max(20, int(cell_h * 0.70))

    x_font_size = x_max_font_size
    x_font = _load_truetype_font(x_font_size)
    if x_font is None:
        x_font = ImageFont.load_default()

    _, max_x_h, _ = _measure_labels(
        measure,
        x_labels,
        x_font,
        max_text_w=x_text_limit_w,
        max_text_h=x_text_limit_h,
        allow_wrap=True,
    )
    x_padding = max(4, int(x_font_size * 0.22))
    header_top_h = max(16, min(x_text_limit_h, max_x_h + (x_padding * 2)))

    if is_x_only_wrapped:
        wrapped_x_count = max(1, min(resolved_x_only_max_columns, x_count))
        wrapped_y_count = max(1, (x_count + wrapped_x_count - 1) // wrapped_x_count)
        wrapped_cell_h = header_top_h + cell_h

        canvas_w = cell_w * wrapped_x_count
        canvas_h = wrapped_cell_h * wrapped_y_count

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color_rgb)
        draw = ImageDraw.Draw(canvas)

        for index, frame in enumerate(pil_frames):
            x = index % wrapped_x_count
            y = index // wrapped_x_count
            ox = x * cell_w
            oy = (y * wrapped_cell_h) + header_top_h
            canvas.paste(frame, (ox, oy))

        border = border_color_rgb
        draw.rectangle((0, 0, canvas_w - 1, canvas_h - 1), outline=border, width=1)

        for x in range(wrapped_x_count + 1):
            ox = x * cell_w
            draw.line((ox, 0, ox, canvas_h), fill=border, width=1)

        for y in range(wrapped_y_count + 1):
            oy = y * wrapped_cell_h
            draw.line((0, oy, canvas_w, oy), fill=border, width=1)

        for y in range(wrapped_y_count):
            oy = (y * wrapped_cell_h) + header_top_h
            draw.line((0, oy, canvas_w, oy), fill=border, width=1)

        for index, label in enumerate(x_labels):
            x = index % wrapped_x_count
            y = index // wrapped_x_count
            left = x * cell_w
            right = left + cell_w
            top = y * wrapped_cell_h
            bottom = top + header_top_h
            _draw_center_text(
                draw,
                (left, top, right, bottom),
                label,
                x_font,
                text_color_rgb,
                allow_wrap=True,
            )

        canvas_np = np.asarray(canvas).astype(np.float32) / 255.0
        return torch.from_numpy(canvas_np)[None,]

    y_font: ImageFont.ImageFont = x_font
    header_left_w = 0
    if has_y_header:
        y_font_size = y_max_font_size
        resolved_y_font = _load_truetype_font(y_font_size)
        if resolved_y_font is not None:
            y_font = resolved_y_font

        max_y_w, _, _ = _measure_labels(
            measure,
            y_labels,
            y_font,
            max_text_w=y_text_limit_w,
            max_text_h=y_text_limit_h,
            allow_wrap=True,
        )
        y_padding = max(8, int(y_font_size * 0.45))
        header_left_w = max(64, min(y_text_limit_w, max_y_w + (y_padding * 2)))

    canvas_w = header_left_w + (cell_w * x_count)
    canvas_h = header_top_h + (cell_h * y_count)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color_rgb)
    draw = ImageDraw.Draw(canvas)

    idx = 0
    for y in range(y_count):
        for x in range(x_count):
            ox = header_left_w + (x * cell_w)
            oy = header_top_h + (y * cell_h)
            canvas.paste(pil_frames[idx], (ox, oy))
            idx += 1

    border = border_color_rgb
    draw.rectangle((0, 0, canvas_w - 1, canvas_h - 1), outline=border, width=1)

    for x in range(x_count + 1):
        ox = header_left_w + (x * cell_w)
        draw.line((ox, 0, ox, canvas_h), fill=border, width=1)

    for y in range(y_count + 1):
        oy = header_top_h + (y * cell_h)
        draw.line((0, oy, canvas_w, oy), fill=border, width=1)

    for x, label in enumerate(x_labels):
        left = header_left_w + (x * cell_w)
        right = left + cell_w
        _draw_center_text(
            draw,
            (left, 0, right, header_top_h),
            label,
            x_font,
            text_color_rgb,
            allow_wrap=True,
        )

    if has_y_header:
        for y, label in enumerate(y_labels):
            top = header_top_h + (y * cell_h)
            bottom = top + cell_h
            _draw_center_text(
                draw,
                (0, top, header_left_w, bottom),
                label,
                y_font,
                text_color_rgb,
                allow_wrap=True,
            )

    canvas_np = np.asarray(canvas).astype(np.float32) / 255.0
    return torch.from_numpy(canvas_np)[None,]


class XYPlotEnd(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-XYPlotEnd",
            display_name="XY Plot End",
            category=Const.CATEGORY_XYPLOT,
            enable_expand=True,
            is_input_list=True,
            hidden=[
                c_io.Hidden.dynprompt,
                c_io.Hidden.unique_id,
            ],
            inputs=[
                c_io.Image.Input(
                    "image",
                    tooltip="Generated image(s)",
                ),
                Const.XY_PLOT_INFO_TYPE.Input(
                    "xy_plot_info",
                    tooltip="XY plot labels and grid shape",
                ),
                Const.LOOP_CONTROL_TYPE.Input(
                    "loop_control",
                    tooltip="Loop control value from XY Plot Start (None means list mode)",
                ),
                c_io.String.Input(
                    "background_color",
                    default=XY_PLOT_BG_COLOR_DEFAULT_HEX,
                    tooltip="XY plot background color (RRGGBB)",
                ),
                c_io.String.Input(
                    "text_color",
                    default=XY_PLOT_TEXT_COLOR_DEFAULT_HEX,
                    tooltip="XY plot text color (RRGGBB)",
                ),
                c_io.String.Input(
                    "border_color",
                    default=XY_PLOT_GRID_COLOR_DEFAULT_HEX,
                    tooltip="XY plot border and grid color (RRGGBB)",
                ),
                c_io.Float.Input(
                    "y_label_width_ratio",
                    default=Y_LABEL_WIDTH_RATIO_DEFAULT,
                    min=Y_LABEL_WIDTH_RATIO_MIN,
                    max=Y_LABEL_WIDTH_RATIO_MAX,
                    step=0.01,
                    tooltip="Y-label width upper ratio against cell width",
                ),
                c_io.Int.Input(
                    "x_only_max_columns",
                    default=X_ONLY_MAX_COLUMNS_DEFAULT,
                    min=X_ONLY_MAX_COLUMNS_MIN,
                    max=X_ONLY_MAX_COLUMNS_MAX,
                    step=1,
                    tooltip="0 keeps single-row mode. >=1 wraps only when Y modifiers are not used.",
                ),
                c_io.AnyType.Input(
                    "accumulated_images",
                    optional=True,
                    advanced=True,
                    tooltip="Internal use only. Leave this unconnected.",
                ),
            ],
            outputs=[
                c_io.Image.Output(
                    Cast.out_id("xy_plot_image"),
                    display_name="xy_plot_image",
                ),
            ],
        )

    @classmethod
    def _explore_dependencies(cls, node_id: str, dynprompt: Any, upstream: dict[str, list[str]]) -> None:
        node_info = dynprompt.get_node(node_id)
        if not isinstance(node_info, dict):
            return
        inputs = node_info.get("inputs", {})
        if not isinstance(inputs, dict):
            return

        for value in inputs.values():
            if not is_link(value):
                continue
            parent_id = value[0]
            if parent_id not in upstream:
                upstream[parent_id] = []
                cls._explore_dependencies(parent_id, dynprompt, upstream)
            upstream[parent_id].append(node_id)

    @classmethod
    def _collect_contained(cls, node_id: str, upstream: dict[str, list[str]], contained: dict[str, bool]) -> None:
        children = upstream.get(node_id)
        if children is None:
            return
        for child_id in children:
            if child_id in contained:
                continue
            contained[child_id] = True
            cls._collect_contained(child_id, upstream, contained)

    @classmethod
    def execute(
        cls,
        image: Any,
        xy_plot_info: Any,
        loop_control: Any,
        background_color: Any = XY_PLOT_BG_COLOR_DEFAULT_HEX,
        text_color: Any = XY_PLOT_TEXT_COLOR_DEFAULT_HEX,
        border_color: Any = XY_PLOT_GRID_COLOR_DEFAULT_HEX,
        y_label_width_ratio: Any = Y_LABEL_WIDTH_RATIO_DEFAULT,
        x_only_max_columns: Any = X_ONLY_MAX_COLUMNS_DEFAULT,
        accumulated_images: Any = None,
    ) -> c_io.NodeOutput:
        info = _xy_plot_info_dict(xy_plot_info)
        if info is None:
            raise ValueError("xy_plot_info is required")

        is_loop_mode = _has_value(loop_control)
        image_frames = _collect_image_frames(image)
        accumulated_frames = _collect_image_frames(accumulated_images)

        if is_loop_mode:
            collected = list(accumulated_frames)
            collected.extend(image_frames)
        else:
            collected = image_frames

        x_count = _int_or_default(info.get("x_count"), 0)
        y_count = _int_or_default(info.get("y_count"), 0)
        expected_cells = x_count * y_count
        if expected_cells <= 0:
            expected_cells = max(1, _int_or_default(info.get("cell_count"), 1))

        normalized_collected, normalize_strategy = _normalize_frames_for_expected_count(
            collected,
            expected_cells,
            prefer_interleaved=not is_loop_mode,
        )
        if normalize_strategy != "none":
            print(
                "[IPT-XYPlotEnd]"
                f" normalized strategy={normalize_strategy}"
                f" before={len(collected)}"
                f" after={len(normalized_collected)}"
            )
        collected = normalized_collected

        y_label_width_ratio_value = _float_or_default(y_label_width_ratio, Y_LABEL_WIDTH_RATIO_DEFAULT)
        if y_label_width_ratio_value < Y_LABEL_WIDTH_RATIO_MIN:
            y_label_width_ratio_value = Y_LABEL_WIDTH_RATIO_MIN
        elif y_label_width_ratio_value > Y_LABEL_WIDTH_RATIO_MAX:
            y_label_width_ratio_value = Y_LABEL_WIDTH_RATIO_MAX

        x_only_max_columns_value = _int_or_default(x_only_max_columns, X_ONLY_MAX_COLUMNS_DEFAULT)
        if x_only_max_columns_value < X_ONLY_MAX_COLUMNS_MIN:
            x_only_max_columns_value = X_ONLY_MAX_COLUMNS_MIN
        elif x_only_max_columns_value > X_ONLY_MAX_COLUMNS_MAX:
            x_only_max_columns_value = X_ONLY_MAX_COLUMNS_MAX

        compose_args = (
            collected,
            info,
            y_label_width_ratio_value,
            x_only_max_columns_value,
            background_color,
            text_color,
            border_color,
        )

        if not is_loop_mode:
            return c_io.NodeOutput(_compose_xy_plot_image(*compose_args))

        hidden = getattr(cls, "hidden", None)
        dynprompt = getattr(hidden, "dynprompt", None)
        unique_id = getattr(hidden, "unique_id", None)
        if dynprompt is None or unique_id is None:
            return c_io.NodeOutput(_compose_xy_plot_image(*compose_args))

        unique_id_str = str(unique_id)
        open_node_id = _resolve_open_node_id(dynprompt, unique_id_str)
        if open_node_id is None:
            return c_io.NodeOutput(_compose_xy_plot_image(*compose_args))

        open_node = dynprompt.get_node(open_node_id)
        if not isinstance(open_node, dict):
            return c_io.NodeOutput(_compose_xy_plot_image(*compose_args))

        open_inputs = open_node.get("inputs", {})
        if not isinstance(open_inputs, dict):
            return c_io.NodeOutput(_compose_xy_plot_image(*compose_args))

        current_index = _int_or_default(open_inputs.get("loop_index"), 0)
        cell_count = max(1, _int_or_default(info.get("cell_count"), 1))
        final_index = cell_count - 1
        if current_index >= final_index:
            return c_io.NodeOutput(_compose_xy_plot_image(*compose_args))

        upstream: dict[str, list[str]] = {}
        cls._explore_dependencies(unique_id_str, dynprompt, upstream)

        contained: dict[str, bool] = {}
        cls._collect_contained(open_node_id, upstream, contained)
        contained[unique_id_str] = True
        contained[open_node_id] = True

        graph = GraphBuilder()
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            if not isinstance(original_node, dict):
                continue

            class_type = original_node.get("class_type")
            if not isinstance(class_type, str):
                continue

            clone_id = "Recurse" if node_id == unique_id_str else node_id
            node = graph.node(class_type, clone_id)
            node.set_override_display_id(node_id)

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            if not isinstance(original_node, dict):
                continue

            clone_id = "Recurse" if node_id == unique_id_str else node_id
            node = graph.lookup_node(clone_id)
            if node is None:
                continue

            original_inputs = original_node.get("inputs", {})
            if not isinstance(original_inputs, dict):
                continue

            for key, value in original_inputs.items():
                if is_link(value) and value[0] in contained:
                    parent_id = "Recurse" if value[0] == unique_id_str else value[0]
                    parent = graph.lookup_node(parent_id)
                    if parent is not None:
                        node.set_input(key, parent.out(value[1]))
                else:
                    node.set_input(key, value)

        new_open = graph.lookup_node(open_node_id)
        recurse_close = graph.lookup_node("Recurse")
        if new_open is None or recurse_close is None:
            return c_io.NodeOutput(_compose_xy_plot_image(*compose_args))

        new_open.set_input("loop_index", current_index + 1)
        recurse_close.set_input("accumulated_images", collected)

        return c_io.NodeOutput(
            recurse_close.out(0),
            expand=graph.finalize(),
        )









