# Copyright 2026 kinorax
from __future__ import annotations

import math

from . import aspect_ratio_size as AspectRatioSize

UNIT_OPTIONS = AspectRatioSize.UNIT_OPTIONS
DEFAULT_WIDTH_RATIO = AspectRatioSize.DEFAULT_WIDTH_RATIO
DEFAULT_HEIGHT_RATIO = AspectRatioSize.DEFAULT_HEIGHT_RATIO
DEFAULT_UNIT = AspectRatioSize.DEFAULT_UNIT
DEFAULT_WIDTH = AspectRatioSize.DEFAULT_WIDTH
DEFAULT_HEIGHT = AspectRatioSize.DEFAULT_HEIGHT
DEFAULT_MARGIN = 0


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _floor_to_unit(value: float, unit: int) -> int:
    return int(math.floor(float(value) / float(unit))) * int(unit)


def _ceil_to_unit(value: float, unit: int) -> int:
    return int(math.ceil(float(value) / float(unit))) * int(unit)


def normalize_margin(value: object) -> int:
    return max(0, _int_or_default(value, DEFAULT_MARGIN))


def margin_sums(
    margin_top: object,
    margin_right: object,
    margin_bottom: object,
    margin_left: object,
) -> tuple[int, int]:
    horizontal = normalize_margin(margin_left) + normalize_margin(margin_right)
    vertical = normalize_margin(margin_top) + normalize_margin(margin_bottom)
    return horizontal, vertical


def _sampling_bounds(
    margin_total: int,
    unit: int,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    minimum_sampling = _ceil_to_unit(minimum + margin_total, unit)
    maximum_sampling = _floor_to_unit(maximum, unit)
    if maximum_sampling < minimum_sampling:
        raise ValueError("margins leave no valid content size")
    return minimum_sampling, maximum_sampling


def _normalize_anchor_content(
    value: object,
    default: int,
    margin_total: int,
    unit: int,
    minimum: int,
    maximum: int,
) -> int:
    minimum_sampling, maximum_sampling = _sampling_bounds(
        margin_total,
        unit,
        minimum,
        maximum,
    )
    maximum_content = maximum_sampling - margin_total
    content = AspectRatioSize.normalize_dimension(value, default, minimum, maximum_content)
    sampling = _floor_to_unit(content + margin_total, unit)
    sampling = _clamp(sampling, minimum_sampling, maximum_sampling)
    return sampling - margin_total


def _best_content_candidate(
    target_content: float,
    margin_total: int,
    unit: int,
    minimum: int,
    maximum: int,
    error_fn,
) -> int:
    minimum_sampling, maximum_sampling = _sampling_bounds(
        margin_total,
        unit,
        minimum,
        maximum,
    )
    target_sampling = target_content + margin_total
    candidates: list[int] = []
    for sampling in (
        _floor_to_unit(target_sampling, unit),
        _ceil_to_unit(target_sampling, unit),
    ):
        bounded_sampling = _clamp(sampling, minimum_sampling, maximum_sampling)
        content = bounded_sampling - margin_total
        if content not in candidates:
            candidates.append(content)

    candidates.sort(key=lambda candidate: (error_fn(candidate), candidate))
    return int(candidates[0])


def resolve_from_width(
    width: object,
    width_ratio: object,
    height_ratio: object,
    min_unit: object,
    margin_top: object,
    margin_right: object,
    margin_bottom: object,
    margin_left: object,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    width_ratio_int = AspectRatioSize.normalize_ratio(width_ratio)
    height_ratio_int = AspectRatioSize.normalize_ratio(height_ratio)
    min_unit_int = AspectRatioSize.normalize_unit(min_unit)
    horizontal_margin, vertical_margin = margin_sums(
        margin_top,
        margin_right,
        margin_bottom,
        margin_left,
    )
    width_int = _normalize_anchor_content(
        width,
        DEFAULT_WIDTH,
        horizontal_margin,
        min_unit_int,
        minimum,
        maximum,
    )
    target_height = (width_int * height_ratio_int) / width_ratio_int
    height_int = _best_content_candidate(
        target_content=target_height,
        margin_total=vertical_margin,
        unit=min_unit_int,
        minimum=minimum,
        maximum=maximum,
        error_fn=lambda candidate: abs((width_int * height_ratio_int) - (candidate * width_ratio_int)),
    )
    return width_int, height_int


def resolve_from_height(
    height: object,
    width_ratio: object,
    height_ratio: object,
    min_unit: object,
    margin_top: object,
    margin_right: object,
    margin_bottom: object,
    margin_left: object,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    width_ratio_int = AspectRatioSize.normalize_ratio(width_ratio)
    height_ratio_int = AspectRatioSize.normalize_ratio(height_ratio)
    min_unit_int = AspectRatioSize.normalize_unit(min_unit)
    horizontal_margin, vertical_margin = margin_sums(
        margin_top,
        margin_right,
        margin_bottom,
        margin_left,
    )
    height_int = _normalize_anchor_content(
        height,
        DEFAULT_HEIGHT,
        vertical_margin,
        min_unit_int,
        minimum,
        maximum,
    )
    target_width = (height_int * width_ratio_int) / height_ratio_int
    width_int = _best_content_candidate(
        target_content=target_width,
        margin_total=horizontal_margin,
        unit=min_unit_int,
        minimum=minimum,
        maximum=maximum,
        error_fn=lambda candidate: abs((candidate * height_ratio_int) - (height_int * width_ratio_int)),
    )
    return width_int, height_int


def infer_anchor(
    width: object,
    height: object,
    width_ratio: object,
    height_ratio: object,
    min_unit: object,
    margin_top: object,
    margin_right: object,
    margin_bottom: object,
    margin_left: object,
    *,
    minimum: int,
    maximum: int,
) -> str:
    horizontal_margin, vertical_margin = margin_sums(
        margin_top,
        margin_right,
        margin_bottom,
        margin_left,
    )
    width_int = AspectRatioSize.normalize_dimension(
        width,
        DEFAULT_WIDTH,
        minimum,
        maximum - horizontal_margin,
    )
    height_int = AspectRatioSize.normalize_dimension(
        height,
        DEFAULT_HEIGHT,
        minimum,
        maximum - vertical_margin,
    )
    width_from_height, _ = resolve_from_height(
        height_int,
        width_ratio,
        height_ratio,
        min_unit,
        margin_top,
        margin_right,
        margin_bottom,
        margin_left,
        minimum=minimum,
        maximum=maximum,
    )
    _, height_from_width = resolve_from_width(
        width_int,
        width_ratio,
        height_ratio,
        min_unit,
        margin_top,
        margin_right,
        margin_bottom,
        margin_left,
        minimum=minimum,
        maximum=maximum,
    )

    width_delta = abs(width_from_height - width_int)
    height_delta = abs(height_from_width - height_int)
    if height_delta < width_delta:
        return "width"
    if width_delta < height_delta:
        return "height"
    return "width"


def resolve_size(
    width: object,
    height: object,
    width_ratio: object,
    height_ratio: object,
    min_unit: object,
    margin_top: object,
    margin_right: object,
    margin_bottom: object,
    margin_left: object,
    *,
    anchor: str | None = None,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    resolved_anchor = anchor if anchor in ("width", "height") else infer_anchor(
        width,
        height,
        width_ratio,
        height_ratio,
        min_unit,
        margin_top,
        margin_right,
        margin_bottom,
        margin_left,
        minimum=minimum,
        maximum=maximum,
    )
    resolver = resolve_from_height if resolved_anchor == "height" else resolve_from_width
    anchor_value = height if resolved_anchor == "height" else width
    return resolver(
        anchor_value,
        width_ratio,
        height_ratio,
        min_unit,
        margin_top,
        margin_right,
        margin_bottom,
        margin_left,
        minimum=minimum,
        maximum=maximum,
    )


def render_actual_ratio(
    width_ratio: object,
    width: object,
    height: object,
    decimals: int = 4,
) -> str:
    return AspectRatioSize.render_actual_ratio(
        width_ratio=width_ratio,
        width=width,
        height=height,
        decimals=decimals,
    )
