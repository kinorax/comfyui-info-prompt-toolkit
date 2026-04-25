# Copyright 2026 kinorax
from __future__ import annotations

import math

UNIT_OPTIONS: tuple[int, ...] = (8, 16, 32, 64)
DEFAULT_RATIO = 1
DEFAULT_WIDTH_RATIO = 10
DEFAULT_HEIGHT_RATIO = 16
DEFAULT_UNIT = 32
DEFAULT_WIDTH = 864
DEFAULT_HEIGHT = 1376


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def normalize_ratio(value: object, default: int = DEFAULT_RATIO) -> int:
    normalized = _int_or_default(value, default)
    if normalized < 1:
        return int(default if default >= 1 else 1)
    return normalized


def normalize_unit(value: object, default: int = DEFAULT_UNIT) -> int:
    normalized = _int_or_default(value, default)
    if normalized in UNIT_OPTIONS:
        return normalized
    return int(default)


def normalize_dimension(
    value: object,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    normalized = _int_or_default(value, default)
    return _clamp(normalized, minimum, maximum)


def _floor_to_unit(value: float, unit: int) -> int:
    return int(math.floor(float(value) / float(unit))) * int(unit)


def _ceil_to_unit(value: float, unit: int) -> int:
    return int(math.ceil(float(value) / float(unit))) * int(unit)


def _unit_bounds(unit: int, minimum: int, maximum: int) -> tuple[int, int]:
    unit_int = max(1, int(unit))
    min_multiple = _ceil_to_unit(minimum, unit_int)
    max_multiple = _floor_to_unit(maximum, unit_int)
    if max_multiple < min_multiple:
        return _clamp(minimum, minimum, maximum), _clamp(maximum, minimum, maximum)
    return min_multiple, max_multiple


def _normalized_anchor_dimension(
    value: object,
    default: int,
    unit: int,
    minimum: int,
    maximum: int,
) -> int:
    base = normalize_dimension(value, default, minimum, maximum)
    floored = _floor_to_unit(base, unit)
    bounded_minimum, bounded_maximum = _unit_bounds(unit, minimum, maximum)
    return _clamp(floored, bounded_minimum, bounded_maximum)


def _best_candidate(
    target: float,
    unit: int,
    minimum: int,
    maximum: int,
    error_fn,
) -> int:
    floor_value = _floor_to_unit(target, unit)
    ceil_value = _ceil_to_unit(target, unit)

    candidates: list[int] = []
    for candidate in (floor_value, ceil_value):
        normalized = _clamp(candidate, minimum, maximum)
        if normalized not in candidates:
            candidates.append(normalized)

    if not candidates:
        return _clamp(unit, minimum, maximum)

    candidates.sort(key=lambda candidate: (error_fn(candidate), candidate))
    return int(candidates[0])


def resolve_from_width(
    width: object,
    width_ratio: object,
    height_ratio: object,
    min_unit: object,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    width_ratio_int = normalize_ratio(width_ratio)
    height_ratio_int = normalize_ratio(height_ratio)
    min_unit_int = normalize_unit(min_unit)
    bounded_minimum, bounded_maximum = _unit_bounds(min_unit_int, minimum, maximum)
    width_int = _normalized_anchor_dimension(
        width,
        DEFAULT_WIDTH,
        min_unit_int,
        minimum,
        maximum,
    )
    target_height = (width_int * height_ratio_int) / width_ratio_int

    height_int = _best_candidate(
        target=target_height,
        unit=min_unit_int,
        minimum=bounded_minimum,
        maximum=bounded_maximum,
        error_fn=lambda candidate: abs((width_int * height_ratio_int) - (candidate * width_ratio_int)),
    )
    return width_int, height_int


def resolve_from_height(
    height: object,
    width_ratio: object,
    height_ratio: object,
    min_unit: object,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    width_ratio_int = normalize_ratio(width_ratio)
    height_ratio_int = normalize_ratio(height_ratio)
    min_unit_int = normalize_unit(min_unit)
    bounded_minimum, bounded_maximum = _unit_bounds(min_unit_int, minimum, maximum)
    height_int = _normalized_anchor_dimension(
        height,
        DEFAULT_HEIGHT,
        min_unit_int,
        minimum,
        maximum,
    )
    target_width = (height_int * width_ratio_int) / height_ratio_int

    width_int = _best_candidate(
        target=target_width,
        unit=min_unit_int,
        minimum=bounded_minimum,
        maximum=bounded_maximum,
        error_fn=lambda candidate: abs((candidate * height_ratio_int) - (height_int * width_ratio_int)),
    )
    return width_int, height_int


def infer_anchor(
    width: object,
    height: object,
    width_ratio: object,
    height_ratio: object,
    min_unit: object,
    *,
    minimum: int,
    maximum: int,
) -> str:
    width_int = normalize_dimension(width, DEFAULT_WIDTH, minimum, maximum)
    height_int = normalize_dimension(height, DEFAULT_HEIGHT, minimum, maximum)
    width_from_height, _ = resolve_from_height(
        height_int,
        width_ratio,
        height_ratio,
        min_unit,
        minimum=minimum,
        maximum=maximum,
    )
    _, height_from_width = resolve_from_width(
        width_int,
        width_ratio,
        height_ratio,
        min_unit,
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
        minimum=minimum,
        maximum=maximum,
    )

    if resolved_anchor == "height":
        return resolve_from_height(
            height,
            width_ratio,
            height_ratio,
            min_unit,
            minimum=minimum,
            maximum=maximum,
        )

    return resolve_from_width(
        width,
        width_ratio,
        height_ratio,
        min_unit,
        minimum=minimum,
        maximum=maximum,
    )


def actual_height_ratio(
    width_ratio: object,
    width: object,
    height: object,
) -> float | None:
    width_ratio_int = normalize_ratio(width_ratio)
    width_int = _int_or_default(width, 0)
    height_int = _int_or_default(height, 0)

    if width_int <= 0 or height_int <= 0:
        return None

    return (height_int * width_ratio_int) / width_int


def format_ratio_value(value: float | None, decimals: int = 4) -> str | None:
    if value is None or not math.isfinite(value):
        return None

    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return str(int(rounded))

    precision = max(0, int(decimals))
    rendered = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    if not rendered:
        return "0"
    return rendered


def render_actual_ratio(
    width_ratio: object,
    width: object,
    height: object,
    decimals: int = 4,
) -> str:
    width_ratio_int = normalize_ratio(width_ratio)
    ratio_value = actual_height_ratio(width_ratio_int, width, height)
    rendered = format_ratio_value(ratio_value, decimals=decimals)
    if rendered is None:
        return "-"
    return f"{width_ratio_int} : {rendered}"
