from __future__ import annotations

import math

UNIT_OPTIONS: tuple[int, ...] = (8, 16, 32, 64)
DEFAULT_UNIT = 32
DEFAULT_SCALE_BY = 1.0
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _float_or_default(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def normalize_unit(value: object, default: int = DEFAULT_UNIT) -> int:
    normalized = _int_or_default(value, default)
    if normalized in UNIT_OPTIONS:
        return normalized
    return int(default)


def normalize_scale_by(value: object, default: float = DEFAULT_SCALE_BY) -> float:
    normalized = _float_or_default(value, default)
    if normalized > 0:
        return normalized
    fallback = float(default)
    if fallback > 0:
        return fallback
    return 1.0


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


def _axis_candidates(target: float, unit: int, minimum: int, maximum: int) -> list[int]:
    floor_value = _floor_to_unit(target, unit)
    ceil_value = _ceil_to_unit(target, unit)

    candidates: list[int] = []
    for candidate in (floor_value, ceil_value):
        normalized = _clamp(candidate, minimum, maximum)
        if normalized not in candidates:
            candidates.append(normalized)

    if candidates:
        return candidates

    return [_clamp(unit, minimum, maximum)]


def _ratio_error(width: int, height: int, base_width: int, base_height: int) -> int:
    return abs((width * base_height) - (height * base_width))


def resolve_scaled_size(
    width: object,
    height: object,
    scale_by: object,
    min_unit: object,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    width_int = normalize_dimension(width, DEFAULT_WIDTH, minimum, maximum)
    height_int = normalize_dimension(height, DEFAULT_HEIGHT, minimum, maximum)
    scale_by_float = normalize_scale_by(scale_by)
    min_unit_int = normalize_unit(min_unit)
    bounded_minimum, bounded_maximum = _unit_bounds(min_unit_int, minimum, maximum)

    target_width = float(width_int) * scale_by_float
    target_height = float(height_int) * scale_by_float
    target_area = target_width * target_height

    width_candidates = _axis_candidates(target_width, min_unit_int, bounded_minimum, bounded_maximum)
    height_candidates = _axis_candidates(target_height, min_unit_int, bounded_minimum, bounded_maximum)

    candidates: list[tuple[int, int]] = []
    for width_candidate in width_candidates:
        for height_candidate in height_candidates:
            pair = (width_candidate, height_candidate)
            if pair not in candidates:
                candidates.append(pair)

    if not candidates:
        return bounded_minimum, bounded_minimum

    candidates.sort(
        key=lambda candidate: (
            _ratio_error(candidate[0], candidate[1], width_int, height_int),
            abs(candidate[0] - target_width) + abs(candidate[1] - target_height),
            abs((candidate[0] * candidate[1]) - target_area),
            candidate[0] * candidate[1],
            candidate[0],
            candidate[1],
        )
    )
    return int(candidates[0][0]), int(candidates[0][1])


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
    base_width: object,
    base_height: object,
    width: object,
    height: object,
    decimals: int = 4,
) -> str:
    base_width_int = _int_or_default(base_width, 0)
    base_height_int = _int_or_default(base_height, 0)
    width_int = _int_or_default(width, 0)
    height_int = _int_or_default(height, 0)

    if base_width_int <= 0 or base_height_int <= 0 or width_int <= 0 or height_int <= 0:
        return "-"

    divisor = math.gcd(base_width_int, base_height_int)
    width_ratio = int(base_width_int // divisor) if divisor > 0 else int(base_width_int)
    if width_ratio <= 0:
        return "-"

    ratio_value = (float(height_int) * float(width_ratio)) / float(width_int)
    rendered = format_ratio_value(ratio_value, decimals=decimals)
    if rendered is None:
        return "-"
    return f"{width_ratio} : {rendered}"
