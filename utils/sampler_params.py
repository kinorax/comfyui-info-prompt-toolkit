# Copyright 2026 kinorax
from __future__ import annotations

import json
import math
from typing import Mapping

from .. import const as Const

SAMPLER_PARAMS_KEY = "sampler_params"
DENOISE_KEY = "denoise"
STEPS_MAX = 10000
SEED_MAX = 0xFFFFFFFFFFFFFFFF
CFG_MAX = 100.0
DENOISE_MIN = 0.01
DENOISE_MAX = 1.0
_EXTRA_SCHEMA_KEY = "schema"
_EXTRA_PAYLOAD_KEY = "payload"
_SAMPLER_PARAMS_EXTRA_SCHEMA = "sampler_params"


def sampler_params_payload_or_error(
    value: object,
) -> tuple[dict[str, object] | None, str | None]:
    if value is None:
        return None, None

    unwrapped = _unwrap_value(value)
    if unwrapped is None:
        return None, None
    if not isinstance(unwrapped, Mapping):
        return None, "sampler_params must be an object"

    sampler = _normalized_option(
        unwrapped.get(Const.IMAGEINFO_SAMPLER),
        Const.SAMPLER_OPTIONS,
    )
    if sampler is None:
        return None, f"{Const.IMAGEINFO_SAMPLER} must be one of the available sampler options"

    scheduler = _normalized_option(
        unwrapped.get(Const.IMAGEINFO_SCHEDULER),
        Const.SCHEDULER_OPTIONS,
    )
    if scheduler is None:
        return None, f"{Const.IMAGEINFO_SCHEDULER} must be one of the available scheduler options"

    steps = _normalized_int(
        unwrapped.get(Const.IMAGEINFO_STEPS),
        minimum=1,
        maximum=STEPS_MAX,
    )
    if steps is None:
        return None, f"{Const.IMAGEINFO_STEPS} must be an integer from 1 to {STEPS_MAX}"

    denoise = _normalized_float(
        unwrapped.get(DENOISE_KEY),
        minimum=DENOISE_MIN,
        maximum=DENOISE_MAX,
    )
    if denoise is None:
        return None, f"{DENOISE_KEY} must be a finite number from {DENOISE_MIN} to {DENOISE_MAX}"

    seed = _normalized_int(
        unwrapped.get(Const.IMAGEINFO_SEED),
        minimum=0,
        maximum=SEED_MAX,
    )
    if seed is None:
        return None, f"{Const.IMAGEINFO_SEED} must be an integer from 0 to {SEED_MAX}"

    cfg = _normalized_float(
        unwrapped.get(Const.IMAGEINFO_CFG),
        minimum=0.0,
        maximum=CFG_MAX,
    )
    if cfg is None:
        return None, f"{Const.IMAGEINFO_CFG} must be a finite number from 0.0 to {CFG_MAX}"

    payload: dict[str, object] = {
        Const.IMAGEINFO_SAMPLER: sampler,
        Const.IMAGEINFO_SCHEDULER: scheduler,
        Const.IMAGEINFO_STEPS: steps,
        DENOISE_KEY: denoise,
        Const.IMAGEINFO_SEED: seed,
        Const.IMAGEINFO_CFG: cfg,
    }
    return payload, None


def serialize_sampler_params_extra_json(sampler_params: object) -> str | None:
    payload, error = sampler_params_payload_or_error(sampler_params)
    if payload is None:
        if error is None:
            return None
        raise RuntimeError(f"Set Sampler Params Extra: {error}")
    envelope = {
        _EXTRA_SCHEMA_KEY: _SAMPLER_PARAMS_EXTRA_SCHEMA,
        _EXTRA_PAYLOAD_KEY: payload,
    }
    return json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))


def deserialize_sampler_params_extra(
    raw_value: object,
    *,
    key: str,
) -> dict[str, object] | None:
    if raw_value is None:
        return None

    parsed = raw_value
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise RuntimeError(
                f"Get Sampler Params Extra: extra '{key}' is not valid Sampler Params JSON"
            ) from exc

    if not isinstance(parsed, Mapping):
        raise RuntimeError(f"Get Sampler Params Extra: extra '{key}' is not valid Sampler Params JSON")

    schema = str(parsed.get(_EXTRA_SCHEMA_KEY) or "").strip()
    if schema != _SAMPLER_PARAMS_EXTRA_SCHEMA:
        raise RuntimeError(f"Get Sampler Params Extra: extra '{key}' is not valid Sampler Params JSON")
    if _EXTRA_PAYLOAD_KEY not in parsed:
        raise RuntimeError(f"Get Sampler Params Extra: extra '{key}' is not valid Sampler Params JSON")

    payload, _error = sampler_params_payload_or_error(parsed.get(_EXTRA_PAYLOAD_KEY))
    if payload is None:
        raise RuntimeError(f"Get Sampler Params Extra: extra '{key}' is not valid Sampler Params JSON")
    return payload


def split_sampler_params_values(
    payload: dict[str, object] | None,
) -> tuple[str | None, str | None, int | None, float | None, int | None, float | None]:
    if not isinstance(payload, dict):
        return None, None, None, None, None, None
    return (
        str(payload.get(Const.IMAGEINFO_SAMPLER)) if payload.get(Const.IMAGEINFO_SAMPLER) is not None else None,
        str(payload.get(Const.IMAGEINFO_SCHEDULER)) if payload.get(Const.IMAGEINFO_SCHEDULER) is not None else None,
        _normalized_int(payload.get(Const.IMAGEINFO_STEPS), minimum=1, maximum=STEPS_MAX),
        _normalized_float(payload.get(DENOISE_KEY), minimum=DENOISE_MIN, maximum=DENOISE_MAX),
        _normalized_int(payload.get(Const.IMAGEINFO_SEED), minimum=0, maximum=SEED_MAX),
        _normalized_float(payload.get(Const.IMAGEINFO_CFG), minimum=0.0, maximum=CFG_MAX),
    )


def _unwrap_value(value: object) -> object:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _unwrap_value(value[0])

    if isinstance(value, Mapping) and "__value__" in value:
        return _unwrap_value(value.get("__value__"))

    return value


def _normalized_option(
    value: object,
    options: tuple[str, ...],
) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text in options:
        return text
    return None


def _normalized_int(
    value: object,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed < minimum or parsed > maximum:
        return None
    return parsed


def _normalized_float(
    value: object,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    if parsed < minimum or parsed > maximum:
        return None
    return float(parsed)
