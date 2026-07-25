# Copyright 2026 kinorax
from __future__ import annotations

import json
import math
from typing import Any, Mapping

MODEL_RUNTIME_SETTING_PROFILE_KEY = "runtime_settings_profile"
MODEL_RUNTIME_SETTING_PROFILE_SDXL = "sdxl"
MODEL_RUNTIME_SETTING_CLIP_LAST_LAYER_KEY = "stop_at_clip_layer"
MODEL_RUNTIME_SETTING_SD3_SHIFT_KEY = "model_sampling_sd3_shift"
SUPPORTED_MODEL_RUNTIME_SETTING_FOLDERS: tuple[str, ...] = ("checkpoints",)
ModelRuntimeSettings = dict[str, str | int | float]


def is_supported_model_runtime_settings_folder(folder_name: object) -> bool:
    normalized = str(folder_name or "").strip()
    return normalized in SUPPORTED_MODEL_RUNTIME_SETTING_FOLDERS


def normalize_model_runtime_settings(
    value: Any,
    *,
    infer_legacy_profile: bool = True,
) -> ModelRuntimeSettings:
    parsed: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}

    if not isinstance(parsed, Mapping):
        return {}

    clip_last_layer = _coerce_clip_last_layer(parsed.get(MODEL_RUNTIME_SETTING_CLIP_LAST_LAYER_KEY))
    sd3_shift = _coerce_float(parsed.get(MODEL_RUNTIME_SETTING_SD3_SHIFT_KEY))

    profile = _coerce_profile(parsed.get(MODEL_RUNTIME_SETTING_PROFILE_KEY))
    if profile is None:
        # Permanent compatibility for a v1 database restored after its one-time
        # migration. When read through a Checkpoint path, either legacy numeric
        # override means the user had opted into the behavior now named "SDXL".
        if (
            infer_legacy_profile
            and MODEL_RUNTIME_SETTING_PROFILE_KEY not in parsed
            and (clip_last_layer is not None or sd3_shift is not None)
        ):
            profile = MODEL_RUNTIME_SETTING_PROFILE_SDXL
        else:
            return {}

    output: ModelRuntimeSettings = {
        MODEL_RUNTIME_SETTING_PROFILE_KEY: profile,
    }
    if clip_last_layer is not None:
        output[MODEL_RUNTIME_SETTING_CLIP_LAST_LAYER_KEY] = clip_last_layer
    if sd3_shift is not None:
        output[MODEL_RUNTIME_SETTING_SD3_SHIFT_KEY] = sd3_shift

    return output


def filter_model_runtime_settings_for_folder(
    folder_name: object,
    settings: Mapping[str, Any] | None,
    *,
    infer_legacy_profile: bool = True,
) -> ModelRuntimeSettings:
    normalized = normalize_model_runtime_settings(
        settings,
        infer_legacy_profile=infer_legacy_profile,
    )
    folder = str(folder_name or "").strip()
    if folder == "checkpoints":
        return normalized
    return {}


def effective_model_runtime_settings(
    settings: Mapping[str, Any] | None,
) -> dict[str, int | float]:
    """Return only values that can alter a loaded Checkpoint runtime."""
    normalized = normalize_model_runtime_settings(settings)
    output: dict[str, int | float] = {}
    clip_last_layer = _coerce_clip_last_layer(
        normalized.get(MODEL_RUNTIME_SETTING_CLIP_LAST_LAYER_KEY)
    )
    if clip_last_layer is not None:
        output[MODEL_RUNTIME_SETTING_CLIP_LAST_LAYER_KEY] = clip_last_layer
    sd3_shift = _coerce_float(normalized.get(MODEL_RUNTIME_SETTING_SD3_SHIFT_KEY))
    if sd3_shift is not None:
        output[MODEL_RUNTIME_SETTING_SD3_SHIFT_KEY] = sd3_shift
    return output


def clip_last_layer_from_settings(settings: Mapping[str, Any] | None) -> int | None:
    normalized = normalize_model_runtime_settings(settings)
    return _coerce_clip_last_layer(normalized.get(MODEL_RUNTIME_SETTING_CLIP_LAST_LAYER_KEY))


def sd3_shift_from_settings(settings: Mapping[str, Any] | None) -> float | None:
    normalized = normalize_model_runtime_settings(settings)
    return _coerce_float(normalized.get(MODEL_RUNTIME_SETTING_SD3_SHIFT_KEY))


def _coerce_profile(value: Any) -> str | None:
    profile = str(value or "").strip().lower()
    if profile == MODEL_RUNTIME_SETTING_PROFILE_SDXL:
        return MODEL_RUNTIME_SETTING_PROFILE_SDXL
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_clip_last_layer(value: Any) -> int | None:
    parsed = _coerce_int(value)
    if parsed is None or parsed > -1:
        return None
    return parsed


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None
