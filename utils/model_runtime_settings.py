from __future__ import annotations

import json
from typing import Any, Mapping

MODEL_RUNTIME_SETTING_CLIP_LAST_LAYER_KEY = "stop_at_clip_layer"
MODEL_RUNTIME_SETTING_SD3_SHIFT_KEY = "model_sampling_sd3_shift"
SUPPORTED_MODEL_RUNTIME_SETTING_FOLDERS: tuple[str, ...] = ("checkpoints", "diffusion_models")


def is_supported_model_runtime_settings_folder(folder_name: object) -> bool:
    normalized = str(folder_name or "").strip()
    return normalized in SUPPORTED_MODEL_RUNTIME_SETTING_FOLDERS


def normalize_model_runtime_settings(value: Any) -> dict[str, int | float]:
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

    output: dict[str, int | float] = {}

    clip_last_layer = _coerce_clip_last_layer(parsed.get(MODEL_RUNTIME_SETTING_CLIP_LAST_LAYER_KEY))
    if clip_last_layer is not None:
        output[MODEL_RUNTIME_SETTING_CLIP_LAST_LAYER_KEY] = clip_last_layer

    sd3_shift = _coerce_float(parsed.get(MODEL_RUNTIME_SETTING_SD3_SHIFT_KEY))
    if sd3_shift is not None:
        output[MODEL_RUNTIME_SETTING_SD3_SHIFT_KEY] = sd3_shift

    return output


def filter_model_runtime_settings_for_folder(
    folder_name: object,
    settings: Mapping[str, Any] | None,
) -> dict[str, int | float]:
    normalized = normalize_model_runtime_settings(settings)
    folder = str(folder_name or "").strip()
    if folder == "checkpoints":
        return normalized
    if folder == "diffusion_models":
        if MODEL_RUNTIME_SETTING_SD3_SHIFT_KEY in normalized:
            return {
                MODEL_RUNTIME_SETTING_SD3_SHIFT_KEY: normalized[MODEL_RUNTIME_SETTING_SD3_SHIFT_KEY],
            }
        return {}
    return {}


def clip_last_layer_from_settings(settings: Mapping[str, Any] | None) -> int | None:
    normalized = normalize_model_runtime_settings(settings)
    return _coerce_clip_last_layer(normalized.get(MODEL_RUNTIME_SETTING_CLIP_LAST_LAYER_KEY))


def sd3_shift_from_settings(settings: Mapping[str, Any] | None) -> float | None:
    normalized = normalize_model_runtime_settings(settings)
    return _coerce_float(normalized.get(MODEL_RUNTIME_SETTING_SD3_SHIFT_KEY))


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
        return float(value)
    except Exception:
        return None
