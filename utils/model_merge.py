# Copyright 2026 kinorax
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Callable, Mapping

MODEL_VALUE_NAME_KEY = "name"
MODEL_VALUE_FOLDER_PATHS_KEY = "folder_paths"
MODEL_FOLDER_PATH_CHECKPOINTS = "checkpoints"
MODEL_FOLDER_PATH_DIFFUSION_MODELS = "diffusion_models"
MODEL_MERGE_PAYLOAD_KEY = "ipt_model_merge"
MODEL_MERGE_VERSION_KEY = "version"
MODEL_MERGE_MODE_KEY = "mode"
MODEL_MERGE_BASE_MODEL_KEY = "base_model"
MODEL_MERGE_MODEL_KEY = "merge_model"
MODEL_MERGE_MODEL_RATIO_KEY = "model_ratio"
MODEL_MERGE_CLIP_RATIO_KEY = "clip_ratio"
MODEL_MERGE_VERSION = 1
MODEL_MERGE_MODE_SIMPLE = "simple"

SUPPORTED_MODEL_MERGE_FOLDERS: tuple[str, ...] = (
    MODEL_FOLDER_PATH_CHECKPOINTS,
    MODEL_FOLDER_PATH_DIFFUSION_MODELS,
)


def model_value_payload_or_none(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict) and "__value__" in value:
        return model_value_payload_or_none(value.get("__value__"))
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return model_value_payload_or_none(value[0])
    return None


def clone_model_value_or_none(value: Any) -> dict[str, Any] | None:
    payload = model_value_payload_or_none(value)
    if payload is None:
        return None
    return deepcopy(payload)


def model_name_or_none(value: Any) -> str | None:
    payload = model_value_payload_or_none(value)
    if payload is None:
        raw_name = value
    else:
        raw_name = payload.get(MODEL_VALUE_NAME_KEY)
    return _text_or_none(raw_name)


def model_folder_or_none(value: Any) -> str | None:
    payload = model_value_payload_or_none(value)
    if payload is None:
        return None
    return _text_or_none(payload.get(MODEL_VALUE_FOLDER_PATHS_KEY))


def model_merge_payload_or_none(value: Any) -> dict[str, Any] | None:
    payload = model_value_payload_or_none(value)
    if payload is None:
        return None

    raw_merge = payload.get(MODEL_MERGE_PAYLOAD_KEY)
    if not isinstance(raw_merge, Mapping):
        return None
    return dict(raw_merge)


def is_model_merge_value(value: Any) -> bool:
    return model_merge_payload_or_none(value) is not None


def validate_model_merge_inputs(base_model: Any, merge_model: Any) -> str | None:
    base_payload = model_value_payload_or_none(base_model)
    merge_payload = model_value_payload_or_none(merge_model)
    if base_payload is None or merge_payload is None:
        return "base_model and merge_model are required"

    base_name = _text_or_none(base_payload.get(MODEL_VALUE_NAME_KEY))
    merge_name = _text_or_none(merge_payload.get(MODEL_VALUE_NAME_KEY))
    if base_name is None or merge_name is None:
        return "base_model and merge_model must be IPT-Model values"

    base_folder = model_folder_or_none(base_payload)
    merge_folder = model_folder_or_none(merge_payload)
    if base_folder not in SUPPORTED_MODEL_MERGE_FOLDERS:
        return "base_model must be checkpoints or diffusion_models"
    if merge_folder not in SUPPORTED_MODEL_MERGE_FOLDERS:
        return "merge_model must be checkpoints or diffusion_models"
    if base_folder != merge_folder:
        return "base_model and merge_model must use the same folder_paths"
    return None


def build_model_merge_value(
    base_model: Any,
    merge_model: Any,
    model_ratio: Any,
    clip_ratio: Any,
) -> dict[str, Any] | None:
    error = validate_model_merge_inputs(base_model, merge_model)
    if error is not None:
        return None

    base_payload = clone_model_value_or_none(base_model)
    merge_payload = clone_model_value_or_none(merge_model)
    if base_payload is None or merge_payload is None:
        return None

    output = deepcopy(base_payload)
    output[MODEL_MERGE_PAYLOAD_KEY] = {
        MODEL_MERGE_VERSION_KEY: MODEL_MERGE_VERSION,
        MODEL_MERGE_MODE_KEY: MODEL_MERGE_MODE_SIMPLE,
        MODEL_MERGE_BASE_MODEL_KEY: base_payload,
        MODEL_MERGE_MODEL_KEY: merge_payload,
        MODEL_MERGE_MODEL_RATIO_KEY: _ratio_or_default(model_ratio, 1.0),
        MODEL_MERGE_CLIP_RATIO_KEY: _ratio_or_default(clip_ratio, 1.0),
    }
    return output


def model_merge_json_or_none(value: Any) -> str | None:
    payload = model_merge_payload_or_none(value)
    if payload is None:
        return None
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def model_value_from_merge_json(text: Any) -> dict[str, Any] | None:
    raw_text = _text_or_none(text)
    if raw_text is None:
        return None

    try:
        parsed = json.loads(raw_text)
    except Exception:
        return None

    if not isinstance(parsed, Mapping):
        return None

    return build_model_merge_value(
        parsed.get(MODEL_MERGE_BASE_MODEL_KEY),
        parsed.get(MODEL_MERGE_MODEL_KEY),
        parsed.get(MODEL_MERGE_MODEL_RATIO_KEY),
        parsed.get(MODEL_MERGE_CLIP_RATIO_KEY),
    )


def model_runtime_settings_tree(
    value: Any,
    settings_resolver: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    payload = model_value_payload_or_none(value)
    if payload is None:
        return {}

    merge_payload = model_merge_payload_or_none(payload)
    if merge_payload is None:
        return dict(settings_resolver(payload) or {})

    output: dict[str, Any] = {
        MODEL_MERGE_MODE_KEY: _text_or_none(merge_payload.get(MODEL_MERGE_MODE_KEY))
        or MODEL_MERGE_MODE_SIMPLE,
        MODEL_MERGE_MODEL_RATIO_KEY: _ratio_or_default(
            merge_payload.get(MODEL_MERGE_MODEL_RATIO_KEY),
            1.0,
        ),
        MODEL_MERGE_CLIP_RATIO_KEY: _ratio_or_default(
            merge_payload.get(MODEL_MERGE_CLIP_RATIO_KEY),
            1.0,
        ),
        MODEL_MERGE_BASE_MODEL_KEY: model_runtime_settings_tree(
            merge_payload.get(MODEL_MERGE_BASE_MODEL_KEY),
            settings_resolver,
        ),
        MODEL_MERGE_MODEL_KEY: model_runtime_settings_tree(
            merge_payload.get(MODEL_MERGE_MODEL_KEY),
            settings_resolver,
        ),
    }
    return output


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ratio_or_default(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed
