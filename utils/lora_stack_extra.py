# Copyright 2026 kinorax
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import const as Const
from .file_hash_cache import normalize_relative_path
from .image_info_hash_extras import (
    EXTRA_CLIP_HASHES,
    EXTRA_DETAILER_HASH,
    EXTRA_HASHES,
    EXTRA_LORA_HASHES,
    EXTRA_MODEL_HASH,
    EXTRA_REFINER_HASH,
    EXTRA_VAE_HASH,
)
from .model_lora_metadata_pipeline import get_shared_metadata_pipeline

_RESERVED_EXTRA_KEYS = frozenset(
    {
        "Steps",
        "Sampler",
        "Schedule type",
        "CFG scale",
        "Seed",
        "Size",
        "Model",
        "Refiner",
        "Detailer",
        "Model folder paths",
        "Clip type",
        "Clip device",
        "VAE",
        "Hashes",
        "Extra info",
        "Model weight dtype",
        "Model ModelSamplingAuraFlow shift",
        "Refiner weight dtype",
        "Refiner ModelSamplingAuraFlow shift",
        "Detailer weight dtype",
        "Detailer ModelSamplingAuraFlow shift",
        EXTRA_MODEL_HASH,
        EXTRA_REFINER_HASH,
        EXTRA_DETAILER_HASH,
        EXTRA_VAE_HASH,
        EXTRA_LORA_HASHES,
        EXTRA_CLIP_HASHES,
        EXTRA_HASHES,
    }
)
_RESERVED_EXTRA_KEYS_CASEFOLD = {key.casefold() for key in _RESERVED_EXTRA_KEYS}
_CLIP_PARAMETER_KEY_RE = re.compile(r"^clip\s+\d+$", re.IGNORECASE)
_HASH_ALGO_SHA256 = "sha256"
_EXTRA_SCHEMA_KEY = "schema"
_EXTRA_PAYLOAD_KEY = "payload"
_LORA_STACK_EXTRA_SCHEMA = "lora_stack"


def normalized_extra_key_or_none(key: object) -> str | None:
    if key is None:
        return None
    normalized = str(key).strip()
    return normalized or None


def is_reserved_extra_key(key: object) -> bool:
    normalized = normalized_extra_key_or_none(key)
    if normalized is None:
        return False
    if normalized.casefold() in _RESERVED_EXTRA_KEYS_CASEFOLD:
        return True
    return _CLIP_PARAMETER_KEY_RE.fullmatch(normalized) is not None


def merge_extra_value(
    extras: dict[str, object] | None,
    key: str,
    value: str | None,
) -> dict[str, object] | None:
    output = dict(extras) if isinstance(extras, dict) else {}
    if value is None:
        output.pop(key, None)
    else:
        output[key] = value
    return output or None


def serialize_lora_stack_extra_json(lora_stack: object) -> str | None:
    if not isinstance(lora_stack, list):
        return None

    lora_options = _normalized_lora_options(Const.get_LORA_OPTIONS())
    pipeline = get_shared_metadata_pipeline(start=True)
    serialized: list[dict[str, object]] = []

    for item in lora_stack:
        if not isinstance(item, Mapping):
            continue

        resolved_relative_path, resolve_error = _resolve_lora_relative_path(
            item.get("name"),
            lora_options,
        )
        if resolved_relative_path is None:
            display_name = _basename(item.get("name"))
            if resolve_error == "multiple":
                raise RuntimeError(
                    f"Set Lora Stack Extra: LoRA '{display_name}' matched multiple local files. "
                    "Use a selector-resolved LoRA and try again."
                )
            raise RuntimeError(
                f"Set Lora Stack Extra: LoRA '{display_name}' was not found in ComfyUI loras"
            )

        sha256 = _normalize_sha256_or_none(
            pipeline.get_hash_by_relative_path(
                Const.MODEL_FOLDER_PATH_LORAS,
                resolved_relative_path,
                _HASH_ALGO_SHA256,
            )
        )
        if sha256 is None:
            pipeline.enqueue_hash_priority(Const.MODEL_FOLDER_PATH_LORAS, resolved_relative_path)
            basename = _basename(resolved_relative_path)
            normalized_relative_path = normalize_relative_path(resolved_relative_path)
            raise RuntimeError(
                f"Set Lora Stack Extra: sha256 is not ready for '{basename}'.\n"
                f"Queued priority hash job for loras/{normalized_relative_path}. "
                "Run again after hashing completes."
            )

        serialized.append(
            {
                "name": _basename(resolved_relative_path),
                "strength": _coerce_float(item.get("strength"), 1.0),
                "sha256": sha256,
            }
        )

    if not serialized:
        return None
    envelope = {
        _EXTRA_SCHEMA_KEY: _LORA_STACK_EXTRA_SCHEMA,
        _EXTRA_PAYLOAD_KEY: serialized,
    }
    return json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))


def deserialize_lora_stack_extra(
    raw_value: object,
    *,
    key: str,
) -> list[dict[str, str | float]] | None:
    items = _parsed_items(raw_value, key=key)
    if not items:
        return None

    lora_options = tuple(str(option) for option in Const.get_LORA_OPTIONS())
    pipeline = get_shared_metadata_pipeline(start=True)
    output: list[dict[str, str | float]] = []

    for item in items:
        if not isinstance(item, Mapping):
            continue

        sha256 = _normalize_sha256_or_none(item.get("sha256"))
        if sha256 is None:
            continue

        hash_matches = pipeline.find_relative_paths_by_hash(
            folder_name=Const.MODEL_FOLDER_PATH_LORAS,
            hash_prefix=sha256,
            preferred_algos=(_HASH_ALGO_SHA256,),
        )
        option_matches = _match_options_by_relative_paths(lora_options, hash_matches)
        if len(option_matches) != 1:
            continue

        normalized_item = Const.make_lora_stack_item(
            option_matches[0],
            _coerce_float(item.get("strength"), 1.0),
        )
        if normalized_item is not None:
            output.append(normalized_item)

    return output or None


def _parsed_items(raw_value: object, *, key: str) -> list[object]:
    if raw_value is None:
        return []

    parsed = raw_value
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise RuntimeError(f"Get Lora Stack Extra: extra '{key}' is not valid LoRA stack JSON") from exc

    if not isinstance(parsed, Mapping):
        raise RuntimeError(f"Get Lora Stack Extra: extra '{key}' is not valid LoRA stack JSON")

    schema = str(parsed.get(_EXTRA_SCHEMA_KEY) or "").strip()
    if schema != _LORA_STACK_EXTRA_SCHEMA:
        raise RuntimeError(f"Get Lora Stack Extra: extra '{key}' is not valid LoRA stack JSON")

    payload = parsed.get(_EXTRA_PAYLOAD_KEY)
    if not isinstance(payload, list):
        raise RuntimeError(f"Get Lora Stack Extra: extra '{key}' is not valid LoRA stack JSON")
    return list(payload)


def _normalized_lora_options(options: Sequence[str]) -> tuple[str, ...]:
    output: list[str] = []
    for option in options:
        normalized = normalize_relative_path(option)
        if normalized and normalized not in output:
            output.append(normalized)
    return tuple(output)


def _resolve_lora_relative_path(
    raw_name: object,
    options: Sequence[str],
) -> tuple[str | None, str]:
    normalized = normalize_relative_path(str(raw_name or ""))
    if not normalized:
        return None, "missing"

    if normalized in options:
        return normalized, ""

    basename = _basename(normalized)
    matches = [option for option in options if _basename(option).casefold() == basename.casefold()]
    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        return None, "multiple"
    return None, "missing"


def _match_options_by_relative_paths(options: Sequence[str], relative_paths: Sequence[str]) -> list[str]:
    option_map: dict[str, str] = {}
    for option in options:
        normalized = normalize_relative_path(option)
        if normalized and normalized not in option_map:
            option_map[normalized] = str(option)

    matches: list[str] = []
    for relative_path in relative_paths:
        normalized = normalize_relative_path(relative_path)
        if not normalized:
            continue
        option = option_map.get(normalized)
        if option and option not in matches:
            matches.append(option)
    return matches


def _basename(value: object) -> str:
    normalized = normalize_relative_path(str(value or ""))
    basename = Path(normalized).name
    return basename or str(value or "").strip()


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_sha256_or_none(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if len(text) != 64:
        return None
    if any(ch not in "0123456789abcdef" for ch in text):
        return None
    return text
