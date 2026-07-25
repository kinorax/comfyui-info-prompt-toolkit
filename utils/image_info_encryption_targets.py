# Copyright 2026 kinorax
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .. import const as Const

VERSION = 1
WIDTH_HEIGHT_TARGET = "width x height"
MAIN_FIELD_TARGETS = (
    Const.IMAGEINFO_MODEL,
    Const.IMAGEINFO_REFINER_MODEL,
    Const.IMAGEINFO_DETAILER_MODEL,
    Const.IMAGEINFO_LORA_STACK,
    Const.IMAGEINFO_CLIP,
    Const.IMAGEINFO_VAE,
    Const.IMAGEINFO_POSITIVE,
    Const.IMAGEINFO_NEGATIVE,
    Const.IMAGEINFO_STEPS,
    Const.IMAGEINFO_SAMPLER,
    Const.IMAGEINFO_SCHEDULER,
    Const.IMAGEINFO_CFG,
    Const.IMAGEINFO_SEED,
    WIDTH_HEIGHT_TARGET,
)
TARGET_TO_IMAGE_INFO_KEYS = {
    Const.IMAGEINFO_MODEL: (Const.IMAGEINFO_MODEL,),
    Const.IMAGEINFO_REFINER_MODEL: (Const.IMAGEINFO_REFINER_MODEL,),
    Const.IMAGEINFO_DETAILER_MODEL: (Const.IMAGEINFO_DETAILER_MODEL,),
    Const.IMAGEINFO_LORA_STACK: (Const.IMAGEINFO_LORA_STACK,),
    Const.IMAGEINFO_CLIP: (Const.IMAGEINFO_CLIP,),
    Const.IMAGEINFO_VAE: (Const.IMAGEINFO_VAE,),
    Const.IMAGEINFO_POSITIVE: (Const.IMAGEINFO_POSITIVE,),
    Const.IMAGEINFO_NEGATIVE: (Const.IMAGEINFO_NEGATIVE,),
    Const.IMAGEINFO_STEPS: (Const.IMAGEINFO_STEPS,),
    Const.IMAGEINFO_SAMPLER: (Const.IMAGEINFO_SAMPLER,),
    Const.IMAGEINFO_SCHEDULER: (Const.IMAGEINFO_SCHEDULER,),
    Const.IMAGEINFO_CFG: (Const.IMAGEINFO_CFG,),
    Const.IMAGEINFO_SEED: (Const.IMAGEINFO_SEED,),
    WIDTH_HEIGHT_TARGET: (Const.IMAGEINFO_WIDTH, Const.IMAGEINFO_HEIGHT),
}
HASH_EXTRA_KEYS_BY_MAIN_TARGET = {
    Const.IMAGEINFO_MODEL: ("Model hash",),
    Const.IMAGEINFO_REFINER_MODEL: ("Refiner hash",),
    Const.IMAGEINFO_DETAILER_MODEL: ("Detailer hash",),
    Const.IMAGEINFO_LORA_STACK: ("Lora hashes",),
    Const.IMAGEINFO_CLIP: ("Clip hashes",),
    Const.IMAGEINFO_VAE: ("VAE hash",),
}


def bool_or_default(value: Any, default: bool = False) -> bool:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return bool_or_default(value[0], default)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def normalize_extra_keys(value: Any) -> tuple[str, ...]:
    raw_rows: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return tuple()
        try:
            raw_rows = json.loads(text)
        except json.JSONDecodeError:
            if text.startswith(("[", "{")):
                return tuple()
            raw_rows = text.split(",")

    if isinstance(raw_rows, Mapping):
        raw_rows = raw_rows.get("items", raw_rows.get("extra_keys", []))
    if not isinstance(raw_rows, (list, tuple)):
        return tuple()

    output: list[str] = []
    seen: set[str] = set()
    for raw_row in raw_rows:
        key_value: Any = raw_row
        enabled = True
        if isinstance(raw_row, Mapping):
            key_value = raw_row.get("key", raw_row.get("name", ""))
            enabled = bool_or_default(raw_row.get("enabled"), True)
        elif isinstance(raw_row, (list, tuple)):
            if not raw_row:
                continue
            key_value = raw_row[0]
            if len(raw_row) >= 2:
                enabled = bool_or_default(raw_row[1], True)

        normalized = str(key_value).strip() if key_value is not None else ""
        if not enabled or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return tuple(output)


def normalize_main_fields(value: Any) -> tuple[str, ...]:
    raw_items: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return tuple()
        try:
            raw_items = json.loads(text)
        except json.JSONDecodeError:
            raw_items = text.split(",")

    if not isinstance(raw_items, (list, tuple, set)):
        return tuple()

    selected = {str(item).strip() for item in raw_items if str(item).strip()}
    return tuple(target for target in MAIN_FIELD_TARGETS if target in selected)


def normalize_encryption_targets(value: Any) -> dict[str, object] | None:
    raw_value = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            raw_value = json.loads(text)
        except json.JSONDecodeError:
            return None

    if not isinstance(raw_value, Mapping):
        return None

    main_fields = normalize_main_fields(raw_value.get("main_fields"))
    extra_keys = normalize_extra_keys(raw_value.get("extra_keys"))
    if not main_fields and not extra_keys:
        return None

    return {
        "version": VERSION,
        "main_fields": list(main_fields),
        "extra_keys": list(extra_keys),
    }


def build_encryption_targets(
    main_field_flags: Mapping[str, Any],
    extra_keys: Any = None,
) -> dict[str, object] | None:
    main_fields = tuple(
        target
        for target in MAIN_FIELD_TARGETS
        if bool_or_default(main_field_flags.get(target), False)
    )
    normalized_extra_keys = normalize_extra_keys(extra_keys)
    if not main_fields and not normalized_extra_keys:
        return None

    return {
        "version": VERSION,
        "main_fields": list(main_fields),
        "extra_keys": list(normalized_extra_keys),
    }


def image_info_with_encryption_targets(
    image_info: Mapping[str, Any] | None,
    encryption_targets: Any,
) -> dict[str, object]:
    output = dict(image_info) if isinstance(image_info, Mapping) else {}
    extras = output.get(Const.IMAGEINFO_EXTRAS)
    if isinstance(extras, Mapping):
        output[Const.IMAGEINFO_EXTRAS] = dict(extras)

    normalized_targets = normalize_encryption_targets(encryption_targets)
    if normalized_targets is None:
        output.pop(Const.IMAGEINFO_ENCRYPTION_TARGETS, None)
    else:
        output[Const.IMAGEINFO_ENCRYPTION_TARGETS] = normalized_targets
    return output


def encryption_targets_from_image_info(
    image_info: Mapping[str, Any] | None,
) -> dict[str, object] | None:
    if not isinstance(image_info, Mapping):
        return None
    return normalize_encryption_targets(image_info.get(Const.IMAGEINFO_ENCRYPTION_TARGETS))


def associated_hash_extra_keys(main_fields: Any) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for target in normalize_main_fields(main_fields):
        for extra_key in HASH_EXTRA_KEYS_BY_MAIN_TARGET.get(target, tuple()):
            if extra_key in seen:
                continue
            seen.add(extra_key)
            output.append(extra_key)
    return tuple(output)
