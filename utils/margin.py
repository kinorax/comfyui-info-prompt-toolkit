# Copyright 2026 kinorax
from __future__ import annotations

import json
from typing import Mapping

from .. import const as Const

MARGIN_KEYS = ("top", "right", "bottom", "left")
_EXTRA_SCHEMA_KEY = "schema"
_EXTRA_PAYLOAD_KEY = "payload"
_MARGIN_EXTRA_SCHEMA = "margin"


def normalized_extra_key_or_none(key: object) -> str | None:
    if key is None:
        return None
    normalized = str(key).strip()
    return normalized or None


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


def margin_payload_or_error(
    value: object,
) -> tuple[dict[str, int] | None, str | None]:
    if value is None:
        return None, None

    unwrapped = _unwrap_value(value)
    if unwrapped is None:
        return None, None
    if not isinstance(unwrapped, Mapping):
        return None, "margin must be an object"

    payload: dict[str, int] = {}
    for key in MARGIN_KEYS:
        parsed = _normalized_int(
            unwrapped.get(key),
            minimum=0,
            maximum=Const.MAX_RESOLUTION,
        )
        if parsed is None:
            return None, f"{key} must be an integer from 0 to {Const.MAX_RESOLUTION}"
        payload[key] = parsed
    return payload, None


def serialize_margin_extra_json(margin: object) -> str | None:
    payload, error = margin_payload_or_error(margin)
    if payload is None:
        if error is None:
            return None
        raise RuntimeError(f"Set Margin Extra: {error}")

    envelope = {
        _EXTRA_SCHEMA_KEY: _MARGIN_EXTRA_SCHEMA,
        _EXTRA_PAYLOAD_KEY: payload,
    }
    return json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))


def deserialize_margin_extra(
    raw_value: object,
    *,
    key: str,
) -> dict[str, int] | None:
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
                f"Get Margin Extra: extra '{key}' is not valid Margin JSON"
            ) from exc

    if not isinstance(parsed, Mapping):
        raise RuntimeError(f"Get Margin Extra: extra '{key}' is not valid Margin JSON")

    schema = str(parsed.get(_EXTRA_SCHEMA_KEY) or "").strip()
    if schema != _MARGIN_EXTRA_SCHEMA or _EXTRA_PAYLOAD_KEY not in parsed:
        raise RuntimeError(f"Get Margin Extra: extra '{key}' is not valid Margin JSON")

    payload, _error = margin_payload_or_error(parsed.get(_EXTRA_PAYLOAD_KEY))
    if payload is None:
        raise RuntimeError(f"Get Margin Extra: extra '{key}' is not valid Margin JSON")
    return payload


def split_margin_values(
    payload: dict[str, int] | None,
) -> tuple[int | None, int | None, int | None, int | None]:
    if not isinstance(payload, dict):
        return None, None, None, None
    return (
        payload.get("top"),
        payload.get("right"),
        payload.get("bottom"),
        payload.get("left"),
    )


def _unwrap_value(value: object) -> object:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _unwrap_value(value[0])
    if isinstance(value, Mapping) and "__value__" in value:
        return _unwrap_value(value.get("__value__"))
    return value


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
