# Copyright 2026 kinorax
from __future__ import annotations

import json
from typing import Any

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast


def _normalized_keys(value: object | None) -> tuple[str, ...]:
    if value is None:
        return tuple()

    output: list[str] = []
    seen: set[str] = set()
    for raw_item in str(value).split(","):
        normalized = raw_item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return tuple(output)


def _bool_or_default(value: Any, default: bool) -> bool:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _bool_or_default(value[0], default)
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


def _normalized_enabled_keys(value: object | None) -> tuple[str, ...]:
    raw_rows: object = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return tuple()
        try:
            raw_rows = json.loads(text)
        except json.JSONDecodeError:
            if text.startswith(("[", "{")):
                return tuple()
            return _normalized_keys(text)

    if isinstance(raw_rows, dict):
        raw_rows = raw_rows.get("items", [])
    if not isinstance(raw_rows, (list, tuple)):
        return tuple()

    output: list[str] = []
    seen: set[str] = set()
    for raw_row in raw_rows:
        enabled = True
        key_value: object = raw_row
        if isinstance(raw_row, dict):
            key_value = raw_row.get("key", raw_row.get("name", ""))
            enabled = _bool_or_default(raw_row.get("enabled"), True)
        elif isinstance(raw_row, (list, tuple)):
            if not raw_row:
                continue
            key_value = raw_row[0]
            if len(raw_row) >= 2:
                enabled = _bool_or_default(raw_row[1], True)

        normalized = str(key_value).strip() if key_value is not None else ""
        if not enabled or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return tuple(output)


def _copied_image_info(image_info: dict[str, object] | None) -> dict[str, object]:
    output = dict(image_info) if isinstance(image_info, dict) else {}
    extras_raw = output.get(Const.IMAGEINFO_EXTRAS)
    if isinstance(extras_raw, dict):
        output[Const.IMAGEINFO_EXTRAS] = dict(extras_raw)
    return output


def _removed_extra_keys(
    image_info: dict[str, object] | None,
    keys: tuple[str, ...],
) -> dict[str, object]:
    output = _copied_image_info(image_info)
    extras = output.get(Const.IMAGEINFO_EXTRAS)
    if not isinstance(extras, dict) or not keys:
        return output

    for key in keys:
        extras.pop(key, None)
    if extras:
        output[Const.IMAGEINFO_EXTRAS] = extras
    else:
        output.pop(Const.IMAGEINFO_EXTRAS, None)
    return output


class RemoveImageInfoExtraKeys(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-RemoveImageInfoExtraKeys",
            display_name="Remove Image Info Extra Keys (Deprecated)",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.IMAGEINFO_TYPE.Input(
                    Const.IMAGEINFO,
                    optional=True,
                ),
                c_io.String.Input(
                    "key",
                    display_name="remove",
                    default="",
                    optional=True,
                    tooltip="Comma-separated keys in image_info.extras to remove. Matches exact keys after trimming each item.",
                ),
            ],
            outputs=[
                Const.IMAGEINFO_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO),
                    display_name=Const.IMAGEINFO,
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        image_info: dict[str, object] | None = None,
        key: object = None,
    ) -> bool | str:
        return True

    @classmethod
    def execute(
        cls,
        image_info: dict[str, object] | None = None,
        key: object = None,
    ) -> c_io.NodeOutput:
        normalized_keys = _normalized_keys(key)
        output = _removed_extra_keys(image_info, normalized_keys)
        return c_io.NodeOutput(output)


class RemoveImageInfoExtraKeysV2(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-RemoveImageInfoExtraKeysV2",
            display_name="Remove Image Info Extra Keys",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.IMAGEINFO_TYPE.Input(
                    Const.IMAGEINFO,
                    optional=True,
                ),
                c_io.String.Input(
                    "items",
                    display_name="remove",
                    default="[]",
                    multiline=True,
                    socketless=True,
                    tooltip="JSON rows used by the key toggle editor. Enabled keys are removed from image_info.extras.",
                ),
            ],
            outputs=[
                Const.IMAGEINFO_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO),
                    display_name=Const.IMAGEINFO,
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        image_info: dict[str, object] | None = None,
        items: object = None,
    ) -> bool | str:
        return True

    @classmethod
    def execute(
        cls,
        image_info: dict[str, object] | None = None,
        items: object = None,
    ) -> c_io.NodeOutput:
        normalized_keys = _normalized_enabled_keys(items)
        output = _removed_extra_keys(image_info, normalized_keys)
        return c_io.NodeOutput(output)
