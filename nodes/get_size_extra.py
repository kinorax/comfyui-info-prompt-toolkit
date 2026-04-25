# Copyright 2026 kinorax
from __future__ import annotations

import re

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast

SIZE_PATTERN = re.compile(r"^\s*(-?\d+)\s*[xX]\s*(-?\d+)\s*$")


def _normalized_key_or_none(key: object) -> str | None:
    if key is None:
        return None
    normalized = str(key).strip()
    return normalized or None


def _resolve(extras: dict[str, object] | None, key: str) -> tuple[int | None, int | None]:
    if not isinstance(extras, dict):
        return None, None
    if key not in extras:
        return None, None

    value = extras.get(key)
    text = None if value is None else str(value)
    if text is None:
        return None, None

    m = SIZE_PATTERN.fullmatch(text)
    if not m:
        return None, None

    try:
        width = int(m.group(1))
        height = int(m.group(2))
        return width, height
    except Exception:
        return None, None


def _size_payload_or_none(width: int | None, height: int | None) -> dict[str, int] | None:
    if width is None or height is None:
        return None
    return {
        "width": int(width),
        "height": int(height),
    }


class GetSizeExtra(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-GetSizeExtra",
            display_name="Get Size Extra",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.String.Input(
                    "key",
                    tooltip="Parameter line key",
                ),
                Const.IMAGEINFO_EXTRAS_TYPE.Input(
                    Const.IMAGEINFO_EXTRAS,
                    optional=True,
                ),
            ],
            outputs=[
                Const.SIZE_TYPE.Output(
                    Cast.out_id("size"),
                    display_name="width x height",
                ),
                c_io.Int.Output(
                    Cast.out_id("width"),
                    display_name="width",
                ),
                c_io.Int.Output(
                    Cast.out_id("height"),
                    display_name="height",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        key: object,
        extras: dict[str, object] | None = None,
    ) -> bool | str:
        normalized_key = _normalized_key_or_none(key)
        if key is not None and normalized_key is None:
            return "key is required"
        return True

    @classmethod
    def execute(
        cls,
        key: object,
        extras: dict[str, object] | None = None,
    ) -> c_io.NodeOutput:
        normalized_key = _normalized_key_or_none(key)
        if normalized_key is None:
            raise ValueError("key is required")
        width_out, height_out = _resolve(extras, normalized_key)
        size_out = _size_payload_or_none(width_out, height_out)
        return c_io.NodeOutput(size_out, width_out, height_out)
