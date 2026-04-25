# Copyright 2026 kinorax
from __future__ import annotations

import re

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast

SIZE_PATTERN = re.compile(r"^\s*(-?\d+)\s*[xX]\s*(-?\d+)\s*$")
_DEFAULT_WIDTH = 512
_DEFAULT_HEIGHT = 512


def _merged_extras(extras: dict[str, object] | None, key: str, value: str) -> dict[str, object]:
    output = dict(extras) if isinstance(extras, dict) else {}
    output[key] = value
    return output


def _normalized_key_or_none(key: object) -> str | None:
    if key is None:
        return None
    normalized = str(key).strip()
    return normalized or None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _size_payload(width: int, height: int) -> dict[str, int]:
    return {
        "width": width,
        "height": height,
    }


def _size_tuple_or_none(value: object) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _size_tuple_or_none(value[0])

    if isinstance(value, dict) and "__value__" in value:
        return _size_tuple_or_none(value.get("__value__"))

    if isinstance(value, dict):
        width = _int_or_none(value.get("width"))
        height = _int_or_none(value.get("height"))
        if width is None:
            width = _int_or_none(value.get("w"))
        if height is None:
            height = _int_or_none(value.get("h"))
        if width is None or height is None:
            return None
        return width, height

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        width = _int_or_none(value[0])
        height = _int_or_none(value[1])
        if width is None or height is None:
            return None
        return width, height

    if isinstance(value, str):
        m = SIZE_PATTERN.fullmatch(value)
        if not m:
            return None
        width = _int_or_none(m.group(1))
        height = _int_or_none(m.group(2))
        if width is None or height is None:
            return None
        return width, height

    return None


class SetSizeExtra(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-SetSizeExtra",
            display_name="Set Size Extra",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.String.Input(
                    "key",
                    tooltip="Parameter line key",
                ),
                Const.SIZE_TYPE.Input(
                    "size",
                    display_name="width x height",
                    extra_dict={
                        "default": _size_payload(_DEFAULT_WIDTH, _DEFAULT_HEIGHT),
                        "min": Const.MIN_RESOLUTION,
                        "max": Const.MAX_RESOLUTION,
                        "step": 1,
                    },
                ),
                Const.IMAGEINFO_EXTRAS_TYPE.Input(
                    Const.IMAGEINFO_EXTRAS,
                    optional=True,
                ),
            ],
            outputs=[
                Const.IMAGEINFO_EXTRAS_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO_EXTRAS),
                    display_name=Const.IMAGEINFO_EXTRAS,
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        key: object,
        size: object,
        extras: dict[str, object] | None = None,
    ) -> bool | str:
        normalized_key = _normalized_key_or_none(key)
        if key is not None and normalized_key is None:
            return "key is required"

        # During Comfy validation, linked values are unresolved and arrive as None.
        # Required-input checks and type checks for links are handled by the framework.
        if size is None:
            return True

        parsed_size = _size_tuple_or_none(size)
        if parsed_size is None:
            return "width x height must contain integer width and height"
        width_int, height_int = parsed_size
        if width_int < Const.MIN_RESOLUTION:
            return f"width must be {Const.MIN_RESOLUTION} or greater"
        if width_int > Const.MAX_RESOLUTION:
            return f"width must be {Const.MAX_RESOLUTION} or less"
        if height_int < Const.MIN_RESOLUTION:
            return f"height must be {Const.MIN_RESOLUTION} or greater"
        if height_int > Const.MAX_RESOLUTION:
            return f"height must be {Const.MAX_RESOLUTION} or less"
        return True

    @classmethod
    def execute(
        cls,
        key: object,
        size: object,
        extras: dict[str, object] | None = None,
    ) -> c_io.NodeOutput:
        normalized_key = _normalized_key_or_none(key)
        if normalized_key is None:
            raise ValueError("key is required")

        parsed_size = _size_tuple_or_none(size)
        if parsed_size is None:
            raise ValueError("width x height must contain integer width and height")

        width_int, height_int = parsed_size
        if width_int < Const.MIN_RESOLUTION or height_int < Const.MIN_RESOLUTION:
            raise ValueError(f"width and height must be {Const.MIN_RESOLUTION} or greater")
        if width_int > Const.MAX_RESOLUTION or height_int > Const.MAX_RESOLUTION:
            raise ValueError(f"width and height must be {Const.MAX_RESOLUTION} or less")
        rendered = f"{width_int}x{height_int}"
        output = _merged_extras(extras, normalized_key, rendered)
        return c_io.NodeOutput(output)
