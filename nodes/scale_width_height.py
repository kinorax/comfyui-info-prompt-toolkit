# Copyright 2026 kinorax
from __future__ import annotations

import math
import re

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils import scale_size as ScaleSize

SIZE_PATTERN = re.compile(r"^\s*(-?\d+)\s*[xX]\s*(-?\d+)\s*$")
_UNIT_OPTIONS: tuple[str, ...] = tuple(str(option) for option in ScaleSize.UNIT_OPTIONS)


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _size_payload(width: int, height: int) -> dict[str, int]:
    return {
        "width": int(width),
        "height": int(height),
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
        matched = SIZE_PATTERN.fullmatch(value)
        if not matched:
            return None
        width = _int_or_none(matched.group(1))
        height = _int_or_none(matched.group(2))
        if width is None or height is None:
            return None
        return width, height

    return None


class ScaleWidthHeight(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-ScaleWidthHeight",
            display_name="Scale (width x height)",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.SIZE_TYPE.Input(
                    "size",
                    display_name="width x height",
                    extra_dict={
                        "default": _size_payload(ScaleSize.DEFAULT_WIDTH, ScaleSize.DEFAULT_HEIGHT),
                        "min": Const.MIN_RESOLUTION,
                        "max": Const.MAX_RESOLUTION,
                        "step": 1,
                    },
                ),
                c_io.Float.Input(
                    "scale_by",
                    default=ScaleSize.DEFAULT_SCALE_BY,
                    min=0.01,
                    max=64.0,
                    step=0.01,
                    tooltip="Scale factor",
                ),
                c_io.Combo.Input(
                    "min_unit",
                    options=_UNIT_OPTIONS,
                    default=str(ScaleSize.DEFAULT_UNIT),
                    tooltip="Minimum size step",
                ),
                c_io.String.Input(
                    "actual_ratio",
                    default="",
                    socketless=True,
                    tooltip="Display only",
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
        size: object,
        scale_by: object,
        min_unit: object,
        actual_ratio: object | None = None,
    ) -> bool | str:
        parsed_scale = _float_or_none(scale_by)
        if scale_by is not None and parsed_scale is None:
            return "scale_by must be a number"
        if parsed_scale is not None and parsed_scale <= 0:
            return "scale_by must be greater than 0"

        min_unit_int = _int_or_none(min_unit)
        if min_unit is not None and min_unit_int is None:
            return "min_unit is invalid"
        if min_unit_int is not None and min_unit_int not in ScaleSize.UNIT_OPTIONS:
            return "min_unit is invalid"

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
        size: object,
        scale_by: object,
        min_unit: object,
        actual_ratio: object | None = None,
    ) -> c_io.NodeOutput:
        parsed_size = _size_tuple_or_none(size)
        if parsed_size is None:
            raise ValueError("width x height must contain integer width and height")

        width_in, height_in = parsed_size
        if width_in < Const.MIN_RESOLUTION or height_in < Const.MIN_RESOLUTION:
            raise ValueError(f"width and height must be {Const.MIN_RESOLUTION} or greater")
        if width_in > Const.MAX_RESOLUTION or height_in > Const.MAX_RESOLUTION:
            raise ValueError(f"width and height must be {Const.MAX_RESOLUTION} or less")

        parsed_scale = _float_or_none(scale_by)
        if parsed_scale is None:
            raise ValueError("scale_by must be a number")
        if parsed_scale <= 0:
            raise ValueError("scale_by must be greater than 0")

        min_unit_int = _int_or_none(min_unit)
        if min_unit_int is None or min_unit_int not in ScaleSize.UNIT_OPTIONS:
            raise ValueError("min_unit is invalid")

        width_out, height_out = ScaleSize.resolve_scaled_size(
            width=width_in,
            height=height_in,
            scale_by=parsed_scale,
            min_unit=min_unit_int,
            minimum=Const.MIN_RESOLUTION,
            maximum=Const.MAX_RESOLUTION,
        )
        ratio_text = ScaleSize.render_actual_ratio(
            base_width=width_in,
            base_height=height_in,
            width=width_out,
            height=height_out,
        )
        return c_io.NodeOutput(
            _size_payload(width_out, height_out),
            width_out,
            height_out,
            ui={"actual_ratio": [ratio_text]},
        )
