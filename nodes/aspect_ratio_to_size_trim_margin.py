# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import aspect_ratio_trim_margin_size as TrimMarginSize
from ..utils import cast as Cast

_RATIO_INPUT_MAX = 4096
_UNIT_OPTIONS: tuple[str, ...] = tuple(str(option) for option in TrimMarginSize.UNIT_OPTIONS)
_MIN_DIMENSION = min(TrimMarginSize.UNIT_OPTIONS)
_MAX_MARGIN = Const.MAX_RESOLUTION - _MIN_DIMENSION


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _size_payload(width: int, height: int) -> dict[str, int]:
    return {"width": int(width), "height": int(height)}


def _margin_payload(top: int, right: int, bottom: int, left: int) -> dict[str, int]:
    return {
        "top": int(top),
        "right": int(right),
        "bottom": int(bottom),
        "left": int(left),
    }


class AspectRatioToSizeTrimMargin(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        socketless = dict(socketless=True)
        margin_input = dict(
            default=TrimMarginSize.DEFAULT_MARGIN,
            min=0,
            max=_MAX_MARGIN,
            step=1,
            **socketless,
        )
        return c_io.Schema(
            node_id="IPT-AspectRatioToSizeTrimMargin",
            display_name="Aspect Ratio to Size (Trim Margin)",
            category=Const.CATEGORY_IMAGEINFO,
            search_aliases=["aspect ratio overscan", "sampling size trim", "trim margin size"],
            inputs=[
                c_io.Int.Input(
                    "width_ratio",
                    default=TrimMarginSize.DEFAULT_WIDTH_RATIO,
                    min=1,
                    max=_RATIO_INPUT_MAX,
                    step=1,
                    tooltip="Aspect ratio width after trimming margins",
                    **socketless,
                ),
                c_io.Int.Input(
                    "height_ratio",
                    default=TrimMarginSize.DEFAULT_HEIGHT_RATIO,
                    min=1,
                    max=_RATIO_INPUT_MAX,
                    step=1,
                    tooltip="Aspect ratio height after trimming margins",
                    **socketless,
                ),
                c_io.Combo.Input(
                    "min_unit",
                    options=_UNIT_OPTIONS,
                    default=str(TrimMarginSize.DEFAULT_UNIT),
                    tooltip="Minimum size step for the sampling size including margins",
                    **socketless,
                ),
                c_io.Int.Input(
                    "margin_top",
                    tooltip="Linked with margin_bottom; their total is 0 or min_unit",
                    **margin_input,
                ),
                c_io.Int.Input(
                    "margin_right",
                    tooltip="Linked with margin_left; their total is 0 or min_unit",
                    **margin_input,
                ),
                c_io.Int.Input(
                    "margin_bottom",
                    tooltip="Linked with margin_top; their total is 0 or min_unit",
                    **margin_input,
                ),
                c_io.Int.Input(
                    "margin_left",
                    tooltip="Linked with margin_right; their total is 0 or min_unit",
                    **margin_input,
                ),
                c_io.Int.Input(
                    "width",
                    default=TrimMarginSize.DEFAULT_WIDTH,
                    min=_MIN_DIMENSION,
                    max=Const.MAX_RESOLUTION,
                    step=1,
                    tooltip="Sampling width including margins (display only)",
                    **socketless,
                ),
                c_io.Int.Input(
                    "height",
                    default=TrimMarginSize.DEFAULT_HEIGHT,
                    min=_MIN_DIMENSION,
                    max=Const.MAX_RESOLUTION,
                    step=1,
                    tooltip="Sampling height including margins (display only)",
                    **socketless,
                ),
                c_io.Int.Input(
                    "actual_width",
                    default=TrimMarginSize.DEFAULT_WIDTH,
                    min=_MIN_DIMENSION,
                    max=Const.MAX_RESOLUTION,
                    step=1,
                    tooltip="Width remaining after trimming margins",
                    **socketless,
                ),
                c_io.Int.Input(
                    "actual_height",
                    default=TrimMarginSize.DEFAULT_HEIGHT,
                    min=_MIN_DIMENSION,
                    max=Const.MAX_RESOLUTION,
                    step=1,
                    tooltip="Height remaining after trimming margins",
                    **socketless,
                ),
                c_io.String.Input(
                    "actual_ratio",
                    default="",
                    tooltip="Actual ratio after trimming margins (display only)",
                    **socketless,
                ),
            ],
            outputs=[
                Const.SIZE_TYPE.Output(Cast.out_id("size"), display_name="width x height"),
                c_io.Int.Output(Cast.out_id("width"), display_name="width"),
                c_io.Int.Output(Cast.out_id("height"), display_name="height"),
                Const.MARGIN_TYPE.Output(Cast.out_id("margin"), display_name="margin"),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        width_ratio: object,
        height_ratio: object,
        min_unit: object,
        margin_top: object,
        margin_right: object,
        margin_bottom: object,
        margin_left: object,
        width: object,
        height: object,
        actual_width: object,
        actual_height: object,
        actual_ratio: object | None = None,
    ) -> bool | str:
        width_ratio_int = _int_or_none(width_ratio)
        if width_ratio_int is None or width_ratio_int < 1:
            return "width_ratio must be an integer of 1 or greater"

        height_ratio_int = _int_or_none(height_ratio)
        if height_ratio_int is None or height_ratio_int < 1:
            return "height_ratio must be an integer of 1 or greater"

        min_unit_int = _int_or_none(min_unit)
        if min_unit_int is None or min_unit_int not in TrimMarginSize.UNIT_OPTIONS:
            return "min_unit is invalid"

        margins: dict[str, int] = {}
        for name, value in (
            ("margin_top", margin_top),
            ("margin_right", margin_right),
            ("margin_bottom", margin_bottom),
            ("margin_left", margin_left),
        ):
            parsed = _int_or_none(value)
            if parsed is None or parsed < 0:
                return f"{name} must be a non-negative integer"
            if parsed > _MAX_MARGIN:
                return f"{name} must be {_MAX_MARGIN} or less"
            if parsed > min_unit_int:
                return f"{name} must be min_unit ({min_unit_int}) or less"
            margins[name] = parsed

        for name, value in (
            ("width", width),
            ("height", height),
            ("actual_width", actual_width),
            ("actual_height", actual_height),
        ):
            parsed = _int_or_none(value)
            if parsed is None:
                return f"{name} must be an integer"
            if parsed < _MIN_DIMENSION:
                return f"{name} must be {_MIN_DIMENSION} or greater"
            if parsed > Const.MAX_RESOLUTION:
                return f"{name} must be {Const.MAX_RESOLUTION} or less"

        horizontal_margin = margins["margin_left"] + margins["margin_right"]
        vertical_margin = margins["margin_top"] + margins["margin_bottom"]
        if horizontal_margin not in (0, min_unit_int):
            return "margin_left and margin_right must total 0 or min_unit"
        if vertical_margin not in (0, min_unit_int):
            return "margin_top and margin_bottom must total 0 or min_unit"
        if horizontal_margin + _MIN_DIMENSION > Const.MAX_RESOLUTION:
            return "left and right margins leave no valid width"
        if vertical_margin + _MIN_DIMENSION > Const.MAX_RESOLUTION:
            return "top and bottom margins leave no valid height"
        return True

    @classmethod
    def execute(
        cls,
        width_ratio: object,
        height_ratio: object,
        min_unit: object,
        margin_top: object,
        margin_right: object,
        margin_bottom: object,
        margin_left: object,
        width: object,
        height: object,
        actual_width: object,
        actual_height: object,
        actual_ratio: object | None = None,
    ) -> c_io.NodeOutput:
        validation = cls.validate_inputs(
            width_ratio,
            height_ratio,
            min_unit,
            margin_top,
            margin_right,
            margin_bottom,
            margin_left,
            width,
            height,
            actual_width,
            actual_height,
            actual_ratio,
        )
        if validation is not True:
            raise ValueError(validation)

        top = int(margin_top)
        right = int(margin_right)
        bottom = int(margin_bottom)
        left = int(margin_left)
        content_width, content_height = TrimMarginSize.resolve_size(
            width=actual_width,
            height=actual_height,
            width_ratio=width_ratio,
            height_ratio=height_ratio,
            min_unit=min_unit,
            margin_top=top,
            margin_right=right,
            margin_bottom=bottom,
            margin_left=left,
            minimum=_MIN_DIMENSION,
            maximum=Const.MAX_RESOLUTION,
        )
        sampling_width = content_width + left + right
        sampling_height = content_height + top + bottom
        return c_io.NodeOutput(
            _size_payload(sampling_width, sampling_height),
            sampling_width,
            sampling_height,
            _margin_payload(top, right, bottom, left),
        )
