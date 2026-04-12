from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import aspect_ratio_size as AspectRatioSize
from ..utils import cast as Cast

_RATIO_INPUT_MAX = 4096
_UNIT_OPTIONS: tuple[str, ...] = tuple(str(option) for option in AspectRatioSize.UNIT_OPTIONS)
_MIN_DIMENSION = min(AspectRatioSize.UNIT_OPTIONS)


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _size_payload(width: int, height: int) -> dict[str, int]:
    return {
        "width": int(width),
        "height": int(height),
    }


class AspectRatioToSize(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        socketless = dict(socketless=True)
        return c_io.Schema(
            node_id="IPT-AspectRatioToSize",
            display_name="Aspect Ratio to Size",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.Int.Input(
                    "width_ratio",
                    default=AspectRatioSize.DEFAULT_WIDTH_RATIO,
                    min=1,
                    max=_RATIO_INPUT_MAX,
                    step=1,
                    tooltip="Aspect ratio width",
                    **socketless,
                ),
                c_io.Int.Input(
                    "height_ratio",
                    default=AspectRatioSize.DEFAULT_HEIGHT_RATIO,
                    min=1,
                    max=_RATIO_INPUT_MAX,
                    step=1,
                    tooltip="Aspect ratio height",
                    **socketless,
                ),
                c_io.Combo.Input(
                    "min_unit",
                    options=_UNIT_OPTIONS,
                    default=str(AspectRatioSize.DEFAULT_UNIT),
                    tooltip="Minimum size step",
                    **socketless,
                ),
                c_io.Int.Input(
                    "width",
                    default=AspectRatioSize.DEFAULT_WIDTH,
                    min=_MIN_DIMENSION,
                    max=Const.MAX_RESOLUTION,
                    step=1,
                    tooltip="Width",
                    **socketless,
                ),
                c_io.Int.Input(
                    "height",
                    default=AspectRatioSize.DEFAULT_HEIGHT,
                    min=_MIN_DIMENSION,
                    max=Const.MAX_RESOLUTION,
                    step=1,
                    tooltip="Height",
                    **socketless,
                ),
                c_io.String.Input(
                    "actual_ratio",
                    default="",
                    tooltip="Display only",
                    **socketless,
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
        width_ratio: object,
        height_ratio: object,
        min_unit: object,
        width: object,
        height: object,
        actual_ratio: object | None = None,
    ) -> bool | str:
        width_ratio_int = _int_or_none(width_ratio)
        if width_ratio is not None and width_ratio_int is None:
            return "width_ratio must be an integer"
        if width_ratio_int is not None and width_ratio_int < 1:
            return "width_ratio must be 1 or greater"

        height_ratio_int = _int_or_none(height_ratio)
        if height_ratio is not None and height_ratio_int is None:
            return "height_ratio must be an integer"
        if height_ratio_int is not None and height_ratio_int < 1:
            return "height_ratio must be 1 or greater"

        min_unit_int = _int_or_none(min_unit)
        if min_unit is not None and min_unit_int is None:
            return "min_unit is invalid"
        if min_unit_int is not None and min_unit_int not in AspectRatioSize.UNIT_OPTIONS:
            return "min_unit is invalid"

        width_int = _int_or_none(width)
        if width is not None and width_int is None:
            return "width must be an integer"
        if width_int is not None:
            if width_int < _MIN_DIMENSION:
                return f"width must be {_MIN_DIMENSION} or greater"
            if width_int > Const.MAX_RESOLUTION:
                return f"width must be {Const.MAX_RESOLUTION} or less"

        height_int = _int_or_none(height)
        if height is not None and height_int is None:
            return "height must be an integer"
        if height_int is not None:
            if height_int < _MIN_DIMENSION:
                return f"height must be {_MIN_DIMENSION} or greater"
            if height_int > Const.MAX_RESOLUTION:
                return f"height must be {Const.MAX_RESOLUTION} or less"

        return True

    @classmethod
    def execute(
        cls,
        width_ratio: object,
        height_ratio: object,
        min_unit: object,
        width: object,
        height: object,
        actual_ratio: object | None = None,
    ) -> c_io.NodeOutput:
        width_out, height_out = AspectRatioSize.resolve_size(
            width=width,
            height=height,
            width_ratio=width_ratio,
            height_ratio=height_ratio,
            min_unit=min_unit,
            minimum=_MIN_DIMENSION,
            maximum=Const.MAX_RESOLUTION,
        )
        return c_io.NodeOutput(_size_payload(width_out, height_out), width_out, height_out)
