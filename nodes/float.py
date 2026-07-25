# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast


MIN_DECIMALS = 0
MAX_DECIMALS = 10


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


class Float(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-Float",
            display_name="Float",
            category=Const.CATEGORY_UTILITY,
            inputs=[
                c_io.Float.Input(
                    "value",
                    default=0.0,
                    step=0.01,
                    tooltip="Float value",
                ),
                c_io.Int.Input(
                    "decimals",
                    default=2,
                    min=MIN_DECIMALS,
                    max=MAX_DECIMALS,
                    step=1,
                    tooltip="Number of digits after decimal point",
                ),
            ],
            outputs=[
                c_io.Float.Output(
                    Cast.out_id("float"),
                    display_name="float",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, value: object, decimals: object) -> bool | str:
        if _float_or_none(value) is None:
            return "value must be a float"

        decimals_int = _int_or_none(decimals)
        if decimals_int is None:
            return "decimals must be an integer"
        if not MIN_DECIMALS <= decimals_int <= MAX_DECIMALS:
            return f"decimals must be between {MIN_DECIMALS} and {MAX_DECIMALS}"
        return True

    @classmethod
    def execute(cls, value: object, decimals: object) -> c_io.NodeOutput:
        value_float = _float_or_none(value)
        if value_float is None:
            raise ValueError("value must be a float")

        decimals_int = _int_or_none(decimals)
        if decimals_int is None:
            raise ValueError("decimals must be an integer")
        if not MIN_DECIMALS <= decimals_int <= MAX_DECIMALS:
            raise ValueError(
                f"decimals must be between {MIN_DECIMALS} and {MAX_DECIMALS}"
            )

        rounded = float(f"{value_float:.{decimals_int}f}")
        return c_io.NodeOutput(rounded)
