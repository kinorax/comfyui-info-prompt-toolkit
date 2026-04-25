# Copyright 2026 kinorax
import math

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast

_INPUT_IDS = (
    "any_01",
    "any_02",
    "any_03",
    "any_04",
    "any_05",
)


def _unwrap_singleton(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _unwrap_singleton(value[0])

    if isinstance(value, dict) and "__value__" in value:
        return _unwrap_singleton(value.get("__value__"))

    return value


def _is_invalid_value(value):
    unwrapped = _unwrap_singleton(value)

    if unwrapped is None:
        return True

    if isinstance(unwrapped, float) and math.isnan(unwrapped):
        return True

    if isinstance(unwrapped, str):
        text = unwrapped.strip()
        if not text:
            return True
        if text.lower() in {"none", "null", "nan"}:
            return True

    return False


class AnySwitchAny(c_io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return c_io.Schema(
            node_id="IPT-AnySwitchAny",
            display_name="Any Switch (Any)",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.AnyType.Input(
                    input_id,
                    optional=True,
                )
                for input_id in _INPUT_IDS
            ],
            outputs=[
                c_io.AnyType.Output(
                    Cast.out_id("any"),
                    display_name="any",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        any_01=None,
        any_02=None,
        any_03=None,
        any_04=None,
        any_05=None,
    ):
        for candidate in (any_01, any_02, any_03, any_04, any_05):
            if not _is_invalid_value(candidate):
                return c_io.NodeOutput(candidate)

        return c_io.NodeOutput(None)

