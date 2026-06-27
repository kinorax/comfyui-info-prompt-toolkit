# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast


class HasValueAny(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-HasValueAny",
            display_name="Has Value (Any)",
            category=Const.CATEGORY_UTILITY,
            inputs=[
                c_io.AnyType.Input(
                    "value",
                    optional=True,
                ),
            ],
            outputs=[
                c_io.Boolean.Output(
                    Cast.out_id("has_value"),
                    display_name="has_value",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, value=None) -> bool:
        return True

    @classmethod
    def execute(cls, value=None) -> c_io.NodeOutput:
        return c_io.NodeOutput(value is not None)
