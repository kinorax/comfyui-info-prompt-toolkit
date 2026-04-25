# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            return list(value[0])
        return list(value)
    return [value]


class LoopImageListRelaySimple(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-LoopImageListRelaySimple",
            display_name="Loop Image List Relay Simple",
            category=Const.CATEGORY_XYPLOT,
            is_input_list=True,
            is_dev_only=True,
            inputs=[
                c_io.Image.Input(
                    "image",
                    tooltip="Internal use only",
                ),
            ],
            outputs=[
                c_io.Image.Output(
                    Cast.out_id("image"),
                    display_name="image",
                ),
            ],
        )

    @classmethod
    def execute(cls, image: Any) -> c_io.NodeOutput:
        return c_io.NodeOutput(_flatten(image))