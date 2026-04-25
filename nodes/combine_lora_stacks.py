# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast


def _lora_stack_or_none(value: object | None) -> list[object] | None:
    if value is None:
        return None

    if not isinstance(value, list):
        return None

    return list(value)


class CombineLoraStacks(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-CombineLoraStacks",
            display_name="Combine Lora Stacks",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.LORA_STACK_TYPE.Input(
                    "lora_stack1",
                    tooltip="First lora stack",
                    optional=True,
                ),
                Const.LORA_STACK_TYPE.Input(
                    "lora_stack2",
                    tooltip="Second lora stack",
                    optional=True,
                ),
            ],
            outputs=[
                Const.LORA_STACK_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO_LORA_STACK),
                    display_name=Const.IMAGEINFO_LORA_STACK,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        lora_stack1: object | None = None,
        lora_stack2: object | None = None,
    ) -> c_io.NodeOutput:
        stack1 = _lora_stack_or_none(lora_stack1)
        stack2 = _lora_stack_or_none(lora_stack2)

        if stack1 is None and stack2 is None:
            return c_io.NodeOutput(None)
        if stack1 is None:
            return c_io.NodeOutput(stack2)
        if stack2 is None:
            return c_io.NodeOutput(stack1)

        merged = list(stack1)
        merged.extend(stack2)
        return c_io.NodeOutput(merged)

