# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.a1111_infotext import extract_lora_stack_from_prompt
from ..utils.image_info_normalizer import normalize_lora_stack_with_comfy_options


class PromptToLoraStack(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-PromptToLoraStack",
            display_name="Prompt To Lora Stack",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Prompt text that may include <lora:...:...> tags",
                ),
            ],
            outputs=[
                c_io.String.Output(
                    Cast.out_id("prompt"),
                    display_name="prompt",
                ),
                Const.LORA_STACK_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO_LORA_STACK),
                    display_name=Const.IMAGEINFO_LORA_STACK,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
    ) -> c_io.NodeOutput:
        prompt_without_lora, lora_stack = extract_lora_stack_from_prompt(prompt)
        lora_stack_out = normalize_lora_stack_with_comfy_options(lora_stack)
        return c_io.NodeOutput(prompt_without_lora, lora_stack_out)
