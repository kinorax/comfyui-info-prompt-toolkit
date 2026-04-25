# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast
from ...utils.prompt_text import normalize_prompt_tokens


class NormalizePromptTokens(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-NormalizePromptTokens",
            display_name="Normalize Prompt Tokens",
            category=Const.CATEGORY_PROMPT,
            inputs=[
                c_io.String.Input(
                    "string",
                    optional=True,
                    force_input=True,
                    tooltip="Prompt text treated as token list (non-structured text)",
                ),
            ],
            outputs=[
                c_io.String.Output(
                    Cast.out_id("string"),
                    display_name="string",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        string: str | None = None,
    ) -> c_io.NodeOutput:
        if string is None:
            return c_io.NodeOutput(None)
        return c_io.NodeOutput(normalize_prompt_tokens(string))
