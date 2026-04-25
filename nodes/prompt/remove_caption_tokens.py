# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast
from ...utils.prompt_text import remove_caption_tokens


def _string_or_none(value: object | None) -> str | None:
    if value is None:
        return None

    try:
        return str(value)
    except Exception:
        return None


class RemoveCaptionTokens(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-RemoveCaptionTokens",
            display_name="Remove Caption Tokens",
            category=Const.CATEGORY_PROMPT,
            inputs=[
                c_io.String.Input(
                    "string",
                    tooltip="Caption token text to filter. Tokens are split on ', ' and '. '.",
                    optional=True,
                    force_input=True,
                ),
                c_io.String.Input(
                    "remove",
                    default="",
                    optional=True,
                    tooltip="Caption tokens to remove from string. Can be typed directly or provided from another node. Split on ',' with trimming for each item and matches exact tokens only.",
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
        string: object | None = None,
        remove: object | None = None,
    ) -> c_io.NodeOutput:
        string_text = _string_or_none(string)
        remove_text = _string_or_none(remove)
        return c_io.NodeOutput(remove_caption_tokens(string_text, remove_text))
