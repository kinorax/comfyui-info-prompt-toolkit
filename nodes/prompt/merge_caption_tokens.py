# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast
from ...utils.prompt_text import merge_caption_tokens


def _string_or_none(value: object | None) -> str | None:
    if value is None:
        return None

    try:
        return str(value)
    except Exception:
        return None


class MergeCaptionTokens(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        fi = dict(optional=True, force_input=True)
        return c_io.Schema(
            node_id="IPT-MergeCaptionTokens",
            display_name="Merge Caption Tokens",
            category=Const.CATEGORY_PROMPT,
            inputs=[
                c_io.String.Input(
                    "start",
                    tooltip="Caption token text placed at the beginning. Duplicates are removed after splitting on ', ' and '. '.",
                    **fi,
                ),
                c_io.String.Input(
                    "end",
                    tooltip="Caption token text appended after start. Duplicates are removed after splitting on ', ' and '. '.",
                    **fi,
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
        start: object | None = None,
        end: object | None = None,
    ) -> c_io.NodeOutput:
        start_text = _string_or_none(start)
        end_text = _string_or_none(end)
        return c_io.NodeOutput(merge_caption_tokens(start_text, end_text))
