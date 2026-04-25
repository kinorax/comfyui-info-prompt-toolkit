# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast
from ...utils.prompt_text import combine_prompt_text


def _string_or_none(value: object | None) -> str | None:
    if value is None:
        return None

    try:
        return str(value)
    except Exception:
        return None


class CombinePrompts(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        fi = dict(optional=True, force_input=True)
        return c_io.Schema(
            node_id="IPT-CombinePrompts",
            display_name="Combine Prompts",
            category=Const.CATEGORY_PROMPT,
            inputs=[
                c_io.String.Input(
                    "start",
                    tooltip="Text placed at the beginning (normalized for prompt separator)",
                    **fi,
                ),
                c_io.String.Input(
                    "end",
                    tooltip="Text appended after start",
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
        base_text: object | None = None,
        append_text: object | None = None,
        prefix: object | None = None,
        suffix: object | None = None,
    ) -> c_io.NodeOutput:
        # Backward compatibility for graphs created before input rename.
        if start is None:
            if base_text is not None:
                start = base_text
            elif prefix is not None:
                start = prefix
        if end is None:
            if append_text is not None:
                end = append_text
            elif suffix is not None:
                end = suffix

        start_text = _string_or_none(start)
        end_text = _string_or_none(end)
        return c_io.NodeOutput(combine_prompt_text(start_text, end_text))
