# Copyright 2026 kinorax
from __future__ import annotations

import random
import threading
import time

from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast
from ...utils.prompt_template import render_prompt_template
from ...utils.prompt_text import normalize_prompt_tokens

_CYCLE_INDEX_LOCK = threading.Lock()
_CYCLE_INDEX_BY_NODE_ID: dict[str, int] = {}
_CYCLE_INDEX_DEFAULT_KEY = "__prompt_template_default__"
_CYCLE_INDEX_RESET_THRESHOLD = 2_147_483_000
_CYCLE_INDEX_START_MIN = 0
_CYCLE_INDEX_START_MAX = 1000


class PromptTemplate(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        fi = dict(optional=True, force_input=True)
        return c_io.Schema(
            node_id="IPT-PromptTemplate",
            not_idempotent=True,
            display_name="Prompt Template",
            category=Const.CATEGORY_PROMPT,
            description="Supports // and /* ... */ comments, Dynamic Prompts-compatible templates, weighted wildcard lines from user/info_prompt_toolkit/wildcards, and custom $key replacement from extras.",
            hidden=[c_io.Hidden.unique_id],
            inputs=[
                c_io.String.Input(
                    "template",
                    default="",
                    multiline=True,
                    tooltip="Prompt template body. Type __ to suggest wildcards, then # after __name__ for item list.",
                ),
                c_io.String.Input(
                    "suffix",
                    tooltip="String to append to template",
                    **fi,
                ),
                Const.IMAGEINFO_EXTRAS_TYPE.Input(
                    Const.IMAGEINFO_EXTRAS,
                    optional=True,
                    tooltip="extras used for $key replacement in Dynamic Prompts",
                ),
                c_io.Int.Input(
                    "seed",
                    tooltip="Optional seed for deterministic random sampler",
                    **fi,
                ),
            ],
            outputs=[
                c_io.String.Output(
                    Cast.out_id("string"),
                    display_name="string",
                ),
                c_io.String.Output(
                    Cast.out_id("normalized_string"),
                    display_name="normalized_string",
                ),
            ],
        )

    @classmethod
    def _resolve_cycle_state_key(cls) -> str:
        unique_id = getattr(cls.hidden, "unique_id", None)
        if unique_id is None:
            return _CYCLE_INDEX_DEFAULT_KEY

        key = str(unique_id).strip()
        if not key:
            return _CYCLE_INDEX_DEFAULT_KEY
        return key

    @classmethod
    def _next_cycle_index(cls) -> int:
        state_key = cls._resolve_cycle_state_key()

        with _CYCLE_INDEX_LOCK:
            current = _CYCLE_INDEX_BY_NODE_ID.get(state_key)
            if current is None or current >= _CYCLE_INDEX_RESET_THRESHOLD:
                current = random.randint(_CYCLE_INDEX_START_MIN, _CYCLE_INDEX_START_MAX)

            _CYCLE_INDEX_BY_NODE_ID[state_key] = current + 1
            return current

    @classmethod
    def fingerprint_inputs(
        cls,
        template: object = "",
        suffix: object | None = None,
        extras: object | None = None,
        seed: object | None = None,
    ) -> int:
        # Dynamic Prompts include random choice, so always re-evaluate.
        return time.time_ns()

    @classmethod
    def execute(
        cls,
        template: str = "",
        suffix: str | None = None,
        extras: dict[str, object] | None = None,
        seed: int | None = None,
    ) -> c_io.NodeOutput:
        seeded_rng = None
        if seed is not None:
            seeded_rng = random.Random(int(seed))

        rendered = render_prompt_template(
            template=template,
            suffix=suffix,
            extras=extras,
            rng=seeded_rng,
            cycle_index=cls._next_cycle_index(),
        )
        normalized_rendered = normalize_prompt_tokens(rendered)
        return c_io.NodeOutput(rendered, normalized_rendered)
