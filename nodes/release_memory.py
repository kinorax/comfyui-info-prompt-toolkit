# Copyright 2026 kinorax
from __future__ import annotations

import time
from typing import Any

from comfy_api.latest import io as c_io
from comfy_execution.graph_utils import is_link

from .. import const as Const
from ..utils import cast as Cast
from ..utils.release_memory import bool_or_default, release_memory, release_memory_summary_text, tagger_runtime_or_default

_MATCH_TEMPLATE = c_io.MatchType.Template("release_memory_passthrough", c_io.AnyType)
_PASSTHROUGH_INPUT_ID = "passthrough"


def _release_options(
    generation_runtime: Any,
    sam3_runtime: Any,
    tagger_runtime: Any,
    gc_cuda_cleanup: Any,
    pixai_tagger_runtime: Any = None,
    oppai_oracle_tagger_runtime: Any = None,
) -> dict[str, bool]:
    return {
        "generation_runtime": bool_or_default(generation_runtime, True),
        "sam3_runtime": bool_or_default(sam3_runtime, True),
        "tagger_runtime": tagger_runtime_or_default(
            tagger_runtime,
            pixai_tagger_runtime,
            oppai_oracle_tagger_runtime,
        ),
        "gc_cuda_cleanup": bool_or_default(gc_cuda_cleanup, True),
    }


def _skipped_result(requested: dict[str, bool]) -> dict[str, Any]:
    return {
        "ok": True,
        "skipped": True,
        "reason": "after_not_connected",
        "requested": dict(requested),
        "steps": [],
        "errors": [],
    }


def _input_has_link(prompt: Any, unique_id: Any, input_id: str) -> bool:
    if not isinstance(prompt, dict) or unique_id is None:
        return False

    node = prompt.get(str(unique_id), {}) or {}
    if not isinstance(node, dict):
        return False

    inputs = node.get("inputs", {}) or {}
    if not isinstance(inputs, dict):
        return False

    return is_link(inputs.get(input_id))


class ReleaseMemory(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-ReleaseMemory",
            display_name="Release Memory",
            category=Const.CATEGORY_UTILITY,
            is_output_node=True,
            not_idempotent=True,
            search_aliases=["free memory", "clear vram", "release vram", "memory cleanup"],
            hidden=[c_io.Hidden.prompt, c_io.Hidden.unique_id],
            inputs=[
                c_io.MatchType.Input(
                    _PASSTHROUGH_INPUT_ID,
                    template=_MATCH_TEMPLATE,
                    display_name="after",
                    optional=True,
                    tooltip="Connect this input to run workflow memory release; the value is returned unchanged",
                ),
                c_io.Boolean.Input(
                    "generation_runtime",
                    default=True,
                    tooltip="Release generation runtime references, including sampler model, CLIP, and VAE runtimes",
                ),
                c_io.Boolean.Input(
                    "sam3_runtime",
                    default=True,
                    tooltip="Release the SAM3 Prompt To Mask runtime cache",
                ),
                c_io.Boolean.Input(
                    "tagger_runtime",
                    display_name="Tagger Runtime",
                    default=True,
                    tooltip="Release the PixAI Tagger and OppaiOracle Tagger runtime caches",
                ),
                c_io.Boolean.Input(
                    "gc_cuda_cleanup",
                    display_name="GC & CUDA Cleanup",
                    default=True,
                    tooltip=(
                        "Run Python garbage collection, then ask ComfyUI and torch "
                        "to clear available CUDA cache memory"
                    ),
                ),
            ],
            outputs=[
                c_io.MatchType.Output(
                    template=_MATCH_TEMPLATE,
                    id=Cast.out_id(_PASSTHROUGH_INPUT_ID),
                    display_name="then",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        passthrough: Any = None,
        generation_runtime: Any = True,
        sam3_runtime: Any = True,
        tagger_runtime: Any = None,
        gc_cuda_cleanup: Any = True,
        pixai_tagger_runtime: Any = None,
        oppai_oracle_tagger_runtime: Any = None,
    ) -> bool | str:
        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        passthrough: Any = None,
        generation_runtime: Any = True,
        sam3_runtime: Any = True,
        tagger_runtime: Any = None,
        gc_cuda_cleanup: Any = True,
        pixai_tagger_runtime: Any = None,
        oppai_oracle_tagger_runtime: Any = None,
    ) -> int:
        return time.time_ns()

    @classmethod
    def execute(
        cls,
        passthrough: Any = None,
        generation_runtime: Any = True,
        sam3_runtime: Any = True,
        tagger_runtime: Any = None,
        gc_cuda_cleanup: Any = True,
        pixai_tagger_runtime: Any = None,
        oppai_oracle_tagger_runtime: Any = None,
    ) -> c_io.NodeOutput:
        requested = _release_options(
            generation_runtime,
            sam3_runtime,
            tagger_runtime,
            gc_cuda_cleanup,
            pixai_tagger_runtime,
            oppai_oracle_tagger_runtime,
        )

        after_connected = _input_has_link(
            getattr(cls.hidden, "prompt", None),
            getattr(cls.hidden, "unique_id", None),
            _PASSTHROUGH_INPUT_ID,
        )
        if not after_connected:
            result = _skipped_result(requested)
            print("[IPT-ReleaseMemory] skipped reason=after_not_connected")
            return c_io.NodeOutput(
                passthrough,
                ui={"release_memory": [result]},
            )

        result = release_memory(**requested)
        print(f"[IPT-ReleaseMemory] ok={bool(result.get('ok'))} {release_memory_summary_text(result)}")
        return c_io.NodeOutput(
            passthrough,
            ui={"release_memory": [result]},
        )
