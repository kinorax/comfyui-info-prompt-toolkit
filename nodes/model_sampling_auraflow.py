# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

from comfy_api.latest import io as c_io

from .. import const as Const


class ModelSamplingAuraFlow(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-ModelSamplingAuraFlow",
            display_name="ModelSamplingAuraFlow",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.MODEL_TYPE.Input(
                    "model",
                    tooltip="Attach ModelSamplingAuraFlow metadata to model",
                ),
                c_io.Float.Input(
                    "shift",
                    default=1.73,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                ),
            ],
            outputs=[
                Const.MODEL_TYPE.Output(
                    "MODEL",
                    display_name="model",
                ),
            ],
        )

    @classmethod
    def execute(cls, model: Any, shift: float) -> c_io.NodeOutput:
        if not isinstance(model, dict):
            raise RuntimeError("model is required")

        output = dict(model)
        output[Const.MODEL_VALUE_AURAFLOW_SHIFT_KEY] = float(shift)
        return c_io.NodeOutput(output)
