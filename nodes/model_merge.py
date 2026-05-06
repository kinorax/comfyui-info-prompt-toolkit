# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.model_merge import build_model_merge_value, validate_model_merge_inputs


class ModelMerge(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-ModelMerge",
            display_name="Model Merge",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.MODEL_TYPE.Input(
                    "base_model",
                    tooltip="Base IPT-Model. ratio=0 keeps this side.",
                ),
                Const.MODEL_TYPE.Input(
                    "merge_model",
                    tooltip="Second IPT-Model. ratio=1 keeps this side.",
                ),
                c_io.Float.Input(
                    "model_ratio",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="ModelMergeSimple ratio passed to runtime merge",
                ),
                c_io.Float.Input(
                    "clip_ratio",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="CLIPMergeSimple ratio. Ignored for diffusion_models",
                ),
            ],
            outputs=[
                Const.MODEL_TYPE.Output(
                    Cast.out_id("model"),
                    display_name="model",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        base_model: object | None = None,
        merge_model: object | None = None,
        model_ratio: object | None = None,
        clip_ratio: object | None = None,
    ) -> bool | str:
        if base_model is None or merge_model is None:
            return True

        error = validate_model_merge_inputs(base_model, merge_model)
        return True if error is None else error

    @classmethod
    def execute(
        cls,
        base_model: object,
        merge_model: object,
        model_ratio: float,
        clip_ratio: float,
    ) -> c_io.NodeOutput:
        error = validate_model_merge_inputs(base_model, merge_model)
        if error is not None:
            raise RuntimeError(error)

        merged = build_model_merge_value(
            base_model=base_model,
            merge_model=merge_model,
            model_ratio=model_ratio,
            clip_ratio=clip_ratio,
        )
        if merged is None:
            raise RuntimeError("failed to build merged IPT-Model")
        return c_io.NodeOutput(merged)
