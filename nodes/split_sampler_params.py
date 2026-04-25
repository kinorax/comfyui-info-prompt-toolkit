# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.sampler_params import (
    SAMPLER_PARAMS_KEY,
    sampler_params_payload_or_error,
    split_sampler_params_values,
)


class SplitSamplerParams(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-SplitSamplerParams",
            display_name="Split Sampler Params",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.SAMPLER_PARAMS_TYPE.Input(
                    SAMPLER_PARAMS_KEY,
                    display_name=SAMPLER_PARAMS_KEY,
                    extra_dict={"forceInput": True},
                ),
            ],
            outputs=[
                c_io.AnyType.Output(
                    Cast.out_id(Const.IMAGEINFO_SAMPLER),
                    display_name=Const.IMAGEINFO_SAMPLER,
                ),
                c_io.AnyType.Output(
                    Cast.out_id(Const.IMAGEINFO_SCHEDULER),
                    display_name=Const.IMAGEINFO_SCHEDULER,
                ),
                c_io.Int.Output(
                    Cast.out_id(Const.IMAGEINFO_STEPS),
                    display_name=Const.IMAGEINFO_STEPS,
                ),
                c_io.Float.Output(
                    Cast.out_id("denoise"),
                    display_name="denoise",
                ),
                c_io.Int.Output(
                    Cast.out_id(Const.IMAGEINFO_SEED),
                    display_name=Const.IMAGEINFO_SEED,
                ),
                c_io.Float.Output(
                    Cast.out_id(Const.IMAGEINFO_CFG),
                    display_name=Const.IMAGEINFO_CFG,
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        sampler_params: object,
    ) -> bool | str:
        # During Comfy validation, linked values are unresolved and arrive as None.
        # Required-input checks and type checks for links are handled by the framework.
        if sampler_params is None:
            return True

        _, error = sampler_params_payload_or_error(sampler_params)
        return True if error is None else error

    @classmethod
    def execute(
        cls,
        sampler_params: object,
    ) -> c_io.NodeOutput:
        payload, error = sampler_params_payload_or_error(sampler_params)
        if payload is None or error is not None:
            raise RuntimeError(f"Split Sampler Params: {error or 'sampler_params is required'}")

        return c_io.NodeOutput(*split_sampler_params_values(payload))
