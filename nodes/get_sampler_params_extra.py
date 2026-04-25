# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.lora_stack_extra import normalized_extra_key_or_none
from ..utils.sampler_params import (
    SAMPLER_PARAMS_KEY,
    deserialize_sampler_params_extra,
    split_sampler_params_values,
)


class GetSamplerParamsExtra(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-GetSamplerParamsExtra",
            display_name="Get Sampler Params Extra",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.IMAGEINFO_EXTRAS_TYPE.Input(
                    Const.IMAGEINFO_EXTRAS,
                    optional=True,
                ),
                c_io.String.Input(
                    "key",
                    tooltip="Parameter line key",
                ),
            ],
            outputs=[
                Const.SAMPLER_PARAMS_TYPE.Output(
                    Cast.out_id(SAMPLER_PARAMS_KEY),
                    display_name=SAMPLER_PARAMS_KEY,
                ),
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
        extras: dict[str, object] | None = None,
        key: object = None,
    ) -> bool | str:
        normalized_key = normalized_extra_key_or_none(key)
        if key is not None and normalized_key is None:
            return "key is required"
        return True

    @classmethod
    def execute(
        cls,
        extras: dict[str, object] | None = None,
        key: object = None,
    ) -> c_io.NodeOutput:
        normalized_key = normalized_extra_key_or_none(key)
        if normalized_key is None:
            raise RuntimeError("Get Sampler Params Extra: key is required")

        raw_value = extras.get(normalized_key) if isinstance(extras, dict) else None
        payload = deserialize_sampler_params_extra(raw_value, key=normalized_key)
        return c_io.NodeOutput(payload, *split_sampler_params_values(payload))
