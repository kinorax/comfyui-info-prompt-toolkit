# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.lora_stack_extra import is_reserved_extra_key, merge_extra_value, normalized_extra_key_or_none
from ..utils.sampler_params import SAMPLER_PARAMS_KEY, serialize_sampler_params_extra_json


class SetSamplerParamsExtra(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-SetSamplerParamsExtra",
            display_name="Set Sampler Params Extra",
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
                Const.SAMPLER_PARAMS_TYPE.Input(
                    SAMPLER_PARAMS_KEY,
                    display_name=SAMPLER_PARAMS_KEY,
                    optional=True,
                    extra_dict={"forceInput": True},
                ),
            ],
            outputs=[
                Const.IMAGEINFO_EXTRAS_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO_EXTRAS),
                    display_name=Const.IMAGEINFO_EXTRAS,
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        key: object = None,
        sampler_params: object | None = None,
        extras: dict[str, object] | None = None,
    ) -> bool | str:
        normalized_key = normalized_extra_key_or_none(key)
        if key is not None and normalized_key is None:
            return "key is required"
        if normalized_key is not None and is_reserved_extra_key(normalized_key):
            return f"key '{normalized_key}' is reserved and cannot be used"
        return True

    @classmethod
    def execute(
        cls,
        key: object = None,
        sampler_params: object | None = None,
        extras: dict[str, object] | None = None,
    ) -> c_io.NodeOutput:
        normalized_key = normalized_extra_key_or_none(key)
        if normalized_key is None:
            raise RuntimeError("Set Sampler Params Extra: key is required")
        if is_reserved_extra_key(normalized_key):
            raise RuntimeError(
                f"Set Sampler Params Extra: key '{normalized_key}' is reserved and cannot be used"
            )

        serialized = serialize_sampler_params_extra_json(sampler_params)
        output = merge_extra_value(extras, normalized_key, serialized)
        return c_io.NodeOutput(output)
