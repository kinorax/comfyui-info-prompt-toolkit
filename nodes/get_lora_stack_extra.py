# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.lora_stack_extra import deserialize_lora_stack_extra, normalized_extra_key_or_none


class GetLoraStackExtra(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-GetLoraStackExtra",
            display_name="Get Lora Stack Extra",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.String.Input(
                    "key",
                    tooltip="Parameter line key",
                ),
                Const.IMAGEINFO_EXTRAS_TYPE.Input(
                    Const.IMAGEINFO_EXTRAS,
                    optional=True,
                ),
            ],
            outputs=[
                Const.LORA_STACK_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO_LORA_STACK),
                    display_name=Const.IMAGEINFO_LORA_STACK,
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        key: object,
        extras: dict[str, object] | None = None,
    ) -> bool | str:
        normalized_key = normalized_extra_key_or_none(key)
        if key is not None and normalized_key is None:
            return "key is required"
        return True

    @classmethod
    def execute(
        cls,
        key: object,
        extras: dict[str, object] | None = None,
    ) -> c_io.NodeOutput:
        normalized_key = normalized_extra_key_or_none(key)
        if normalized_key is None:
            raise RuntimeError("Get Lora Stack Extra: key is required")

        raw_value = extras.get(normalized_key) if isinstance(extras, dict) else None
        value_out = deserialize_lora_stack_extra(raw_value, key=normalized_key)
        return c_io.NodeOutput(value_out)
