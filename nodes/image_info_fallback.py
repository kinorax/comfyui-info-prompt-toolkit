# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.image_info_defaults_merge import merge_image_info_missing_values


class ImageInfoFallback(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-ImageInfoFallback",
            display_name="Image Info Fallback",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.IMAGEINFO_TYPE.Input(
                    Const.IMAGEINFO,
                    display_name=Const.IMAGEINFO,
                ),
                Const.IMAGEINFO_TYPE.Input(
                    Const.IMAGEINFO_FALLBACK,
                    display_name="image_info_fallback",
                ),
            ],
            outputs=[
                Const.IMAGEINFO_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO),
                    display_name=Const.IMAGEINFO,
                ),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> c_io.NodeOutput:
        base_image_info = kwargs.get(Const.IMAGEINFO)
        fallback_image_info = kwargs.get(Const.IMAGEINFO_FALLBACK)
        merged = merge_image_info_missing_values(
            base_image_info,
            fallback_image_info,
            extras_key=Const.IMAGEINFO_EXTRAS,
            positive_key=Const.IMAGEINFO_POSITIVE,
            lora_stack_key=Const.IMAGEINFO_LORA_STACK,
            preserve_lora_stack_when_positive_present=True,
        )
        return c_io.NodeOutput(merged)
