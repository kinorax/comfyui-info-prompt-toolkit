# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.a1111_infotext import a1111_infotext_to_image_info
from ..utils.image_info_normalizer import normalize_image_info_with_comfy_options


class InfotextToImageInfo(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-InfotextToImageInfo",
            display_name="Infotext To Image Info",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.String.Input(
                    "infotext",
                    force_input=True,
                    tooltip="A1111 infotext string",
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
    def execute(
        cls,
        infotext: str,
    ) -> c_io.NodeOutput:
        image_info = a1111_infotext_to_image_info(infotext)
        image_info = normalize_image_info_with_comfy_options(image_info)
        return c_io.NodeOutput(image_info)
