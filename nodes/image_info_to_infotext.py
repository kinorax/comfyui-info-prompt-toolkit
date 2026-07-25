# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils.metadata_encryption import image_info_to_public_infotext


class ImageInfoToInfotext(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-ImageInfoToInfotext",
            display_name="Image Info To Infotext",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.IMAGEINFO_TYPE.Input(
                    Const.IMAGEINFO,
                    optional=True,
                ),
            ],
            outputs=[
                c_io.String.Output(
                    "INFOTEXT",
                    display_name="infotext",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image_info: dict[str, object] | None = None,
    ) -> c_io.NodeOutput:
        infotext = image_info_to_public_infotext(image_info)
        return c_io.NodeOutput(infotext)
