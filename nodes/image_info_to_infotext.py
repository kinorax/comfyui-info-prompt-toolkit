# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils.a1111_infotext import image_info_to_a1111_infotext
from ..utils.image_info_hash_extras import (
    add_civitai_hash_extras,
    clear_representative_hash_extras,
)


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
        image_info_without_hashes = clear_representative_hash_extras(image_info)
        image_info_with_hashes = add_civitai_hash_extras(image_info_without_hashes)
        infotext = image_info_to_a1111_infotext(image_info_with_hashes)
        return c_io.NodeOutput(infotext)
