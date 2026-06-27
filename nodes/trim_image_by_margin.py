# Copyright 2026 kinorax
from __future__ import annotations

import torch
from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.margin import margin_payload_or_error


class TrimImageByMargin(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-TrimImageByMargin",
            display_name="Trim Image by Margin",
            category=Const.CATEGORY_IMAGEINFO,
            search_aliases=["crop image by margin", "trim image edges", "border trim"],
            inputs=[
                c_io.Image.Input("image"),
                Const.MARGIN_TYPE.Input(
                    "margin",
                    display_name="margin",
                    optional=True,
                    extra_dict={"forceInput": True},
                ),
            ],
            outputs=[
                c_io.Image.Output(
                    Cast.out_id("image"),
                    display_name="image",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        image: object,
        margin: object | None = None,
    ) -> bool | str:
        if margin is None:
            return True
        _payload, error = margin_payload_or_error(margin)
        return error or True

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        margin: object | None = None,
    ) -> c_io.NodeOutput:
        if margin is None:
            return c_io.NodeOutput(image)

        payload, error = margin_payload_or_error(margin)
        if payload is None:
            raise ValueError(f"Trim Image by Margin: {error or 'margin is invalid'}")
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("Trim Image by Margin: image must be a rank-4 IMAGE tensor")

        top = payload["top"]
        right = payload["right"]
        bottom = payload["bottom"]
        left = payload["left"]
        if top == 0 and right == 0 and bottom == 0 and left == 0:
            return c_io.NodeOutput(image)

        height = int(image.shape[1])
        width = int(image.shape[2])
        if left + right >= width:
            raise ValueError(
                "Trim Image by Margin: left and right margins must leave at least one pixel "
                f"(image width: {width})"
            )
        if top + bottom >= height:
            raise ValueError(
                "Trim Image by Margin: top and bottom margins must leave at least one pixel "
                f"(image height: {height})"
            )

        trimmed = image[:, top : height - bottom, left : width - right, :]
        return c_io.NodeOutput(trimmed)
