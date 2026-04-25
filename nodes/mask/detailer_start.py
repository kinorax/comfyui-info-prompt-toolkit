# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast
from ._detailer_utils import prepare_detailer_batch

_MINI_UNIT_OPTIONS: tuple[str, ...] = ("8", "16", "32", "64")


class DetailerStart(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-DetailerStart",
            display_name="Detailer Start",
            category=Const.CATEGORY_MASK,
            hidden=[
                c_io.Hidden.unique_id,
            ],
            search_aliases=["detailer crop", "detailer open", "detailer prepare"],
            inputs=[
                c_io.Image.Input(
                    "image",
                    tooltip="Source image. Active detailer crops are processed one item at a time between Detailer Start and Detailer End.",
                ),
                c_io.Mask.Input(
                    "mask",
                    tooltip="Soft mask used to determine crop bounds. Any value above 0.0 becomes fully masked in the output inpaint mask.",
                ),
                c_io.Float.Input(
                    "crop_margin_scale",
                    default=1.5,
                    min=1.0,
                    max=16.0,
                    step=0.05,
                    tooltip="Crop expansion multiplier applied directly to the detected mask bounding-box width and height.",
                ),
                c_io.Float.Input(
                    "upscale_factor",
                    default=2.0,
                    min=1.0,
                    max=16.0,
                    step=0.05,
                    tooltip="Pre-upscale factor applied to the cropped image and binary inpaint mask before mini-unit padding.",
                ),
                c_io.Combo.Input(
                    "mini_unit",
                    options=_MINI_UNIT_OPTIONS,
                    default="32",
                    tooltip="Round the output inpaint image and mask size up to a width and height divisible by this unit.",
                ),
                c_io.Int.Input(
                    "detailer_index",
                    default=0,
                    min=0,
                    advanced=True,
                    tooltip="Internal index for recursive per-item detailer execution.",
                ),
            ],
            outputs=[
                Const.DETAILER_CONTROL_TYPE.Output(
                    Cast.out_id("detailer_control"),
                    display_name="detailer_control",
                ),
                c_io.Image.Output(
                    Cast.out_id("inpaint_image"),
                    display_name="inpaint_image",
                ),
                c_io.Mask.Output(
                    Cast.out_id("inpaint_mask"),
                    display_name="inpaint_mask",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        image: Any = None,
        mask: Any = None,
        crop_margin_scale: object = 1.5,
        upscale_factor: object = 2.0,
        mini_unit: object = "32",
        detailer_index: object = 0,
    ) -> bool | str:
        try:
            crop_margin_scale_value = float(crop_margin_scale)
            upscale_factor_value = float(upscale_factor)
            mini_unit_value = int(mini_unit)
            detailer_index_value = int(detailer_index)
        except Exception:
            return "crop_margin_scale, upscale_factor, mini_unit, and detailer_index must be numbers"

        if crop_margin_scale_value < 1.0:
            return "crop_margin_scale must be 1.0 or greater"
        if upscale_factor_value < 1.0:
            return "upscale_factor must be 1.0 or greater"
        if mini_unit_value not in (8, 16, 32, 64):
            return "mini_unit must be one of 8, 16, 32, or 64"
        if detailer_index_value < 0:
            return "detailer_index must be 0 or greater"
        return True

    @classmethod
    def execute(
        cls,
        image: Any,
        mask: Any,
        crop_margin_scale: float,
        upscale_factor: float,
        mini_unit: Any,
        detailer_index: int = 0,
    ) -> c_io.NodeOutput:
        detailer_control, inpaint_image, inpaint_mask = prepare_detailer_batch(
            image,
            mask,
            crop_margin_scale=float(crop_margin_scale),
            upscale_factor=float(upscale_factor),
            mini_unit=int(mini_unit),
            detailer_index=int(detailer_index),
            open_node_id=str(getattr(cls.hidden, "unique_id", "")),
        )
        return c_io.NodeOutput(detailer_control, inpaint_image, inpaint_mask)
