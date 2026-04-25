# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast
from ._mask_utils import remove_small_regions


class RemoveSmallMaskRegions(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-RemoveSmallMaskRegions",
            display_name="Remove Small Mask Regions",
            category=Const.CATEGORY_MASK,
            inputs=[
                c_io.Mask.Input(
                    "mask",
                    tooltip="Soft or binary input mask. The node thresholds it before removing islands.",
                ),
                c_io.Float.Input(
                    "min_image_area_percent",
                    default=0.004,
                    min=0.0,
                    max=100.0,
                    step=0.001,
                    tooltip="Connected foreground islands smaller than this image-area percentage are removed.",
                ),
                c_io.Float.Input(
                    "threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Pixels above this value are treated as foreground before filtering.",
                ),
            ],
            outputs=[
                c_io.Mask.Output(
                    Cast.out_id("mask"),
                    display_name="mask",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        mask: Any = None,
        min_image_area_percent: object = 0.004,
        threshold: object = 0.5,
    ) -> bool | str:
        try:
            min_area_value = float(min_image_area_percent)
            threshold_value = float(threshold)
        except Exception:
            return "min_image_area_percent and threshold must be numbers"

        if min_area_value < 0.0 or min_area_value > 100.0:
            return "min_image_area_percent must be between 0.0 and 100.0"
        if threshold_value < 0.0 or threshold_value > 1.0:
            return "threshold must be between 0.0 and 1.0"
        return True

    @classmethod
    def execute(
        cls,
        mask: Any,
        min_image_area_percent: float,
        threshold: float,
    ) -> c_io.NodeOutput:
        output_mask = remove_small_regions(
            mask,
            min_image_area_percent=float(min_image_area_percent),
            threshold=float(threshold),
        )
        return c_io.NodeOutput(output_mask)
