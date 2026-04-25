# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast
from ._mask_utils import grow_mask


class GrowMask(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-GrowMask",
            display_name="Grow Mask",
            category=Const.CATEGORY_MASK,
            search_aliases=["grow soft mask", "grow mask with blur", "mask blur grow"],
            inputs=[
                c_io.Mask.Input(
                    "mask",
                    tooltip="Soft or binary input mask.",
                ),
                c_io.Float.Input(
                    "threshold",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Pixels above this value define the foreground used for radius estimation and hard-core preservation.",
                ),
                c_io.Float.Input(
                    "region_area_scale",
                    default=1.0,
                    min=1.0,
                    max=64.0,
                    step=0.05,
                    tooltip="Area multiplier used to derive the shared grow radius from the weighted-median representative region size.",
                ),
                c_io.Float.Input(
                    "blur_radius_scale",
                    default=0.5,
                    min=0.0,
                    max=16.0,
                    step=0.05,
                    tooltip="Log-damped area ratio used to derive the shared blur radius from the weighted-median representative region radius.",
                ),
                c_io.Float.Input(
                    "solid_core_scale",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Inside fill strength after blur. Higher values make the interior more solid without expanding the outer silhouette by itself.",
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
        threshold: object = 0.1,
        region_area_scale: object = 1.0,
        blur_radius_scale: object = 0.5,
        solid_core_scale: object = 1.0,
    ) -> bool | str:
        try:
            threshold_value = float(threshold)
            region_area_scale_value = float(region_area_scale)
            blur_radius_scale_value = float(blur_radius_scale)
            solid_core_scale_value = float(solid_core_scale)
        except Exception:
            return "threshold, region_area_scale, blur_radius_scale, and solid_core_scale must be numbers"

        if threshold_value < 0.0 or threshold_value > 1.0:
            return "threshold must be between 0.0 and 1.0"
        if region_area_scale_value < 1.0:
            return "region_area_scale must be 1.0 or greater"
        if blur_radius_scale_value < 0.0:
            return "blur_radius_scale must be 0.0 or greater"
        if solid_core_scale_value < 0.0 or solid_core_scale_value > 2.0:
            return "solid_core_scale must be between 0.0 and 2.0"
        return True

    @classmethod
    def execute(
        cls,
        mask: Any,
        threshold: float,
        region_area_scale: float,
        blur_radius_scale: float,
        solid_core_scale: float,
    ) -> c_io.NodeOutput:
        tapered_corners = True
        output_mask = grow_mask(
            mask,
            threshold=float(threshold),
            region_area_scale=float(region_area_scale),
            blur_radius_scale=float(blur_radius_scale),
            solid_core_scale=float(solid_core_scale),
            tapered_corners=tapered_corners,
        )
        return c_io.NodeOutput(output_mask)
