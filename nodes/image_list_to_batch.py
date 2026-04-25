# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        output: list[Any] = []
        for item in value:
            output.extend(_flatten(item))
        return output
    return [value]


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "ndim") and hasattr(value, "shape")


def _merge_images_to_batch(image: Any) -> torch.Tensor:
    values = _flatten(image)
    if len(values) == 0:
        raise ValueError("image is required")

    batches: list[torch.Tensor] = []
    for value in values:
        if not _is_tensor_like(value):
            raise ValueError("image input must be IMAGE")
        if value.ndim == 3:
            batches.append(value.unsqueeze(0))
            continue
        if value.ndim == 4:
            batches.append(value)
            continue
        raise ValueError(f"unsupported image tensor shape: {tuple(value.shape)}")

    if len(batches) == 1:
        return batches[0]

    try:
        return torch.cat(batches, dim=0)
    except Exception as exc:
        raise ValueError("all image items must have matching height, width, and channels") from exc


class ImageListToBatch(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-ImageListToBatch",
            display_name="Image List To Batch",
            category=Const.CATEGORY_IMAGEINFO,
            description="Collect list-expanded IMAGE inputs into one rank-4 IMAGE batch. Useful before VFI nodes.",
            search_aliases=[
                "image list to batch",
                "image sequence to batch",
                "merge image list",
                "vae decode to vfi",
                "vfi image batch",
            ],
            is_input_list=True,
            inputs=[
                c_io.Image.Input(
                    "image",
                    tooltip="Frame-by-frame IMAGE inputs to merge into one batch",
                ),
            ],
            outputs=[
                c_io.Image.Output(
                    Cast.out_id("image"),
                    display_name="image",
                ),
                c_io.Int.Output(
                    Cast.out_id("frame_count"),
                    display_name="frame_count",
                ),
            ],
        )

    @classmethod
    def execute(cls, image: Any) -> c_io.NodeOutput:
        batch = _merge_images_to_batch(image)
        return c_io.NodeOutput(
            batch,
            int(batch.shape[0]),
        )
