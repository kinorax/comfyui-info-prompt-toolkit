# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        output: list[Any] = []
        for item in value:
            output.extend(_flatten(item))
        return output
    return [value]


def _bool_or_default(value: Any, default: bool) -> bool:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _bool_or_default(value[0], default)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _shape_text(shapes: dict[tuple[int, ...], int]) -> str:
    if len(shapes) == 0:
        return "-"

    parts: list[str] = []
    for shape in sorted(shapes.keys()):
        parts.append(f"{shape}x{shapes[shape]}")
    return ", ".join(parts)


def _summarize_image_payload(value: Any) -> dict[str, Any]:
    tensor_items = 0
    frames = 0
    rank3 = 0
    rank4 = 0
    non_tensor_items = 0
    unsupported_tensor_ranks = 0
    shapes: dict[tuple[int, ...], int] = {}

    for item in _flatten(value):
        if not isinstance(item, torch.Tensor):
            non_tensor_items += 1
            continue

        tensor_items += 1
        shape = tuple(int(x) for x in item.shape)
        shapes[shape] = shapes.get(shape, 0) + 1

        if item.ndim == 4:
            rank4 += 1
            frames += max(0, int(item.shape[0]))
            continue

        if item.ndim == 3:
            rank3 += 1
            frames += 1
            continue

        unsupported_tensor_ranks += 1

    return {
        "frames": frames,
        "tensor_items": tensor_items,
        "rank3": rank3,
        "rank4": rank4,
        "non_tensor_items": non_tensor_items,
        "unsupported_tensor_ranks": unsupported_tensor_ranks,
        "shape_text": _shape_text(shapes),
    }


class ImageBatchCountDebug(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-ImageBatchCountDebug",
            display_name="Image Batch Count Debug",
            category=Const.CATEGORY_DEBUG,
            hidden=[
                c_io.Hidden.unique_id,
            ],
            inputs=[
                c_io.Image.Input(
                    "image",
                    tooltip="Pass-through image payload",
                ),
                c_io.String.Input(
                    "label",
                    default="",
                    optional=True,
                    tooltip="Optional prefix shown in console logs",
                ),
                c_io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip="If false, this node logs nothing and only relays image",
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
    def execute(
        cls,
        image: Any,
        label: Any = "",
        enabled: Any = True,
    ) -> c_io.NodeOutput:
        if _bool_or_default(enabled, True):
            summary = _summarize_image_payload(image)
            unique_id = getattr(cls.hidden, "unique_id", None)
            label_text = str(label).strip() if label is not None else ""
            if len(label_text) == 0:
                label_text = "-"

            print(
                "[IPT-ImageBatchCountDebug]"
                f" node={unique_id}"
                f" label={label_text}"
                f" frames={summary['frames']}"
                f" tensors={summary['tensor_items']}"
                f" rank3={summary['rank3']}"
                f" rank4={summary['rank4']}"
                f" non_tensors={summary['non_tensor_items']}"
                f" unsupported_ranks={summary['unsupported_tensor_ranks']}"
                f" shapes={summary['shape_text']}"
            )

        return c_io.NodeOutput(image)
