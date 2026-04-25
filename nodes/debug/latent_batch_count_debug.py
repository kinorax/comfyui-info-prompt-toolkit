# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast

LATENT_RUNTIME_TYPE = c_io.Custom("LATENT")


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


def _record_shape(shapes: dict[tuple[int, ...], int], tensor: torch.Tensor) -> None:
    shape = tuple(int(x) for x in tensor.shape)
    shapes[shape] = shapes.get(shape, 0) + 1


def _summarize_latent_payload(value: Any) -> dict[str, Any]:
    entries = 0
    dict_items = 0
    none_items = 0
    non_dict_items = 0
    samples_missing = 0
    sample_tensors = 0
    sample_nested = 0
    rank4 = 0
    rank5 = 0
    unsupported_tensor_ranks = 0
    nested_children = 0
    batches = 0
    shapes: dict[tuple[int, ...], int] = {}

    for item in _flatten(value):
        entries += 1

        if item is None:
            none_items += 1
            continue

        if not isinstance(item, dict):
            non_dict_items += 1
            continue

        dict_items += 1
        samples = item.get("samples")
        if samples is None:
            samples_missing += 1
            continue

        if isinstance(samples, torch.Tensor):
            sample_tensors += 1
            _record_shape(shapes, samples)
            if samples.ndim == 4:
                rank4 += 1
                batches += max(0, int(samples.shape[0]))
            elif samples.ndim == 5:
                rank5 += 1
                batches += max(0, int(samples.shape[0]))
            else:
                unsupported_tensor_ranks += 1
            continue

        if bool(getattr(samples, "is_nested", False)) and callable(getattr(samples, "unbind", None)):
            sample_nested += 1
            for child in samples.unbind():
                if not isinstance(child, torch.Tensor):
                    continue
                nested_children += 1
                batches += max(0, int(child.shape[0])) if child.ndim >= 1 else 0
                _record_shape(shapes, child)
            continue

        non_dict_items += 1

    return {
        "entries": entries,
        "dict_items": dict_items,
        "none_items": none_items,
        "non_dict_items": non_dict_items,
        "samples_missing": samples_missing,
        "sample_tensors": sample_tensors,
        "sample_nested": sample_nested,
        "rank4": rank4,
        "rank5": rank5,
        "unsupported_tensor_ranks": unsupported_tensor_ranks,
        "nested_children": nested_children,
        "batches": batches,
        "shape_text": _shape_text(shapes),
    }


class LatentBatchCountDebug(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-LatentBatchCountDebug",
            display_name="Latent Batch Count Debug",
            category=Const.CATEGORY_DEBUG,
            hidden=[
                c_io.Hidden.unique_id,
            ],
            inputs=[
                LATENT_RUNTIME_TYPE.Input(
                    "latent",
                    tooltip="Pass-through latent payload",
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
                    tooltip="If false, this node logs nothing and only relays latent",
                ),
            ],
            outputs=[
                LATENT_RUNTIME_TYPE.Output(
                    Cast.out_id("latent"),
                    display_name="latent",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent: Any,
        label: Any = "",
        enabled: Any = True,
    ) -> c_io.NodeOutput:
        if _bool_or_default(enabled, True):
            summary = _summarize_latent_payload(latent)
            unique_id = getattr(cls.hidden, "unique_id", None)
            label_text = str(label).strip() if label is not None else ""
            if len(label_text) == 0:
                label_text = "-"

            print(
                "[IPT-LatentBatchCountDebug]"
                f" node={unique_id}"
                f" label={label_text}"
                f" entries={summary['entries']}"
                f" dicts={summary['dict_items']}"
                f" none={summary['none_items']}"
                f" non_dicts={summary['non_dict_items']}"
                f" samples_missing={summary['samples_missing']}"
                f" tensors={summary['sample_tensors']}"
                f" nested={summary['sample_nested']}"
                f" nested_children={summary['nested_children']}"
                f" batches={summary['batches']}"
                f" rank4={summary['rank4']}"
                f" rank5={summary['rank5']}"
                f" unsupported_ranks={summary['unsupported_tensor_ranks']}"
                f" shapes={summary['shape_text']}"
            )

        return c_io.NodeOutput(latent)
