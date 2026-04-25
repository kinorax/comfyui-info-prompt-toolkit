# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast


def _flatten(value: Any) -> list[Any]:
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


def _short_text(value: Any, max_length: int = 24) -> str:
    try:
        text = str(value)
    except Exception:
        text = type(value).__name__

    text = text.replace("\n", "\\n")
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}..."


def _counts_text(counts: dict[str, int], limit: int = 8) -> str:
    if len(counts) == 0:
        return "-"

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    parts = [f"{key}x{count}" for key, count in ordered[:limit]]
    if len(ordered) > limit:
        parts.append(f"...+{len(ordered) - limit}")
    return ", ".join(parts)


def _summarize_image_info_payload(value: Any) -> dict[str, Any]:
    entries = 0
    dict_items = 0
    none_items = 0
    non_dict_items = 0
    key_counts: dict[str, int] = {}
    seed_counts: dict[str, int] = {}
    cell_index_counts: dict[str, int] = {}

    for item in _flatten(value):
        entries += 1

        if item is None:
            none_items += 1
            continue

        if not isinstance(item, dict):
            non_dict_items += 1
            continue

        dict_items += 1

        for key in item.keys():
            key_text = _short_text(key)
            key_counts[key_text] = key_counts.get(key_text, 0) + 1

        if Const.IMAGEINFO_SEED in item:
            seed_text = _short_text(item.get(Const.IMAGEINFO_SEED))
            seed_counts[seed_text] = seed_counts.get(seed_text, 0) + 1

        extras = item.get(Const.IMAGEINFO_EXTRAS)
        if isinstance(extras, dict) and "xy.cell_index" in extras:
            cell_text = _short_text(extras.get("xy.cell_index"))
            cell_index_counts[cell_text] = cell_index_counts.get(cell_text, 0) + 1

    return {
        "entries": entries,
        "dict_items": dict_items,
        "none_items": none_items,
        "non_dict_items": non_dict_items,
        "key_text": _counts_text(key_counts),
        "seed_text": _counts_text(seed_counts),
        "cell_index_text": _counts_text(cell_index_counts),
    }


class ImageInfoCountDebug(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-ImageInfoCountDebug",
            display_name="Image Info Count Debug",
            category=Const.CATEGORY_DEBUG,
            hidden=[
                c_io.Hidden.unique_id,
            ],
            inputs=[
                Const.IMAGEINFO_TYPE.Input(
                    Const.IMAGEINFO,
                    optional=True,
                    tooltip="Pass-through image_info payload",
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
                    tooltip="If false, this node logs nothing and only relays image_info",
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
        image_info: Any = None,
        label: Any = "",
        enabled: Any = True,
    ) -> c_io.NodeOutput:
        if _bool_or_default(enabled, True):
            summary = _summarize_image_info_payload(image_info)
            unique_id = getattr(cls.hidden, "unique_id", None)
            label_text = str(label).strip() if label is not None else ""
            if len(label_text) == 0:
                label_text = "-"

            print(
                "[IPT-ImageInfoCountDebug]"
                f" node={unique_id}"
                f" label={label_text}"
                f" entries={summary['entries']}"
                f" dicts={summary['dict_items']}"
                f" none={summary['none_items']}"
                f" non_dicts={summary['non_dict_items']}"
                f" keys={summary['key_text']}"
                f" seeds={summary['seed_text']}"
                f" xy_cell_index={summary['cell_index_text']}"
            )

        return c_io.NodeOutput(image_info)
