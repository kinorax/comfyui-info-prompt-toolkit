# Copyright 2026 kinorax
from __future__ import annotations

import json
import math

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.selector_resolution import resolve_selector_value


_MIN_STRENGTH = -100.0
_MAX_STRENGTH = 100.0


def _parse_rows(value: object) -> list[object]:
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Power Lora Selector: loras contains invalid JSON: {exc.msg}") from exc

    if isinstance(parsed, dict):
        parsed = parsed.get("items")
    if parsed is None:
        return []
    if not isinstance(parsed, list):
        raise RuntimeError("Power Lora Selector: loras must be a JSON array")
    return list(parsed)


def _is_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if value is None:
        return True
    return str(value).strip().casefold() not in {"", "0", "false", "no", "off"}


def _normalize_strength(value: object, *, row_number: int) -> float:
    try:
        strength = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Power Lora Selector: row {row_number} strength must be a number"
        ) from exc
    if not math.isfinite(strength):
        raise RuntimeError(f"Power Lora Selector: row {row_number} strength must be finite")
    return min(_MAX_STRENGTH, max(_MIN_STRENGTH, strength))


class PowerLoraSelector(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        lora_options = Const.get_LORA_OPTIONS()
        return c_io.Schema(
            node_id="IPT-PowerLoraSelector",
            display_name="Power Lora Selector",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.LORA_STACK_TYPE.Input(
                    "lora_stack",
                    optional=True,
                    tooltip="Optional LoRA stack to append the enabled rows to",
                ),
                c_io.String.Input(
                    "items",
                    display_name="loras",
                    default="[]",
                    multiline=True,
                    socketless=True,
                    tooltip="JSON rows used by the Power Lora Selector editor",
                    extra_dict={"lora_options": list(lora_options)},
                ),
            ],
            outputs=[
                Const.LORA_STACK_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO_LORA_STACK),
                    display_name=Const.IMAGEINFO_LORA_STACK,
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        lora_stack: object | None = None,
        items: object = None,
    ) -> bool | str:
        return True

    @classmethod
    def execute(
        cls,
        lora_stack: object | None = None,
        items: object = None,
    ) -> c_io.NodeOutput:
        output_stack = list(lora_stack) if isinstance(lora_stack, list) else []
        lora_options = Const.get_LORA_OPTIONS()

        for row_number, raw_row in enumerate(_parse_rows(items), start=1):
            if not isinstance(raw_row, dict):
                raise RuntimeError(f"Power Lora Selector: row {row_number} must be an object")
            if not _is_enabled(raw_row.get("enabled", True)):
                continue

            selected_name = raw_row.get("lora_name", raw_row.get("name"))
            if selected_name is None or not str(selected_name).strip():
                raise RuntimeError(f"Power Lora Selector: row {row_number} lora_name is required")

            resolved_name = resolve_selector_value(
                selected_name,
                lora_options,
                value_label=f"Power Lora Selector row {row_number} lora_name",
                folder_name=Const.MODEL_FOLDER_PATH_LORAS,
                sha256=raw_row.get("sha256"),
            )
            strength = _normalize_strength(raw_row.get("strength", 1.0), row_number=row_number)
            stack_item = Const.make_lora_stack_item(resolved_name, strength)
            if stack_item is not None:
                output_stack.append(stack_item)

        return c_io.NodeOutput(output_stack)
