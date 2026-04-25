# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast
from ...utils.a1111_infotext import extract_lora_stack_from_prompt
from ...utils.image_info_normalizer import normalize_lora_stack_with_comfy_options
from ...utils.prompt_text import normalize_prompt_tokens

_SIZE_INPUT_ID = "size"


def _normalized_label_or_none(value: Any) -> str | None:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _normalized_label_or_none(value[0])
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _unwrap_modifiers(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            return list(value[0])
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _size_pair_or_none(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _size_pair_or_none(value[0])

    if isinstance(value, dict) and "__value__" in value:
        return _size_pair_or_none(value.get("__value__"))

    if isinstance(value, dict):
        width = _int_or_none(value.get("width"))
        height = _int_or_none(value.get("height"))
        if width is None:
            width = _int_or_none(value.get("w"))
        if height is None:
            height = _int_or_none(value.get("h"))
        if width is None or height is None:
            return None
        return width, height

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        width = _int_or_none(value[0])
        height = _int_or_none(value[1])
        if width is None or height is None:
            return None
        return width, height

    if isinstance(value, str):
        parts = value.lower().replace(" ", "").split("x")
        if len(parts) != 2:
            return None
        width = _int_or_none(parts[0])
        height = _int_or_none(parts[1])
        if width is None or height is None:
            return None
        return width, height

    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None

    try:
        return str(value)
    except Exception:
        return None


def _normalized_prompt_or_none(value: Any) -> str | None:
    text = _string_or_none(value)
    if text is None:
        return None
    return normalize_prompt_tokens(text)


def _lora_stack_or_none(value: Any) -> list[Any] | None:
    if not isinstance(value, list):
        return None
    return list(value)


def _merged_lora_stack(base_lora_stack: Any, appended_lora_stack: list[dict[str, str | float]] | None) -> list[Any] | None:
    base = _lora_stack_or_none(base_lora_stack)
    if base is None and appended_lora_stack is None:
        return None
    if base is None:
        return list(appended_lora_stack) if isinstance(appended_lora_stack, list) else None
    if appended_lora_stack is None:
        return base

    merged = list(base)
    merged.extend(appended_lora_stack)
    return merged


def _positive_prompt_and_lora_stack(value: Any) -> tuple[str | None, list[dict[str, str | float]] | None]:
    text = _string_or_none(value)
    if text is None:
        return None, None

    prompt_without_lora, lora_stack = extract_lora_stack_from_prompt(text)
    normalized_prompt = normalize_prompt_tokens(prompt_without_lora)
    normalized_lora_stack = normalize_lora_stack_with_comfy_options(lora_stack)
    return normalized_prompt, normalized_lora_stack


def _connected_input_ids(cls: type[c_io.ComfyNode]) -> set[str]:
    prompt = getattr(cls.hidden, "prompt", None)
    unique_id = getattr(cls.hidden, "unique_id", None)
    if not isinstance(prompt, dict) or unique_id is None:
        return set()

    node = prompt.get(str(unique_id), {}) or {}
    inputs = node.get("inputs", {}) or {}
    if not isinstance(inputs, dict):
        return set()
    return set(inputs.keys())


def _copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return value


class XYPlotModifier(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        fi = dict(optional=True, force_input=True)

        return c_io.Schema(
            node_id="IPT-XYPlotModifier",
            display_name="XY Plot Modifier",
            category=Const.CATEGORY_XYPLOT,
            hidden=[c_io.Hidden.prompt, c_io.Hidden.unique_id],
            inputs=[
                Const.XY_PLOT_MODIFIER_TYPE.Input(
                    "modifiers",
                    optional=True,
                    tooltip="Modifier array chain input",
                ),
                Const.MODEL_TYPE.Input(
                    Const.IMAGEINFO_MODEL,
                    optional=True,
                ),
                Const.MODEL_TYPE.Input(
                    Const.IMAGEINFO_REFINER_MODEL,
                    optional=True,
                ),
                Const.MODEL_TYPE.Input(
                    Const.IMAGEINFO_DETAILER_MODEL,
                    optional=True,
                ),
                Const.LORA_STACK_TYPE.Input(
                    Const.IMAGEINFO_LORA_STACK,
                    optional=True,
                ),
                Const.CLIP_TYPE.Input(
                    Const.IMAGEINFO_CLIP,
                    optional=True,
                ),
                c_io.String.Input(
                    Const.IMAGEINFO_VAE,
                    **fi,
                ),
                c_io.String.Input(
                    Const.IMAGEINFO_POSITIVE,
                    **fi,
                ),
                c_io.String.Input(
                    Const.IMAGEINFO_NEGATIVE,
                    **fi,
                ),
                c_io.Int.Input(
                    Const.IMAGEINFO_STEPS,
                    min=0,
                    max=100000,
                    **fi,
                ),
                c_io.String.Input(
                    Const.IMAGEINFO_SAMPLER,
                    **fi,
                ),
                c_io.String.Input(
                    Const.IMAGEINFO_SCHEDULER,
                    **fi,
                ),
                c_io.Float.Input(
                    Const.IMAGEINFO_CFG,
                    **fi,
                ),
                c_io.Int.Input(
                    Const.IMAGEINFO_SEED,
                    min=Const.INT64_MIN,
                    max=Const.INT64_MAX,
                    **fi,
                ),
                Const.SIZE_TYPE.Input(
                    _SIZE_INPUT_ID,
                    display_name="width x height",
                    optional=True,
                    extra_dict={"forceInput": True},
                ),
                Const.IMAGEINFO_EXTRAS_TYPE.Input(
                    Const.IMAGEINFO_EXTRAS,
                    optional=True,
                ),
                c_io.String.Input(
                    "label",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Optional axis label. Leave empty for auto label.",
                ),
            ],
            outputs=[
                Const.XY_PLOT_MODIFIER_TYPE.Output(
                    Cast.out_id("modifiers"),
                    display_name="modifiers",
                ),
            ],
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> c_io.NodeOutput:
        current = _unwrap_modifiers(kwargs.get("modifiers"))
        connected = _connected_input_ids(cls)

        changes: dict[str, Any] = {}

        positive_value: str | None = None
        positive_lora_stack: list[dict[str, str | float]] | None = None
        include_positive = Const.IMAGEINFO_POSITIVE in connected or (not connected and kwargs.get(Const.IMAGEINFO_POSITIVE) is not None)
        if include_positive:
            positive_value, positive_lora_stack = _positive_prompt_and_lora_stack(kwargs.get(Const.IMAGEINFO_POSITIVE))

        include_negative = Const.IMAGEINFO_NEGATIVE in connected or (not connected and kwargs.get(Const.IMAGEINFO_NEGATIVE) is not None)
        if include_negative:
            changes[Const.IMAGEINFO_NEGATIVE] = _normalized_prompt_or_none(kwargs.get(Const.IMAGEINFO_NEGATIVE))

        include_lora_stack = Const.IMAGEINFO_LORA_STACK in connected or (not connected and kwargs.get(Const.IMAGEINFO_LORA_STACK) is not None)
        if include_lora_stack or positive_lora_stack is not None:
            changes[Const.IMAGEINFO_LORA_STACK] = _merged_lora_stack(
                kwargs.get(Const.IMAGEINFO_LORA_STACK) if include_lora_stack else None,
                positive_lora_stack,
            )
        if include_positive:
            changes[Const.IMAGEINFO_POSITIVE] = positive_value

        for key in (
            Const.IMAGEINFO_MODEL,
            Const.IMAGEINFO_REFINER_MODEL,
            Const.IMAGEINFO_DETAILER_MODEL,
            Const.IMAGEINFO_CLIP,
            Const.IMAGEINFO_VAE,
            Const.IMAGEINFO_STEPS,
            Const.IMAGEINFO_SAMPLER,
            Const.IMAGEINFO_SCHEDULER,
            Const.IMAGEINFO_CFG,
            Const.IMAGEINFO_SEED,
            Const.IMAGEINFO_EXTRAS,
        ):
            value = kwargs.get(key)
            include = key in connected or (not connected and value is not None)
            if not include:
                continue
            changes[key] = _copy_value(value)

        size_value = kwargs.get(_SIZE_INPUT_ID)
        include_size = _SIZE_INPUT_ID in connected or (not connected and size_value is not None)
        if include_size:
            size_pair = _size_pair_or_none(size_value)
            if size_pair is not None:
                changes[Const.IMAGEINFO_WIDTH], changes[Const.IMAGEINFO_HEIGHT] = size_pair

        entry = {
            "label": _normalized_label_or_none(kwargs.get("label")),
            "changes": changes,
        }

        output = list(current)
        output.append(entry)
        return c_io.NodeOutput(output)
