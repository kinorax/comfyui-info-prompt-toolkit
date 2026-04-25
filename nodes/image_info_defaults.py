# Copyright 2026 kinorax
from __future__ import annotations

import re

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.a1111_infotext import extract_lora_stack_from_prompt
from ..utils.image_info_defaults_merge import is_unset, merge_image_info_missing_values
from ..utils.image_info_normalizer import normalize_lora_stack_with_comfy_options

SIZE_PATTERN = re.compile(r"^\s*(-?\d+)\s*[xX]\s*(-?\d+)\s*$")
SIZE_INPUT_ID = "size"

_INT_INPUT_MAX = 0xFFFFFFFFFFFFFFFF
_STEPS_INPUT_MAX = 10000
_DEFAULT_STEPS = 20
_DEFAULT_CFG = 7.0
_DEFAULT_SEED = 0
_DEFAULT_WIDTH = 512
_DEFAULT_HEIGHT = 512

_SOCKET_ONLY_KEYS = (
    Const.IMAGEINFO_MODEL,
    Const.IMAGEINFO_REFINER_MODEL,
    Const.IMAGEINFO_DETAILER_MODEL,
    Const.IMAGEINFO_CLIP,
    Const.IMAGEINFO_VAE,
    Const.IMAGEINFO_EXTRAS,
)

_WIDGET_DEFAULT_KEYS = (
    Const.IMAGEINFO_NEGATIVE,
    Const.IMAGEINFO_STEPS,
    Const.IMAGEINFO_SAMPLER,
    Const.IMAGEINFO_SCHEDULER,
    Const.IMAGEINFO_CFG,
    Const.IMAGEINFO_SEED,
)


def _connected_input_keys(prompt: object, unique_id: object) -> set[str]:
    if not isinstance(prompt, dict) or unique_id is None:
        return set()

    node = prompt.get(str(unique_id), {}) or {}
    inputs = node.get("inputs", {}) or {}
    return set(inputs.keys())


def _normalize_option(value: object, options: tuple[str, ...]) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text in options:
        return text
    return None


def _split_positive_and_lora_stack(positive_value: object) -> tuple[str | None, list[dict[str, str | float]] | None]:
    if positive_value is None:
        return None, None

    prompt_without_lora, lora_stack = extract_lora_stack_from_prompt(str(positive_value))
    normalized_lora_stack = normalize_lora_stack_with_comfy_options(lora_stack)
    return prompt_without_lora, normalized_lora_stack


def _merged_lora_stack(
    base_lora_stack: object,
    appended_lora_stack: list[dict[str, str | float]] | None,
) -> list[object] | None:
    output = list(base_lora_stack) if isinstance(base_lora_stack, list) else []
    if isinstance(appended_lora_stack, list):
        output.extend(appended_lora_stack)
    if output:
        return output
    return None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _size_tuple_or_none(value: object) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _size_tuple_or_none(value[0])

    if isinstance(value, dict) and "__value__" in value:
        return _size_tuple_or_none(value.get("__value__"))

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
        m = SIZE_PATTERN.fullmatch(value)
        if not m:
            return None
        width = _int_or_none(m.group(1))
        height = _int_or_none(m.group(2))
        if width is None or height is None:
            return None
        return width, height

    return None


def _size_payload(width: int, height: int) -> dict[str, int]:
    return {
        "width": width,
        "height": height,
    }


class ImageInfoDefaults(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        socket_only_string = dict(optional=True, force_input=True)
        sampler_default = Const.SAMPLER_OPTIONS[0] if Const.SAMPLER_OPTIONS else ""
        scheduler_default = Const.SCHEDULER_OPTIONS[0] if Const.SCHEDULER_OPTIONS else ""

        return c_io.Schema(
            node_id="IPT-ImageInfoDefaults",
            display_name="Image Info Defaults",
            category=Const.CATEGORY_IMAGEINFO,
            hidden=[c_io.Hidden.prompt, c_io.Hidden.unique_id],
            inputs=[
                Const.IMAGEINFO_TYPE.Input(Const.IMAGEINFO, optional=True),
                Const.MODEL_TYPE.Input(Const.IMAGEINFO_MODEL, optional=True),
                Const.MODEL_TYPE.Input(Const.IMAGEINFO_REFINER_MODEL, optional=True),
                Const.MODEL_TYPE.Input(Const.IMAGEINFO_DETAILER_MODEL, optional=True),
                Const.LORA_STACK_TYPE.Input(Const.IMAGEINFO_LORA_STACK, optional=True),
                Const.CLIP_TYPE.Input(Const.IMAGEINFO_CLIP, optional=True),
                c_io.String.Input(Const.IMAGEINFO_VAE, **socket_only_string),
                c_io.String.Input(
                    Const.IMAGEINFO_POSITIVE,
                    default="",
                    multiline=True,
                    tooltip="Default positive prompt when image_info.positive is None",
                ),
                c_io.String.Input(
                    Const.IMAGEINFO_NEGATIVE,
                    default="",
                    multiline=True,
                    tooltip="Default negative prompt when image_info.negative is None",
                ),
                c_io.Int.Input(
                    Const.IMAGEINFO_STEPS,
                    default=_DEFAULT_STEPS,
                    min=1,
                    max=_STEPS_INPUT_MAX,
                    tooltip="Default steps when image_info.steps is None",
                ),
                c_io.Combo.Input(
                    Const.IMAGEINFO_SAMPLER,
                    options=Const.SAMPLER_OPTIONS,
                    default=sampler_default,
                    tooltip="Default sampler when image_info.sampler is None",
                ),
                c_io.Combo.Input(
                    Const.IMAGEINFO_SCHEDULER,
                    options=Const.SCHEDULER_OPTIONS,
                    default=scheduler_default,
                    tooltip="Default scheduler when image_info.scheduler is None",
                ),
                c_io.Float.Input(
                    Const.IMAGEINFO_CFG,
                    default=_DEFAULT_CFG,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    tooltip="Default CFG scale when image_info.cfg is None",
                ),
                c_io.Int.Input(
                    Const.IMAGEINFO_SEED,
                    default=_DEFAULT_SEED,
                    min=0,
                    max=_INT_INPUT_MAX,
                    control_after_generate=False,
                    tooltip="Default seed when image_info.seed is None",
                ),
                Const.SIZE_TYPE.Input(
                    SIZE_INPUT_ID,
                    display_name="width x height",
                    extra_dict={
                        "default": _size_payload(_DEFAULT_WIDTH, _DEFAULT_HEIGHT),
                        "min": Const.MIN_RESOLUTION,
                        "max": Const.MAX_RESOLUTION,
                        "step": 1,
                    },
                    tooltip="Default size when image_info.width or image_info.height is None",
                ),
                Const.IMAGEINFO_EXTRAS_TYPE.Input(Const.IMAGEINFO_EXTRAS, optional=True),
            ],
            outputs=[
                Const.IMAGEINFO_TYPE.Output(Cast.out_id(Const.IMAGEINFO), display_name=Const.IMAGEINFO),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        size: object,
    ) -> bool | str:
        # During Comfy validation, linked values are unresolved and arrive as None.
        # Required-input checks and type checks for links are handled by the framework.
        if size is None:
            return True

        parsed_size = _size_tuple_or_none(size)
        if parsed_size is None:
            return "width x height must contain integer width and height"

        width_int, height_int = parsed_size
        if width_int < Const.MIN_RESOLUTION:
            return f"width must be {Const.MIN_RESOLUTION} or greater"
        if width_int > Const.MAX_RESOLUTION:
            return f"width must be {Const.MAX_RESOLUTION} or less"
        if height_int < Const.MIN_RESOLUTION:
            return f"height must be {Const.MIN_RESOLUTION} or greater"
        if height_int > Const.MAX_RESOLUTION:
            return f"height must be {Const.MAX_RESOLUTION} or less"
        return True

    @classmethod
    def execute(cls, **kwargs) -> c_io.NodeOutput:
        image_info = kwargs.get(Const.IMAGEINFO)
        base = dict(image_info) if isinstance(image_info, dict) else {}

        prompt = getattr(cls.hidden, "prompt", None)
        unique_id = getattr(cls.hidden, "unique_id", None)
        connected = _connected_input_keys(prompt, unique_id)

        socket_defaults: dict[str, object] = {}
        for key in _SOCKET_ONLY_KEYS:
            if key in connected:
                socket_defaults[key] = kwargs.get(key)

        base = merge_image_info_missing_values(
            base,
            socket_defaults,
            extras_key=Const.IMAGEINFO_EXTRAS,
            positive_key=Const.IMAGEINFO_POSITIVE,
            lora_stack_key=Const.IMAGEINFO_LORA_STACK,
            preserve_lora_stack_when_positive_present=True,
        )

        positive_cleaned, lora_stack_from_positive = _split_positive_and_lora_stack(kwargs.get(Const.IMAGEINFO_POSITIVE))

        computed_defaults: dict[str, object] = {}
        if positive_cleaned is not None:
            computed_defaults[Const.IMAGEINFO_POSITIVE] = positive_cleaned

        if is_unset(base.get(Const.IMAGEINFO_LORA_STACK)):
            base_lora_stack = kwargs.get(Const.IMAGEINFO_LORA_STACK) if Const.IMAGEINFO_LORA_STACK in connected else None
            merged_lora_stack = _merged_lora_stack(base_lora_stack, lora_stack_from_positive)
            if merged_lora_stack is not None:
                computed_defaults[Const.IMAGEINFO_LORA_STACK] = merged_lora_stack

        parsed_size = _size_tuple_or_none(kwargs.get(SIZE_INPUT_ID))
        if parsed_size is not None:
            parsed_width, parsed_height = parsed_size
            computed_defaults[Const.IMAGEINFO_WIDTH] = parsed_width
            computed_defaults[Const.IMAGEINFO_HEIGHT] = parsed_height

        for key in _WIDGET_DEFAULT_KEYS:
            if not is_unset(base.get(key)):
                continue

            value = kwargs.get(key)
            if key == Const.IMAGEINFO_SAMPLER:
                value = _normalize_option(value, Const.SAMPLER_OPTIONS)
            elif key == Const.IMAGEINFO_SCHEDULER:
                value = _normalize_option(value, Const.SCHEDULER_OPTIONS)

            if value is not None:
                computed_defaults[key] = value

        base = merge_image_info_missing_values(
            base,
            computed_defaults,
            extras_key=Const.IMAGEINFO_EXTRAS,
            positive_key=Const.IMAGEINFO_POSITIVE,
            lora_stack_key=Const.IMAGEINFO_LORA_STACK,
            preserve_lora_stack_when_positive_present=True,
        )

        return c_io.NodeOutput(base)
