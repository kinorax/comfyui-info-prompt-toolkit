# Copyright 2026 kinorax
from __future__ import annotations

import re

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.a1111_infotext import extract_lora_stack_from_prompt
from ..utils.image_info_normalizer import normalize_lora_stack_with_comfy_options
from ..utils.sampler_params import DENOISE_KEY, sampler_params_payload_or_error

SIZE_PATTERN = re.compile(r"^\s*(-?\d+)\s*[xX]\s*(-?\d+)\s*$")
SIZE_INPUT_ID = "size"
BASE_SAMPLER_PARAMS_OUTPUT_ID = "base_sampler_params"

FIELDS = (
    (Const.IMAGEINFO_POSITIVE, c_io.String, c_io.String, Cast.str_or_none),
    (Const.IMAGEINFO_NEGATIVE, c_io.String, c_io.String, Cast.str_or_none),
    (Const.IMAGEINFO_STEPS, c_io.Int, c_io.Int, Cast.int_or_none),
    # sampler/scheduler are string values, but AnyType output keeps compatibility with combo-like sockets.
    (Const.IMAGEINFO_SAMPLER, c_io.String, c_io.AnyType, Cast.str_or_none),
    (Const.IMAGEINFO_SCHEDULER, c_io.String, c_io.AnyType, Cast.str_or_none),
    (Const.IMAGEINFO_CFG, c_io.Float, c_io.Float, Cast.float_or_none),
    (Const.IMAGEINFO_SEED, c_io.Int, c_io.Int, Cast.int_or_none),
)
TOP_KEYS = (
    Const.IMAGEINFO_MODEL,
    Const.IMAGEINFO_REFINER_MODEL,
    Const.IMAGEINFO_DETAILER_MODEL,
    Const.IMAGEINFO_LORA_STACK,
    Const.IMAGEINFO_CLIP,
    Const.IMAGEINFO_VAE,
    Const.IMAGEINFO_EXTRAS,
)


def _merge_extras(base_extras: object, input_extras: object) -> dict[str, object] | None:
    output = dict(base_extras) if isinstance(base_extras, dict) else {}
    if isinstance(input_extras, dict):
        output.update(input_extras)
    if output:
        return output
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


def _size_payload_or_none(width: object, height: object) -> dict[str, int] | None:
    width_int = _int_or_none(width)
    height_int = _int_or_none(height)
    if width_int is None or height_int is None:
        return None
    return {
        "width": width_int,
        "height": height_int,
    }


def _include_input_key(connected: set[str], kwargs: dict[str, object], key: str) -> bool:
    if key in connected:
        return True
    if len(connected) > 0:
        return False
    return kwargs.get(key) is not None


def _base_sampler_params_or_none(base: dict[str, object]) -> dict[str, object] | None:
    payload, error = sampler_params_payload_or_error(
        {
            Const.IMAGEINFO_SAMPLER: base.get(Const.IMAGEINFO_SAMPLER),
            Const.IMAGEINFO_SCHEDULER: base.get(Const.IMAGEINFO_SCHEDULER),
            Const.IMAGEINFO_STEPS: base.get(Const.IMAGEINFO_STEPS),
            DENOISE_KEY: 1.0,
            Const.IMAGEINFO_SEED: base.get(Const.IMAGEINFO_SEED),
            Const.IMAGEINFO_CFG: base.get(Const.IMAGEINFO_CFG),
        }
    )
    if error is not None:
        return None
    return payload


class ImageInfoContext(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        fi = dict(optional=True, force_input=True)  # 「点」固定（widget を出さず socket 入力のみ）

        return c_io.Schema(
            node_id="IPT-ImageInfoContext",
            display_name="Image Info Context",
            category=Const.CATEGORY_IMAGEINFO,
            # v1 の hidden: {"prompt":"PROMPT","unique_id":"UNIQUE_ID"} 相当
            hidden=[c_io.Hidden.prompt, c_io.Hidden.unique_id],
            inputs=[
                Const.IMAGEINFO_TYPE.Input(Const.IMAGEINFO, optional=True),
                Const.MODEL_TYPE.Input(Const.IMAGEINFO_MODEL, optional=True),
                Const.MODEL_TYPE.Input(Const.IMAGEINFO_REFINER_MODEL, optional=True),
                Const.MODEL_TYPE.Input(Const.IMAGEINFO_DETAILER_MODEL, optional=True),
                Const.LORA_STACK_TYPE.Input(Const.IMAGEINFO_LORA_STACK, optional=True),
                Const.CLIP_TYPE.Input(Const.IMAGEINFO_CLIP, optional=True),
                c_io.String.Input(Const.IMAGEINFO_VAE, **fi),
                *(input_io_type.Input(key, **fi) for key, input_io_type, _, _ in FIELDS),
                Const.SIZE_TYPE.Input(
                    SIZE_INPUT_ID,
                    display_name="width x height",
                    optional=True,
                    extra_dict={"forceInput": True},
                ),
                Const.IMAGEINFO_EXTRAS_TYPE.Input(Const.IMAGEINFO_EXTRAS, optional=True),
            ],
            outputs=[
                Const.IMAGEINFO_TYPE.Output(Cast.out_id(Const.IMAGEINFO), display_name=Const.IMAGEINFO),
                Const.MODEL_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO_MODEL),
                    display_name=Const.IMAGEINFO_MODEL,
                ),
                Const.MODEL_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO_REFINER_MODEL),
                    display_name=Const.IMAGEINFO_REFINER_MODEL,
                ),
                Const.MODEL_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO_DETAILER_MODEL),
                    display_name=Const.IMAGEINFO_DETAILER_MODEL,
                ),
                Const.LORA_STACK_TYPE.Output(Cast.out_id(Const.IMAGEINFO_LORA_STACK), display_name=Const.IMAGEINFO_LORA_STACK),
                Const.CLIP_TYPE.Output(Cast.out_id(Const.IMAGEINFO_CLIP), display_name=Const.IMAGEINFO_CLIP),
                c_io.AnyType.Output(Cast.out_id(Const.IMAGEINFO_VAE), display_name=Const.IMAGEINFO_VAE),
                *(output_io_type.Output(Cast.out_id(key), display_name=key) for key, _, output_io_type, _ in FIELDS),
                Const.SIZE_TYPE.Output(
                    Cast.out_id(SIZE_INPUT_ID),
                    display_name="width x height",
                ),
                Const.IMAGEINFO_EXTRAS_TYPE.Output(Cast.out_id(Const.IMAGEINFO_EXTRAS), display_name=Const.IMAGEINFO_EXTRAS),
                Const.SAMPLER_PARAMS_TYPE.Output(
                    Cast.out_id(BASE_SAMPLER_PARAMS_OUTPUT_ID),
                    display_name=BASE_SAMPLER_PARAMS_OUTPUT_ID,
                ),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> c_io.NodeOutput:
        image_info = kwargs.get(Const.IMAGEINFO)
        base = dict(image_info) if isinstance(image_info, dict) else {}

        prompt = getattr(cls.hidden, "prompt", None)
        unique_id = getattr(cls.hidden, "unique_id", None)

        connected: set[str] = set()
        if isinstance(prompt, dict) and unique_id is not None:
            node = prompt.get(str(unique_id), {}) or {}
            inputs = node.get("inputs", {}) or {}
            connected = set(inputs.keys())

        for key in TOP_KEYS:
            if _include_input_key(connected, kwargs, key):
                if key == Const.IMAGEINFO_EXTRAS:
                    merged_extras = _merge_extras(base.get(Const.IMAGEINFO_EXTRAS), kwargs.get(key))
                    if merged_extras is None:
                        base.pop(Const.IMAGEINFO_EXTRAS, None)
                    else:
                        base[Const.IMAGEINFO_EXTRAS] = merged_extras
                    continue
                base[key] = kwargs.get(key)

        for key, _, _, _ in FIELDS:
            if _include_input_key(connected, kwargs, key):
                if key == Const.IMAGEINFO_POSITIVE:
                    positive_cleaned, lora_stack_from_positive = _split_positive_and_lora_stack(kwargs.get(key))
                    base[key] = positive_cleaned

                    if _include_input_key(connected, kwargs, Const.IMAGEINFO_LORA_STACK):
                        base_lora_stack = kwargs.get(Const.IMAGEINFO_LORA_STACK)
                        base[Const.IMAGEINFO_LORA_STACK] = _merged_lora_stack(base_lora_stack, lora_stack_from_positive)
                    elif lora_stack_from_positive is not None:
                        # positive から抽出した LoRA は入力 lora_stack と同等に扱う。
                        # lora_stack 入力が未接続なら、image_info 側の既存値ではなく抽出値で上書きする。
                        base[Const.IMAGEINFO_LORA_STACK] = lora_stack_from_positive
                    continue
                base[key] = kwargs.get(key)

        if _include_input_key(connected, kwargs, SIZE_INPUT_ID):
            parsed_size = _size_tuple_or_none(kwargs.get(SIZE_INPUT_ID))
            if parsed_size is not None:
                base[Const.IMAGEINFO_WIDTH], base[Const.IMAGEINFO_HEIGHT] = parsed_size

        model_value = base.get(Const.IMAGEINFO_MODEL)
        refiner_model_value = base.get(Const.IMAGEINFO_REFINER_MODEL)
        detailer_model_value = base.get(Const.IMAGEINFO_DETAILER_MODEL)

        size_out = _size_payload_or_none(base.get(Const.IMAGEINFO_WIDTH), base.get(Const.IMAGEINFO_HEIGHT))
        base_sampler_params_out = _base_sampler_params_or_none(base)

        return c_io.NodeOutput(
            base,
            model_value,
            refiner_model_value,
            detailer_model_value,
            base.get(Const.IMAGEINFO_LORA_STACK),
            base.get(Const.IMAGEINFO_CLIP),
            base.get(Const.IMAGEINFO_VAE),
            *(caster(base.get(key)) for key, _, _, caster in FIELDS),
            size_out,
            base.get(Const.IMAGEINFO_EXTRAS),
            base_sampler_params_out,
        )
