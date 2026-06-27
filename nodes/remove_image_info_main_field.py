# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.image_info_hash_extras import clear_model_reference_hash_extras

MAIN_FIELD_KEYS = (
    Const.IMAGEINFO_MODEL,
    Const.IMAGEINFO_REFINER_MODEL,
    Const.IMAGEINFO_DETAILER_MODEL,
    Const.IMAGEINFO_LORA_STACK,
    Const.IMAGEINFO_CLIP,
    Const.IMAGEINFO_VAE,
    Const.IMAGEINFO_POSITIVE,
    Const.IMAGEINFO_NEGATIVE,
    Const.IMAGEINFO_SAMPLER,
    Const.IMAGEINFO_SCHEDULER,
    Const.IMAGEINFO_STEPS,
    Const.IMAGEINFO_CFG,
    Const.IMAGEINFO_SEED,
    Const.IMAGEINFO_WIDTH,
    Const.IMAGEINFO_HEIGHT,
)

SAMPLER_SETTINGS_INPUT = "sampler_settings"
SIZE_INPUT = "size"
SAMPLER_SETTINGS_FIELD_KEYS = (
    Const.IMAGEINFO_SAMPLER,
    Const.IMAGEINFO_SCHEDULER,
    Const.IMAGEINFO_STEPS,
    Const.IMAGEINFO_CFG,
)
MODEL_REFERENCE_FIELD_KEYS = (
    Const.IMAGEINFO_MODEL,
    Const.IMAGEINFO_REFINER_MODEL,
    Const.IMAGEINFO_DETAILER_MODEL,
    Const.IMAGEINFO_LORA_STACK,
    Const.IMAGEINFO_CLIP,
    Const.IMAGEINFO_VAE,
)
MAIN_FIELD_INPUTS = (
    (
        Const.IMAGEINFO_MODEL,
        f"If true, remove image_info.{Const.IMAGEINFO_MODEL}.",
    ),
    (
        Const.IMAGEINFO_REFINER_MODEL,
        f"If true, remove image_info.{Const.IMAGEINFO_REFINER_MODEL}.",
    ),
    (
        Const.IMAGEINFO_DETAILER_MODEL,
        f"If true, remove image_info.{Const.IMAGEINFO_DETAILER_MODEL}.",
    ),
    (
        Const.IMAGEINFO_LORA_STACK,
        f"If true, remove image_info.{Const.IMAGEINFO_LORA_STACK}.",
    ),
    (
        Const.IMAGEINFO_CLIP,
        f"If true, remove image_info.{Const.IMAGEINFO_CLIP}.",
    ),
    (
        Const.IMAGEINFO_VAE,
        f"If true, remove image_info.{Const.IMAGEINFO_VAE}.",
    ),
    (
        Const.IMAGEINFO_POSITIVE,
        f"If true, remove image_info.{Const.IMAGEINFO_POSITIVE}.",
    ),
    (
        Const.IMAGEINFO_NEGATIVE,
        f"If true, remove image_info.{Const.IMAGEINFO_NEGATIVE}.",
    ),
    (
        SAMPLER_SETTINGS_INPUT,
        "If true, remove image_info.sampler, image_info.scheduler, image_info.steps, and image_info.cfg.",
    ),
    (
        Const.IMAGEINFO_SEED,
        f"If true, remove image_info.{Const.IMAGEINFO_SEED}.",
    ),
)


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


def _copied_image_info(image_info: dict[str, object] | None) -> dict[str, object]:
    return dict(image_info) if isinstance(image_info, dict) else {}


def _removed_main_fields(
    image_info: dict[str, object] | None,
    remove_flags: dict[str, bool],
) -> dict[str, object]:
    output = _copied_image_info(image_info)
    removed_model_reference_fields: list[str] = []
    for key in MAIN_FIELD_KEYS:
        if remove_flags.get(key, False):
            output.pop(key, None)
            if key in MODEL_REFERENCE_FIELD_KEYS:
                removed_model_reference_fields.append(key)
    if removed_model_reference_fields:
        output = clear_model_reference_hash_extras(
            output,
            fields=removed_model_reference_fields,
        )
    return output


class RemoveImageInfoMainFields(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-RemoveImageInfoMainFields",
            display_name="Remove Image Info Main Fields",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.IMAGEINFO_TYPE.Input(
                    Const.IMAGEINFO,
                    optional=True,
                ),
                *(
                    c_io.Boolean.Input(
                        key,
                        default=False,
                        tooltip=tooltip,
                    )
                    for key, tooltip in MAIN_FIELD_INPUTS
                ),
                c_io.Boolean.Input(
                    SIZE_INPUT,
                    display_name="width x height",
                    default=False,
                    tooltip="If true, remove image_info.width and image_info.height.",
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
    def validate_inputs(
        cls,
        image_info: dict[str, object] | None = None,
        model: Any = False,
        refiner: Any = False,
        detailer: Any = False,
        lora_stack: Any = False,
        clip: Any = False,
        vae: Any = False,
        positive: Any = False,
        negative: Any = False,
        sampler_settings: Any = False,
        seed: Any = False,
        size: Any = False,
    ) -> bool | str:
        return True

    @classmethod
    def execute(
        cls,
        image_info: dict[str, object] | None = None,
        model: Any = False,
        refiner: Any = False,
        detailer: Any = False,
        lora_stack: Any = False,
        clip: Any = False,
        vae: Any = False,
        positive: Any = False,
        negative: Any = False,
        sampler_settings: Any = False,
        seed: Any = False,
        size: Any = False,
    ) -> c_io.NodeOutput:
        remove_sampler_settings = _bool_or_default(sampler_settings, False)
        remove_size = _bool_or_default(size, False)
        remove_flags = {
            Const.IMAGEINFO_MODEL: _bool_or_default(model, False),
            Const.IMAGEINFO_REFINER_MODEL: _bool_or_default(refiner, False),
            Const.IMAGEINFO_DETAILER_MODEL: _bool_or_default(detailer, False),
            Const.IMAGEINFO_LORA_STACK: _bool_or_default(lora_stack, False),
            Const.IMAGEINFO_CLIP: _bool_or_default(clip, False),
            Const.IMAGEINFO_VAE: _bool_or_default(vae, False),
            Const.IMAGEINFO_POSITIVE: _bool_or_default(positive, False),
            Const.IMAGEINFO_NEGATIVE: _bool_or_default(negative, False),
            Const.IMAGEINFO_SEED: _bool_or_default(seed, False),
            Const.IMAGEINFO_WIDTH: remove_size,
            Const.IMAGEINFO_HEIGHT: remove_size,
        }
        for key in SAMPLER_SETTINGS_FIELD_KEYS:
            remove_flags[key] = remove_sampler_settings

        output = _removed_main_fields(image_info, remove_flags)
        return c_io.NodeOutput(output)
