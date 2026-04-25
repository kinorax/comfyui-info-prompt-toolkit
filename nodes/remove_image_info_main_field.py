# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast

MAIN_FIELD_KEYS = (
    Const.IMAGEINFO_MODEL,
    Const.IMAGEINFO_REFINER_MODEL,
    Const.IMAGEINFO_DETAILER_MODEL,
    Const.IMAGEINFO_LORA_STACK,
    Const.IMAGEINFO_CLIP,
    Const.IMAGEINFO_VAE,
    Const.IMAGEINFO_POSITIVE,
    Const.IMAGEINFO_NEGATIVE,
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
    for key in MAIN_FIELD_KEYS:
        if remove_flags.get(key, False):
            output.pop(key, None)
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
                        tooltip=f"If true, remove image_info.{key}.",
                    )
                    for key in MAIN_FIELD_KEYS
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
    ) -> c_io.NodeOutput:
        remove_flags = {
            Const.IMAGEINFO_MODEL: _bool_or_default(model, False),
            Const.IMAGEINFO_REFINER_MODEL: _bool_or_default(refiner, False),
            Const.IMAGEINFO_DETAILER_MODEL: _bool_or_default(detailer, False),
            Const.IMAGEINFO_LORA_STACK: _bool_or_default(lora_stack, False),
            Const.IMAGEINFO_CLIP: _bool_or_default(clip, False),
            Const.IMAGEINFO_VAE: _bool_or_default(vae, False),
            Const.IMAGEINFO_POSITIVE: _bool_or_default(positive, False),
            Const.IMAGEINFO_NEGATIVE: _bool_or_default(negative, False),
        }
        output = _removed_main_fields(image_info, remove_flags)
        return c_io.NodeOutput(output)
