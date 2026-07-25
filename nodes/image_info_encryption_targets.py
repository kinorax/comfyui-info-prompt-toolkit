# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.image_info_encryption_targets import (
    WIDTH_HEIGHT_TARGET,
    build_encryption_targets,
    image_info_with_encryption_targets,
)

ENCRYPTION_TARGETS = "encryption_targets"
EXTRA_KEYS_INPUT = "extra_keys"
WIDTH_HEIGHT_INPUT = "width_x_height"

MAIN_FIELD_INPUTS = (
    (
        Const.IMAGEINFO_MODEL,
        Const.IMAGEINFO_MODEL,
        f"If true, save image_info.{Const.IMAGEINFO_MODEL} in the encrypted metadata area.",
    ),
    (
        Const.IMAGEINFO_REFINER_MODEL,
        Const.IMAGEINFO_REFINER_MODEL,
        f"If true, save image_info.{Const.IMAGEINFO_REFINER_MODEL} in the encrypted metadata area.",
    ),
    (
        Const.IMAGEINFO_DETAILER_MODEL,
        Const.IMAGEINFO_DETAILER_MODEL,
        f"If true, save image_info.{Const.IMAGEINFO_DETAILER_MODEL} in the encrypted metadata area.",
    ),
    (
        Const.IMAGEINFO_LORA_STACK,
        Const.IMAGEINFO_LORA_STACK,
        f"If true, save image_info.{Const.IMAGEINFO_LORA_STACK} in the encrypted metadata area.",
    ),
    (
        Const.IMAGEINFO_CLIP,
        Const.IMAGEINFO_CLIP,
        f"If true, save image_info.{Const.IMAGEINFO_CLIP} in the encrypted metadata area.",
    ),
    (
        Const.IMAGEINFO_VAE,
        Const.IMAGEINFO_VAE,
        f"If true, save image_info.{Const.IMAGEINFO_VAE} in the encrypted metadata area.",
    ),
    (
        Const.IMAGEINFO_POSITIVE,
        Const.IMAGEINFO_POSITIVE,
        f"If true, save image_info.{Const.IMAGEINFO_POSITIVE} in the encrypted metadata area.",
    ),
    (
        Const.IMAGEINFO_NEGATIVE,
        Const.IMAGEINFO_NEGATIVE,
        f"If true, save image_info.{Const.IMAGEINFO_NEGATIVE} in the encrypted metadata area.",
    ),
    (
        Const.IMAGEINFO_STEPS,
        Const.IMAGEINFO_STEPS,
        f"If true, save image_info.{Const.IMAGEINFO_STEPS} in the encrypted metadata area.",
    ),
    (
        Const.IMAGEINFO_SAMPLER,
        Const.IMAGEINFO_SAMPLER,
        f"If true, save image_info.{Const.IMAGEINFO_SAMPLER} in the encrypted metadata area.",
    ),
    (
        Const.IMAGEINFO_SCHEDULER,
        Const.IMAGEINFO_SCHEDULER,
        f"If true, save image_info.{Const.IMAGEINFO_SCHEDULER} in the encrypted metadata area.",
    ),
    (
        Const.IMAGEINFO_CFG,
        Const.IMAGEINFO_CFG,
        f"If true, save image_info.{Const.IMAGEINFO_CFG} in the encrypted metadata area.",
    ),
    (
        Const.IMAGEINFO_SEED,
        Const.IMAGEINFO_SEED,
        f"If true, save image_info.{Const.IMAGEINFO_SEED} in the encrypted metadata area.",
    ),
    (
        WIDTH_HEIGHT_INPUT,
        WIDTH_HEIGHT_TARGET,
        "If true, save image_info.width and image_info.height in the encrypted metadata area.",
    ),
)


class ImageInfoEncryptionTargets(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-ImageInfoEncryptionTargets",
            display_name="Image Info Encryption Targets",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                *(
                    c_io.Boolean.Input(
                        input_id,
                        display_name=display_name,
                        default=False,
                        tooltip=tooltip,
                    )
                    for input_id, display_name, tooltip in MAIN_FIELD_INPUTS
                ),
                c_io.String.Input(
                    EXTRA_KEYS_INPUT,
                    display_name="extra keys",
                    default="[]",
                    multiline=True,
                    socketless=True,
                    tooltip="JSON rows used by the extra key toggle editor. Enabled keys in image_info.extras are encrypted.",
                ),
            ],
            outputs=[
                Const.IMAGEINFO_ENCRYPTION_TARGETS_TYPE.Output(
                    Cast.out_id(ENCRYPTION_TARGETS),
                    display_name=ENCRYPTION_TARGETS,
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        return True

    @classmethod
    def execute(cls, **kwargs) -> c_io.NodeOutput:
        main_field_flags = {
            display_name: kwargs.get(input_id)
            for input_id, display_name, _tooltip in MAIN_FIELD_INPUTS
        }
        targets = build_encryption_targets(
            main_field_flags,
            kwargs.get(EXTRA_KEYS_INPUT),
        )
        return c_io.NodeOutput(targets)


class SetImageInfoEncryptionTargets(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-SetImageInfoEncryptionTargets",
            display_name="Set Image Info Encryption Targets",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.IMAGEINFO_TYPE.Input(
                    Const.IMAGEINFO,
                    optional=True,
                ),
                Const.IMAGEINFO_ENCRYPTION_TARGETS_TYPE.Input(
                    ENCRYPTION_TARGETS,
                    optional=True,
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
        encryption_targets: Any = None,
    ) -> bool | str:
        return True

    @classmethod
    def execute(
        cls,
        image_info: dict[str, object] | None = None,
        encryption_targets: Any = None,
    ) -> c_io.NodeOutput:
        output = image_info_with_encryption_targets(image_info, encryption_targets)
        return c_io.NodeOutput(output)
