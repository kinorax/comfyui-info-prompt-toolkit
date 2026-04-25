# Copyright 2026 kinorax
from comfy_api.latest import io as c_io
from .. import const as Const
from ..utils.selector_resolution import resolve_selector_value


class VaeSelector(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        vae_options = Const.get_VAE_OPTIONS()
        default_vae = vae_options[0] if vae_options else ""
        return c_io.Schema(
            node_id="IPT-VaeSelector",
            display_name="Vae Selector",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.Combo.Input(
                    "vae",
                    options=vae_options,
                    default=default_vae,
                    tooltip="Select VAE",
                ),
                c_io.String.Input(
                    "sha256",
                    default="",
                    socketless=True,
                    optional=True,
                    tooltip="Cached SHA256 used by View Model Info fallback",
                ),
            ],
            outputs=[
                c_io.AnyType.Output(
                    "VAE",
                    display_name="vae",
                ),
            ],
        )

    @classmethod
    def execute(cls, vae: str, sha256: str | None = None) -> c_io.NodeOutput:
        vae_options = Const.get_VAE_OPTIONS()
        value = resolve_selector_value(
            vae,
            vae_options,
            value_label="vae",
            folder_name=Const.MODEL_FOLDER_PATH_VAE,
            sha256=sha256,
        )
        return c_io.NodeOutput(value)
