# Copyright 2026 kinorax
from comfy_api.latest import io as c_io
from .. import const as Const
from ..utils.selector_resolution import resolve_selector_value


class UnetModelSelector(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        unet_model_options = Const.get_UNET_MODEL_OPTIONS()
        default_unet_model = unet_model_options[0] if unet_model_options else ""
        return c_io.Schema(
            node_id="IPT-UnetModelSelector",
            display_name="Unet Model Selector",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.Combo.Input(
                    "unet_model",
                    display_name="unet_name",
                    options=unet_model_options,
                    default=default_unet_model,
                    tooltip="Select UNet model",
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
                Const.MODEL_TYPE.Output(
                    "UNET_MODEL",
                    display_name="model",
                ),
                c_io.AnyType.Output(
                    "UNET_MODEL_COMBO",
                    display_name="unet_name",
                ),
            ],
        )

    @classmethod
    def execute(cls, unet_model: str, sha256: str | None = None) -> c_io.NodeOutput:
        unet_model_options = Const.get_UNET_MODEL_OPTIONS()
        value = resolve_selector_value(
            unet_model,
            unet_model_options,
            value_label="unet_model",
            folder_name=Const.MODEL_FOLDER_PATH_UNET,
            sha256=sha256,
        )
        model = Const.make_model_value(value, Const.MODEL_FOLDER_PATH_UNET)
        return c_io.NodeOutput(model, value)
