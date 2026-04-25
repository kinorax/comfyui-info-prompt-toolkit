# Copyright 2026 kinorax
from comfy_api.latest import io as c_io
from .. import const as Const
from ..utils.selector_resolution import resolve_selector_value


class DiffusionModelSelector(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        diffusion_model_options = Const.get_DIFFUSION_MODEL_OPTIONS()
        weight_dtype_options = Const.get_WEIGHT_DTYPE_OPTIONS()
        default_diffusion_model = diffusion_model_options[0] if diffusion_model_options else ""
        return c_io.Schema(
            node_id="IPT-DiffusionModelSelector",
            display_name="Diffusion Model Selector",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.Combo.Input(
                    "diffusion_model",
                    display_name="unet_name",
                    options=diffusion_model_options,
                    default=default_diffusion_model,
                    tooltip="Select diffusion model",
                ),
                c_io.Combo.Input(
                    "weight_dtype",
                    options=weight_dtype_options,
                    default=weight_dtype_options[0],
                    tooltip="Weight dtype passed by Load Diffusion Model",
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
                    "DIFFUSION_MODEL",
                    display_name="model",
                ),
                c_io.AnyType.Output(
                    "DIFFUSION_MODEL_COMBO",
                    display_name="unet_name",
                ),
            ],
        )

    @classmethod
    def execute(cls, diffusion_model: str, weight_dtype: str, sha256: str | None = None) -> c_io.NodeOutput:
        diffusion_model_options = Const.get_DIFFUSION_MODEL_OPTIONS()
        weight_dtype_options = Const.get_WEIGHT_DTYPE_OPTIONS()
        value = resolve_selector_value(
            diffusion_model,
            diffusion_model_options,
            value_label="diffusion_model",
            folder_name=Const.MODEL_FOLDER_PATH_DIFFUSION_MODELS,
            sha256=sha256,
        )
        normalized_weight_dtype = (
            weight_dtype if weight_dtype in weight_dtype_options else weight_dtype_options[0]
        )
        model = Const.make_model_value(
            value,
            Const.MODEL_FOLDER_PATH_DIFFUSION_MODELS,
            {
                Const.MODEL_VALUE_WEIGHT_DTYPE_KEY: normalized_weight_dtype,
            },
        )
        return c_io.NodeOutput(model, value)
