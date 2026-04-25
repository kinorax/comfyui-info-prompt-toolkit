# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.sampler_params import SAMPLER_PARAMS_KEY, sampler_params_payload_or_error
from ..utils.tiled_sampling import (
    SpatialTiledModelWrapper,
    nonnegative_int_or_none,
    passthrough_top_level_controls,
    pixel_overlap_to_latent_overlap,
    positive_int_or_none,
)
from .sampler_custom_from_params import (
    CONDITIONING_RUNTIME_TYPE,
    LATENT_RUNTIME_TYPE,
    MODEL_RUNTIME_TYPE,
    _sample_with_sampler_custom,
    _sampler_from_name,
    _sigmas_from_scheduler,
)

DEFAULT_TILE_COLUMNS = 2
DEFAULT_TILE_ROWS = 2
DEFAULT_TILE_OVERLAP = 128
DEFAULT_MINI_UNIT = "32"
_MINI_UNIT_OPTIONS: tuple[str, ...] = ("8", "16", "32", "64")


def _spatial_scale_from_model(model: object) -> int:
    get_model_object = getattr(model, "get_model_object", None)
    if not callable(get_model_object):
        return 8

    try:
        latent_format = get_model_object("latent_format")
    except Exception:
        return 8

    try:
        scale = int(getattr(latent_format, "spacial_downscale_ratio", 8))
    except Exception:
        return 8
    return max(1, scale)


def _clone_model_with_tiled_wrapper(
    model: object,
    *,
    tile_columns: int,
    tile_rows: int,
    tile_overlap: int,
    mini_unit: int,
) -> object:
    clone = getattr(model, "clone", None)
    if not callable(clone):
        raise RuntimeError("SamplerCustom (Sampler Params, Tiled): model.clone() is unavailable")

    wrapped_model = clone()
    old_wrapper = None
    model_options = getattr(wrapped_model, "model_options", None)
    if isinstance(model_options, dict):
        candidate = model_options.get("model_function_wrapper")
        if callable(candidate):
            old_wrapper = candidate

    set_wrapper = getattr(wrapped_model, "set_model_unet_function_wrapper", None)
    if not callable(set_wrapper):
        raise RuntimeError("SamplerCustom (Sampler Params, Tiled): set_model_unet_function_wrapper() is unavailable")

    spatial_scale = _spatial_scale_from_model(wrapped_model)
    set_wrapper(
        SpatialTiledModelWrapper(
            tile_rows=tile_rows,
            tile_columns=tile_columns,
            tile_overlap=tile_overlap,
            mini_unit=pixel_overlap_to_latent_overlap(mini_unit, spatial_scale),
            spatial_scale=spatial_scale,
            old_wrapper=old_wrapper,
        )
    )
    return wrapped_model


class SamplerCustomFromParamsTiled(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        socket_force_input = {"forceInput": True}
        return c_io.Schema(
            node_id="IPT-SamplerCustomFromParamsTiled",
            display_name="SamplerCustom (Sampler Params, Tiled)",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                MODEL_RUNTIME_TYPE.Input(
                    "model",
                    extra_dict=socket_force_input,
                    tooltip="Loaded runtime MODEL input",
                ),
                CONDITIONING_RUNTIME_TYPE.Input(
                    "positive",
                    extra_dict=socket_force_input,
                    tooltip="Positive conditioning",
                ),
                CONDITIONING_RUNTIME_TYPE.Input(
                    "negative",
                    extra_dict=socket_force_input,
                    tooltip="Negative conditioning",
                ),
                Const.SAMPLER_PARAMS_TYPE.Input(
                    SAMPLER_PARAMS_KEY,
                    display_name=SAMPLER_PARAMS_KEY,
                    extra_dict=socket_force_input,
                    tooltip="Sampler Params bundle used to build sampler and sigmas",
                ),
                LATENT_RUNTIME_TYPE.Input(
                    "latent_image",
                    extra_dict=socket_force_input,
                    tooltip="Input latent image",
                ),
                c_io.Int.Input(
                    "tile_columns",
                    default=DEFAULT_TILE_COLUMNS,
                    min=1,
                    max=256,
                    tooltip="Number of horizontal tiles",
                ),
                c_io.Int.Input(
                    "tile_rows",
                    default=DEFAULT_TILE_ROWS,
                    min=1,
                    max=256,
                    tooltip="Number of vertical tiles",
                ),
                c_io.Int.Input(
                    "tile_overlap",
                    default=DEFAULT_TILE_OVERLAP,
                    min=0,
                    max=Const.MAX_RESOLUTION,
                    tooltip="Tile overlap in pixel-space units",
                ),
                c_io.Combo.Input(
                    "mini_unit",
                    options=_MINI_UNIT_OPTIONS,
                    default=DEFAULT_MINI_UNIT,
                    tooltip="Round tiled sampling window size up to a width and height divisible by this unit.",
                ),
            ],
            outputs=[
                LATENT_RUNTIME_TYPE.Output(
                    Cast.out_id("output"),
                    display_name="output",
                ),
                LATENT_RUNTIME_TYPE.Output(
                    Cast.out_id("denoised_output"),
                    display_name="denoised_output",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        model: object | None = None,
        positive: object | None = None,
        negative: object | None = None,
        sampler_params: object | None = None,
        latent_image: object | None = None,
        tile_columns: object = DEFAULT_TILE_COLUMNS,
        tile_rows: object = DEFAULT_TILE_ROWS,
        tile_overlap: object = DEFAULT_TILE_OVERLAP,
        mini_unit: object = DEFAULT_MINI_UNIT,
    ) -> bool | str:
        if tile_columns is not None and positive_int_or_none(tile_columns) is None:
            return "tile_columns must be an integer greater than 0"
        if tile_rows is not None and positive_int_or_none(tile_rows) is None:
            return "tile_rows must be an integer greater than 0"
        if tile_overlap is not None and nonnegative_int_or_none(tile_overlap) is None:
            return "tile_overlap must be an integer 0 or greater"
        mini_unit_value = positive_int_or_none(mini_unit)
        if mini_unit is not None and (mini_unit_value is None or mini_unit_value not in (8, 16, 32, 64)):
            return "mini_unit must be one of 8, 16, 32, or 64"
        if None in (model, positive, negative, sampler_params, latent_image):
            return True

        _, error = sampler_params_payload_or_error(sampler_params)
        return True if error is None else error

    @classmethod
    def execute(
        cls,
        model: object,
        positive: object,
        negative: object,
        sampler_params: object,
        latent_image: object,
        tile_columns: object = DEFAULT_TILE_COLUMNS,
        tile_rows: object = DEFAULT_TILE_ROWS,
        tile_overlap: object = DEFAULT_TILE_OVERLAP,
        mini_unit: object = DEFAULT_MINI_UNIT,
    ) -> c_io.NodeOutput:
        payload, error = sampler_params_payload_or_error(sampler_params)
        if payload is None or error is not None:
            raise RuntimeError(f"SamplerCustom (Sampler Params, Tiled): {error or 'sampler_params is required'}")

        tile_columns_value = positive_int_or_none(tile_columns)
        if tile_columns_value is None:
            raise RuntimeError("SamplerCustom (Sampler Params, Tiled): tile_columns must be an integer greater than 0")

        tile_rows_value = positive_int_or_none(tile_rows)
        if tile_rows_value is None:
            raise RuntimeError("SamplerCustom (Sampler Params, Tiled): tile_rows must be an integer greater than 0")

        tile_overlap_value = nonnegative_int_or_none(tile_overlap)
        if tile_overlap_value is None:
            raise RuntimeError("SamplerCustom (Sampler Params, Tiled): tile_overlap must be an integer 0 or greater")

        mini_unit_value = positive_int_or_none(mini_unit)
        if mini_unit_value is None or mini_unit_value not in (8, 16, 32, 64):
            raise RuntimeError("SamplerCustom (Sampler Params, Tiled): mini_unit must be one of 8, 16, 32, or 64")

        sampler = _sampler_from_name(str(payload[Const.IMAGEINFO_SAMPLER]))
        sigmas = _sigmas_from_scheduler(
            model,
            str(payload[Const.IMAGEINFO_SCHEDULER]),
            int(payload[Const.IMAGEINFO_STEPS]),
            float(payload["denoise"]),
        )
        wrapped_model = _clone_model_with_tiled_wrapper(
            model,
            tile_columns=tile_columns_value,
            tile_rows=tile_rows_value,
            tile_overlap=tile_overlap_value,
            mini_unit=mini_unit_value,
        )

        with passthrough_top_level_controls(positive, negative):
            output, denoised_output = _sample_with_sampler_custom(
                model=wrapped_model,
                noise_seed=int(payload[Const.IMAGEINFO_SEED]),
                cfg=float(payload[Const.IMAGEINFO_CFG]),
                positive=positive,
                negative=negative,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latent_image,
            )
        return c_io.NodeOutput(output, denoised_output)
