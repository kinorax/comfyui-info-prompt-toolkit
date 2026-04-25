# Copyright 2026 kinorax
from __future__ import annotations

import time
from threading import Lock

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.model_lora_metadata_pipeline import get_shared_metadata_pipeline
from ..utils.model_runtime_settings import clip_last_layer_from_settings, sd3_shift_from_settings
from ._runtime_loader import (
    CHECKPOINT_LOADER_KEYS,
    CLIP_LOADER_KEYS_BY_COUNT,
    CLIP_SET_LAST_LAYER_KEYS,
    DIFFUSION_MODEL_LOADER_KEYS,
    MODEL_SAMPLING_SD3_KEYS,
    VAE_LOADER_KEYS,
    apply_clip_last_layer_with_core_node,
    apply_model_sampling_sd3_with_core_node,
    core_nodes_module_or_none,
    is_checkpoint_model,
    is_diffusion_model,
    load_checkpoint_with_core_loader,
    load_clip_with_core_loader,
    load_diffusion_model_with_core_loader,
    load_vae_with_core_loader,
    loader_class_or_none,
    normalized_clip_device_or_none,
    normalized_clip_last_layer_or_none,
    normalized_clip_names,
    normalized_clip_payload_or_none,
    normalized_clip_type_or_none,
    normalized_model_name_or_none,
    normalized_model_weight_dtype_or_none,
    normalized_vae_name_or_none,
)

MODEL_RUNTIME_TYPE = c_io.Custom("MODEL")
CLIP_RUNTIME_TYPE = c_io.Custom("CLIP")
VAE_RUNTIME_TYPE = c_io.Custom("VAE")

_VAE_CACHE_LOCK = Lock()
_LAST_VAE_NAME: str | None = None
_LAST_VAE_RUNTIME: object | None = None


def _load_cached_vae(vae_name: str, core_nodes_module: object) -> object:
    global _LAST_VAE_NAME
    global _LAST_VAE_RUNTIME

    with _VAE_CACHE_LOCK:
        if _LAST_VAE_NAME == vae_name and _LAST_VAE_RUNTIME is not None:
            return _LAST_VAE_RUNTIME

    vae_loader_class = loader_class_or_none(core_nodes_module, VAE_LOADER_KEYS)
    if vae_loader_class is None:
        raise RuntimeError("Load VAE node is unavailable")

    runtime_vae = load_vae_with_core_loader(vae_loader_class(), vae_name)
    if runtime_vae is not None:
        with _VAE_CACHE_LOCK:
            _LAST_VAE_NAME = vae_name
            _LAST_VAE_RUNTIME = runtime_vae
    return runtime_vae


def _resolve_vae_runtime(value: object, core_nodes_module: object) -> object | None:
    if value is None:
        return None

    vae_name = normalized_vae_name_or_none(value)
    if vae_name is not None:
        return _load_cached_vae(vae_name, core_nodes_module)
    return value


def _runtime_settings_for_model(model: object) -> dict[str, int | float]:
    model_name = normalized_model_name_or_none(model)
    if model_name is None:
        return {}

    if is_checkpoint_model(model):
        folder_name = Const.MODEL_FOLDER_PATH_CHECKPOINTS
    elif is_diffusion_model(model):
        folder_name = Const.MODEL_FOLDER_PATH_DIFFUSION_MODELS
    else:
        return {}

    pipeline = get_shared_metadata_pipeline(start=True)
    return pipeline.get_model_runtime_settings_by_relative_path(
        folder_name=folder_name,
        relative_path=model_name,
    )


def _apply_clip_last_layer_if_needed(core_nodes_module: object, clip: object | None, stop_at_clip_layer: int | None) -> object | None:
    if clip is None or stop_at_clip_layer is None:
        return clip

    clip_set_last_layer_class = loader_class_or_none(core_nodes_module, CLIP_SET_LAST_LAYER_KEYS)
    if clip_set_last_layer_class is None:
        raise RuntimeError("CLIP Set Last Layer node is unavailable")
    return apply_clip_last_layer_with_core_node(
        clip_set_last_layer_class(),
        clip,
        stop_at_clip_layer,
    )


def _apply_model_sampling_sd3_if_needed(core_nodes_module: object, model: object, shift: float | None) -> object:
    if shift is None:
        return model

    model_sampling_sd3_class = loader_class_or_none(core_nodes_module, MODEL_SAMPLING_SD3_KEYS)
    if model_sampling_sd3_class is None:
        raise RuntimeError("ModelSamplingSD3 node is unavailable")
    return apply_model_sampling_sd3_with_core_node(
        model_sampling_sd3_class(),
        model,
        shift,
    )


def _load_diffusion_clip(core_nodes_module: object, clip: object) -> object | None:
    payload = normalized_clip_payload_or_none(clip)
    if payload is None:
        return None

    clip_names = normalized_clip_names(payload)
    if not clip_names:
        return None

    loader_keys = CLIP_LOADER_KEYS_BY_COUNT.get(len(clip_names))
    if loader_keys is None:
        raise RuntimeError(f"Unsupported CLIP count: {len(clip_names)}")

    clip_loader_class = loader_class_or_none(core_nodes_module, loader_keys)
    if clip_loader_class is None:
        raise RuntimeError("Load CLIP node is unavailable")

    runtime_clip = load_clip_with_core_loader(
        clip_loader_class(),
        clip_names,
        clip_type=normalized_clip_type_or_none(payload),
        device=normalized_clip_device_or_none(payload),
    )
    return _apply_clip_last_layer_if_needed(
        core_nodes_module,
        runtime_clip,
        normalized_clip_last_layer_or_none(payload),
    )


class LoadNewModel(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        socket_force_input = {"forceInput": True}
        return c_io.Schema(
            node_id="IPT-LoadNewModel",
            display_name="Load New Model",
            category=Const.CATEGORY_IMAGEINFO,
            not_idempotent=True,
            hidden=[
                c_io.Hidden.unique_id,
            ],
            inputs=[
                Const.MODEL_TYPE.Input(
                    "model",
                    tooltip="Load model runtime directly from IPT-Model",
                ),
                Const.CLIP_TYPE.Input(
                    "clip",
                    optional=True,
                    extra_dict=socket_force_input,
                    tooltip="Optional external CLIP reference used for diffusion models",
                ),
                c_io.AnyType.Input(
                    "vae",
                    optional=True,
                    extra_dict=socket_force_input,
                    tooltip="Optional VAE override. Non-empty input always takes priority",
                ),
                c_io.AnyType.Input(
                    "vae_fallback",
                    display_name="vae_fallback",
                    optional=True,
                    extra_dict=socket_force_input,
                    tooltip="Fallback VAE reference used when no VAE is available",
                ),
            ],
            outputs=[
                MODEL_RUNTIME_TYPE.Output(
                    Cast.out_id("model"),
                    display_name="model",
                ),
                CLIP_RUNTIME_TYPE.Output(
                    Cast.out_id("clip"),
                    display_name="clip",
                ),
                VAE_RUNTIME_TYPE.Output(
                    Cast.out_id("vae"),
                    display_name="vae",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        model: object | None = None,
        clip: object | None = None,
        vae: object | None = None,
        vae_fallback: object | None = None,
    ) -> bool | str:
        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        model: object | None = None,
        clip: object | None = None,
        vae: object | None = None,
        vae_fallback: object | None = None,
    ) -> int:
        return time.time_ns()

    @classmethod
    def execute(
        cls,
        model: object,
        clip: object | None = None,
        vae: object | None = None,
        vae_fallback: object | None = None,
    ) -> c_io.NodeOutput:
        model_name = normalized_model_name_or_none(model)
        if model_name is None:
            raise RuntimeError("model is required")

        core_nodes_module = core_nodes_module_or_none()
        if core_nodes_module is None:
            raise RuntimeError("ComfyUI core nodes module is unavailable")

        runtime_settings = _runtime_settings_for_model(model)
        runtime_model: object
        runtime_clip: object | None = None
        runtime_vae: object | None = None

        if is_checkpoint_model(model):
            checkpoint_loader_class = loader_class_or_none(core_nodes_module, CHECKPOINT_LOADER_KEYS)
            if checkpoint_loader_class is None:
                raise RuntimeError("Load Checkpoint node is unavailable")

            runtime_model, runtime_clip, runtime_vae = load_checkpoint_with_core_loader(
                checkpoint_loader_class(),
                model_name,
            )
            runtime_clip = _apply_clip_last_layer_if_needed(
                core_nodes_module,
                runtime_clip,
                clip_last_layer_from_settings(runtime_settings),
            )
            runtime_model = _apply_model_sampling_sd3_if_needed(
                core_nodes_module,
                runtime_model,
                sd3_shift_from_settings(runtime_settings),
            )
        elif is_diffusion_model(model):
            diffusion_loader_class = loader_class_or_none(core_nodes_module, DIFFUSION_MODEL_LOADER_KEYS)
            if diffusion_loader_class is None:
                raise RuntimeError("Load Diffusion Model node is unavailable")

            runtime_model = load_diffusion_model_with_core_loader(
                diffusion_loader_class(),
                model_name,
                weight_dtype=normalized_model_weight_dtype_or_none(model),
            )
            runtime_model = _apply_model_sampling_sd3_if_needed(
                core_nodes_module,
                runtime_model,
                sd3_shift_from_settings(runtime_settings),
            )
            if clip is not None:
                runtime_clip = _load_diffusion_clip(core_nodes_module, clip)
        else:
            raise RuntimeError("Only checkpoint and diffusion_models are supported")

        explicit_vae = _resolve_vae_runtime(vae, core_nodes_module)
        if explicit_vae is not None:
            runtime_vae = explicit_vae
        elif runtime_vae is None:
            runtime_vae = _resolve_vae_runtime(vae_fallback, core_nodes_module)

        return c_io.NodeOutput(runtime_model, runtime_clip, runtime_vae)
