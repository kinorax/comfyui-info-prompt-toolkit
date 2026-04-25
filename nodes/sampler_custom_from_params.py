# Copyright 2026 kinorax
from __future__ import annotations

import importlib

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.sampler_params import SAMPLER_PARAMS_KEY, sampler_params_payload_or_error
from ._runtime_loader import core_nodes_module_or_none, loader_class_or_none, loader_method_or_none

MODEL_RUNTIME_TYPE = c_io.Custom("MODEL")
CONDITIONING_RUNTIME_TYPE = c_io.Custom("CONDITIONING")
LATENT_RUNTIME_TYPE = c_io.Custom("LATENT")

K_SAMPLER_SELECT_KEYS: tuple[str, ...] = ("KSamplerSelect",)
BASIC_SCHEDULER_KEYS: tuple[str, ...] = ("BasicScheduler",)
SAMPLER_CUSTOM_KEYS: tuple[str, ...] = ("SamplerCustom",)


def _custom_sampler_module_or_none() -> object | None:
    try:
        return importlib.import_module("comfy_extras.nodes_custom_sampler")
    except Exception:
        return None


def _core_node_class_or_none(keys: tuple[str, ...]) -> type | None:
    core_nodes_module = core_nodes_module_or_none()
    if core_nodes_module is not None:
        node_class = loader_class_or_none(core_nodes_module, keys)
        if node_class is not None:
            return node_class

    custom_sampler_module = _custom_sampler_module_or_none()
    if custom_sampler_module is not None:
        return loader_class_or_none(custom_sampler_module, keys)

    return None


def _first_result_value(result: object) -> object:
    result_values = getattr(result, "result", None)
    if isinstance(result_values, tuple):
        return result_values[0] if result_values else None
    if isinstance(result, (list, tuple)):
        return result[0] if result else None
    return result


def _call_core_node(
    node_class: type,
    *,
    error_prefix: str,
    fallback_names: tuple[str, ...],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    method = loader_method_or_none(node_class(), fallback_names)
    if method is None:
        raise RuntimeError(f"{error_prefix} function is unavailable")

    try:
        return method(*args)
    except TypeError:
        return method(**kwargs)


def _sampler_from_name(sampler_name: str) -> object:
    node_class = _core_node_class_or_none(K_SAMPLER_SELECT_KEYS)
    if node_class is None:
        raise RuntimeError("KSamplerSelect node is unavailable")

    result = _call_core_node(
        node_class,
        error_prefix="KSamplerSelect",
        fallback_names=("get_sampler",),
        args=(sampler_name,),
        kwargs={"sampler_name": sampler_name},
    )
    return _first_result_value(result)


def _sigmas_from_scheduler(
    model: object,
    scheduler: str,
    steps: int,
    denoise: float,
) -> object:
    node_class = _core_node_class_or_none(BASIC_SCHEDULER_KEYS)
    if node_class is None:
        raise RuntimeError("BasicScheduler node is unavailable")

    result = _call_core_node(
        node_class,
        error_prefix="BasicScheduler",
        fallback_names=("get_sigmas",),
        args=(model, scheduler, steps, denoise),
        kwargs={
            "model": model,
            "scheduler": scheduler,
            "steps": steps,
            "denoise": denoise,
        },
    )
    return _first_result_value(result)


def _sample_with_sampler_custom(
    *,
    model: object,
    noise_seed: int,
    cfg: float,
    positive: object,
    negative: object,
    sampler: object,
    sigmas: object,
    latent_image: object,
) -> tuple[object, object]:
    node_class = _core_node_class_or_none(SAMPLER_CUSTOM_KEYS)
    if node_class is None:
        raise RuntimeError("SamplerCustom node is unavailable")

    result = _call_core_node(
        node_class,
        error_prefix="SamplerCustom",
        fallback_names=("sample",),
        args=(model, True, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image),
        kwargs={
            "model": model,
            "add_noise": True,
            "noise_seed": noise_seed,
            "cfg": cfg,
            "positive": positive,
            "negative": negative,
            "sampler": sampler,
            "sigmas": sigmas,
            "latent_image": latent_image,
        },
    )
    result_values = getattr(result, "result", result)
    if isinstance(result_values, tuple):
        result = result_values
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        raise RuntimeError("SamplerCustom returned unexpected outputs")
    return result[0], result[1]


class SamplerCustomFromParams(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        socket_force_input = {"forceInput": True}
        return c_io.Schema(
            node_id="IPT-SamplerCustomFromParams",
            display_name="SamplerCustom (Sampler Params)",
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
    ) -> bool | str:
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
    ) -> c_io.NodeOutput:
        payload, error = sampler_params_payload_or_error(sampler_params)
        if payload is None or error is not None:
            raise RuntimeError(f"SamplerCustom (Sampler Params): {error or 'sampler_params is required'}")

        sampler = _sampler_from_name(str(payload[Const.IMAGEINFO_SAMPLER]))
        sigmas = _sigmas_from_scheduler(
            model,
            str(payload[Const.IMAGEINFO_SCHEDULER]),
            int(payload[Const.IMAGEINFO_STEPS]),
            float(payload["denoise"]),
        )
        output, denoised_output = _sample_with_sampler_custom(
            model=model,
            noise_seed=int(payload[Const.IMAGEINFO_SEED]),
            cfg=float(payload[Const.IMAGEINFO_CFG]),
            positive=positive,
            negative=negative,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent_image,
        )
        return c_io.NodeOutput(output, denoised_output)
