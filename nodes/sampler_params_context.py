# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.sampler_params import (
    DENOISE_KEY,
    SAMPLER_PARAM_KEYS,
    SAMPLER_PARAMS_KEY,
    normalized_sampler_param_value_or_error,
    sampler_params_payload_or_error,
    split_sampler_params_values,
)


def _connected_input_keys(cls: type) -> set[str]:
    hidden = getattr(cls, "hidden", None)
    prompt = getattr(hidden, "prompt", None)
    unique_id = getattr(hidden, "unique_id", None)
    if isinstance(prompt, dict) and unique_id is not None:
        node = prompt.get(str(unique_id), {}) or {}
        inputs = node.get("inputs", {}) or {}
        return set(inputs.keys())
    return set()


def _include_input_key(connected: set[str], kwargs: dict[str, object], key: str) -> bool:
    if key in connected:
        return True
    if len(connected) > 0:
        return False
    return kwargs.get(key) is not None


class SamplerParamsContext(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        socket_only = dict(optional=True, force_input=True)
        sampler_params_socket = {"forceInput": True}

        return c_io.Schema(
            node_id="IPT-SamplerParamsContext",
            display_name="Sampler Params Context",
            category=Const.CATEGORY_IMAGEINFO,
            hidden=[c_io.Hidden.prompt, c_io.Hidden.unique_id],
            inputs=[
                Const.SAMPLER_PARAMS_TYPE.Input(
                    SAMPLER_PARAMS_KEY,
                    display_name=SAMPLER_PARAMS_KEY,
                    optional=True,
                    extra_dict=sampler_params_socket,
                ),
                c_io.String.Input(Const.IMAGEINFO_SAMPLER, **socket_only),
                c_io.String.Input(Const.IMAGEINFO_SCHEDULER, **socket_only),
                c_io.Int.Input(Const.IMAGEINFO_STEPS, **socket_only),
                c_io.Float.Input(DENOISE_KEY, **socket_only),
                c_io.Int.Input(Const.IMAGEINFO_SEED, **socket_only),
                c_io.Float.Input(Const.IMAGEINFO_CFG, **socket_only),
            ],
            outputs=[
                Const.SAMPLER_PARAMS_TYPE.Output(
                    Cast.out_id(SAMPLER_PARAMS_KEY),
                    display_name=SAMPLER_PARAMS_KEY,
                ),
                c_io.AnyType.Output(
                    Cast.out_id(Const.IMAGEINFO_SAMPLER),
                    display_name=Const.IMAGEINFO_SAMPLER,
                ),
                c_io.AnyType.Output(
                    Cast.out_id(Const.IMAGEINFO_SCHEDULER),
                    display_name=Const.IMAGEINFO_SCHEDULER,
                ),
                c_io.Int.Output(
                    Cast.out_id(Const.IMAGEINFO_STEPS),
                    display_name=Const.IMAGEINFO_STEPS,
                ),
                c_io.Float.Output(
                    Cast.out_id(DENOISE_KEY),
                    display_name=DENOISE_KEY,
                ),
                c_io.Int.Output(
                    Cast.out_id(Const.IMAGEINFO_SEED),
                    display_name=Const.IMAGEINFO_SEED,
                ),
                c_io.Float.Output(
                    Cast.out_id(Const.IMAGEINFO_CFG),
                    display_name=Const.IMAGEINFO_CFG,
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        sampler_params: object | None = None,
        sampler: object | None = None,
        scheduler: object | None = None,
        steps: object | None = None,
        denoise: object | None = None,
        seed: object | None = None,
        cfg: object | None = None,
    ) -> bool | str:
        if sampler_params is not None:
            _, error = sampler_params_payload_or_error(sampler_params, require_all=False)
            if error is not None:
                return error

        for key, value in (
            (Const.IMAGEINFO_SAMPLER, sampler),
            (Const.IMAGEINFO_SCHEDULER, scheduler),
            (Const.IMAGEINFO_STEPS, steps),
            (DENOISE_KEY, denoise),
            (Const.IMAGEINFO_SEED, seed),
            (Const.IMAGEINFO_CFG, cfg),
        ):
            if value is None:
                continue
            _, error = normalized_sampler_param_value_or_error(key, value)
            if error is not None:
                return error
        return True

    @classmethod
    def execute(cls, **kwargs) -> c_io.NodeOutput:
        connected = _connected_input_keys(cls)
        output: dict[str, object] = {}

        if _include_input_key(connected, kwargs, SAMPLER_PARAMS_KEY):
            base_sampler_params = kwargs.get(SAMPLER_PARAMS_KEY)
            if base_sampler_params is not None:
                payload, error = sampler_params_payload_or_error(base_sampler_params, require_all=False)
                if error is not None:
                    raise RuntimeError(f"Sampler Params Context: {error}")
                if payload is not None:
                    output.update(payload)

        for key in SAMPLER_PARAM_KEYS:
            if not _include_input_key(connected, kwargs, key):
                continue

            value = kwargs.get(key)
            if value is None:
                output.pop(key, None)
                continue

            normalized, error = normalized_sampler_param_value_or_error(key, value)
            if error is not None:
                raise RuntimeError(f"Sampler Params Context: {error}")
            output[key] = normalized

        payload_out = output or None
        return c_io.NodeOutput(payload_out, *split_sampler_params_values(payload_out))
