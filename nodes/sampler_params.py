# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.sampler_params import (
    CFG_MAX,
    DENOISE_KEY,
    DENOISE_MAX,
    DENOISE_MIN,
    SAMPLER_PARAMS_KEY,
    SEED_MAX,
    STEPS_MAX,
    sampler_params_payload_or_error,
)

_DEFAULT_STEPS = 20
_DEFAULT_DENOISE = 1.0
_DEFAULT_SEED = 0
_DEFAULT_CFG = 7.0


class SamplerParams(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        sampler_default = Const.SAMPLER_OPTIONS[0] if Const.SAMPLER_OPTIONS else ""
        scheduler_default = Const.SCHEDULER_OPTIONS[0] if Const.SCHEDULER_OPTIONS else ""

        return c_io.Schema(
            node_id="IPT-SamplerParams",
            display_name="Sampler Params",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.Combo.Input(
                    Const.IMAGEINFO_SAMPLER,
                    options=Const.SAMPLER_OPTIONS,
                    default=sampler_default,
                    tooltip="Select sampler",
                ),
                c_io.Combo.Input(
                    Const.IMAGEINFO_SCHEDULER,
                    options=Const.SCHEDULER_OPTIONS,
                    default=scheduler_default,
                    tooltip="Select scheduler",
                ),
                c_io.Int.Input(
                    Const.IMAGEINFO_STEPS,
                    default=_DEFAULT_STEPS,
                    min=1,
                    max=STEPS_MAX,
                    tooltip="Sampling steps",
                ),
                c_io.Float.Input(
                    DENOISE_KEY,
                    default=_DEFAULT_DENOISE,
                    min=DENOISE_MIN,
                    max=DENOISE_MAX,
                    step=0.01,
                    tooltip="Denoise strength",
                ),
                c_io.Int.Input(
                    Const.IMAGEINFO_SEED,
                    default=_DEFAULT_SEED,
                    min=0,
                    max=SEED_MAX,
                    control_after_generate=False,
                    tooltip="Sampling seed",
                ),
                c_io.Float.Input(
                    Const.IMAGEINFO_CFG,
                    default=_DEFAULT_CFG,
                    min=0.0,
                    max=CFG_MAX,
                    step=0.1,
                    tooltip="CFG scale",
                ),
            ],
            outputs=[
                Const.SAMPLER_PARAMS_TYPE.Output(
                    Cast.out_id(SAMPLER_PARAMS_KEY),
                    display_name=SAMPLER_PARAMS_KEY,
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        sampler: object = None,
        scheduler: object = None,
        steps: object = None,
        denoise: object = None,
        seed: object = None,
        cfg: object = None,
    ) -> bool | str:
        # During Comfy validation, linked values are unresolved and arrive as None.
        # Required-input checks and type checks for links are handled by the framework.
        if None in (sampler, scheduler, steps, denoise, seed, cfg):
            return True

        _, error = sampler_params_payload_or_error(
            {
                Const.IMAGEINFO_SAMPLER: sampler,
                Const.IMAGEINFO_SCHEDULER: scheduler,
                Const.IMAGEINFO_STEPS: steps,
                DENOISE_KEY: denoise,
                Const.IMAGEINFO_SEED: seed,
                Const.IMAGEINFO_CFG: cfg,
            }
        )
        return True if error is None else error

    @classmethod
    def execute(
        cls,
        sampler: object,
        scheduler: object,
        steps: object,
        denoise: object,
        seed: object,
        cfg: object,
    ) -> c_io.NodeOutput:
        payload, error = sampler_params_payload_or_error(
            {
                Const.IMAGEINFO_SAMPLER: sampler,
                Const.IMAGEINFO_SCHEDULER: scheduler,
                Const.IMAGEINFO_STEPS: steps,
                DENOISE_KEY: denoise,
                Const.IMAGEINFO_SEED: seed,
                Const.IMAGEINFO_CFG: cfg,
            }
        )
        if payload is None or error is not None:
            raise RuntimeError(f"Sampler Params: {error or 'sampler_params is required'}")
        return c_io.NodeOutput(payload)
