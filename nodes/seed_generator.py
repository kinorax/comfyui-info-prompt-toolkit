# Copyright 2026 kinorax
from __future__ import annotations

import random
import time

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast

MODE_SAMPLER_SAFE_32 = "Sampler Safe (0-4294967293)"
MODE_KSAMPLER_STANDARD = "KSampler Standard"
MODE_FIXED = "Fixed"

MODE_OPTIONS: tuple[str, ...] = (
    MODE_KSAMPLER_STANDARD,
    MODE_SAMPLER_SAFE_32,
    MODE_FIXED,
)

MODE_RANGES: dict[str, tuple[int, int]] = {
    MODE_KSAMPLER_STANDARD: (0, 0xFFFFFFFFFFFFFFFF),
    MODE_SAMPLER_SAFE_32: (0, 4294967293),
}

_SEED_MIN = 0
_SEED_MAX = 0xFFFFFFFFFFFFFFFF

_RNG = random.SystemRandom()


def _normalized_mode_or_none(mode: object) -> str | None:
    if mode is None:
        return None
    text = str(mode)
    if text in MODE_OPTIONS:
        return text
    return None


def _seed_int_or_none(value: object) -> int | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = int(text)
    except Exception:
        return None

    if parsed < _SEED_MIN or parsed > _SEED_MAX:
        return None

    return parsed


class SeedGenerator(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-SeedGenerator",
            display_name="Seed Generator",
            category=Const.CATEGORY_IMAGEINFO,
            not_idempotent=True,
            inputs=[
                c_io.Combo.Input(
                    "generation_mode",
                    options=MODE_OPTIONS,
                    default=MODE_OPTIONS[0],
                    tooltip="Select seed generation mode",
                ),
                c_io.String.Input(
                    "generated_seed",
                    default="",
                    socketless=True,
                    tooltip="Generated seed (copy-only display)",
                ),
            ],
            outputs=[
                c_io.Int.Output(
                    Cast.out_id(Const.IMAGEINFO_SEED),
                    display_name=Const.IMAGEINFO_SEED,
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        generation_mode: object,
        generated_seed: object | None = None,
    ) -> bool | str:
        normalized_mode = _normalized_mode_or_none(generation_mode)
        if normalized_mode is None:
            return "generation_mode is invalid"
        if normalized_mode == MODE_FIXED and _seed_int_or_none(generated_seed) is None:
            return "generated_seed must be an integer from 0 to 18446744073709551615 when generation_mode is Fixed"
        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        generation_mode: object,
        generated_seed: object | None = None,
    ) -> int:
        # Always regenerate so this node behaves as a pure seed source.
        return time.time_ns()

    @classmethod
    def execute(cls, generation_mode: str, generated_seed: object | None = None) -> c_io.NodeOutput:
        normalized_mode = _normalized_mode_or_none(generation_mode)
        if normalized_mode is None:
            raise ValueError("generation_mode is invalid")

        if normalized_mode == MODE_FIXED:
            seed_value = _seed_int_or_none(generated_seed)
            if seed_value is None:
                raise ValueError(
                    "generated_seed must be an integer from 0 to 18446744073709551615 when generation_mode is Fixed"
                )
        else:
            minimum, maximum = MODE_RANGES[normalized_mode]
            seed_value = _RNG.randint(minimum, maximum)

        return c_io.NodeOutput(seed_value, ui={"generated_seed": [str(seed_value)]})
