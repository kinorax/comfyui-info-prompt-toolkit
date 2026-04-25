# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast


def _merged_extras(extras: dict[str, object] | None, key: str, value: str) -> dict[str, object]:
    output = dict(extras) if isinstance(extras, dict) else {}
    output[key] = value
    return output


def _normalized_key_or_none(key: object) -> str | None:
    if key is None:
        return None
    normalized = str(key).strip()
    return normalized or None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


class SetFloatExtra(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-SetFloatExtra",
            display_name="Set Float Extra",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.String.Input(
                    "key",
                    tooltip="Parameter line key",
                ),
                c_io.Float.Input(
                    "value",
                    default=0.0,
                    step=0.01,
                    tooltip="Parameter line value",
                ),
                c_io.Int.Input(
                    "decimals",
                    default=2,
                    min=0,
                    max=10,
                    step=1,
                    tooltip="Number of digits after decimal point",
                ),
                Const.IMAGEINFO_EXTRAS_TYPE.Input(
                    Const.IMAGEINFO_EXTRAS,
                    optional=True,
                ),
            ],
            outputs=[
                Const.IMAGEINFO_EXTRAS_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO_EXTRAS),
                    display_name=Const.IMAGEINFO_EXTRAS,
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        key: object,
        value: object,
        decimals: object,
        extras: dict[str, object] | None = None,
    ) -> bool | str:
        normalized_key = _normalized_key_or_none(key)
        if key is not None and normalized_key is None:
            return "key is required"

        decimals_int = _int_or_none(decimals)
        if decimals is not None and decimals_int is None:
            return "decimals must be an integer"
        if decimals_int is not None and decimals_int < 0:
            return "decimals must be 0 or greater"
        return True

    @classmethod
    def execute(
        cls,
        key: object,
        value: float,
        decimals: int,
        extras: dict[str, object] | None = None,
    ) -> c_io.NodeOutput:
        normalized_key = _normalized_key_or_none(key)
        if normalized_key is None:
            raise ValueError("key is required")

        precision = max(0, min(10, int(decimals)))
        rendered = f"{float(value):.{precision}f}"
        output = _merged_extras(extras, normalized_key, rendered)
        return c_io.NodeOutput(output)
