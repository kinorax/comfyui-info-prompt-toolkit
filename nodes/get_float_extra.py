# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast


def _normalized_key_or_none(key: object) -> str | None:
    if key is None:
        return None
    normalized = str(key).strip()
    return normalized or None


def _resolve(extras: dict[str, object] | None, key: str) -> float | None:
    if not isinstance(extras, dict):
        return None
    if key not in extras:
        return None

    value = extras.get(key)
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


class GetFloatExtra(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-GetFloatExtra",
            display_name="Get Float Extra",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.String.Input(
                    "key",
                    tooltip="Parameter line key",
                ),
                Const.IMAGEINFO_EXTRAS_TYPE.Input(
                    Const.IMAGEINFO_EXTRAS,
                    optional=True,
                ),
            ],
            outputs=[
                c_io.Float.Output(
                    Cast.out_id("float"),
                    display_name="float",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        key: object,
        extras: dict[str, object] | None = None,
    ) -> bool | str:
        normalized_key = _normalized_key_or_none(key)
        if key is not None and normalized_key is None:
            return "key is required"
        return True

    @classmethod
    def execute(
        cls,
        key: object,
        extras: dict[str, object] | None = None,
    ) -> c_io.NodeOutput:
        normalized_key = _normalized_key_or_none(key)
        if normalized_key is None:
            raise ValueError("key is required")
        value_out = _resolve(extras, normalized_key)
        return c_io.NodeOutput(value_out)
