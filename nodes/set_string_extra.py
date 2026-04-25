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


class SetStringExtra(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-SetStringExtra",
            display_name="Set String Extra",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.String.Input(
                    "key",
                    tooltip="Parameter line key",
                ),
                c_io.String.Input(
                    "value",
                    tooltip="Parameter line value",
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
        value: object,
        extras: dict[str, object] | None = None,
    ) -> c_io.NodeOutput:
        normalized_key = _normalized_key_or_none(key)
        if normalized_key is None:
            raise ValueError("key is required")

        value_str = str(value)
        output = _merged_extras(extras, normalized_key, value_str)
        return c_io.NodeOutput(output)
