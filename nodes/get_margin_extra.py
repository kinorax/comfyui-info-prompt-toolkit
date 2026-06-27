# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.margin import (
    deserialize_margin_extra,
    normalized_extra_key_or_none,
    split_margin_values,
)


class GetMarginExtra(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-GetMarginExtra",
            display_name="Get Margin Extra",
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
                Const.MARGIN_TYPE.Output(
                    Cast.out_id("margin"),
                    display_name="margin",
                ),
                c_io.Int.Output(Cast.out_id("top"), display_name="top"),
                c_io.Int.Output(Cast.out_id("right"), display_name="right"),
                c_io.Int.Output(Cast.out_id("bottom"), display_name="bottom"),
                c_io.Int.Output(Cast.out_id("left"), display_name="left"),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        key: object,
        extras: dict[str, object] | None = None,
    ) -> bool | str:
        normalized_key = normalized_extra_key_or_none(key)
        if key is not None and normalized_key is None:
            return "key is required"
        return True

    @classmethod
    def execute(
        cls,
        key: object,
        extras: dict[str, object] | None = None,
    ) -> c_io.NodeOutput:
        normalized_key = normalized_extra_key_or_none(key)
        if normalized_key is None:
            raise RuntimeError("Get Margin Extra: key is required")

        raw_value = extras.get(normalized_key) if isinstance(extras, dict) else None
        payload = deserialize_margin_extra(raw_value, key=normalized_key)
        return c_io.NodeOutput(payload, *split_margin_values(payload))
