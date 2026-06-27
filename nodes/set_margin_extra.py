# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.margin import (
    margin_payload_or_error,
    merge_extra_value,
    normalized_extra_key_or_none,
    serialize_margin_extra_json,
)


class SetMarginExtra(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-SetMarginExtra",
            display_name="Set Margin Extra",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.IMAGEINFO_EXTRAS_TYPE.Input(
                    Const.IMAGEINFO_EXTRAS,
                    optional=True,
                ),
                c_io.String.Input(
                    "key",
                    tooltip="Parameter line key",
                ),
                Const.MARGIN_TYPE.Input(
                    "margin",
                    display_name="margin",
                    optional=True,
                    extra_dict={"forceInput": True},
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
        margin: object | None = None,
        extras: dict[str, object] | None = None,
    ) -> bool | str:
        normalized_key = normalized_extra_key_or_none(key)
        if key is not None and normalized_key is None:
            return "key is required"

        # Linked custom values are unresolved during Comfy validation.
        if margin is None:
            return True
        _payload, error = margin_payload_or_error(margin)
        return error or True

    @classmethod
    def execute(
        cls,
        key: object,
        margin: object | None = None,
        extras: dict[str, object] | None = None,
    ) -> c_io.NodeOutput:
        normalized_key = normalized_extra_key_or_none(key)
        if normalized_key is None:
            raise RuntimeError("Set Margin Extra: key is required")

        serialized = serialize_margin_extra_json(margin)
        output = merge_extra_value(extras, normalized_key, serialized)
        return c_io.NodeOutput(output)
