# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast


def _extras_or_none(value: object | None) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return dict(value)


def _combined_extras_or_none(
    extras1: object | None,
    extras2: object | None,
) -> dict[str, object] | None:
    left = _extras_or_none(extras1)
    right = _extras_or_none(extras2)

    if left is None and right is None:
        return None
    if left is None:
        return right
    if right is None:
        return left

    merged = dict(left)
    merged.update(right)
    return merged


class CombineExtras(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-CombineExtras",
            display_name="Combine Extras",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.IMAGEINFO_EXTRAS_TYPE.Input(
                    "extras1",
                    tooltip="Base extras",
                    optional=True,
                ),
                Const.IMAGEINFO_EXTRAS_TYPE.Input(
                    "extras2",
                    tooltip="Merged extras. Keys here overwrite matching keys from extras1.",
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
    def execute(
        cls,
        extras1: object | None = None,
        extras2: object | None = None,
    ) -> c_io.NodeOutput:
        return c_io.NodeOutput(_combined_extras_or_none(extras1, extras2))
