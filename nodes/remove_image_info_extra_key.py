# Copyright 2026 kinorax
from __future__ import annotations

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast


def _normalized_keys(value: object | None) -> tuple[str, ...]:
    if value is None:
        return tuple()

    output: list[str] = []
    seen: set[str] = set()
    for raw_item in str(value).split(","):
        normalized = raw_item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return tuple(output)


def _copied_image_info(image_info: dict[str, object] | None) -> dict[str, object]:
    output = dict(image_info) if isinstance(image_info, dict) else {}
    extras_raw = output.get(Const.IMAGEINFO_EXTRAS)
    if isinstance(extras_raw, dict):
        output[Const.IMAGEINFO_EXTRAS] = dict(extras_raw)
    return output


def _removed_extra_keys(
    image_info: dict[str, object] | None,
    keys: tuple[str, ...],
) -> dict[str, object]:
    output = _copied_image_info(image_info)
    extras = output.get(Const.IMAGEINFO_EXTRAS)
    if not isinstance(extras, dict) or not keys:
        return output

    for key in keys:
        extras.pop(key, None)
    if extras:
        output[Const.IMAGEINFO_EXTRAS] = extras
    else:
        output.pop(Const.IMAGEINFO_EXTRAS, None)
    return output


class RemoveImageInfoExtraKeys(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-RemoveImageInfoExtraKeys",
            display_name="Remove Image Info Extra Keys",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                Const.IMAGEINFO_TYPE.Input(
                    Const.IMAGEINFO,
                    optional=True,
                ),
                c_io.String.Input(
                    "key",
                    display_name="remove",
                    default="",
                    optional=True,
                    tooltip="Comma-separated keys in image_info.extras to remove. Matches exact keys after trimming each item.",
                ),
            ],
            outputs=[
                Const.IMAGEINFO_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO),
                    display_name=Const.IMAGEINFO,
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        image_info: dict[str, object] | None = None,
        key: object = None,
    ) -> bool | str:
        return True

    @classmethod
    def execute(
        cls,
        image_info: dict[str, object] | None = None,
        key: object = None,
    ) -> c_io.NodeOutput:
        normalized_keys = _normalized_keys(key)
        output = _removed_extra_keys(image_info, normalized_keys)
        return c_io.NodeOutput(output)
