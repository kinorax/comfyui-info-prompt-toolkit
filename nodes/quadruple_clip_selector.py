# Copyright 2026 kinorax
from comfy_api.latest import io as c_io

from .. import const as Const
from .clip_selector_common import resolve_clip_selector_values


class QuadrupleClipSelector(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        clip_name_options = Const.get_CLIP_NAME_OPTIONS()
        default_clip_name = clip_name_options[0] if clip_name_options else ""
        return c_io.Schema(
            node_id="IPT-QuadrupleClipSelector",
            display_name="Quadruple CLIP Selector",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.Combo.Input(
                    "clip_name1",
                    display_name="clip_name1",
                    options=clip_name_options,
                    default=default_clip_name,
                    tooltip="Select first CLIP",
                ),
                c_io.Combo.Input(
                    "clip_name2",
                    display_name="clip_name2",
                    options=clip_name_options,
                    default=default_clip_name,
                    tooltip="Select second CLIP",
                ),
                c_io.Combo.Input(
                    "clip_name3",
                    display_name="clip_name3",
                    options=clip_name_options,
                    default=default_clip_name,
                    tooltip="Select third CLIP",
                ),
                c_io.Combo.Input(
                    "clip_name4",
                    display_name="clip_name4",
                    options=clip_name_options,
                    default=default_clip_name,
                    tooltip="Select fourth CLIP",
                ),
                c_io.String.Input(
                    "sha256_1",
                    default="",
                    socketless=True,
                    optional=True,
                    tooltip="Cached SHA256 for clip_name1 used by View Model Info fallback",
                ),
                c_io.String.Input(
                    "sha256_2",
                    default="",
                    socketless=True,
                    optional=True,
                    tooltip="Cached SHA256 for clip_name2 used by View Model Info fallback",
                ),
                c_io.String.Input(
                    "sha256_3",
                    default="",
                    socketless=True,
                    optional=True,
                    tooltip="Cached SHA256 for clip_name3 used by View Model Info fallback",
                ),
                c_io.String.Input(
                    "sha256_4",
                    default="",
                    socketless=True,
                    optional=True,
                    tooltip="Cached SHA256 for clip_name4 used by View Model Info fallback",
                ),
            ],
            outputs=[
                Const.CLIP_TYPE.Output(
                    "CLIP",
                    display_name="clip",
                ),
                c_io.AnyType.Output(
                    "CLIP_NAME1_COMBO",
                    display_name="clip_name1",
                ),
                c_io.AnyType.Output(
                    "CLIP_NAME2_COMBO",
                    display_name="clip_name2",
                ),
                c_io.AnyType.Output(
                    "CLIP_NAME3_COMBO",
                    display_name="clip_name3",
                ),
                c_io.AnyType.Output(
                    "CLIP_NAME4_COMBO",
                    display_name="clip_name4",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip_name1: str,
        clip_name2: str,
        clip_name3: str,
        clip_name4: str,
        sha256_1: str | None = None,
        sha256_2: str | None = None,
        sha256_3: str | None = None,
        sha256_4: str | None = None,
    ) -> c_io.NodeOutput:
        values = resolve_clip_selector_values(
            [clip_name1, clip_name2, clip_name3, clip_name4],
            sha256_values=[sha256_1, sha256_2, sha256_3, sha256_4],
        )
        clip_names = [value for value in values if value is not None]
        clip = Const.make_clip_value(clip_names if len(clip_names) == len(values) else None)
        value1 = values[0] if len(values) >= 1 else None
        value2 = values[1] if len(values) >= 2 else None
        value3 = values[2] if len(values) >= 3 else None
        value4 = values[3] if len(values) >= 4 else None
        return c_io.NodeOutput(clip, value1, value2, value3, value4)
