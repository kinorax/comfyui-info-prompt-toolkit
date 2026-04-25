# Copyright 2026 kinorax
from comfy_api.latest import io as c_io

from .. import const as Const
from .clip_selector_common import resolve_clip_selector_values


class DualClipSelector(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        clip_name_options = Const.get_CLIP_NAME_OPTIONS()
        clip_type_options = Const.get_DUAL_CLIP_TYPE_OPTIONS()
        clip_device_options = Const.get_DUAL_CLIP_DEVICE_OPTIONS()
        default_clip_name = clip_name_options[0] if clip_name_options else ""
        return c_io.Schema(
            node_id="IPT-DualClipSelector",
            display_name="Dual CLIP Selector",
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
                    "type",
                    display_name="type",
                    options=clip_type_options,
                    default=clip_type_options[0],
                    tooltip="CLIP type passed by DualCLIPLoader",
                ),
                c_io.Combo.Input(
                    "device",
                    display_name="device",
                    options=clip_device_options,
                    default=clip_device_options[0],
                    tooltip="CLIP device passed by DualCLIPLoader",
                    advanced=True,
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
            ],
        )

    @classmethod
    def execute(
        cls,
        clip_name1: str,
        clip_name2: str,
        type: str,
        device: str,
        sha256_1: str | None = None,
        sha256_2: str | None = None,
    ) -> c_io.NodeOutput:
        clip_type_options = Const.get_DUAL_CLIP_TYPE_OPTIONS()
        clip_device_options = Const.get_DUAL_CLIP_DEVICE_OPTIONS()
        values = resolve_clip_selector_values(
            [clip_name1, clip_name2],
            sha256_values=[sha256_1, sha256_2],
        )
        normalized_type = type if type in clip_type_options else clip_type_options[0]
        normalized_device = device if device in clip_device_options else clip_device_options[0]
        clip_names = [value for value in values if value is not None]
        clip = Const.make_clip_value(
            clip_names if len(clip_names) == len(values) else None,
            {
                Const.MODEL_VALUE_TYPE_KEY: normalized_type,
                Const.MODEL_VALUE_DEVICE_KEY: normalized_device,
            },
        )
        value1 = values[0] if len(values) >= 1 else None
        value2 = values[1] if len(values) >= 2 else None
        return c_io.NodeOutput(clip, value1, value2)
