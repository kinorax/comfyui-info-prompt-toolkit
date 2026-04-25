# Copyright 2026 kinorax
from comfy_api.latest import io as c_io
from .. import const as Const
from ..utils.selector_resolution import resolve_selector_value


class ClipSelector(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        clip_name_options = Const.get_CLIP_NAME_OPTIONS()
        clip_type_options = Const.get_CLIP_TYPE_OPTIONS()
        clip_device_options = Const.get_CLIP_DEVICE_OPTIONS()
        default_clip_name = clip_name_options[0] if clip_name_options else ""
        return c_io.Schema(
            node_id="IPT-ClipSelector",
            display_name="CLIP Selector",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.Combo.Input(
                    "clip_name",
                    display_name="clip_name",
                    options=clip_name_options,
                    default=default_clip_name,
                    tooltip="Select CLIP",
                ),
                c_io.Combo.Input(
                    "type",
                    display_name="type",
                    options=clip_type_options,
                    default=clip_type_options[0],
                    tooltip="CLIP type passed by Load CLIP",
                ),
                c_io.Combo.Input(
                    "device",
                    display_name="device",
                    options=clip_device_options,
                    default=clip_device_options[0],
                    tooltip="CLIP device passed by Load CLIP",
                    advanced=True,
                ),
                c_io.String.Input(
                    "sha256",
                    default="",
                    socketless=True,
                    optional=True,
                    tooltip="Cached SHA256 used by View Model Info fallback",
                ),
            ],
            outputs=[
                Const.CLIP_TYPE.Output(
                    "CLIP",
                    display_name="clip",
                ),
                c_io.AnyType.Output(
                    "CLIP_NAME_COMBO",
                    display_name="clip_name",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip_name: str,
        type: str,
        device: str,
        sha256: str | None = None,
    ) -> c_io.NodeOutput:
        clip_name_options = Const.get_CLIP_NAME_OPTIONS()
        clip_type_options = Const.get_CLIP_TYPE_OPTIONS()
        clip_device_options = Const.get_CLIP_DEVICE_OPTIONS()
        value = resolve_selector_value(
            clip_name,
            clip_name_options,
            value_label="clip_name",
            folder_name=Const.MODEL_FOLDER_PATH_TEXT_ENCODERS,
            sha256=sha256,
        )
        normalized_type = type if type in clip_type_options else clip_type_options[0]
        normalized_device = device if device in clip_device_options else clip_device_options[0]
        clip = Const.make_clip_value(
            [value] if value is not None else None,
            {
                Const.MODEL_VALUE_TYPE_KEY: normalized_type,
                Const.MODEL_VALUE_DEVICE_KEY: normalized_device,
            },
        )
        return c_io.NodeOutput(clip, value)
