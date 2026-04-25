# Copyright 2026 kinorax
from comfy_api.latest import io as c_io
from .. import const as Const
from ..utils.selector_resolution import resolve_selector_value


class CheckpointSelector(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        checkpoint_options = Const.get_CHECKPOINT_OPTIONS()
        default_checkpoint = checkpoint_options[0] if checkpoint_options else ""
        return c_io.Schema(
            node_id="IPT-CheckpointSelector",
            display_name="Checkpoint Selector",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.Combo.Input(
                    "checkpoint",
                    display_name="ckpt_name",
                    options=checkpoint_options,
                    default=default_checkpoint,
                    tooltip="Select checkpoint",
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
                Const.MODEL_TYPE.Output(
                    "CHECKPOINT",
                    display_name="model",
                ),
                c_io.AnyType.Output(
                    "CHECKPOINT_COMBO",
                    display_name="ckpt_name",
                ),
            ],
        )

    @classmethod
    def execute(cls, checkpoint: str, sha256: str | None = None) -> c_io.NodeOutput:
        checkpoint_options = Const.get_CHECKPOINT_OPTIONS()
        value = resolve_selector_value(
            checkpoint,
            checkpoint_options,
            value_label="checkpoint",
            folder_name=Const.MODEL_FOLDER_PATH_CHECKPOINTS,
            sha256=sha256,
        )
        model = Const.make_model_value(value, Const.MODEL_FOLDER_PATH_CHECKPOINTS)
        return c_io.NodeOutput(model, value, ui={"checkpoint": [value or ""]})
