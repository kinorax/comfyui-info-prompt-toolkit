# Copyright 2026 kinorax
from comfy_api.latest import io as c_io
from .. import const as Const
from ..utils.selector_resolution import resolve_selector_value


class LoraSelector(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        lora_options = Const.get_LORA_OPTIONS()
        default_lora = lora_options[0] if lora_options else ""
        return c_io.Schema(
            node_id="IPT-LoraSelector",
            display_name="Lora Selector",
            category=Const.CATEGORY_IMAGEINFO,
            hidden=[c_io.Hidden.prompt, c_io.Hidden.unique_id],
            inputs=[
                c_io.Combo.Input(
                    "lora",
                    display_name="lora_name",
                    options=lora_options,
                    default=default_lora,
                    tooltip="Select LoRA",
                ),
                c_io.Float.Input(
                    "strength",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip="Set LoRA strength",
                ),
                c_io.String.Input(
                    "sha256",
                    default="",
                    socketless=True,
                    optional=True,
                    tooltip="Cached SHA256 used by View Model Info fallback",
                ),
                Const.LORA_STACK_TYPE.Input(
                    "lora_stack",
                    optional=True,
                ),
            ],
            outputs=[
                Const.LORA_STACK_TYPE.Output(
                    "LORA_STACK",
                    display_name="lora_stack",
                ),
                c_io.AnyType.Output(
                    "LORA",
                    display_name="lora_name",
                ),
                c_io.Float.Output(
                    "STRENGTH",
                    display_name="strength",
                ),
            ],
        )

    @classmethod
    def _is_lora_stack_connected(cls) -> bool:
        prompt = getattr(cls.hidden, "prompt", None)
        unique_id = getattr(cls.hidden, "unique_id", None)
        if not isinstance(prompt, dict) or unique_id is None:
            return False

        node = prompt.get(str(unique_id), {}) or {}
        inputs = node.get("inputs", {}) or {}
        return "lora_stack" in inputs

    @classmethod
    def execute(
        cls,
        lora: str,
        strength: float,
        sha256: str | None = None,
        lora_stack: list[dict[str, str | float]] | None = None,
    ) -> c_io.NodeOutput:
        lora_options = Const.get_LORA_OPTIONS()
        value = resolve_selector_value(
            lora,
            lora_options,
            value_label="lora",
            folder_name=Const.MODEL_FOLDER_PATH_LORAS,
            sha256=sha256,
        )
        stack_item = Const.make_lora_stack_item(value, strength)

        if cls._is_lora_stack_connected() and isinstance(lora_stack, list):
            output_stack = list(lora_stack)
        else:
            output_stack = []

        if stack_item is not None:
            output_stack.append(stack_item)

        return c_io.NodeOutput(output_stack, value, float(strength))
