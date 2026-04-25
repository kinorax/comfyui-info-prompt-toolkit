# Copyright 2026 kinorax

from comfy_api.latest import io as c_io
from .. import const as Const

class SamplerSelector(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-SamplerSelector",
            display_name="Sampler Selector",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.Combo.Input(
                    "sampler",
                    options=Const.SAMPLER_OPTIONS,
                    default=Const.SAMPLER_OPTIONS[0],
                    tooltip="Select sampler",
                ),
            ],
            outputs=[
                c_io.AnyType.Output(
                    "SAMPLER",
                    display_name="sampler",
                ),
            ],
        )

    @classmethod
    def execute(cls, sampler: str) -> c_io.NodeOutput:
        value = sampler if sampler in Const.SAMPLER_OPTIONS else None
        return c_io.NodeOutput(value)
