# Copyright 2026 kinorax

from comfy_api.latest import io as c_io
from .. import const as Const

class SchedulerSelector(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-SchedulerSelector",
            display_name="Scheduler Selector",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                c_io.Combo.Input(
                    "scheduler",
                    options=Const.SCHEDULER_OPTIONS,
                    default=Const.SCHEDULER_OPTIONS[0],
                    tooltip="Select scheduler",
                ),
            ],
            outputs=[
                c_io.AnyType.Output(
                    "SCHEDULER",
                    display_name="scheduler",
                ),
            ],
        )

    @classmethod
    def execute(cls, scheduler: str) -> c_io.NodeOutput:
        value = scheduler if scheduler in Const.SCHEDULER_OPTIONS else None
        return c_io.NodeOutput(value)
