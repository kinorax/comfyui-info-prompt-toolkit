# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast

MODE_LOOP = "Loop (list 1)"
MODE_BATCH = "Batch (list all)"
MODE_OPTIONS: tuple[str, ...] = (
    MODE_LOOP,
    MODE_BATCH,
)


def _first_value(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            nested = value[0]
            return nested[0] if nested else None
        return value[0] if value else None
    return value


def _normalized_mode(mode: object) -> str:
    text = str(mode) if mode is not None else MODE_LOOP
    if text in MODE_OPTIONS:
        return text
    return MODE_LOOP


class LoopStartSimple(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-LoopStartSimple",
            display_name="Loop Start Simple",
            category=Const.CATEGORY_XYPLOT,
            inputs=[
                Const.IMAGEINFO_TYPE.Input(
                    "image_info_01",
                    tooltip="First image_info",
                ),
                Const.IMAGEINFO_TYPE.Input(
                    "image_info_02",
                    tooltip="Second image_info",
                ),
                Const.IMAGEINFO_TYPE.Input(
                    "image_info_03",
                    tooltip="Third image_info",
                ),
                c_io.Combo.Input(
                    "mode",
                    options=MODE_OPTIONS,
                    default=MODE_LOOP,
                    tooltip="Loop mode returns list(1). Batch mode returns list(all).",
                ),
                c_io.Int.Input(
                    "loop_index",
                    default=0,
                    min=0,
                    max=2,
                    advanced=True,
                    tooltip="Internal index for recursive loop expansion",
                ),
            ],
            outputs=[
                Const.IMAGEINFO_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO),
                    display_name=Const.IMAGEINFO,
                    is_output_list=True,
                ),
                Const.LOOP_CONTROL_TYPE.Output(
                    Cast.out_id("loop_control"),
                    display_name="loop_control",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image_info_01: Any,
        image_info_02: Any,
        image_info_03: Any,
        mode: str = MODE_LOOP,
        loop_index: int = 0,
    ) -> c_io.NodeOutput:
        image_infos = [
            _first_value(image_info_01),
            _first_value(image_info_02),
            _first_value(image_info_03),
        ]
        normalized_mode = _normalized_mode(mode)

        index = int(loop_index)
        if index < 0:
            index = 0
        if index >= len(image_infos):
            index = len(image_infos) - 1

        flow_control = ("ipt_loop_start_simple", index)
        if normalized_mode == MODE_BATCH:
            return c_io.NodeOutput(image_infos, None)
        return c_io.NodeOutput([image_infos[index]], flow_control)
