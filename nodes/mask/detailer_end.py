# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

from comfy_api.latest import io as c_io
from comfy_execution.graph_utils import GraphBuilder, is_link

from ... import const as Const
from ...utils import cast as Cast
from ._detailer_utils import (
    detailer_has_next_item,
    detailer_requires_inpainted_image,
    restore_detailer_batch,
)

_MISSING = object()


class DetailerEnd(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-DetailerEnd",
            display_name="Detailer End",
            category=Const.CATEGORY_MASK,
            enable_expand=True,
            hidden=[
                c_io.Hidden.dynprompt,
                c_io.Hidden.unique_id,
            ],
            search_aliases=["detailer close", "detailer composite", "detailer paste back"],
            inputs=[
                Const.DETAILER_CONTROL_TYPE.Input(
                    "detailer_control",
                    tooltip="Control object produced by Detailer Start.",
                ),
                c_io.Image.Input(
                    "inpainted_image",
                    optional=True,
                    lazy=True,
                    tooltip="Inpainted crop image. When Detailer Start produced no active mask, this input is not evaluated.",
                ),
                c_io.Image.Input(
                    "accumulated_image",
                    optional=True,
                    advanced=True,
                    tooltip="Internal use only. Carries the progressively composited full image batch during per-item execution.",
                ),
            ],
            outputs=[
                c_io.Image.Output(
                    Cast.out_id("image"),
                    display_name="image",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        detailer_control: Any = None,
        inpainted_image: Any = None,
        accumulated_image: Any = None,
    ) -> bool | str:
        return True

    @classmethod
    def check_lazy_status(
        cls,
        detailer_control: Any,
        inpainted_image: object = _MISSING,
        accumulated_image: object = _MISSING,
    ) -> list[str]:
        if not detailer_requires_inpainted_image(detailer_control):
            return []
        return ["inpainted_image"]

    @classmethod
    def _explore_dependencies(cls, node_id: str, dynprompt: Any, upstream: dict[str, list[str]]) -> None:
        node_info = dynprompt.get_node(node_id)
        if not isinstance(node_info, dict):
            return
        inputs = node_info.get("inputs", {})
        if not isinstance(inputs, dict):
            return
        for value in inputs.values():
            if not is_link(value):
                continue
            parent_id = value[0]
            if parent_id not in upstream:
                upstream[parent_id] = []
                cls._explore_dependencies(parent_id, dynprompt, upstream)
            upstream[parent_id].append(node_id)

    @classmethod
    def _collect_contained(cls, node_id: str, upstream: dict[str, list[str]], contained: dict[str, bool]) -> None:
        children = upstream.get(node_id)
        if children is None:
            return
        for child_id in children:
            if child_id in contained:
                continue
            contained[child_id] = True
            cls._collect_contained(child_id, upstream, contained)

    @classmethod
    def execute(
        cls,
        detailer_control: Any,
        inpainted_image: object = _MISSING,
        accumulated_image: Any = None,
    ) -> c_io.NodeOutput:
        if detailer_requires_inpainted_image(detailer_control):
            if inpainted_image is _MISSING or inpainted_image is None:
                raise RuntimeError("inpainted_image input is required when Detailer Start produced an active mask")
            output_image = restore_detailer_batch(
                detailer_control,
                inpainted_image,
                accumulated_image=accumulated_image,
            )
        else:
            output_image = restore_detailer_batch(
                detailer_control,
                None,
                accumulated_image=accumulated_image,
            )

        if not detailer_has_next_item(detailer_control):
            return c_io.NodeOutput(output_image)

        hidden = getattr(cls, "hidden", None)
        dynprompt = getattr(hidden, "dynprompt", None)
        unique_id = getattr(hidden, "unique_id", None)
        if dynprompt is None or unique_id is None:
            return c_io.NodeOutput(output_image)

        open_node_id = detailer_control.get("open_node_id")
        if not isinstance(open_node_id, str) or not open_node_id:
            return c_io.NodeOutput(output_image)

        unique_id_str = str(unique_id)
        upstream: dict[str, list[str]] = {}
        cls._explore_dependencies(unique_id_str, dynprompt, upstream)

        contained: dict[str, bool] = {}
        cls._collect_contained(open_node_id, upstream, contained)
        contained[unique_id_str] = True
        contained[open_node_id] = True

        graph = GraphBuilder()
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            if not isinstance(original_node, dict):
                continue

            class_type = original_node.get("class_type")
            if not isinstance(class_type, str):
                continue

            clone_id = "Recurse" if node_id == unique_id_str else node_id
            node = graph.node(class_type, clone_id)
            node.set_override_display_id(node_id)

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            if not isinstance(original_node, dict):
                continue

            clone_id = "Recurse" if node_id == unique_id_str else node_id
            node = graph.lookup_node(clone_id)
            if node is None:
                continue

            original_inputs = original_node.get("inputs", {})
            if not isinstance(original_inputs, dict):
                continue

            for key, value in original_inputs.items():
                if is_link(value) and value[0] in contained:
                    parent_id = "Recurse" if value[0] == unique_id_str else value[0]
                    parent = graph.lookup_node(parent_id)
                    if parent is not None:
                        node.set_input(key, parent.out(value[1]))
                else:
                    node.set_input(key, value)

        new_open = graph.lookup_node(open_node_id)
        recurse_close = graph.lookup_node("Recurse")
        if new_open is None or recurse_close is None:
            return c_io.NodeOutput(output_image)

        next_index = int(detailer_control.get("current_index", 0)) + 1
        new_open.set_input("detailer_index", next_index)
        recurse_close.set_input("accumulated_image", output_image)

        return c_io.NodeOutput(
            recurse_close.out(0),
            expand=graph.finalize(),
        )
