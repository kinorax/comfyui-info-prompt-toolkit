# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

from comfy_api.latest import io as c_io
from comfy_execution.graph_utils import GraphBuilder, is_link

from .. import const as Const
from ..utils import cast as Cast

_RELAY_NODE_ID = "IPT-LoopImageListRelaySimple"


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [value]


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        for item in value:
            if _has_value(item):
                return True
        return False
    return True


def _extract_raw_link(value: Any) -> list[Any] | None:
    if is_link(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            raw_link = _extract_raw_link(item)
            if raw_link is not None:
                return raw_link
    return None


def _resolve_open_node_id(dynprompt: Any, close_node_id: str) -> str | None:
    close_node = dynprompt.get_node(close_node_id)
    if not isinstance(close_node, dict):
        return None

    close_inputs = close_node.get("inputs", {})
    if not isinstance(close_inputs, dict):
        return None

    raw_link = _extract_raw_link(close_inputs.get("loop_control"))
    if raw_link is None:
        return None

    return str(raw_link[0])


class LoopEndSimple(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-LoopEndSimple",
            display_name="Loop End Simple",
            category=Const.CATEGORY_XYPLOT,
            enable_expand=True,
            is_input_list=True,
            hidden=[
                c_io.Hidden.dynprompt,
                c_io.Hidden.unique_id,
            ],
            inputs=[
                c_io.Image.Input(
                    "image",
                    tooltip="Generated image",
                ),
                Const.LOOP_CONTROL_TYPE.Input(
                    "loop_control",
                    tooltip="Loop control value from Loop Start Simple (None means batch mode)",
                ),
                c_io.AnyType.Input(
                    "accumulated_images",
                    optional=True,
                    advanced=True,
                    tooltip="Internal use only. Leave this unconnected.",
                ),
            ],
            outputs=[
                c_io.Image.Output(
                    Cast.out_id("image"),
                    display_name="image",
                    is_output_list=True,
                ),
            ],
        )

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
        image: Any,
        loop_control: Any,
        accumulated_images: Any = None,
    ) -> c_io.NodeOutput:
        collected_images = _flatten(accumulated_images)
        collected_images.extend(_flatten(image))

        if not _has_value(loop_control):
            return c_io.NodeOutput(collected_images)

        hidden = getattr(cls, "hidden", None)
        dynprompt = getattr(hidden, "dynprompt", None)
        unique_id = getattr(hidden, "unique_id", None)

        if dynprompt is None or unique_id is None:
            return c_io.NodeOutput(collected_images)

        unique_id_str = str(unique_id)
        open_node_id = _resolve_open_node_id(dynprompt, unique_id_str)
        if open_node_id is None:
            return c_io.NodeOutput(collected_images)

        open_node = dynprompt.get_node(open_node_id)
        if not isinstance(open_node, dict):
            return c_io.NodeOutput(collected_images)

        open_inputs = open_node.get("inputs", {})
        if not isinstance(open_inputs, dict):
            return c_io.NodeOutput(collected_images)

        current_index = int(open_inputs.get("loop_index", 0))
        final_index = 2
        if current_index >= final_index:
            return c_io.NodeOutput(collected_images)

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
            return c_io.NodeOutput(collected_images)

        new_open.set_input("loop_index", current_index + 1)
        recurse_close.set_input("accumulated_images", collected_images)

        relay = graph.node(_RELAY_NODE_ID, "Relay", image=recurse_close.out(0))

        return c_io.NodeOutput(
            relay.out(0),
            expand=graph.finalize(),
        )