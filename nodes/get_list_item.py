# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast

_MATCH_TEMPLATE = c_io.MatchType.Template("get_list_item", c_io.AnyType)


def _first_value(value: Any, default: Any = None) -> Any:
    while isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        value = value[0]
    return value


def _normalized_index(value: Any) -> int | None:
    value = _first_value(value, 0)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _get_list_item(items: Any, index: Any) -> Any:
    if not isinstance(items, (list, tuple)):
        return None

    normalized_index = _normalized_index(index)
    if normalized_index is None or normalized_index < 0 or normalized_index >= len(items):
        return None
    return items[normalized_index]


class GetListItem(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-GetListItem",
            display_name="Get List Item",
            category=Const.CATEGORY_UTILITY,
            description="Return one item from a list by zero-based index, or None when unavailable.",
            search_aliases=[
                "list item",
                "list index",
                "get item",
                "first item",
            ],
            is_input_list=True,
            inputs=[
                c_io.MatchType.Input(
                    "items",
                    template=_MATCH_TEMPLATE,
                    display_name="list",
                    optional=True,
                    tooltip="Optional list to select from",
                ),
                c_io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    tooltip="Zero-based item index; unavailable indices return None",
                ),
            ],
            outputs=[
                c_io.MatchType.Output(
                    template=_MATCH_TEMPLATE,
                    id=Cast.out_id("item"),
                    display_name="item",
                    tooltip="Selected item, or None when the list or index is unavailable",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, items: Any = None, index: Any = 0) -> bool:
        return True

    @classmethod
    def execute(cls, items: Any = None, index: Any = 0) -> c_io.NodeOutput:
        return c_io.NodeOutput(_get_list_item(items, index))
