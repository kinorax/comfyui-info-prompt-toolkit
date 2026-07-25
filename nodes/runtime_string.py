# Copyright 2026 kinorax
from __future__ import annotations

from threading import RLock
from typing import Any

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast

_RUNTIME_STRINGS: dict[str, str] = {}
_RUNTIME_STRINGS_LOCK = RLock()


def _unwrap_singleton(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _unwrap_singleton(value[0])
    return value


def _normalized_key_or_none(key: Any) -> str | None:
    value = _unwrap_singleton(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _required_key(key: Any) -> str:
    normalized = _normalized_key_or_none(key)
    if normalized is None:
        raise ValueError("key is required")
    return normalized


def _required_string(value: Any) -> str:
    unwrapped = _unwrap_singleton(value)
    if unwrapped is None:
        raise ValueError("string is required")
    return str(unwrapped)


def _set_runtime_string(key: str, value: str) -> None:
    with _RUNTIME_STRINGS_LOCK:
        _RUNTIME_STRINGS[key] = value


def _get_runtime_string(key: str) -> str:
    with _RUNTIME_STRINGS_LOCK:
        if key not in _RUNTIME_STRINGS:
            raise ValueError(f"runtime string not found for key: {key}")
        return _RUNTIME_STRINGS[key]


def _clear_runtime_strings() -> None:
    with _RUNTIME_STRINGS_LOCK:
        _RUNTIME_STRINGS.clear()


class SetRuntimeString(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-SetRuntimeString",
            display_name="Set Runtime String",
            category=Const.CATEGORY_UTILITY,
            description="Store a string by key in process memory until ComfyUI restarts.",
            search_aliases=["store runtime string", "save runtime string", "memory string set"],
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                c_io.String.Input(
                    "key",
                    tooltip="Non-empty process-wide storage key",
                ),
                c_io.String.Input(
                    "string",
                    force_input=True,
                    tooltip="String to store; a value stored under the same key is overwritten",
                ),
            ],
            outputs=[
                c_io.AnyType.Output(
                    Cast.out_id("string"),
                    display_name="string",
                    tooltip="Stored string, returned for downstream execution ordering",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, key: Any, string: Any = None) -> bool | str:
        if _normalized_key_or_none(key) is None:
            return "key is required"
        return True

    @classmethod
    def execute(cls, key: Any, string: Any = None) -> c_io.NodeOutput:
        normalized_key = _required_key(key)
        string_value = _required_string(string)
        _set_runtime_string(normalized_key, string_value)
        return c_io.NodeOutput(string_value)


class GetRuntimeString(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-GetRuntimeString",
            display_name="Get Runtime String",
            category=Const.CATEGORY_UTILITY,
            description="Load a process-memory string by key after an optional upstream dependency completes.",
            search_aliases=["load runtime string", "read runtime string", "memory string get"],
            not_idempotent=True,
            inputs=[
                c_io.AnyType.Input(
                    "after",
                    optional=True,
                    tooltip="Optional upstream dependency; its value is ignored",
                ),
                c_io.String.Input(
                    "key",
                    tooltip="Process-wide storage key to load",
                ),
            ],
            outputs=[
                c_io.AnyType.Output(
                    Cast.out_id("string"),
                    display_name="string",
                    tooltip="Stored string; compatible with combo-style path inputs",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, after: Any = None, key: Any = None) -> bool | str:
        if _normalized_key_or_none(key) is None:
            return "key is required"
        return True

    @classmethod
    def execute(cls, after: Any = None, key: Any = None) -> c_io.NodeOutput:
        normalized_key = _required_key(key)
        string_value = _get_runtime_string(normalized_key)
        return c_io.NodeOutput(string_value)
