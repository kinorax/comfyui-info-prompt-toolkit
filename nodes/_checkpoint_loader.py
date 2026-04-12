from __future__ import annotations

import importlib
import sys
from typing import Callable

CHECKPOINT_LOADER_KEYS: tuple[str, ...] = ("CheckpointLoaderSimple",)


def normalized_checkpoint_name_or_none(value: object) -> str | None:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return normalized_checkpoint_name_or_none(value[0])

    if isinstance(value, dict) and "__value__" in value:
        return normalized_checkpoint_name_or_none(value.get("__value__"))

    if isinstance(value, dict):
        name = value.get("name")
        if name is None:
            return None
        normalized = str(name).strip()
        return normalized or None

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def core_nodes_module_or_none() -> object | None:
    module = sys.modules.get("nodes")
    if module is not None and isinstance(getattr(module, "NODE_CLASS_MAPPINGS", None), dict):
        return module

    try:
        module = importlib.import_module("nodes")
    except Exception:
        return None

    if not isinstance(getattr(module, "NODE_CLASS_MAPPINGS", None), dict):
        return None
    return module


def loader_class_or_none(module: object, keys: tuple[str, ...]) -> type | None:
    mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
    if isinstance(mappings, dict):
        for key in keys:
            value = mappings.get(key)
            if isinstance(value, type):
                return value

    for key in keys:
        value = getattr(module, key, None)
        if isinstance(value, type):
            return value
    return None


def loader_method_or_none(loader_instance: object) -> Callable[..., object] | None:
    function_name = getattr(loader_instance, "FUNCTION", None)
    if isinstance(function_name, str):
        fn = getattr(loader_instance, function_name, None)
        if callable(fn):
            return fn

    fn = getattr(loader_instance, "load_checkpoint", None)
    if callable(fn):
        return fn
    return None


def load_checkpoint_with_core_loader(loader_instance: object, checkpoint_name: str) -> tuple[object, object, object]:
    method = loader_method_or_none(loader_instance)
    if method is None:
        raise RuntimeError("Failed to resolve Load Checkpoint function")

    try:
        result = method(checkpoint_name)
    except TypeError:
        result = method(ckpt_name=checkpoint_name)

    if not isinstance(result, (list, tuple)) or len(result) < 3:
        raise RuntimeError("Load Checkpoint returned unexpected outputs")

    return result[0], result[1], result[2]
