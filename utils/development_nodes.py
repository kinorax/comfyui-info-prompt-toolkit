# Copyright 2026 kinorax
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

_DEVELOPMENT_PACKAGE = ".nodes._development"
_DEVELOPMENT_PACKAGE_PARTS = ("nodes", "_development")
_DEVELOPMENT_NODE_LIST_NAME = "DEVELOPMENT_NODE_LIST"


def _development_package_init_path(anchor_file: str) -> Path:
    return Path(anchor_file).resolve().parent.joinpath(*_DEVELOPMENT_PACKAGE_PARTS, "__init__.py")


def load_development_node_list(package_name: str | None, anchor_file: str) -> list[type[Any]]:
    """Load private development-only nodes when the local registry package exists."""
    if not package_name:
        return []

    if not _development_package_init_path(anchor_file).is_file():
        return []

    module = importlib.import_module(_DEVELOPMENT_PACKAGE, package=package_name)
    registry = getattr(module, _DEVELOPMENT_NODE_LIST_NAME, None)
    if registry is None:
        get_node_list = getattr(module, "get_node_list", None)
        if not callable(get_node_list):
            return []
        registry = get_node_list()

    if not isinstance(registry, (list, tuple)):
        raise TypeError(f"nodes._development.{_DEVELOPMENT_NODE_LIST_NAME} must be a list or tuple")

    node_list: list[type[Any]] = []
    for node_class in registry:
        if not isinstance(node_class, type):
            raise TypeError("nodes._development entries must be node classes")
        node_list.append(node_class)

    return node_list
