# Copyright 2026 kinorax
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import folder_paths

IMAGE_FILE_REF_VERSION = 1
IMAGE_FILE_REF_VERSION_KEY = "version"
IMAGE_FILE_REF_PATH_KEY = "path"
IMAGE_FILE_REF_FILENAME_KEY = "filename"
IMAGE_FILE_REF_FILE_STEM_KEY = "file_stem"
IMAGE_FILE_REF_EXTENSION_KEY = "extension"
IMAGE_FILE_REF_FORMAT_KEY = "format"
IMAGE_FILE_REF_FOLDER_TYPE_KEY = "folder_type"
IMAGE_FILE_REF_RELATIVE_PATH_KEY = "relative_path"
IMAGE_FILE_REF_SIZE_KEY = "size"
IMAGE_FILE_REF_MTIME_NS_KEY = "mtime_ns"

_UNKNOWN_FOLDER_TYPE = "unknown"
_INPUT_FOLDER_TYPE = "input"
_OUTPUT_FOLDER_TYPE = "output"
_TEMP_FOLDER_TYPE = "temp"


def _safe_resolve_directory(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    try:
        return Path(raw_path).resolve()
    except Exception:
        return None


def _get_comfy_roots() -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    for folder_type, getter_name in (
        (_INPUT_FOLDER_TYPE, "get_input_directory"),
        (_OUTPUT_FOLDER_TYPE, "get_output_directory"),
        (_TEMP_FOLDER_TYPE, "get_temp_directory"),
    ):
        getter = getattr(folder_paths, getter_name, None)
        if not callable(getter):
            continue
        root = _safe_resolve_directory(str(getter() or ""))
        if root is not None:
            roots.append((folder_type, root))
    return roots


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        common = Path(os.path.commonpath((str(root), str(path))))
    except ValueError:
        return False
    return common == root


def classify_comfy_path(path: str | Path) -> tuple[str, str]:
    resolved = Path(path).resolve()
    for folder_type, root in _get_comfy_roots():
        if not _is_relative_to(resolved, root):
            continue
        try:
            return folder_type, resolved.relative_to(root).as_posix()
        except Exception:
            return folder_type, resolved.name
    return _UNKNOWN_FOLDER_TYPE, resolved.name


def build_image_file_ref(
    path: str | Path,
    *,
    format_name: str | None = None,
    folder_type: str | None = None,
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    detected_folder_type, relative_path = classify_comfy_path(resolved)
    normalized_folder_type = folder_type or detected_folder_type
    if normalized_folder_type == _UNKNOWN_FOLDER_TYPE and folder_type:
        normalized_folder_type = str(folder_type)

    return {
        IMAGE_FILE_REF_VERSION_KEY: IMAGE_FILE_REF_VERSION,
        IMAGE_FILE_REF_PATH_KEY: str(resolved),
        IMAGE_FILE_REF_FILENAME_KEY: resolved.name,
        IMAGE_FILE_REF_FILE_STEM_KEY: resolved.stem,
        IMAGE_FILE_REF_EXTENSION_KEY: resolved.suffix.lower(),
        IMAGE_FILE_REF_FORMAT_KEY: str(format_name or "").strip().lower(),
        IMAGE_FILE_REF_FOLDER_TYPE_KEY: normalized_folder_type,
        IMAGE_FILE_REF_RELATIVE_PATH_KEY: relative_path,
        IMAGE_FILE_REF_SIZE_KEY: int(stat.st_size),
        IMAGE_FILE_REF_MTIME_NS_KEY: int(stat.st_mtime_ns),
    }


def resolve_image_file_ref(value: Any) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError("image_reference must be an ImageFileRef payload")

    raw_path = value.get(IMAGE_FILE_REF_PATH_KEY)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("image_reference.path is required")

    resolved = Path(raw_path).resolve()
    if not resolved.is_file():
        raise ValueError(f"referenced image file was not found: {resolved}")

    if not any(_is_relative_to(resolved, root) for _, root in _get_comfy_roots()):
        raise ValueError("image_reference.path must be under a ComfyUI input/output/temp directory")

    return resolved


def resolve_caption_path(image_path: str | Path) -> Path:
    return Path(image_path).with_suffix(".txt")


def read_caption_text(image_path: str | Path) -> str:
    caption_path = resolve_caption_path(image_path)
    if not caption_path.is_file():
        return ""
    return caption_path.read_text(encoding="utf-8-sig", errors="replace")
