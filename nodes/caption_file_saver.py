# Copyright 2026 kinorax
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from .batch_image_reader import (
    _list_directory_options,
    _resolve_base_directory,
    _resolve_target_directory,
)

_PATH_SOURCE_OPTIONS = ("input", "output")
_INVALID_FILE_STEM_CHARS = set('<>:"|?*')
_WINDOWS_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def _unwrap_input_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            return list(value[0])
        return list(value)
    return [value]


def _resolve_single_input(raw: Any, name: str) -> Any:
    values = _unwrap_input_list(raw)
    if len(values) == 0:
        return None
    if len(values) > 1:
        raise ValueError(f"{name} must be a single value")
    return values[0]


def _normalize_caption_values(raw: Any) -> list[Any]:
    values = _unwrap_input_list(raw)
    if len(values) == 0:
        return [""]
    return values


def _normalize_file_stem_values(raw: Any) -> list[Any]:
    values = _unwrap_input_list(raw)
    if len(values) == 0:
        raise ValueError("file_stem is required")
    return values


def _resolve_batch_count(*value_lists: list[Any]) -> int:
    count = 1
    for values in value_lists:
        if len(values) > count:
            count = len(values)
    return count


def _resolve_caption_batch(values: list[Any], batch_count: int) -> list[str]:
    if len(values) == 1:
        text = "" if values[0] is None else str(values[0])
        return [text] * batch_count
    if len(values) == batch_count:
        return ["" if value is None else str(value) for value in values]
    raise ValueError(f"caption must have length 1 or {batch_count}, got {len(values)}")


def _resolve_file_stem_batch(values: list[Any], batch_count: int) -> list[Any]:
    if batch_count <= 1:
        if len(values) != 1:
            raise ValueError(f"file_stem must have length 1, got {len(values)}")
        return values
    if len(values) != batch_count:
        raise ValueError(f"file_stem must have length {batch_count} for batch save, got {len(values)}")
    return values


def _validate_file_stem(value: Any) -> str:
    stem = str(value or "").strip()
    if not stem:
        raise ValueError("file_stem must not be empty")
    if stem in (".", ".."):
        raise ValueError("file_stem must not be '.' or '..'")
    if "/" in stem or "\\" in stem:
        raise ValueError("file_stem must not contain path separators")
    if any(ord(ch) < 32 for ch in stem):
        raise ValueError("file_stem must not contain control characters")
    invalid_chars = sorted(ch for ch in set(stem) if ch in _INVALID_FILE_STEM_CHARS)
    if invalid_chars:
        raise ValueError(f"file_stem contains invalid characters: {''.join(invalid_chars)}")
    if stem.endswith(".") or stem.endswith(" "):
        raise ValueError("file_stem must not end with '.' or space")

    reserved_base = stem.split(".", 1)[0].upper()
    if reserved_base in _WINDOWS_RESERVED_BASENAMES:
        raise ValueError(f"file_stem uses a reserved Windows name: {reserved_base}")
    return stem


def _relative_to_base(path: Path, base_dir: Path) -> str:
    try:
        rel = path.resolve().relative_to(base_dir)
    except Exception as exc:
        raise ValueError("saved file path is outside selected base directory") from exc
    return rel.as_posix()


def _prepare_save_batch(
    *,
    caption: Any,
    file_stem: Any,
    path_source: Any,
    path: Any,
) -> tuple[list[str], list[str], list[Path], list[str]]:
    caption_values = _normalize_caption_values(caption)
    file_stem_values = _normalize_file_stem_values(file_stem)
    batch_count = _resolve_batch_count(caption_values, file_stem_values)

    captions = _resolve_caption_batch(caption_values, batch_count)
    raw_file_stems = _resolve_file_stem_batch(file_stem_values, batch_count)

    path_source_value = str(_resolve_single_input(path_source, "path_source"))
    path_value_raw = _resolve_single_input(path, "path")
    path_value = "" if path_value_raw is None else str(path_value_raw)

    target_dir = _resolve_target_directory(path_value, path_source_value)
    base_dir, _ = _resolve_base_directory(path_source_value)

    target_paths: list[Path] = []
    normalized_file_stems: list[str] = []
    seen_targets: set[str] = set()

    for raw_stem in raw_file_stems:
        stem = _validate_file_stem(raw_stem)
        target_path = (target_dir / f"{stem}.txt").resolve()

        try:
            common = Path(os.path.commonpath((str(target_dir), str(target_path))))
        except ValueError as exc:
            raise ValueError("file_stem must resolve under selected path") from exc
        if common != target_dir:
            raise ValueError("file_stem must resolve under selected path")

        target_key = str(target_path).casefold()
        if target_key in seen_targets:
            raise ValueError(f"duplicate output file path in batch: {stem}.txt")
        seen_targets.add(target_key)

        normalized_file_stems.append(stem)
        target_paths.append(target_path)

    relative_paths = [_relative_to_base(target_path, base_dir) for target_path in target_paths]
    return captions, normalized_file_stems, target_paths, relative_paths


class CaptionFileSaver(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        path_options = _list_directory_options("input")
        default_path = path_options[0] if path_options else "."

        return c_io.Schema(
            node_id="IPT-CaptionFileSaver",
            display_name="Caption File Saver",
            category=Const.CATEGORY_IMAGEINFO,
            is_output_node=True,
            is_input_list=True,
            description="Save caption txt files to an existing input/output subdirectory.",
            search_aliases=[
                "caption file saver",
                "caption saver",
                "save caption txt",
                "txt file saver",
            ],
            inputs=[
                c_io.String.Input(
                    "file_stem",
                    default="caption",
                    force_input=True,
                    tooltip="Output filename without extension; for batch save, provide one stem per item",
                ),
                c_io.String.Input(
                    "caption",
                    default="",
                    force_input=True,
                    tooltip="Caption text or caption list to save as txt files",
                ),
                c_io.Combo.Input(
                    "path_source",
                    options=list(_PATH_SOURCE_OPTIONS),
                    default="input",
                    tooltip="Base directory for path",
                ),
                c_io.Combo.Input(
                    "path",
                    options=path_options,
                    default=default_path,
                    tooltip="Directory path relative to selected base directory",
                ),
            ],
            outputs=[
                c_io.String.Output(
                    Cast.out_id("caption"),
                    display_name="caption",
                    is_output_list=True,
                ),
                c_io.String.Output(
                    Cast.out_id("file_stem"),
                    display_name="file_stem",
                    is_output_list=True,
                ),
                c_io.String.Output(
                    Cast.out_id("file_path"),
                    display_name="file_path",
                    is_output_list=True,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        file_stem: Any,
        caption: Any,
        path_source: Any,
        path: Any,
    ) -> c_io.NodeOutput:
        captions, file_stems, target_paths, relative_paths = _prepare_save_batch(
            caption=caption,
            file_stem=file_stem,
            path_source=path_source,
            path=path,
        )

        for caption_text, target_path in zip(captions, target_paths):
            try:
                target_path.write_text(caption_text, encoding="utf-8")
            except Exception as exc:
                raise RuntimeError(f"failed to save caption file: {target_path.name}") from exc

        return c_io.NodeOutput(captions, file_stems, relative_paths)

    @classmethod
    def validate_inputs(
        cls,
        file_stem: Any,
        caption: Any,
        path_source: Any,
        path: Any,
    ):
        try:
            _prepare_save_batch(
                caption=caption,
                file_stem=file_stem,
                path_source=path_source,
                path=path,
            )
        except Exception as exc:
            return str(exc)
        return True
