# Copyright 2026 kinorax
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import folder_paths
from PIL import Image
from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils import exif as Exif
from ..utils.image_file_reference import build_image_file_ref, resolve_image_file_ref
from ..utils.image_metadata_writer import write_a1111_text_metadata_only, write_ipt_private_metadata_only
from ..utils.metadata_encryption import prepare_image_info_metadata
from .image_saver import (
    _MISSING,
    _OUTPUT_SUBDIR_OPTIONS,
    _normalize_caption_values,
    _normalize_optional_file_stem_values,
    _relative_to_output_root,
    _render_file_stem,
    _resolve_output_dir,
    _resolve_output_root,
    _resolve_per_image_values,
    _resolve_single_input,
    _resolve_subdir,
    _safe_filename_suffix,
    _unwrap_input_list,
    _validate_file_stem,
)

_SUPPORTED_SOURCE_EXTS = (".png", ".webp", ".jpg", ".jpeg")


def _build_infotext(image_info: Any) -> str:
    return prepare_image_info_metadata(image_info).infotext


def _resolve_forced_file_stems(raw: Any, count: int) -> list[str] | None:
    values = _normalize_optional_file_stem_values(raw)
    if values is None:
        return None
    if count <= 1:
        if len(values) != 1:
            raise ValueError(f"file_stem must have length 1, got {len(values)}")
    elif len(values) != count:
        raise ValueError(f"file_stem must have length {count} for batch save, got {len(values)}")
    return [_validate_file_stem(value) for value in values]


def _find_next_counter(folder: Path, date_prefix: str) -> int:
    current_max = 0
    expected_prefix = f"{date_prefix}-"
    for entry in os.scandir(folder):
        if not entry.is_file():
            continue

        name = entry.name
        if not name.startswith(expected_prefix):
            continue
        if Path(name).suffix.lower() not in _SUPPORTED_SOURCE_EXTS:
            continue

        rest = name[len(expected_prefix):]
        if len(rest) < 6 or rest[5] != "-":
            continue
        counter_text = rest[:5]
        if not counter_text.isdigit():
            continue
        counter = int(counter_text)
        if counter > current_max:
            current_max = counter
    return current_max + 1


def _resolve_source_paths(image_reference: Any) -> list[Path]:
    refs = _unwrap_input_list(image_reference)
    if len(refs) == 0:
        raise ValueError("image_reference is required")

    source_paths: list[Path] = []
    for ref in refs:
        source_path = resolve_image_file_ref(ref)
        if source_path.suffix.lower() not in _SUPPORTED_SOURCE_EXTS:
            raise ValueError(f"unsupported referenced image format: {source_path.suffix or source_path.name}")
        source_paths.append(source_path)
    return source_paths


def _resolve_caption_text(caption_value: Any, image_info: Any) -> str:
    if caption_value is not None:
        caption = str(caption_value).strip()
        if caption:
            return caption
    if isinstance(image_info, Mapping):
        positive = image_info.get(Const.IMAGEINFO_POSITIVE)
        if positive is not None:
            return str(positive)
    return ""


def _copy_source_to_target(source_path: Path, target_path: Path) -> None:
    if source_path.resolve() == target_path.resolve():
        return
    shutil.copy2(source_path, target_path)


def _validate_infotext_metadata(path: Path, infotext: str) -> None:
    if not infotext:
        return
    try:
        with Image.open(path) as image:
            loaded_infotext = Exif.extract_a1111_text(image)
        if loaded_infotext is None:
            raise RuntimeError("missing infotext in rewritten image metadata")
    except Exception as exc:
        raise RuntimeError(f"failed to validate infotext metadata: {path}") from exc


class ReferencedImageSaver(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-ReferencedImageSaver",
            display_name="Referenced Image Saver",
            category=Const.CATEGORY_IMAGEINFO,
            is_output_node=True,
            is_input_list=True,
            inputs=[
                Const.IMAGE_FILE_REF_TYPE.Input(
                    "image_reference",
                    tooltip="Referenced image file to copy and update without re-encoding pixels",
                ),
                Const.IMAGEINFO_TYPE.Input(
                    Const.IMAGEINFO,
                    optional=True,
                    tooltip="Image info or image-info list for each referenced image",
                ),
                c_io.String.Input(
                    "filename_suffix",
                    default="image",
                    tooltip="Suffix part of output filename",
                ),
                c_io.String.Input(
                    "file_stem",
                    optional=True,
                    force_input=True,
                    tooltip="Optional output filename stem list; when connected, save as <file_stem><source extension>",
                ),
                c_io.String.Input(
                    "caption",
                    default="",
                    optional=True,
                    force_input=True,
                    tooltip="Caption text or caption list for each referenced image",
                ),
                c_io.String.Input(
                    "output_dir",
                    default="",
                    tooltip="Base output directory under ComfyUI output",
                ),
                c_io.Combo.Input(
                    "output_subdir",
                    options=list(_OUTPUT_SUBDIR_OPTIONS),
                    default="none",
                    tooltip="Output sub directory format",
                ),
                c_io.Boolean.Input(
                    "write_caption",
                    default=False,
                    tooltip="Write caption txt file",
                ),
            ],
            outputs=[
                Const.IMAGE_FILE_REF_TYPE.Output(
                    Cast.out_id("image_reference"),
                    display_name="image_reference",
                    is_output_list=True,
                ),
                Const.IMAGEINFO_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO),
                    display_name=Const.IMAGEINFO,
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
        image_reference: Any,
        image_info: Any = None,
        filename_suffix: Any = _MISSING,
        file_stem: Any = None,
        caption: Any = None,
        output_dir: Any = _MISSING,
        output_subdir: Any = _MISSING,
        write_caption: Any = _MISSING,
    ) -> c_io.NodeOutput:
        source_paths = _resolve_source_paths(image_reference)
        image_count = len(source_paths)
        if image_count == 0:
            raise ValueError("image_reference is required")

        image_info_values = _unwrap_input_list(image_info)
        image_info_mapped = _resolve_per_image_values(image_info_values, image_count, "image_info")
        caption_values = _normalize_caption_values(caption)
        caption_mapped = _resolve_per_image_values(caption_values, image_count, "caption")
        forced_file_stems = _resolve_forced_file_stems(file_stem, image_count)

        if filename_suffix is _MISSING and forced_file_stems is None:
            raise ValueError("filename_suffix is required")
        if output_dir is _MISSING:
            raise ValueError("output_dir is required")
        if output_subdir is _MISSING:
            raise ValueError("output_subdir is required")
        if write_caption is _MISSING:
            raise ValueError("write_caption is required")

        output_dir_value = _resolve_single_input(output_dir, "output_dir")
        output_subdir_value = str(_resolve_single_input(output_subdir, "output_subdir"))
        write_caption_value = bool(_resolve_single_input(write_caption, "write_caption"))
        suffix = ""
        if forced_file_stems is None:
            suffix = _safe_filename_suffix(_resolve_single_input(filename_suffix, "filename_suffix"))

        if output_subdir_value not in _OUTPUT_SUBDIR_OPTIONS:
            raise ValueError(f"unsupported output_subdir: {output_subdir_value}")

        now = datetime.now()
        output_root = _resolve_output_root()
        base_output_dir = _resolve_output_dir(None if output_dir_value is None else str(output_dir_value), output_root)
        subdir = _resolve_subdir(now, output_subdir_value)
        target_dir = (base_output_dir / subdir).resolve() if subdir else base_output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        date_prefix = now.strftime("%Y%m%d")
        counter = _find_next_counter(target_dir, date_prefix) if forced_file_stems is None else 0
        seen_targets: set[str] = set()
        ui_images: list[dict[str, str]] = []
        output_refs: list[dict[str, Any]] = []
        saved_paths: list[str] = []

        for idx, source_path in enumerate(source_paths):
            output_ext = source_path.suffix.lower()
            if forced_file_stems is None:
                stem = _render_file_stem(date_prefix, counter, suffix)
            else:
                stem = forced_file_stems[idx]
            target_path = (target_dir / f"{stem}{output_ext}").resolve()
            target_key = str(target_path).casefold()
            if target_key in seen_targets:
                raise ValueError(f"duplicate output file path in batch: {target_path.name}")
            seen_targets.add(target_key)

            try:
                _copy_source_to_target(source_path, target_path)
            except Exception as exc:
                raise RuntimeError(f"failed to copy referenced image: {source_path.name}") from exc

            info_item = image_info_mapped[idx]
            prepared_metadata = prepare_image_info_metadata(info_item)
            infotext = prepared_metadata.infotext
            try:
                write_a1111_text_metadata_only(target_path, infotext)
                write_ipt_private_metadata_only(target_path, prepared_metadata.encrypted_payload)
            except Exception as exc:
                raise RuntimeError(f"failed to rewrite image metadata: {target_path}") from exc
            _validate_infotext_metadata(target_path, infotext)

            if write_caption_value:
                caption_text = _resolve_caption_text(caption_mapped[idx], info_item)
                caption_path = target_dir / f"{stem}.txt"
                caption_path.write_text(caption_text, encoding="utf-8")

            rel_file_path = _relative_to_output_root(target_path, output_root)
            saved_paths.append(rel_file_path)
            output_refs.append(build_image_file_ref(target_path, folder_type="output"))

            rel_parent = target_path.parent.resolve().relative_to(output_root).as_posix()
            ui_images.append(
                {
                    "filename": target_path.name,
                    "subfolder": "" if rel_parent == "." else rel_parent,
                    "type": "output",
                }
            )

            if forced_file_stems is None:
                counter += 1

        if image_info is None:
            image_info_out_values = [{}]
        else:
            image_info_out_values = _unwrap_input_list(image_info)

        return c_io.NodeOutput(
            output_refs,
            image_info_out_values,
            saved_paths,
            ui={"images": ui_images},
        )
