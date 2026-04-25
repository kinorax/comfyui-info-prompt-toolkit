# Copyright 2026 kinorax
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import folder_paths
import numpy as np
import torch
from PIL import Image
from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils.a1111_infotext import image_info_to_a1111_infotext
from ..utils.image_info_hash_extras import (
    add_civitai_hash_extras,
    clear_representative_hash_extras,
)
from ..utils import cast as Cast
from ..utils import exif as Exif

_OUTPUT_SUBDIR_OPTIONS = ("none", "year", "year_month", "iso_week", "year_month_day")
_WEBP_METHOD = 6
_WEBP_LOSSLESS_QUALITY = 80
_WEBP_EXT = ".webp"
_MISSING = object()
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
        return [None]
    return values


def _split_images_from_input(image_input: Any) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    image_values = _unwrap_input_list(image_input)
    if len(image_values) == 0:
        raise ValueError("image is required")

    images: list[torch.Tensor] = []
    merged_tensors: list[torch.Tensor] = []
    merged_shape: tuple[int, ...] | None = None
    can_merge = True

    for value in image_values:
        if not isinstance(value, torch.Tensor):
            raise ValueError("image input must be IMAGE")
        if value.ndim == 3:
            batch = value.unsqueeze(0)
        elif value.ndim == 4:
            batch = value
        else:
            raise ValueError(f"unsupported image tensor shape: {tuple(value.shape)}")

        for idx in range(batch.shape[0]):
            images.append(batch[idx])

        if not can_merge:
            continue

        batch_shape = tuple(batch.shape[1:])
        if merged_shape is None:
            merged_shape = batch_shape
            merged_tensors.append(batch)
            continue
        if batch_shape == merged_shape:
            merged_tensors.append(batch)
            continue

        # Keep per-image items even when mixed resolutions are provided.
        # IMAGE outputs in ComfyUI lists can contain tensors with different H/W.
        can_merge = False
        merged_tensors.clear()

    merged_batch: torch.Tensor | None = None
    if can_merge and len(merged_tensors) > 0:
        if len(merged_tensors) == 1:
            merged_batch = merged_tensors[0]
        else:
            merged_batch = torch.cat(merged_tensors, dim=0)

    return images, merged_batch


def _resolve_output_root() -> Path:
    return Path(folder_paths.get_output_directory()).resolve()


def _resolve_output_dir(output_dir: str | None, output_root: Path) -> Path:
    text = (output_dir or "").strip()
    base = Path(text) if text else output_root
    if not base.is_absolute():
        base = output_root / base
    resolved = base.resolve()

    try:
        common = Path(os.path.commonpath((str(output_root), str(resolved))))
    except ValueError as exc:
        raise ValueError("output_dir must be under ComfyUI output directory") from exc
    if common != output_root:
        raise ValueError("output_dir must be under ComfyUI output directory")
    return resolved


def _resolve_subdir(now: datetime, output_subdir: str) -> str:
    if output_subdir == "none":
        return ""
    if output_subdir == "year":
        return now.strftime("%Y")
    if output_subdir == "year_month":
        return now.strftime("%Y%m")
    if output_subdir == "year_month_day":
        return now.strftime("%Y%m%d")
    if output_subdir == "iso_week":
        iso = now.isocalendar()
        return f"{iso.year}W{iso.week:02d}"
    raise ValueError(f"unsupported output_subdir: {output_subdir}")


def _find_next_counter(folder: Path, date_prefix: str) -> int:
    current_max = 0
    expected_prefix = f"{date_prefix}-"
    for entry in os.scandir(folder):
        if not entry.is_file():
            continue

        name = entry.name
        if not name.startswith(expected_prefix):
            continue
        if not name.lower().endswith(_WEBP_EXT):
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


def _resolve_per_image_values(values: list[Any], count: int, name: str) -> list[Any]:
    if len(values) == 0:
        return [None] * count
    if len(values) == 1:
        return [values[0]] * count
    if len(values) == count:
        return values
    raise ValueError(f"{name} must have length 1 or {count}, got {len(values)}")


def _safe_filename_suffix(value: Any) -> str:
    suffix = str(value or "").strip()
    if "/" in suffix or "\\" in suffix:
        raise ValueError("filename_suffix must not contain path separators")
    return suffix


def _normalize_optional_file_stem_values(raw: Any) -> list[Any] | None:
    values = _unwrap_input_list(raw)
    if len(values) == 0:
        return None
    if len(values) == 1 and values[0] is None:
        return None
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


def _resolve_forced_file_stems(raw: Any, image_count: int) -> list[str] | None:
    values = _normalize_optional_file_stem_values(raw)
    if values is None:
        return None

    if image_count <= 1:
        if len(values) != 1:
            raise ValueError(f"file_stem must have length 1, got {len(values)}")
    elif len(values) != image_count:
        raise ValueError(f"file_stem must have length {image_count} for batch save, got {len(values)}")

    stems: list[str] = []
    seen: set[str] = set()
    for raw_stem in values:
        stem = _validate_file_stem(raw_stem)
        key = stem.casefold()
        if key in seen:
            raise ValueError(f"duplicate output file path in batch: {stem}{_WEBP_EXT}")
        seen.add(key)
        stems.append(stem)
    return stems


def _build_infotext(image_info: Any) -> str:
    if not isinstance(image_info, Mapping):
        return ""
    image_info_without_hashes = clear_representative_hash_extras(image_info)
    image_info_with_hashes = add_civitai_hash_extras(image_info_without_hashes)
    return image_info_to_a1111_infotext(image_info_with_hashes)


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


def _to_pil(image: torch.Tensor) -> Image.Image:
    image_np = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    return Image.fromarray(image_np)


def _render_file_stem(date_prefix: str, counter: int, suffix: str) -> str:
    return f"{date_prefix}-{counter:05d}-{suffix}"


def _relative_to_output_root(path: Path, output_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(output_root)
    except Exception as exc:
        raise ValueError("saved file path is outside ComfyUI output directory") from exc
    return rel.as_posix()


def _build_webp_save_kwargs(quality: int, infotext: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "method": _WEBP_METHOD,
        "quality": int(quality),
    }
    if quality == 100:
        kwargs["lossless"] = True
        kwargs["quality"] = _WEBP_LOSSLESS_QUALITY

    if infotext:
        user_comment = b"UNICODE\x00" + infotext.encode("utf-16-be")
        exif = Image.Exif()
        exif[Exif.EXIF_USERCOMMENT_TAG] = user_comment
        kwargs["exif"] = exif.tobytes()
    return kwargs


class ImageSaver(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-ImageSaver",
            display_name="Image Saver",
            category=Const.CATEGORY_IMAGEINFO,
            is_output_node=True,
            is_input_list=True,
            inputs=[
                c_io.Image.Input(
                    "image",
                    tooltip="Image or image batch to save",
                ),
                Const.IMAGEINFO_TYPE.Input(
                    Const.IMAGEINFO,
                    optional=True,
                    tooltip="Image info or image-info list for each image",
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
                    tooltip="Optional output filename stem list; when connected, save as <file_stem>.webp",
                ),
                c_io.String.Input(
                    "caption",
                    default="",
                    optional=True,
                    force_input=True,
                    tooltip="Caption text or caption list for each image",
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
                c_io.Int.Input(
                    "quality",
                    default=100,
                    min=0,
                    max=100,
                    tooltip="WebP quality",
                ),
                c_io.Boolean.Input(
                    "write_caption",
                    default=False,
                    tooltip="Write caption txt file",
                ),
            ],
            outputs=[
                c_io.Image.Output(
                    Cast.out_id("image"),
                    display_name="image",
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
        image: Any,
        image_info: Any = None,
        filename_suffix: Any = _MISSING,
        file_stem: Any = None,
        caption: Any = None,
        output_dir: Any = _MISSING,
        output_subdir: Any = _MISSING,
        quality: Any = _MISSING,
        write_caption: Any = _MISSING,
    ) -> c_io.NodeOutput:
        image_batch_items, _ = _split_images_from_input(image)
        image_count = len(image_batch_items)
        if image_count == 0:
            raise ValueError("image is required")

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
        if quality is _MISSING:
            raise ValueError("quality is required")
        if write_caption is _MISSING:
            raise ValueError("write_caption is required")

        output_dir_value = _resolve_single_input(output_dir, "output_dir")
        output_subdir_value = str(_resolve_single_input(output_subdir, "output_subdir"))
        quality_value = int(_resolve_single_input(quality, "quality"))
        write_caption_value = bool(_resolve_single_input(write_caption, "write_caption"))
        suffix = ""
        if forced_file_stems is None:
            suffix = _safe_filename_suffix(_resolve_single_input(filename_suffix, "filename_suffix"))

        if quality_value < 0 or quality_value > 100:
            raise ValueError("quality must be in range 0..100")
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

        ui_images: list[dict[str, str]] = []
        saved_paths: list[str] = []

        for idx, image_item in enumerate(image_batch_items):
            if forced_file_stems is None:
                stem = _render_file_stem(date_prefix, counter, suffix)
            else:
                stem = forced_file_stems[idx]
            image_path = target_dir / f"{stem}{_WEBP_EXT}"
            info_item = image_info_mapped[idx]
            infotext = _build_infotext(info_item)

            img = _to_pil(image_item)
            save_kwargs = _build_webp_save_kwargs(quality_value, infotext)
            try:
                img.save(image_path, **save_kwargs)
            except Exception as exc:
                raise RuntimeError(f"failed to save image: {image_path}") from exc

            if infotext:
                try:
                    with Image.open(image_path) as reloaded:
                        loaded_infotext = Exif.extract_a1111_text(reloaded)
                    if loaded_infotext is None:
                        raise RuntimeError("missing infotext in saved webp metadata")
                except Exception as exc:
                    raise RuntimeError(f"failed to validate infotext metadata: {image_path}") from exc

            if write_caption_value:
                caption_text = _resolve_caption_text(caption_mapped[idx], info_item)
                caption_path = target_dir / f"{stem}.txt"
                caption_path.write_text(caption_text, encoding="utf-8")

            rel_file_path = _relative_to_output_root(image_path, output_root)
            saved_paths.append(rel_file_path)

            rel_parent = image_path.parent.resolve().relative_to(output_root).as_posix()
            ui_images.append(
                {
                    "filename": image_path.name,
                    "subfolder": "" if rel_parent == "." else rel_parent,
                    "type": "output",
                }
            )
            if forced_file_stems is None:
                counter += 1

        image_out_values = _unwrap_input_list(image)
        if image_info is None:
            image_info_out_values = [{}]
        else:
            image_info_out_values = _unwrap_input_list(image_info)

        return c_io.NodeOutput(
            image_out_values,
            image_info_out_values,
            saved_paths,
            ui={"images": ui_images},
        )
