# Copyright 2026 kinorax
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import folder_paths
import node_helpers
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils import exif as Exif
from ..utils.a1111_infotext import a1111_infotext_to_image_info
from ..utils.image_info_normalizer import normalize_image_info_with_comfy_options

_PATH_SOURCE_OPTIONS = ("input", "output")


def _resolve_base_directory(path_source: str) -> tuple[Path, str]:
    normalized = str(path_source or "").strip().lower()
    if normalized == "input":
        return Path(folder_paths.get_input_directory()).resolve(), "ComfyUI input directory"
    if normalized == "output":
        return Path(folder_paths.get_output_directory()).resolve(), "ComfyUI output directory"
    raise ValueError("path_source must be one of: input, output")


def _list_directory_options(path_source: str) -> list[str]:
    base_dir, _ = _resolve_base_directory(path_source)
    discovered: set[str] = {"."}

    for root, dirs, _ in os.walk(base_dir, topdown=True, followlinks=False):
        root_path = Path(root).resolve()

        try:
            root_common = Path(os.path.commonpath((str(base_dir), str(root_path))))
        except ValueError:
            continue
        if root_common != base_dir:
            continue

        rel_root = root_path.relative_to(base_dir).as_posix()
        if rel_root not in ("", "."):
            discovered.add(rel_root)

        safe_dirs: list[str] = []
        for dir_name in dirs:
            candidate_path = (root_path / dir_name).resolve()
            try:
                common = Path(os.path.commonpath((str(base_dir), str(candidate_path))))
            except ValueError:
                continue
            if common != base_dir:
                continue

            rel_dir = candidate_path.relative_to(base_dir).as_posix()
            if rel_dir not in ("", "."):
                discovered.add(rel_dir)
            safe_dirs.append(dir_name)

        dirs[:] = safe_dirs

    return sorted(discovered, key=lambda item: (item != ".", item.casefold()))


def _resolve_target_directory(path: str, path_source: str) -> Path:
    base_dir, base_label = _resolve_base_directory(path_source)
    normalized = (path or "").strip()

    if normalized and Path(normalized).is_absolute():
        raise ValueError(f"path must be relative to {base_label}")

    if normalized in ("", "."):
        target = base_dir
    else:
        target = (base_dir / normalized).resolve()

    try:
        common = Path(os.path.commonpath((str(base_dir), str(target))))
    except ValueError as exc:
        raise ValueError(f"path must be under {base_label}") from exc
    if common != base_dir:
        raise ValueError(f"path must be under {base_label}")

    if not target.exists():
        raise ValueError(f"directory not found: {normalized or '.'}")
    if not target.is_dir():
        raise ValueError(f"path is not a directory: {normalized or '.'}")
    return target


def _list_image_files(target_dir: Path) -> list[Path]:
    names = [entry.name for entry in target_dir.iterdir() if entry.is_file()]
    image_names = folder_paths.filter_files_content_types(names, ["image"])
    return [target_dir / name for name in sorted(image_names, key=str.casefold)]


def _slice_image_files(
    image_files: list[Path],
    start_index: int,
    image_load_limit: int,
) -> list[Path]:
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if image_load_limit < 0:
        raise ValueError("image_load_limit must be >= 0")

    if start_index >= len(image_files):
        return []

    sliced = image_files[start_index:]
    if image_load_limit > 0:
        sliced = sliced[:image_load_limit]
    return sliced


def _read_image_and_info(image_path: Path) -> tuple[torch.Tensor, Any]:
    img = node_helpers.pillow(Image.open, str(image_path))

    a1111_text = Exif.extract_a1111_text(img)
    image_info = a1111_infotext_to_image_info(a1111_text)
    image_info = normalize_image_info_with_comfy_options(image_info)

    frames: list[torch.Tensor] = []
    width, height = None, None

    for frame in ImageSequence.Iterator(img):
        frame = node_helpers.pillow(ImageOps.exif_transpose, frame)

        if frame.mode == "I":
            frame = frame.point(lambda i: i * (1 / 255))

        rgb = frame.convert("RGB")

        if len(frames) == 0:
            width, height = rgb.size
        if rgb.size[0] != width or rgb.size[1] != height:
            continue

        image_np = np.array(rgb).astype(np.float32) / 255.0
        frames.append(torch.from_numpy(image_np)[None,])

        if getattr(img, "format", None) == "MPO":
            break

    if len(frames) == 0:
        raise ValueError("no valid image frames")
    if len(frames) == 1:
        return frames[0], image_info
    return torch.cat(frames, dim=0), image_info


def _build_selection(path: str, path_source: str, image_load_limit: int, start_index: int) -> list[Path]:
    target_dir = _resolve_target_directory(path, path_source)
    image_files = _list_image_files(target_dir)
    selected = _slice_image_files(image_files, start_index, image_load_limit)
    return selected


def _normalize_path_source_value(path_source: str) -> str:
    normalized = str(path_source or "").strip().lower()
    if normalized == "output":
        return "output"
    if normalized == "input":
        return "input"
    raise ValueError("path_source must be one of: input, output")


def _normalize_path_value(path: str) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    if normalized in ("", "."):
        return "."
    return normalized


def _resolve_caption_path(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def _read_caption_text(image_path: Path) -> str:
    caption_path = _resolve_caption_path(image_path)
    if not caption_path.is_file():
        return ""
    return caption_path.read_text(encoding="utf-8-sig", errors="replace")


def _update_digest_with_caption_state(digest: Any, image_path: Path) -> None:
    caption_path = _resolve_caption_path(image_path)
    if not caption_path.is_file():
        digest.update(b"caption:none")
        return

    stat = caption_path.stat()
    digest.update(b"caption:file")
    digest.update(caption_path.name.encode("utf-8"))
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))


class ImageDirectoryReader(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        path_options = _list_directory_options("input")
        default_path = path_options[0] if path_options else "."

        return c_io.Schema(
            node_id="IPT-ImageDirectoryReader",
            display_name="Image Directory Reader",
            category=Const.CATEGORY_IMAGEINFO,
            description="Load images and optional sidecar captions from input/output subdirectories as list outputs.",
            search_aliases=[
                "image directory reader",
                "image directory",
                "batch image reader",
                "image batch reader",
                "image folder reader",
                "load images from directory",
            ],
            inputs=[
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
                c_io.Int.Input(
                    "image_load_limit",
                    default=0,
                    min=0,
                    tooltip="0 = unlimited",
                ),
                c_io.Int.Input(
                    "start_index",
                    default=0,
                    min=0,
                    tooltip="Sorted file index to start from",
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
                    Cast.out_id("file_stem"),
                    display_name="file_stem",
                    is_output_list=True,
                ),
                c_io.String.Output(
                    Cast.out_id("caption"),
                    display_name="caption",
                    is_output_list=True,
                ),
                c_io.AnyType.Output(
                    Cast.out_id("path_source"),
                    display_name="path_source",
                ),
                c_io.AnyType.Output(
                    Cast.out_id("path"),
                    display_name="path",
                ),
            ],
        )

    @classmethod
    def execute(cls, path: str, path_source: str, image_load_limit: int, start_index: int) -> c_io.NodeOutput:
        selected = _build_selection(path, str(path_source), int(image_load_limit), int(start_index))
        if len(selected) == 0:
            raise ValueError("no image files found for current path/start_index/image_load_limit")

        images: list[torch.Tensor] = []
        image_infos: list[Any] = []
        file_stems: list[str] = []
        captions: list[str] = []

        for image_path in selected:
            try:
                image_t, image_info = _read_image_and_info(image_path)
            except Exception as exc:
                raise RuntimeError(f"failed to read image: {image_path.name}") from exc

            try:
                caption = _read_caption_text(image_path)
            except Exception as exc:
                raise RuntimeError(f"failed to read caption: {_resolve_caption_path(image_path).name}") from exc

            images.append(image_t)
            image_infos.append(image_info)
            file_stems.append(image_path.stem)
            captions.append(caption)

        normalized_path_source = _normalize_path_source_value(path_source)
        normalized_path = _normalize_path_value(path)
        return c_io.NodeOutput(images, image_infos, file_stems, captions, normalized_path_source, normalized_path)

    @classmethod
    def fingerprint_inputs(cls, path: str, path_source: str, image_load_limit: int, start_index: int):
        selected = _build_selection(path, str(path_source), int(image_load_limit), int(start_index))
        digest = hashlib.sha256()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(path_source).encode("utf-8"))
        digest.update(str(image_load_limit).encode("utf-8"))
        digest.update(str(start_index).encode("utf-8"))
        for image_path in selected:
            stat = image_path.stat()
            digest.update(image_path.name.encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
            _update_digest_with_caption_state(digest, image_path)
        return digest.hexdigest()

    @classmethod
    def validate_inputs(cls, path: str, path_source: str, image_load_limit: int, start_index: int):
        try:
            selected = _build_selection(path, str(path_source), int(image_load_limit), int(start_index))
        except Exception as exc:
            return str(exc)
        if len(selected) == 0:
            return "no image files found for current path/start_index/image_load_limit"
        return True
