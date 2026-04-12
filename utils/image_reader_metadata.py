from __future__ import annotations

import re
from typing import Optional

import folder_paths
import node_helpers
from PIL import Image

from . import clipspace_bridge as ClipspaceBridge
from . import exif as Exif


def is_clipspace_masked_image(path: str) -> bool:
    norm = path.replace("\\", "/").lower()
    return "/clipspace/" in norm and bool(re.search(r"clipspace-painted-masked-\d+\.png$", norm))


def _read_metadata_from_annotated_source(source_annotated: Optional[str]) -> Optional[str]:
    if not source_annotated:
        return None
    if not folder_paths.exists_annotated_filepath(source_annotated):
        return None
    try:
        source_path = folder_paths.get_annotated_filepath(source_annotated)
        source_img = node_helpers.pillow(Image.open, source_path)
        return Exif.extract_a1111_text(source_img)
    except Exception:
        return None


def recover_metadata_for_clipspace(
    image: str,
    image_path: str,
    current_text: Optional[str],
) -> Optional[str]:
    if current_text:
        return current_text
    if not is_clipspace_masked_image(image_path):
        return current_text

    inherited = (
        _read_metadata_from_annotated_source(ClipspaceBridge.get_source_annotated(image))
        or _read_metadata_from_annotated_source(ClipspaceBridge.get_source_annotated(image_path))
    )
    if not inherited:
        return None

    Exif.inject_a1111_text_png(image_path, inherited)
    return inherited


def read_a1111_text_from_image_selection(image: str) -> tuple[str, Optional[str]]:
    image_path = folder_paths.get_annotated_filepath(image)
    img = node_helpers.pillow(Image.open, image_path)
    text = Exif.extract_a1111_text(img)
    text = recover_metadata_for_clipspace(image, image_path, text)
    return image_path, text
