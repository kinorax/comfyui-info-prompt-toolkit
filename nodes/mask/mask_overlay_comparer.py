from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

import folder_paths
import numpy as np
import torch
from PIL import Image
from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast
from ._mask_utils import normalize_mask_batch

_PREVIEW_SUBDIR_NAME = "ipt_mask_overlay_comparer"
_PREVIEW_EXT = ".png"
_WASH_STRENGTH = 0.42
_DEFAULT_SPLIT_RATIO = 0.5
_FULL_MASK_EPSILON = 1.0e-6
_SOFT_MASK_RGB = (0.0, 0.45, 1.0)


def _normalized_image_batch(image: Any) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise RuntimeError("image input must be a ComfyUI IMAGE tensor")
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim != 4:
        raise RuntimeError("image input must have shape [B,H,W,C]")
    return image.detach().float().cpu()


def _first_preview_image(image: Any) -> torch.Tensor:
    image_batch = _normalized_image_batch(image)
    if int(image_batch.shape[0]) <= 0:
        raise RuntimeError("image batch is empty")
    return image_batch[0]


def _first_preview_mask(mask: Any) -> torch.Tensor:
    mask_batch = normalize_mask_batch(mask)
    if int(mask_batch.shape[0]) <= 0:
        raise RuntimeError("mask batch is empty")
    return mask_batch[0]


def _ensure_preview_shapes(image: torch.Tensor, mask: torch.Tensor) -> None:
    if image.ndim != 3 or int(image.shape[-1]) < 3:
        raise RuntimeError("image input must have shape [H,W,C] with at least 3 channels")
    if mask.ndim != 2:
        raise RuntimeError("mask input must have shape [H,W]")

    image_height = int(image.shape[0])
    image_width = int(image.shape[1])
    mask_height = int(mask.shape[0])
    mask_width = int(mask.shape[1])
    if image_height != mask_height or image_width != mask_width:
        raise RuntimeError(
            "image and mask must have the same resolution for preview "
            f"(image={image_width}x{image_height}, mask={mask_width}x{mask_height})"
        )


def _washed_image_rgb(image: torch.Tensor) -> torch.Tensor:
    rgb = image[..., :3].clamp(0.0, 1.0)
    return rgb + ((1.0 - rgb) * _WASH_STRENGTH)


def _overlay_image_rgb(washed_rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    clamped_mask = mask.clamp(0.0, 1.0)
    full_mask = clamped_mask >= (1.0 - _FULL_MASK_EPSILON)
    soft_mask = torch.where(full_mask, torch.zeros_like(clamped_mask), clamped_mask)

    soft_alpha = soft_mask.unsqueeze(-1)
    soft_rgb = torch.tensor(
        _SOFT_MASK_RGB,
        dtype=washed_rgb.dtype,
        device=washed_rgb.device,
    ).view(1, 1, 3)
    tinted_rgb = (washed_rgb * (1.0 - soft_alpha)) + (soft_rgb * soft_alpha)

    return torch.where(full_mask.unsqueeze(-1), torch.zeros_like(tinted_rgb), tinted_rgb).clamp(0.0, 1.0)


def _to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().numpy()
    array_uint8 = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array_uint8, mode="RGB")


def _preview_root_and_type() -> tuple[Path, str]:
    get_temp_directory = getattr(folder_paths, "get_temp_directory", None)
    if callable(get_temp_directory):
        return Path(get_temp_directory()).resolve(), "temp"

    get_output_directory = getattr(folder_paths, "get_output_directory", None)
    if callable(get_output_directory):
        return Path(get_output_directory()).resolve(), "output"

    fallback = (Path.cwd() / ".tmp_mask_overlay_comparer").resolve()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback, "output"


def _save_preview_image(image: Image.Image) -> dict[str, str]:
    root_dir, view_type = _preview_root_and_type()
    target_dir = (root_dir / _PREVIEW_SUBDIR_NAME).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"mask_overlay_{uuid.uuid4().hex}{_PREVIEW_EXT}"
    target_path = target_dir / filename
    image.save(target_path, format="PNG")

    try:
        relative_parent = target_dir.relative_to(root_dir).as_posix()
    except Exception:
        relative_parent = _PREVIEW_SUBDIR_NAME

    return {
        "filename": filename,
        "subfolder": "" if relative_parent == "." else relative_parent,
        "type": view_type,
    }


def _preview_payload(image: Any, mask: Any) -> dict[str, object]:
    first_image = _first_preview_image(image)
    first_mask = _first_preview_mask(mask)
    _ensure_preview_shapes(first_image, first_mask)

    washed_rgb = _washed_image_rgb(first_image)
    overlay_rgb = _overlay_image_rgb(washed_rgb, first_mask)
    image_batch = _normalized_image_batch(image)
    mask_batch = normalize_mask_batch(mask)

    batch_note = "Preview uses the first batch item only."
    if int(image_batch.shape[0]) <= 1 and int(mask_batch.shape[0]) <= 1:
        batch_note = "Preview reflects the current image and mask."

    return {
        "base_image": _save_preview_image(_to_pil(washed_rgb)),
        "overlay_image": _save_preview_image(_to_pil(overlay_rgb)),
        "split_ratio": _DEFAULT_SPLIT_RATIO,
        "status": batch_note,
    }


class MaskOverlayComparer(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-MaskOverlayComparer",
            display_name="Mask Overlay Comparer",
            category=Const.CATEGORY_MASK,
            is_output_node=True,
            search_aliases=["image comparer", "mask compare", "overlay compare", "mask viewer"],
            inputs=[
                c_io.Image.Input(
                    "image",
                    tooltip="Background image used for mask preview. When batched, only the first item is shown in the comparer.",
                ),
                c_io.Mask.Input(
                    "mask",
                    tooltip="Soft or binary mask used for preview. Fully masked pixels render as black, while softer values are tinted blue.",
                ),
            ],
            outputs=[
                c_io.Image.Output(
                    Cast.out_id("image"),
                    display_name="image",
                ),
                c_io.Mask.Output(
                    Cast.out_id("mask"),
                    display_name="mask",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: Any,
        mask: Any,
    ) -> c_io.NodeOutput:
        payload = _preview_payload(image, mask)
        return c_io.NodeOutput(
            image,
            mask,
            ui={"mask_overlay_comparer": [payload]},
        )
