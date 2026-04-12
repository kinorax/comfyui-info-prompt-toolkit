from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from ._mask_utils import normalize_mask_batch

_DETAILER_CONTROL_VERSION = 1
_EMPTY_MASK_EPSILON = 0.0


def normalize_image_batch(image: Any, *, move_to_cpu: bool = True) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise RuntimeError("image input must be a ComfyUI IMAGE tensor")

    tensor = image.detach().float()
    if move_to_cpu:
        tensor = tensor.cpu()
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise RuntimeError("image input must have shape [B,H,W,C]")
    return tensor


def prepare_detailer_batch(
    image: Any,
    mask: Any,
    *,
    crop_margin_scale: float,
    upscale_factor: float,
    mini_unit: int,
    detailer_index: int = 0,
    open_node_id: str | None = None,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    image_batch, mask_batch = _aligned_image_and_mask_batches(image, mask)
    batch_size = int(image_batch.shape[0])
    image_height = int(image_batch.shape[1])
    image_width = int(image_batch.shape[2])
    channels = int(image_batch.shape[3])
    mini_unit_value = max(1, int(mini_unit))

    regions: list[dict[str, Any]] = []
    for batch_index in range(batch_size):
        soft_mask = mask_batch[batch_index].clamp(0.0, 1.0)
        crop_box = _crop_box_for_mask(
            soft_mask,
            image_width=image_width,
            image_height=image_height,
            crop_margin_scale=float(crop_margin_scale),
        )
        if crop_box is None:
            continue

        x0, y0, x1, y1 = crop_box
        crop_width = x1 - x0
        crop_height = y1 - y0
        scaled_width = max(1, int(math.ceil(float(crop_width) * float(upscale_factor))))
        scaled_height = max(1, int(math.ceil(float(crop_height) * float(upscale_factor))))
        canvas_width = _round_up(scaled_width, mini_unit_value)
        canvas_height = _round_up(scaled_height, mini_unit_value)
        content_y0 = max(0, (canvas_height - scaled_height) // 2)
        content_x0 = max(0, (canvas_width - scaled_width) // 2)
        content_y1 = content_y0 + scaled_height
        content_x1 = content_x0 + scaled_width

        regions.append(
            {
                "batch_index": int(batch_index),
                "crop_box": (int(x0), int(y0), int(x1), int(y1)),
                "crop_width": int(crop_width),
                "crop_height": int(crop_height),
                "scaled_width": int(scaled_width),
                "scaled_height": int(scaled_height),
                "canvas_width": int(canvas_width),
                "canvas_height": int(canvas_height),
                "content_box": (int(content_x0), int(content_y0), int(content_x1), int(content_y1)),
                "soft_mask": soft_mask[y0:y1, x0:x1].clone(),
            }
        )

    original_image = image_batch.clone()
    current_index = 0
    if len(regions) > 0:
        current_index = max(0, min(int(detailer_index), len(regions) - 1))

    control: dict[str, Any] = {
        "version": _DETAILER_CONTROL_VERSION,
        "enabled": len(regions) > 0,
        "open_node_id": str(open_node_id) if open_node_id is not None else None,
        "original_image": original_image,
        "original_batch_size": batch_size,
        "original_height": image_height,
        "original_width": image_width,
        "image_channels": channels,
        "active_indices": [int(region["batch_index"]) for region in regions],
        "active_count": len(regions),
        "current_index": current_index,
        "regions": regions,
    }

    if len(regions) <= 0:
        placeholder_image = torch.zeros((1, mini_unit_value, mini_unit_value, channels), dtype=torch.float32)
        placeholder_mask = torch.zeros((1, mini_unit_value, mini_unit_value), dtype=torch.float32)
        return control, placeholder_image, placeholder_mask

    output_image, output_mask = _render_detailer_region(
        image_batch,
        regions[current_index],
        channels,
    )
    return control, output_image, output_mask


def restore_detailer_batch(
    control: Any,
    inpainted_image: Any,
    *,
    accumulated_image: Any = None,
) -> torch.Tensor:
    normalized_control = normalize_detailer_control(control)
    output_image = _base_detailer_image(normalized_control, accumulated_image)
    regions = normalized_control["regions"]

    if not bool(normalized_control["enabled"]) or len(regions) <= 0:
        return output_image

    current_index = max(0, min(int(normalized_control["current_index"]), len(regions) - 1))
    region = regions[current_index]
    expected_height = int(region["canvas_height"])
    expected_width = int(region["canvas_width"])
    expected_channels = int(normalized_control["image_channels"])

    image_batch = normalize_image_batch(inpainted_image)
    if int(image_batch.shape[0]) != 1:
        raise RuntimeError(
            "inpainted_image must contain exactly one detailer crop "
            f"(actual batch={int(image_batch.shape[0])})"
        )
    if int(image_batch.shape[-1]) != expected_channels:
        raise RuntimeError(
            "inpainted_image channels must match the original image "
            f"(expected={expected_channels}, actual={int(image_batch.shape[-1])})"
        )

    if int(image_batch.shape[1]) != expected_height or int(image_batch.shape[2]) != expected_width:
        resized_frames = [
            _resize_image_frame(frame, expected_height, expected_width)
            for frame in image_batch
        ]
        image_batch = torch.stack(resized_frames, dim=0)

    content_x0, content_y0, content_x1, content_y1 = region["content_box"]
    crop_x0, crop_y0, crop_x1, crop_y1 = region["crop_box"]
    crop_width = int(region["crop_width"])
    crop_height = int(region["crop_height"])
    batch_index = int(region["batch_index"])

    content_frame = image_batch[0, content_y0:content_y1, content_x0:content_x1, :]
    restored_frame = _resize_image_frame(content_frame, crop_height, crop_width)

    soft_mask = region["soft_mask"].float().clamp(0.0, 1.0).unsqueeze(-1)
    original_crop = output_image[batch_index, crop_y0:crop_y1, crop_x0:crop_x1, :]
    blended = (original_crop * (1.0 - soft_mask)) + (restored_frame * soft_mask)
    output_image[batch_index, crop_y0:crop_y1, crop_x0:crop_x1, :] = blended.clamp(0.0, 1.0)
    return output_image


def normalize_detailer_control(control: Any) -> dict[str, Any]:
    if not isinstance(control, dict):
        raise RuntimeError("detailer_control must be a detailer control object")
    if int(control.get("version", 0)) != _DETAILER_CONTROL_VERSION:
        raise RuntimeError("detailer_control version is unsupported")
    if not isinstance(control.get("original_image"), torch.Tensor):
        raise RuntimeError("detailer_control.original_image is required")
    if not isinstance(control.get("regions"), list):
        raise RuntimeError("detailer_control.regions is required")
    if not isinstance(control.get("active_count"), int):
        raise RuntimeError("detailer_control.active_count is required")
    if not isinstance(control.get("current_index"), int):
        raise RuntimeError("detailer_control.current_index is required")
    return control


def detailer_requires_inpainted_image(control: Any) -> bool:
    normalized_control = normalize_detailer_control(control)
    return bool(normalized_control.get("enabled")) and len(normalized_control.get("regions", [])) > 0


def detailer_has_next_item(control: Any) -> bool:
    normalized_control = normalize_detailer_control(control)
    if not bool(normalized_control.get("enabled")):
        return False
    current_index = int(normalized_control.get("current_index", 0))
    active_count = int(normalized_control.get("active_count", 0))
    return current_index + 1 < active_count


def _aligned_image_and_mask_batches(image: Any, mask: Any) -> tuple[torch.Tensor, torch.Tensor]:
    image_batch = normalize_image_batch(image)
    mask_batch = normalize_mask_batch(mask)

    image_batch_size = int(image_batch.shape[0])
    mask_batch_size = int(mask_batch.shape[0])
    if image_batch_size != mask_batch_size:
        if image_batch_size == 1:
            image_batch = image_batch.repeat(mask_batch_size, 1, 1, 1)
        elif mask_batch_size == 1:
            mask_batch = mask_batch.repeat(image_batch_size, 1, 1)
        else:
            raise RuntimeError(
                "image and mask batch sizes must match, or one of them must have batch size 1 "
                f"(image={image_batch_size}, mask={mask_batch_size})"
            )

    image_height = int(image_batch.shape[1])
    image_width = int(image_batch.shape[2])
    mask_height = int(mask_batch.shape[1])
    mask_width = int(mask_batch.shape[2])
    if image_height != mask_height or image_width != mask_width:
        raise RuntimeError(
            "image and mask must have the same resolution "
            f"(image={image_width}x{image_height}, mask={mask_width}x{mask_height})"
        )
    return image_batch, mask_batch


def _crop_box_for_mask(
    mask_frame: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
    crop_margin_scale: float,
) -> tuple[int, int, int, int] | None:
    foreground = mask_frame > _EMPTY_MASK_EPSILON
    if not bool(foreground.any()):
        return None

    coordinates = torch.nonzero(foreground, as_tuple=False)
    y0 = int(coordinates[:, 0].min().item())
    y1 = int(coordinates[:, 0].max().item()) + 1
    x0 = int(coordinates[:, 1].min().item())
    x1 = int(coordinates[:, 1].max().item()) + 1
    return _expanded_crop_box(
        x0,
        y0,
        x1,
        y1,
        image_width=image_width,
        image_height=image_height,
        crop_margin_scale=float(crop_margin_scale),
    )


def _expanded_crop_box(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    image_width: int,
    image_height: int,
    crop_margin_scale: float,
) -> tuple[int, int, int, int]:
    crop_width = max(1, int(x1) - int(x0))
    crop_height = max(1, int(y1) - int(y0))
    target_width = max(crop_width, int(math.ceil(float(crop_width) * float(crop_margin_scale))))
    target_height = max(crop_height, int(math.ceil(float(crop_height) * float(crop_margin_scale))))

    center_x = (float(x0) + float(x1)) * 0.5
    center_y = (float(y0) + float(y1)) * 0.5
    expanded_x0, expanded_x1 = _fit_centered_interval(center_x, target_width, image_width)
    expanded_y0, expanded_y1 = _fit_centered_interval(center_y, target_height, image_height)
    return expanded_x0, expanded_y0, expanded_x1, expanded_y1


def _fit_centered_interval(center: float, size: int, limit: int) -> tuple[int, int]:
    interval_size = max(1, min(int(size), int(limit)))
    start = int(math.floor(float(center) - (float(interval_size) * 0.5)))
    end = start + interval_size
    if start < 0:
        end = min(limit, end - start)
        start = 0
    if end > limit:
        start = max(0, start - (end - limit))
        end = limit
    return int(start), int(end)


def _round_up(value: int, unit: int) -> int:
    rounded_unit = max(1, int(unit))
    rounded_value = max(1, int(value))
    return int(math.ceil(float(rounded_value) / float(rounded_unit)) * rounded_unit)


def _render_detailer_region(
    image_batch: torch.Tensor,
    region: dict[str, Any],
    channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0, y0, x1, y1 = region["crop_box"]
    scaled_height = int(region["scaled_height"])
    scaled_width = int(region["scaled_width"])
    canvas_height = int(region["canvas_height"])
    canvas_width = int(region["canvas_width"])
    content_x0, content_y0, content_x1, content_y1 = region["content_box"]

    crop_image = image_batch[int(region["batch_index"]), y0:y1, x0:x1, :].clone()
    crop_soft_mask = region["soft_mask"].clone()
    crop_binary_mask = (crop_soft_mask > _EMPTY_MASK_EPSILON).float()

    resized_image = _resize_image_frame(crop_image, scaled_height, scaled_width)
    resized_mask = _resize_mask_frame(crop_binary_mask, scaled_height, scaled_width, mode="nearest")

    canvas_image = torch.zeros((canvas_height, canvas_width, channels), dtype=torch.float32)
    canvas_mask = torch.zeros((canvas_height, canvas_width), dtype=torch.float32)
    canvas_image[content_y0:content_y1, content_x0:content_x1, :] = resized_image
    canvas_mask[content_y0:content_y1, content_x0:content_x1] = resized_mask
    return canvas_image.unsqueeze(0), canvas_mask.unsqueeze(0)


def _base_detailer_image(
    control: dict[str, Any],
    accumulated_image: Any,
) -> torch.Tensor:
    original_image = control["original_image"].clone()
    if accumulated_image is None:
        return original_image

    accumulated_batch = normalize_image_batch(accumulated_image)
    expected_shape = tuple(int(v) for v in original_image.shape)
    actual_shape = tuple(int(v) for v in accumulated_batch.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            "accumulated_image must match the original image batch shape "
            f"(expected={expected_shape}, actual={actual_shape})"
        )
    return accumulated_batch.clone()


def _resize_image_frame(frame: torch.Tensor, height: int, width: int) -> torch.Tensor:
    height_value = max(1, int(height))
    width_value = max(1, int(width))
    if int(frame.shape[0]) == height_value and int(frame.shape[1]) == width_value:
        return frame.detach().float().cpu().clamp(0.0, 1.0)

    work = frame.detach().float().permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(
        work,
        size=(height_value, width_value),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).permute(1, 2, 0).cpu().clamp(0.0, 1.0)


def _resize_mask_frame(frame: torch.Tensor, height: int, width: int, *, mode: str) -> torch.Tensor:
    height_value = max(1, int(height))
    width_value = max(1, int(width))
    if int(frame.shape[0]) == height_value and int(frame.shape[1]) == width_value:
        return frame.detach().float().cpu().clamp(0.0, 1.0)

    work = frame.detach().float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        work,
        size=(height_value, width_value),
        mode=mode,
    )
    return resized.squeeze(0).squeeze(0).cpu().clamp(0.0, 1.0)
