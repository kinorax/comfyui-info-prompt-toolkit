from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
import scipy.ndimage
import torch


@dataclass(frozen=True)
class ConnectedComponent:
    label: int
    area: int
    y0: int
    y1: int
    x0: int
    x1: int


_NEIGHBOR_OFFSETS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

_COMPONENT_STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)
_BLUR_RADIUS_LOG_COEFFICIENT = 0.22
_SOLID_CORE_UI_MAX_SCALE = 2.0
_SOLID_CORE_FILLED_BLUR_RATIO = 0.60
_SOLID_CORE_DENSE_BLUR_RATIO = 0.30
_SOLID_CORE_FILL_BLEND_EXPONENT = 0.50
_TAPERED_KERNEL = np.array(
    (
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 0.0),
    ),
    dtype=np.float32,
)
_FULL_KERNEL = np.ones((3, 3), dtype=np.float32)


def normalize_mask_batch(mask: Any, *, move_to_cpu: bool = True) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise RuntimeError("mask input must be a ComfyUI MASK tensor")

    tensor = mask.detach().float()
    if move_to_cpu:
        tensor = tensor.cpu()
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4 and int(tensor.shape[1]) == 1:
        return tensor.squeeze(1)
    if tensor.ndim != 3:
        raise RuntimeError("mask input must have shape [B,H,W]")
    return tensor


def threshold_mask(mask_batch: torch.Tensor, threshold: float) -> torch.Tensor:
    return mask_batch > float(threshold)


def image_area_percent_to_pixels(height: int, width: int, percent: float) -> int:
    if percent <= 0.0:
        return 0
    pixel_count = int(math.ceil((float(height) * float(width) * float(percent)) / 100.0))
    return max(1, pixel_count)


def connected_components_8(mask: np.ndarray) -> tuple[np.ndarray, list[ConnectedComponent]]:
    if mask.ndim != 2:
        raise RuntimeError("connected_components_8 expects a 2D mask")

    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    components: list[ConnectedComponent] = []
    next_label = 1

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue

            stack = [(y, x)]
            labels[y, x] = next_label
            area = 0
            min_y = max_y = y
            min_x = max_x = x

            while stack:
                current_y, current_x = stack.pop()
                area += 1
                min_y = min(min_y, current_y)
                max_y = max(max_y, current_y)
                min_x = min(min_x, current_x)
                max_x = max(max_x, current_x)

                for offset_y, offset_x in _NEIGHBOR_OFFSETS_8:
                    neighbor_y = current_y + offset_y
                    neighbor_x = current_x + offset_x
                    if neighbor_y < 0 or neighbor_y >= height or neighbor_x < 0 or neighbor_x >= width:
                        continue
                    if not mask[neighbor_y, neighbor_x] or labels[neighbor_y, neighbor_x] != 0:
                        continue
                    labels[neighbor_y, neighbor_x] = next_label
                    stack.append((neighbor_y, neighbor_x))

            components.append(
                ConnectedComponent(
                    label=next_label,
                    area=area,
                    y0=min_y,
                    y1=max_y + 1,
                    x0=min_x,
                    x1=max_x + 1,
                )
            )
            next_label += 1

    return labels, components


def remove_small_regions(
    mask: Any,
    *,
    min_image_area_percent: float,
    threshold: float,
) -> torch.Tensor:
    mask_batch = normalize_mask_batch(mask)
    binary_batch = threshold_mask(mask_batch, float(threshold))
    if min_image_area_percent <= 0.0:
        return binary_batch.float()

    outputs = [
        _remove_small_regions_frame(frame, float(min_image_area_percent))
        for frame in binary_batch
    ]
    return torch.stack(outputs, dim=0)


def remove_small_soft_regions(
    mask: Any,
    *,
    min_image_area_percent: float,
    threshold: float,
) -> torch.Tensor:
    mask_batch = normalize_mask_batch(mask)
    binary_batch = threshold_mask(mask_batch, float(threshold))
    if min_image_area_percent <= 0.0:
        return mask_batch.clone()

    outputs = [
        _remove_small_soft_regions_frame(
            soft_frame,
            binary_frame,
            float(min_image_area_percent),
        )
        for soft_frame, binary_frame in zip(mask_batch, binary_batch)
    ]
    return torch.stack(outputs, dim=0)


def grow_mask(
    mask: Any,
    *,
    threshold: float,
    region_area_scale: float,
    blur_radius_scale: float,
    solid_core_scale: float,
    tapered_corners: bool,
) -> torch.Tensor:
    mask_batch = normalize_mask_batch(mask, move_to_cpu=False)
    binary_batch = threshold_mask(mask_batch, float(threshold))

    outputs = [
        _grow_mask_frame(
            binary_frame,
            region_area_scale=float(region_area_scale),
            blur_radius_scale=float(blur_radius_scale),
            solid_core_scale=float(solid_core_scale),
            tapered_corners=bool(tapered_corners),
        )
        for binary_frame in binary_batch
    ]
    return torch.stack(outputs, dim=0)


def _remove_small_regions_frame(frame: torch.Tensor, min_image_area_percent: float) -> torch.Tensor:
    height = int(frame.shape[0])
    width = int(frame.shape[1])
    min_area_pixels = image_area_percent_to_pixels(height, width, min_image_area_percent)
    if min_area_pixels <= 1:
        return frame.float()

    keep_mask = _keep_component_mask(frame, min_area_pixels)
    if keep_mask is None:
        return frame.float()
    return keep_mask.float()


def _remove_small_soft_regions_frame(
    soft_frame: torch.Tensor,
    binary_frame: torch.Tensor,
    min_image_area_percent: float,
) -> torch.Tensor:
    height = int(binary_frame.shape[0])
    width = int(binary_frame.shape[1])
    min_area_pixels = image_area_percent_to_pixels(height, width, min_image_area_percent)
    if min_area_pixels <= 1:
        return soft_frame.clone()

    keep_mask = _keep_component_mask(binary_frame, min_area_pixels)
    if keep_mask is None:
        return soft_frame.clone()
    return soft_frame * keep_mask.float()


def _keep_component_mask(frame: torch.Tensor, min_area_pixels: int) -> torch.Tensor | None:
    labels, components = connected_components_8(frame.numpy().astype(bool, copy=False))
    if not components:
        return None

    keep_by_label = np.zeros(len(components) + 1, dtype=np.bool_)
    for component in components:
        keep_by_label[component.label] = component.area >= min_area_pixels
    filtered = keep_by_label[labels]
    return torch.from_numpy(filtered.astype(np.bool_, copy=False))


def _grow_mask_frame(
    binary_frame: torch.Tensor,
    *,
    region_area_scale: float,
    blur_radius_scale: float,
    solid_core_scale: float,
    tapered_corners: bool,
) -> torch.Tensor:
    binary_frame_cpu = binary_frame.detach().cpu().numpy().astype(np.bool_, copy=False)
    if not bool(binary_frame_cpu.any()):
        return torch.zeros(binary_frame_cpu.shape, dtype=torch.float32)

    representative_radius = _weighted_median_representative_radius(binary_frame_cpu)
    if representative_radius <= 0.0:
        return torch.zeros(binary_frame_cpu.shape, dtype=torch.float32)

    grow_radius_px = max(
        0,
        int(round(representative_radius * (math.sqrt(float(region_area_scale)) - 1.0))),
    )
    blur_radius = max(
        0.0,
        representative_radius * _scaled_blur_radius_factor(float(blur_radius_scale)),
    )

    soft_halo_binary = _dilate_mask(binary_frame, grow_radius_px, tapered_corners=tapered_corners)
    soft_halo = _blur_mask(soft_halo_binary, blur_radius)

    if blur_radius <= 0.0 or solid_core_scale <= 0.0:
        return soft_halo.clamp_(0.0, 1.0)

    return _apply_solid_core_fill(
        soft_halo,
        soft_halo_binary,
        blur_radius=blur_radius,
        solid_core_scale=float(solid_core_scale),
    )


def _weighted_median_representative_radius(binary_frame: np.ndarray) -> float:
    labels, component_count = scipy.ndimage.label(binary_frame, structure=_COMPONENT_STRUCTURE_8)
    if component_count <= 0:
        return 0.0

    areas = np.bincount(labels.reshape(-1))[1:].astype(np.float64, copy=False)
    if areas.size <= 0:
        return 0.0

    sort_order = np.argsort(areas, kind="stable")
    sorted_areas = areas[sort_order]
    cumulative_weights = np.cumsum(sorted_areas, dtype=np.float64)
    cutoff = cumulative_weights[-1] * 0.5
    area_index = int(np.searchsorted(cumulative_weights, cutoff, side="left"))
    representative_area = float(sorted_areas[min(area_index, sorted_areas.size - 1)])
    if representative_area <= 0.0:
        return 0.0
    return math.sqrt(representative_area / math.pi)


def _scaled_blur_radius_factor(scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return math.log1p(float(scale)) * _BLUR_RADIUS_LOG_COEFFICIENT


def _apply_solid_core_fill(
    soft_halo: torch.Tensor,
    soft_halo_binary: torch.Tensor,
    *,
    blur_radius: float,
    solid_core_scale: float,
) -> torch.Tensor:
    soft_halo_cpu = _to_cpu_mask(soft_halo)
    filled_binary = _fill_mask_holes(soft_halo_binary)
    filled_soft = _blur_mask(
        filled_binary,
        max(0.5, float(blur_radius) * _SOLID_CORE_FILLED_BLUR_RATIO),
    )

    fill_blend = max(0.0, min(1.0, float(solid_core_scale)))
    fill_blend = math.pow(fill_blend, _SOLID_CORE_FILL_BLEND_EXPONENT)
    output = torch.lerp(soft_halo_cpu, filled_soft, fill_blend)

    extra_strength = max(0.0, min(1.0, float(solid_core_scale) - 1.0))
    if extra_strength <= 0.0:
        return output.clamp_(0.0, 1.0)

    dense_core = _blur_mask(
        filled_binary,
        max(0.5, float(blur_radius) * _SOLID_CORE_DENSE_BLUR_RATIO),
    )
    gated_dense_core = torch.minimum(dense_core, filled_soft)
    return (output + ((1.0 - output) * gated_dense_core * extra_strength)).clamp_(0.0, 1.0)


def _fill_mask_holes(frame: torch.Tensor) -> torch.Tensor:
    filled = scipy.ndimage.binary_fill_holes(
        frame.detach().cpu().numpy().astype(np.bool_, copy=False)
    )
    return torch.from_numpy(filled.astype(np.float32, copy=False))


def _dilate_mask(frame: torch.Tensor, steps: int, *, tapered_corners: bool) -> torch.Tensor:
    return _apply_binary_morphology(frame, steps, tapered_corners=tapered_corners, mode="dilate")


def _apply_binary_morphology(
    frame: torch.Tensor,
    steps: int,
    *,
    tapered_corners: bool,
    mode: str,
) -> torch.Tensor:
    if steps <= 0:
        return frame.float().clamp(0.0, 1.0)

    morphology_ops = _get_kornia_morphology_ops()
    if morphology_ops is not None:
        return _apply_kornia_morphology(
            frame,
            steps,
            tapered_corners=tapered_corners,
            mode=mode,
            dilation_op=morphology_ops[0],
            erosion_op=morphology_ops[1],
        )
    return _apply_scipy_morphology(frame, steps, tapered_corners=tapered_corners, mode=mode)


@lru_cache(maxsize=1)
def _get_kornia_morphology_ops() -> tuple[object, object] | None:
    try:
        from kornia.morphology import dilation, erosion

        return dilation, erosion
    except Exception:
        return None


def _apply_kornia_morphology(
    frame: torch.Tensor,
    steps: int,
    *,
    tapered_corners: bool,
    mode: str,
    dilation_op: object,
    erosion_op: object,
) -> torch.Tensor:
    kernel = torch.tensor(
        _select_kernel(tapered_corners),
        dtype=frame.dtype if frame.dtype.is_floating_point else torch.float32,
        device=frame.device,
    )
    work = frame.float().unsqueeze(0).unsqueeze(0)

    if mode == "dilate":
        for _ in range(steps):
            work = dilation_op(work, kernel)
    else:
        raise RuntimeError(f"unsupported morphology mode: {mode}")

    return work.squeeze(0).squeeze(0).clamp(0.0, 1.0)


def _apply_scipy_morphology(
    frame: torch.Tensor,
    steps: int,
    *,
    tapered_corners: bool,
    mode: str,
) -> torch.Tensor:
    footprint = _select_kernel(tapered_corners)
    work = frame.detach().cpu().numpy().astype(np.float32, copy=False)

    if mode == "dilate":
        for _ in range(steps):
            work = scipy.ndimage.grey_dilation(work, footprint=footprint)
    else:
        raise RuntimeError(f"unsupported morphology mode: {mode}")

    return torch.from_numpy(np.clip(work, 0.0, 1.0).astype(np.float32, copy=False))


def _select_kernel(tapered_corners: bool) -> np.ndarray:
    return _TAPERED_KERNEL if tapered_corners else _FULL_KERNEL


def _blur_mask(frame: torch.Tensor, blur_radius: float) -> torch.Tensor:
    mask_cpu = _to_cpu_mask(frame)
    if blur_radius <= 0.0:
        return mask_cpu

    image = Image.fromarray(
        np.clip(mask_cpu.numpy() * 255.0, 0.0, 255.0).astype(np.uint8),
        mode="L",
    )
    blurred = image.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    blurred_array = np.asarray(blurred, dtype=np.float32) / 255.0
    return torch.from_numpy(blurred_array.copy()).clamp_(0.0, 1.0)


def _to_cpu_mask(frame: torch.Tensor) -> torch.Tensor:
    return frame.detach().float().cpu().clamp(0.0, 1.0)
