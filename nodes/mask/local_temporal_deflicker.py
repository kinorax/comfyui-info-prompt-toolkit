# Copyright 2026 kinorax
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast

_DEFAULT_STRENGTH = 0.7
_TEMPORAL_RADIUS = 2
_WORK_MAX_SIDE = 256
_FLOW_LEVELS = 4
_FLOW_ITERATIONS = 3
_FLOW_PAIR_CHUNK = 4
_MAX_CORRECTION_STOPS = 0.20
_MAX_CORRECTION_LOG = math.log(2.0) * _MAX_CORRECTION_STOPS
_EPSILON = 1.0e-6


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        output: list[Any] = []
        for item in value:
            output.extend(_flatten(item))
        return output
    return [value]


def _split_image_frames(value: Any, *, input_name: str) -> list[torch.Tensor]:
    output: list[torch.Tensor] = []
    for item in _flatten(value):
        if not isinstance(item, torch.Tensor):
            raise ValueError(f"{input_name} must contain IMAGE tensors")
        if item.ndim == 3:
            item = item.unsqueeze(0)
        if item.ndim != 4:
            raise ValueError(f"{input_name} items must have shape [B,H,W,C]")
        if int(item.shape[-1]) < 3:
            raise ValueError(f"{input_name} items must have at least three color channels")
        output.extend(item[index:index + 1] for index in range(int(item.shape[0])))
    if not output:
        raise ValueError(f"{input_name} is required")
    return output


def _split_mask_frames(value: Any) -> list[torch.Tensor]:
    output: list[torch.Tensor] = []
    for item in _flatten(value):
        if not isinstance(item, torch.Tensor):
            raise ValueError("masks must contain MASK tensors")
        tensor = item
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4 and int(tensor.shape[1]) == 1:
            tensor = tensor.squeeze(1)
        elif tensor.ndim == 4 and int(tensor.shape[-1]) == 1:
            tensor = tensor.squeeze(-1)
        if tensor.ndim != 3:
            raise ValueError("masks items must have shape [B,H,W]")
        output.extend(tensor[index:index + 1] for index in range(int(tensor.shape[0])))
    if not output:
        raise ValueError("masks is required")
    return output


def _read_strength(value: Any) -> float:
    values = _flatten(value)
    if not values:
        return _DEFAULT_STRENGTH
    try:
        strength = float(values[0])
    except Exception as exc:
        raise ValueError("strength must be a number") from exc
    if not math.isfinite(strength):
        raise ValueError("strength must be finite")
    return max(0.0, min(1.0, strength))


def _validate_sequences(
    original_frames: list[torch.Tensor],
    processed_frames: list[torch.Tensor],
    mask_frames: list[torch.Tensor],
) -> tuple[int, int]:
    frame_count = len(processed_frames)
    if len(original_frames) != frame_count or len(mask_frames) != frame_count:
        raise ValueError(
            "original_images, processed_images, and masks must contain the same number of frames"
        )

    height = int(processed_frames[0].shape[1])
    width = int(processed_frames[0].shape[2])
    channel_count = int(processed_frames[0].shape[3])
    for index, (original, processed, mask) in enumerate(
        zip(original_frames, processed_frames, mask_frames)
    ):
        if tuple(processed.shape[1:]) != (height, width, channel_count):
            raise ValueError("all processed_images frames must have matching dimensions")
        if int(original.shape[1]) != height or int(original.shape[2]) != width:
            raise ValueError(
                f"original_images frame {index} does not match processed_images dimensions"
            )
        if tuple(mask.shape[1:]) != (height, width):
            raise ValueError(f"masks frame {index} does not match processed_images dimensions")
    return height, width


def _select_work_size(height: int, width: int) -> tuple[int, int]:
    longest_side = max(height, width)
    if longest_side <= _WORK_MAX_SIDE:
        return height, width
    scale = float(_WORK_MAX_SIDE) / float(longest_side)
    return max(16, int(round(height * scale))), max(16, int(round(width * scale)))


def _select_mask_crop(
    mask_frames: list[torch.Tensor],
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    minimum_y = height
    minimum_x = width
    maximum_y = 0
    maximum_x = 0
    for mask in mask_frames:
        coordinates = torch.nonzero(
            mask.detach().to(device="cpu", dtype=torch.float32)[0] > 0.01,
            as_tuple=False,
        )
        if coordinates.numel() == 0:
            continue
        minimum_y = min(minimum_y, int(coordinates[:, 0].min().item()))
        minimum_x = min(minimum_x, int(coordinates[:, 1].min().item()))
        maximum_y = max(maximum_y, int(coordinates[:, 0].max().item()) + 1)
        maximum_x = max(maximum_x, int(coordinates[:, 1].max().item()) + 1)

    if maximum_y <= minimum_y or maximum_x <= minimum_x:
        return 0, height, 0, width

    region_height = maximum_y - minimum_y
    region_width = maximum_x - minimum_x
    padding_y = max(8, int(round(region_height * 0.5)))
    padding_x = max(8, int(round(region_width * 0.5)))
    return (
        max(0, minimum_y - padding_y),
        min(height, maximum_y + padding_y),
        max(0, minimum_x - padding_x),
        min(width, maximum_x + padding_x),
    )


def _select_compute_device(frames: list[torch.Tensor]) -> torch.device:
    for frame in frames:
        if frame.device.type != "cpu":
            return frame.device
    try:
        import comfy.model_management as model_management

        device = torch.device(model_management.get_torch_device())
        if device.type != "cuda" or torch.cuda.is_available():
            return device
    except Exception:
        pass
    return torch.device("cpu")


def _srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.04045,
        value / 12.92,
        torch.pow((value + 0.055) / 1.055, 2.4),
    )


def _linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.0031308,
        value * 12.92,
        (1.055 * torch.pow(value, 1.0 / 2.4)) - 0.055,
    )


def _resize_rgb_frame(
    frame: torch.Tensor,
    size: tuple[int, int],
    crop: tuple[int, int, int, int],
) -> torch.Tensor:
    y0, y1, x0, x1 = crop
    rgb = frame.detach().to(device="cpu", dtype=torch.float32)[:, y0:y1, x0:x1, :3]
    rgb_nchw = rgb.permute(0, 3, 1, 2)
    if tuple(rgb_nchw.shape[-2:]) != size:
        rgb_nchw = F.interpolate(rgb_nchw, size=size, mode="bilinear", align_corners=False)
    return rgb_nchw


def _resize_mask_frame(
    frame: torch.Tensor,
    size: tuple[int, int],
    crop: tuple[int, int, int, int],
) -> torch.Tensor:
    y0, y1, x0, x1 = crop
    mask = (
        frame.detach()
        .to(device="cpu", dtype=torch.float32)[:, y0:y1, x0:x1]
        .unsqueeze(1)
        .clamp(0.0, 1.0)
    )
    if tuple(mask.shape[-2:]) != size:
        mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
    return mask


def _prepare_work_sequences(
    original_frames: list[torch.Tensor],
    processed_frames: list[torch.Tensor],
    mask_frames: list[torch.Tensor],
    size: tuple[int, int],
    crop: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_gray_items: list[torch.Tensor] = []
    processed_log_luminance_items: list[torch.Tensor] = []
    mask_items: list[torch.Tensor] = []

    luminance_weights = torch.tensor(
        (0.2126, 0.7152, 0.0722),
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    for original, processed, mask in zip(original_frames, processed_frames, mask_frames):
        original_rgb = _resize_rgb_frame(original, size, crop)
        processed_rgb = _resize_rgb_frame(processed, size, crop)
        source_gray_items.append((original_rgb * luminance_weights).sum(dim=1, keepdim=True))

        linear_rgb = _srgb_to_linear(processed_rgb)
        luminance = (linear_rgb * luminance_weights).sum(dim=1, keepdim=True)
        processed_log_luminance_items.append(torch.log(luminance.clamp_min(_EPSILON)))
        mask_items.append(_resize_mask_frame(mask, size, crop))

    return (
        torch.cat(source_gray_items, dim=0),
        torch.cat(processed_log_luminance_items, dim=0),
        torch.cat(mask_items, dim=0),
    )


def _representative_mask_short_side(mask_sequence: torch.Tensor) -> float:
    short_sides: list[int] = []
    for mask in mask_sequence:
        coordinates = torch.nonzero(mask[0] > 0.05, as_tuple=False)
        if coordinates.numel() == 0:
            continue
        height = int(coordinates[:, 0].max().item() - coordinates[:, 0].min().item() + 1)
        width = int(coordinates[:, 1].max().item() - coordinates[:, 1].min().item() + 1)
        short_sides.append(min(height, width))
    if not short_sides:
        return 0.0
    short_sides.sort()
    middle = len(short_sides) // 2
    if len(short_sides) % 2:
        return float(short_sides[middle])
    return 0.5 * float(short_sides[middle - 1] + short_sides[middle])


def _gaussian_blur(value: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return value
    height, width = int(value.shape[-2]), int(value.shape[-1])
    radius = min(int(math.ceil(3.0 * sigma)), height - 1, width - 1)
    if radius <= 0:
        return value

    coordinates = torch.arange(
        -radius,
        radius + 1,
        device=value.device,
        dtype=value.dtype,
    )
    kernel = torch.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(_EPSILON)
    channel_count = int(value.shape[1])
    horizontal = kernel.view(1, 1, 1, -1).repeat(channel_count, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channel_count, 1, 1, 1)

    padding_mode = "reflect" if radius < min(height, width) else "replicate"
    output = F.pad(value, (radius, radius, 0, 0), mode=padding_mode)
    output = F.conv2d(output, horizontal, groups=channel_count)
    output = F.pad(output, (0, 0, radius, radius), mode=padding_mode)
    return F.conv2d(output, vertical, groups=channel_count)


def _normalize_flow_gray(gray: torch.Tensor) -> torch.Tensor:
    local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
    centered = gray - local_mean
    local_variance = F.avg_pool2d(centered * centered, kernel_size=9, stride=1, padding=4)
    return centered / torch.sqrt(local_variance + 0.01)


def _build_pyramid(value: torch.Tensor) -> list[torch.Tensor]:
    pyramid = [value]
    while len(pyramid) < _FLOW_LEVELS and min(pyramid[-1].shape[-2:]) >= 32:
        current = pyramid[-1]
        next_size = (
            max(8, int(round(int(current.shape[-2]) * 0.5))),
            max(8, int(round(int(current.shape[-1]) * 0.5))),
        )
        if next_size == tuple(current.shape[-2:]):
            break
        pyramid.append(F.interpolate(current, size=next_size, mode="bilinear", align_corners=True))
    return list(reversed(pyramid))


def _warp(value: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, height, width = value.shape
    y_coordinates, x_coordinates = torch.meshgrid(
        torch.arange(height, device=value.device, dtype=value.dtype),
        torch.arange(width, device=value.device, dtype=value.dtype),
        indexing="ij",
    )
    x = x_coordinates.unsqueeze(0) + flow[:, 0]
    y = y_coordinates.unsqueeze(0) + flow[:, 1]
    valid = (x >= 0.0) & (x <= float(width - 1)) & (y >= 0.0) & (y <= float(height - 1))

    if width > 1:
        x = ((2.0 * x) / float(width - 1)) - 1.0
    else:
        x = torch.zeros_like(x)
    if height > 1:
        y = ((2.0 * y) / float(height - 1)) - 1.0
    else:
        y = torch.zeros_like(y)
    grid = torch.stack((x, y), dim=-1)
    warped = F.grid_sample(
        value,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped, valid.view(batch_size, 1, height, width).to(value.dtype)


def _image_gradients(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    horizontal_kernel = torch.tensor(
        ((0.0, 0.0, 0.0), (-0.5, 0.0, 0.5), (0.0, 0.0, 0.0)),
        device=value.device,
        dtype=value.dtype,
    ).view(1, 1, 3, 3)
    vertical_kernel = horizontal_kernel.transpose(-1, -2)
    return (
        F.conv2d(value, horizontal_kernel, padding=1),
        F.conv2d(value, vertical_kernel, padding=1),
    )


def _estimate_flow(target: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
    target_pyramid = _build_pyramid(target)
    moving_pyramid = _build_pyramid(moving)
    flow: torch.Tensor | None = None

    for target_level, moving_level in zip(target_pyramid, moving_pyramid):
        level_height, level_width = int(target_level.shape[-2]), int(target_level.shape[-1])
        if flow is None:
            flow = torch.zeros(
                (int(target_level.shape[0]), 2, level_height, level_width),
                device=target_level.device,
                dtype=target_level.dtype,
            )
        elif tuple(flow.shape[-2:]) != (level_height, level_width):
            old_height, old_width = int(flow.shape[-2]), int(flow.shape[-1])
            flow = F.interpolate(flow, size=(level_height, level_width), mode="bilinear", align_corners=True)
            flow[:, 0] *= float(level_width) / float(old_width)
            flow[:, 1] *= float(level_height) / float(old_height)

        for _ in range(_FLOW_ITERATIONS):
            warped, valid = _warp(moving_level, flow)
            gradient_x, gradient_y = _image_gradients(warped)
            error = warped - target_level

            a11 = F.avg_pool2d(gradient_x * gradient_x, 5, stride=1, padding=2) + 0.01
            a22 = F.avg_pool2d(gradient_y * gradient_y, 5, stride=1, padding=2) + 0.01
            a12 = F.avg_pool2d(gradient_x * gradient_y, 5, stride=1, padding=2)
            b1 = F.avg_pool2d(gradient_x * error, 5, stride=1, padding=2)
            b2 = F.avg_pool2d(gradient_y * error, 5, stride=1, padding=2)
            determinant = (a11 * a22) - (a12 * a12)

            delta_x = ((a12 * b2) - (a22 * b1)) / determinant.clamp_min(_EPSILON)
            delta_y = ((a12 * b1) - (a11 * b2)) / determinant.clamp_min(_EPSILON)
            delta = torch.cat((delta_x, delta_y), dim=1).clamp(-1.5, 1.5)
            delta = F.avg_pool2d(delta, 3, stride=1, padding=1) * valid
            flow = flow + delta

    if flow is None:
        raise RuntimeError("failed to initialize optical flow")
    return flow


def _flow_confidence(
    flow: torch.Tensor,
    reverse_flow: torch.Tensor,
    target_gray: torch.Tensor,
    moving_gray: torch.Tensor,
) -> torch.Tensor:
    warped_reverse, valid_reverse = _warp(reverse_flow, flow)
    consistency_error = torch.linalg.vector_norm(flow + warped_reverse, dim=1, keepdim=True)
    flow_magnitude = torch.linalg.vector_norm(flow, dim=1, keepdim=True)
    consistency_scale = 0.75 + (0.05 * flow_magnitude)
    consistency = torch.exp(-torch.square(consistency_error / consistency_scale.clamp_min(0.25)))

    warped_moving, valid_image = _warp(moving_gray, flow)
    photometric_error = _gaussian_blur(torch.abs(target_gray - warped_moving), 0.8)
    photometric = torch.exp(-photometric_error / 0.12)
    return (consistency * photometric * valid_reverse * valid_image).clamp(0.0, 1.0)


def _compute_adjacent_motion(
    source_gray: torch.Tensor,
    compute_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pair_count = int(source_gray.shape[0]) - 1
    if pair_count <= 0:
        empty_flow = torch.empty((0, 2, *source_gray.shape[-2:]), dtype=torch.float16)
        empty_confidence = torch.empty((0, 1, *source_gray.shape[-2:]), dtype=torch.float16)
        return empty_flow, empty_flow.clone(), empty_confidence, empty_confidence.clone()

    normalized_gray = _normalize_flow_gray(source_gray)
    forward_flows: list[torch.Tensor] = []
    backward_flows: list[torch.Tensor] = []
    forward_confidences: list[torch.Tensor] = []
    backward_confidences: list[torch.Tensor] = []

    for start in range(0, pair_count, _FLOW_PAIR_CHUNK):
        end = min(pair_count, start + _FLOW_PAIR_CHUNK)
        first = normalized_gray[start:end].to(compute_device)
        second = normalized_gray[start + 1:end + 1].to(compute_device)
        first_raw = source_gray[start:end].to(compute_device)
        second_raw = source_gray[start + 1:end + 1].to(compute_device)

        forward = _estimate_flow(first, second)
        backward = _estimate_flow(second, first)
        forward_confidence = _flow_confidence(forward, backward, first_raw, second_raw)
        backward_confidence = _flow_confidence(backward, forward, second_raw, first_raw)

        forward_flows.append(forward.detach().to(device="cpu", dtype=torch.float16))
        backward_flows.append(backward.detach().to(device="cpu", dtype=torch.float16))
        forward_confidences.append(
            forward_confidence.detach().to(device="cpu", dtype=torch.float16)
        )
        backward_confidences.append(
            backward_confidence.detach().to(device="cpu", dtype=torch.float16)
        )

    return (
        torch.cat(forward_flows, dim=0),
        torch.cat(backward_flows, dim=0),
        torch.cat(forward_confidences, dim=0),
        torch.cat(backward_confidences, dim=0),
    )


def _load_motion_item(value: torch.Tensor, index: int, device: torch.device) -> torch.Tensor:
    return value[index:index + 1].to(device=device, dtype=torch.float32)


def _compose_motion(
    frame_index: int,
    neighbor_index: int,
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    forward_confidence: torch.Tensor,
    backward_confidence: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if neighbor_index == frame_index:
        raise ValueError("neighbor_index must differ from frame_index")

    if neighbor_index > frame_index:
        pair_indices = range(frame_index, neighbor_index)
        flow_values = forward_flow
        confidence_values = forward_confidence
    else:
        pair_indices = range(frame_index - 1, neighbor_index - 1, -1)
        flow_values = backward_flow
        confidence_values = backward_confidence

    composed_flow: torch.Tensor | None = None
    composed_confidence: torch.Tensor | None = None
    for pair_index in pair_indices:
        next_flow = _load_motion_item(flow_values, pair_index, device)
        next_confidence = _load_motion_item(confidence_values, pair_index, device)
        if composed_flow is None:
            composed_flow = next_flow
            composed_confidence = next_confidence
            continue

        warped_flow, valid_flow = _warp(next_flow, composed_flow)
        warped_confidence, valid_confidence = _warp(next_confidence, composed_flow)
        composed_flow = composed_flow + warped_flow
        composed_confidence = (
            composed_confidence * warped_confidence * valid_flow * valid_confidence
        )

    if composed_flow is None or composed_confidence is None:
        raise RuntimeError("failed to compose optical flow")
    return composed_flow, composed_confidence


def _calculate_correction_maps(
    log_luminance: torch.Tensor,
    masks: torch.Tensor,
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    forward_confidence: torch.Tensor,
    backward_confidence: torch.Tensor,
    compute_device: torch.device,
    spatial_sigma: float,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    base_luminance = _gaussian_blur(log_luminance, spatial_sigma)
    feathered_masks = masks * _gaussian_blur(masks, max(0.75, spatial_sigma * 0.5))
    frame_count = int(base_luminance.shape[0])
    corrections: list[torch.Tensor] = []

    for frame_index in range(frame_count):
        current = base_luminance[frame_index:frame_index + 1].to(compute_device)
        current_mask = masks[frame_index:frame_index + 1].to(compute_device)
        weighted_delta = torch.zeros_like(current)
        total_weight = torch.zeros_like(current)

        for distance in range(1, _TEMPORAL_RADIUS + 1):
            temporal_weight = 1.0 / float(distance)
            for neighbor_index in (frame_index - distance, frame_index + distance):
                if neighbor_index < 0 or neighbor_index >= frame_count:
                    continue
                flow, confidence = _compose_motion(
                    frame_index,
                    neighbor_index,
                    forward_flow,
                    backward_flow,
                    forward_confidence,
                    backward_confidence,
                    compute_device,
                )
                neighbor = base_luminance[neighbor_index:neighbor_index + 1].to(compute_device)
                neighbor_mask = masks[neighbor_index:neighbor_index + 1].to(compute_device)
                warped_neighbor, valid_neighbor = _warp(neighbor, flow)
                warped_mask, valid_mask = _warp(neighbor_mask, flow)

                weight = (
                    temporal_weight
                    * confidence
                    * valid_neighbor
                    * valid_mask
                    * current_mask
                    * warped_mask
                )
                neighbor_delta = (warped_neighbor - current).clamp(
                    -2.0 * _MAX_CORRECTION_LOG,
                    2.0 * _MAX_CORRECTION_LOG,
                )
                weighted_delta = weighted_delta + (neighbor_delta * weight)
                total_weight = total_weight + weight

        correction = torch.where(
            total_weight > 0.05,
            weighted_delta / total_weight.clamp_min(_EPSILON),
            torch.zeros_like(weighted_delta),
        )
        correction = _gaussian_blur(correction, max(0.5, spatial_sigma * 0.35))
        corrections.append(
            correction.clamp(-_MAX_CORRECTION_LOG, _MAX_CORRECTION_LOG).detach().cpu()
        )

    return corrections, feathered_masks


def _apply_correction(
    frame: torch.Tensor,
    mask: torch.Tensor,
    correction: torch.Tensor,
    feathered_mask: torch.Tensor,
    strength: float,
    crop: tuple[int, int, int, int],
) -> torch.Tensor:
    y0, y1, x0, x1 = crop
    crop_height = y1 - y0
    crop_width = x1 - x0
    device = frame.device
    correction_full = F.interpolate(
        correction.to(device=device, dtype=torch.float32),
        size=(crop_height, crop_width),
        mode="bilinear",
        align_corners=False,
    )
    feather_full = F.interpolate(
        feathered_mask.to(device=device, dtype=torch.float32),
        size=(crop_height, crop_width),
        mode="bilinear",
        align_corners=False,
    )
    input_mask = (
        mask.detach()
        .to(device=device, dtype=torch.float32)[:, y0:y1, x0:x1]
        .unsqueeze(1)
        .clamp(0.0, 1.0)
    )
    blend = (feather_full * input_mask).clamp(0.0, 1.0).permute(0, 2, 3, 1)

    rgb = frame[:, y0:y1, x0:x1, :3].to(dtype=torch.float32)
    linear_rgb = _srgb_to_linear(rgb)
    gain = torch.exp(correction_full.permute(0, 2, 3, 1) * float(strength))
    corrected_rgb = _linear_to_srgb(linear_rgb * gain)

    output = frame.clone()
    output[:, y0:y1, x0:x1, :3] = torch.lerp(rgb, corrected_rgb, blend).to(
        dtype=frame.dtype
    )
    return output


def stabilize_local_luminance(
    original_images: Any,
    processed_images: Any,
    masks: Any,
    strength: Any = _DEFAULT_STRENGTH,
) -> list[torch.Tensor]:
    original_frames = _split_image_frames(original_images, input_name="original_images")
    processed_frames = _split_image_frames(processed_images, input_name="processed_images")
    mask_frames = _split_mask_frames(masks)
    strength_value = _read_strength(strength)
    height, width = _validate_sequences(original_frames, processed_frames, mask_frames)

    if strength_value <= 0.0 or len(processed_frames) < 2:
        return [frame.clone() for frame in processed_frames]
    if not any(bool((mask.detach() > 0.0).any().item()) for mask in mask_frames):
        return [frame.clone() for frame in processed_frames]

    crop = _select_mask_crop(mask_frames, height, width)
    crop_height = crop[1] - crop[0]
    crop_width = crop[3] - crop[2]
    work_size = _select_work_size(crop_height, crop_width)
    source_gray, log_luminance, work_masks = _prepare_work_sequences(
        original_frames,
        processed_frames,
        mask_frames,
        work_size,
        crop,
    )
    mask_short_side = _representative_mask_short_side(work_masks)
    spatial_sigma = max(1.0, min(12.0, mask_short_side / 12.0))
    compute_device = _select_compute_device(original_frames)

    forward_flow, backward_flow, forward_confidence, backward_confidence = (
        _compute_adjacent_motion(source_gray, compute_device)
    )
    corrections, feathered_masks = _calculate_correction_maps(
        log_luminance,
        work_masks,
        forward_flow,
        backward_flow,
        forward_confidence,
        backward_confidence,
        compute_device,
        spatial_sigma,
    )

    return [
        _apply_correction(
            processed,
            mask,
            correction,
            feathered_masks[index:index + 1],
            strength_value,
            crop,
        )
        for index, (processed, mask, correction) in enumerate(
            zip(processed_frames, mask_frames, corrections)
        )
    ]


class LocalTemporalDeflicker(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-LocalTemporalDeflicker",
            display_name="Local Temporal Deflicker",
            category=Const.CATEGORY_MASK,
            description=(
                "Stabilize low-frequency luminance inside a mask across an IMAGE list. "
                "Original frames are used only to estimate motion."
            ),
            search_aliases=[
                "local temporal deflicker",
                "face brightness stabilization",
                "masked video flicker",
                "optical flow luminance",
            ],
            is_input_list=True,
            inputs=[
                c_io.Image.Input(
                    "original_images",
                    tooltip="Original frame list used only for optical-flow motion estimation.",
                ),
                c_io.Image.Input(
                    "processed_images",
                    tooltip="Detailer and Color Match output frame list to stabilize.",
                ),
                c_io.Mask.Input(
                    "masks",
                    tooltip="Per-frame soft mask list that limits analysis and correction.",
                ),
                c_io.Float.Input(
                    "strength",
                    default=_DEFAULT_STRENGTH,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Amount of detected temporal luminance variation to correct.",
                ),
            ],
            outputs=[
                c_io.Image.Output(
                    Cast.out_id("images"),
                    display_name="images",
                    is_output_list=True,
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        original_images: Any = None,
        processed_images: Any = None,
        masks: Any = None,
        strength: Any = _DEFAULT_STRENGTH,
    ) -> bool | str:
        try:
            _read_strength(strength)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def execute(
        cls,
        original_images: Any,
        processed_images: Any,
        masks: Any,
        strength: Any = _DEFAULT_STRENGTH,
    ) -> c_io.NodeOutput:
        return c_io.NodeOutput(
            stabilize_local_luminance(
                original_images,
                processed_images,
                masks,
                strength,
            )
        )
