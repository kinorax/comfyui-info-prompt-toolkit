from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence


@dataclass(frozen=True)
class SpatialTile:
    row: int
    column: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def height(self) -> int:
        return max(0, int(self.y1) - int(self.y0))

    @property
    def width(self) -> int:
        return max(0, int(self.x1) - int(self.x0))


@dataclass(frozen=True)
class SpatialTilePlan:
    tile: SpatialTile
    sample_y0: int
    sample_y1: int
    sample_x0: int
    sample_x1: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    content_y0: int
    content_y1: int
    content_x0: int
    content_x1: int

    @property
    def target_height(self) -> int:
        return max(
            1,
            int(self.sample_y1) - int(self.sample_y0) + int(self.pad_top) + int(self.pad_bottom),
        )

    @property
    def target_width(self) -> int:
        return max(
            1,
            int(self.sample_x1) - int(self.sample_x0) + int(self.pad_left) + int(self.pad_right),
        )

def int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def positive_int_or_none(value: object) -> int | None:
    parsed = int_or_none(value)
    if parsed is None or parsed < 1:
        return None
    return parsed


def nonnegative_int_or_none(value: object) -> int | None:
    parsed = int_or_none(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def pixel_overlap_to_latent_overlap(pixel_overlap: int, spatial_scale: int) -> int:
    overlap_value = max(0, int(pixel_overlap))
    scale_value = max(1, int(spatial_scale))
    return int(math.ceil(float(overlap_value) / float(scale_value)))


def build_spatial_tiles(
    *,
    height: int,
    width: int,
    tile_rows: int,
    tile_columns: int,
    overlap: int,
) -> list[SpatialTile]:
    height_value = max(1, int(height))
    width_value = max(1, int(width))
    rows_value = max(1, min(int(tile_rows), height_value))
    columns_value = max(1, min(int(tile_columns), width_value))
    overlap_value = max(0, int(overlap))

    row_spans = _partition_axis(height_value, rows_value, overlap_value)
    column_spans = _partition_axis(width_value, columns_value, overlap_value)

    output: list[SpatialTile] = []
    for row_index, (y0, y1) in enumerate(row_spans):
        for column_index, (x0, x1) in enumerate(column_spans):
            output.append(
                SpatialTile(
                    row=row_index,
                    column=column_index,
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                )
            )
    return output


def build_spatial_tile_plans(
    tiles: Sequence[SpatialTile],
    *,
    full_height: int,
    full_width: int,
    mini_unit: int,
) -> list[SpatialTilePlan]:
    height_value = max(1, int(full_height))
    width_value = max(1, int(full_width))
    mini_unit_value = max(1, int(mini_unit))

    output: list[SpatialTilePlan] = []
    for tile in tiles:
        target_height = _round_up(max(1, int(tile.height)), mini_unit_value)
        target_width = _round_up(max(1, int(tile.width)), mini_unit_value)

        sample_y0, sample_y1 = _expand_interval_to_target(
            int(tile.y0),
            int(tile.y1),
            height_value,
            target_height,
        )
        sample_x0, sample_x1 = _expand_interval_to_target(
            int(tile.x0),
            int(tile.x1),
            width_value,
            target_width,
        )

        sample_height = max(1, sample_y1 - sample_y0)
        sample_width = max(1, sample_x1 - sample_x0)
        pad_height = max(0, target_height - sample_height)
        pad_width = max(0, target_width - sample_width)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        content_y0 = (int(tile.y0) - sample_y0) + pad_top
        content_x0 = (int(tile.x0) - sample_x0) + pad_left
        content_y1 = content_y0 + int(tile.height)
        content_x1 = content_x0 + int(tile.width)

        output.append(
            SpatialTilePlan(
                tile=tile,
                sample_y0=sample_y0,
                sample_y1=sample_y1,
                sample_x0=sample_x0,
                sample_x1=sample_x1,
                pad_top=pad_top,
                pad_bottom=pad_bottom,
                pad_left=pad_left,
                pad_right=pad_right,
                content_y0=content_y0,
                content_y1=content_y1,
                content_x0=content_x0,
                content_x1=content_x1,
            )
        )
    return output


def conditioning_controls(conditioning: object) -> list[object]:
    output: list[object] = []
    if not isinstance(conditioning, list):
        return output

    seen_ids: set[int] = set()
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        payload = item[1]
        if not isinstance(payload, dict):
            continue
        control = payload.get("control")
        if control is None:
            continue
        control_id = id(control)
        if control_id in seen_ids:
            continue
        seen_ids.add(control_id)
        output.append(control)
    return output


@contextmanager
def passthrough_top_level_controls(
    positive: object,
    negative: object,
) -> Iterator[None]:
    controls = conditioning_controls(positive) + conditioning_controls(negative)
    patched: list[object] = []
    seen_ids: set[int] = set()
    for control in controls:
        control_id = id(control)
        if control_id in seen_ids:
            continue
        seen_ids.add(control_id)
        get_control = getattr(control, "get_control", None)
        if not callable(get_control) or hasattr(control, "get_control_orig"):
            continue

        setattr(control, "get_control_orig", get_control)
        setattr(control, "get_control", _passthrough_get_control(control))
        patched.append(control)

    try:
        yield
    finally:
        for control in reversed(patched):
            original = getattr(control, "get_control_orig", None)
            if callable(original):
                setattr(control, "get_control", original)
            try:
                delattr(control, "get_control_orig")
            except Exception:
                pass


class SpatialTiledModelWrapper:
    def __init__(
        self,
        *,
        tile_rows: int,
        tile_columns: int,
        tile_overlap: int,
        mini_unit: int,
        spatial_scale: int,
        old_wrapper: object | None = None,
    ) -> None:
        self.tile_rows = max(1, int(tile_rows))
        self.tile_columns = max(1, int(tile_columns))
        self.tile_overlap = max(0, int(tile_overlap))
        self.mini_unit = max(1, int(mini_unit))
        self.spatial_scale = max(1, int(spatial_scale))
        self.old_wrapper = old_wrapper if callable(old_wrapper) else None
        self._weight_cache: dict[tuple[int, int, str, str], Any] = {}

    def __call__(self, apply_model: object, args: dict[str, object]):
        torch = _import_torch()
        comfy_utils = _import_comfy_utils()

        x_in = args.get("input")
        t_in = args.get("timestep")
        c_in = args.get("c")
        cond_or_uncond = args.get("cond_or_uncond")

        if not _is_tensor_like(x_in) or not _is_tensor_like(t_in) or not isinstance(c_in, dict):
            return self._delegate(apply_model, args)
        if not isinstance(cond_or_uncond, list):
            cond_or_uncond = []

        if int(getattr(x_in, "ndim", 0)) not in (4, 5):
            return self._delegate(apply_model, args)

        height = int(x_in.shape[-2])
        width = int(x_in.shape[-1])
        overlap_latent = pixel_overlap_to_latent_overlap(self.tile_overlap, self.spatial_scale)
        tiles = build_spatial_tiles(
            height=height,
            width=width,
            tile_rows=self.tile_rows,
            tile_columns=self.tile_columns,
            overlap=overlap_latent,
        )
        tile_plans = build_spatial_tile_plans(
            tiles,
            full_height=height,
            full_width=width,
            mini_unit=self.mini_unit,
        )
        if len(tile_plans) <= 1:
            only_plan = tile_plans[0]
            if (
                only_plan.tile.y0 == 0
                and only_plan.tile.y1 == height
                and only_plan.tile.x0 == 0
                and only_plan.tile.x1 == width
                and only_plan.sample_y0 == 0
                and only_plan.sample_y1 == height
                and only_plan.sample_x0 == 0
                and only_plan.sample_x1 == width
                and only_plan.pad_top == 0
                and only_plan.pad_bottom == 0
                and only_plan.pad_left == 0
                and only_plan.pad_right == 0
            ):
                return self._delegate(apply_model, args)

        base_batch = int(x_in.shape[0])
        accum_dtype = torch.float32 if x_in.dtype in (torch.float16, torch.bfloat16) else x_in.dtype
        output_buffer = torch.zeros_like(x_in, dtype=accum_dtype)
        weight_buffer = torch.zeros_like(output_buffer)

        full_shape = tuple(int(v) for v in x_in.shape)
        for tile_plan in tile_plans:
            x_tile = tile_tensor_batch(
                x_in,
                [tile_plan],
                full_height=height,
                full_width=width,
            )
            t_tile = comfy_utils.repeat_to_batch_size(t_in, int(x_tile.shape[0]))
            repeated_cond_or_uncond = list(cond_or_uncond)
            c_tile = self._tile_conditioning_dict(
                c_in,
                x_in=x_in,
                x_tile=x_tile,
                tile_plans=[tile_plan],
                repeated_cond_or_uncond=repeated_cond_or_uncond,
                timestep=t_tile,
            )

            control = c_in.get("control")
            if control is not None and hasattr(control, "get_control_orig"):
                c_tile["control"] = build_tiled_control(
                    control,
                    x_tile=x_tile,
                    t_tile=t_tile,
                    c_tile=c_tile,
                    batched_number=max(1, len(cond_or_uncond)),
                    transformer_options=c_tile.get("transformer_options"),
                    full_shape=full_shape,
                    tile_plans=[tile_plan],
                )

            tile_args = {
                "input": x_tile,
                "timestep": t_tile,
                "c": c_tile,
                "cond_or_uncond": repeated_cond_or_uncond,
            }
            tile_output = self._delegate(apply_model, tile_args)
            if not _is_tensor_like(tile_output):
                return tile_output

            if int(tile_output.shape[0]) != base_batch:
                tile_output = tile_output[:base_batch]
            tile_tensor = crop_tensor_to_plan_content(tile_output, tile_plan).to(dtype=accum_dtype)
            tile_weight = self._tile_weight(
                height=tile_plan.tile.height,
                width=tile_plan.tile.width,
                device=tile_tensor.device,
                dtype=accum_dtype,
                dimensions=int(tile_tensor.ndim),
            )
            output_buffer = add_weighted_tile(output_buffer, tile_tensor, tile_plan.tile, tile_weight)
            weight_buffer = add_weighted_tile(weight_buffer, tile_weight.expand_as(tile_tensor), tile_plan.tile, None)

        safe_weights = torch.where(weight_buffer > 0, weight_buffer, torch.ones_like(weight_buffer))
        return (output_buffer / safe_weights).to(dtype=x_in.dtype)

    def _delegate(self, apply_model: object, args: dict[str, object]):
        if self.old_wrapper is not None:
            return self.old_wrapper(apply_model, args)

        input_tensor = args.get("input")
        timestep = args.get("timestep")
        conditioning = args.get("c")
        if callable(apply_model) and _is_tensor_like(input_tensor) and _is_tensor_like(timestep) and isinstance(conditioning, dict):
            return apply_model(input_tensor, timestep, **conditioning)
        raise RuntimeError("model_function_wrapper received invalid inputs")

    def _tile_conditioning_dict(
        self,
        conditioning: dict[str, object],
        *,
        x_in: Any,
        x_tile: Any,
        tile_plans: Sequence[SpatialTilePlan],
        repeated_cond_or_uncond: list[object],
        timestep: Any,
    ) -> dict[str, object]:
        comfy_utils = _import_comfy_utils()
        output: dict[str, object] = {}
        for key, value in conditioning.items():
            if key == "control":
                continue
            if key == "transformer_options" and isinstance(value, dict):
                output[key] = repeat_transformer_options(value, repeated_cond_or_uncond, timestep)
                continue
            if _is_tensor_like(value) and int(getattr(value, "ndim", 0)) == int(getattr(x_in, "ndim", 0)):
                tiled = tile_tensor_batch(
                    value,
                    tile_plans,
                    full_height=int(x_in.shape[-2]),
                    full_width=int(x_in.shape[-1]),
                )
                if int(tiled.shape[0]) != int(x_tile.shape[0]):
                    tiled = comfy_utils.repeat_to_batch_size(tiled, int(x_tile.shape[0]))
                output[key] = tiled
                continue
            output[key] = value
        return output

    def _tile_weight(
        self,
        *,
        height: int,
        width: int,
        device: object,
        dtype: object,
        dimensions: int,
    ):
        torch = _import_torch()

        key = (int(height), int(width), str(device), str(dtype))
        weight = self._weight_cache.get(key)
        if weight is None:
            weight = gaussian_weight_2d(height, width, device=device, dtype=dtype)
            self._weight_cache[key] = weight

        if dimensions == 5:
            return weight.view(1, 1, 1, int(height), int(width))
        return weight.view(1, 1, int(height), int(width))


def tile_tensor_batch(
    tensor: Any,
    tile_plans: Sequence[SpatialTilePlan],
    *,
    full_height: int,
    full_width: int,
):
    torch = _import_torch()
    if not _is_tensor_like(tensor) or int(getattr(tensor, "ndim", 0)) < 3:
        return tensor

    outputs = []
    for tile_plan in tile_plans:
        outputs.append(
            tile_tensor_for_plan(
                tensor,
                tile_plan,
                full_height=full_height,
                full_width=full_width,
            )
        )

    if len(outputs) <= 0:
        return tensor
    if len(outputs) == 1:
        return outputs[0]

    max_height = max(int(output.shape[-2]) for output in outputs)
    max_width = max(int(output.shape[-1]) for output in outputs)
    normalized_outputs = [
        _pad_tensor_to_size(output, target_height=max_height, target_width=max_width)
        for output in outputs
    ]
    return torch.cat(normalized_outputs, dim=0)


def tile_tensor_for_plan(
    tensor: Any,
    tile_plan: SpatialTilePlan,
    *,
    full_height: int,
    full_width: int,
):
    if not _is_tensor_like(tensor):
        return tensor

    target_height = int(tensor.shape[-2])
    target_width = int(tensor.shape[-1])
    y0, y1 = _scaled_bounds(
        int(tile_plan.sample_y0),
        int(tile_plan.sample_y1),
        int(full_height),
        target_height,
    )
    x0, x1 = _scaled_bounds(
        int(tile_plan.sample_x0),
        int(tile_plan.sample_x1),
        int(full_width),
        target_width,
    )
    sampled = slice_tensor_spatial(tensor, SpatialTile(tile_plan.tile.row, tile_plan.tile.column, y0, y1, x0, x1))
    sampled_height = int(sampled.shape[-2])
    sampled_width = int(sampled.shape[-1])
    target_tile_height = _scaled_length(
        int(tile_plan.target_height),
        int(full_height),
        target_height,
        minimum=sampled_height,
    )
    target_tile_width = _scaled_length(
        int(tile_plan.target_width),
        int(full_width),
        target_width,
        minimum=sampled_width,
    )
    return _pad_tensor_to_size(sampled, target_height=target_tile_height, target_width=target_tile_width)


def crop_tensor_to_plan_content(
    tensor: Any,
    tile_plan: SpatialTilePlan,
):
    if not _is_tensor_like(tensor):
        return tensor

    content_y0 = max(0, int(tile_plan.content_y0))
    content_y1 = max(content_y0 + 1, int(tile_plan.content_y1))
    content_x0 = max(0, int(tile_plan.content_x0))
    content_x1 = max(content_x0 + 1, int(tile_plan.content_x1))

    index = [slice(None)] * int(tensor.ndim)
    index[-2] = slice(content_y0, content_y1)
    index[-1] = slice(content_x0, content_x1)
    return tensor[tuple(index)]


def repeat_transformer_options(
    transformer_options: dict[str, object],
    repeated_cond_or_uncond: list[object],
    timestep: Any,
) -> dict[str, object]:
    output = dict(transformer_options)
    output["cond_or_uncond"] = list(repeated_cond_or_uncond)

    uuids = transformer_options.get("uuids")
    if isinstance(uuids, list) and len(uuids) > 0:
        repeat_count = max(1, len(repeated_cond_or_uncond) // len(uuids))
        output["uuids"] = uuids * repeat_count

    if _is_tensor_like(transformer_options.get("sigmas")):
        output["sigmas"] = timestep
    return output


def build_tiled_control(
    control: object,
    *,
    x_tile: Any,
    t_tile: Any,
    c_tile: dict[str, object],
    batched_number: int,
    transformer_options: object,
    full_shape: Sequence[int],
    tile_plans: Sequence[SpatialTilePlan],
):
    states = prepare_control_for_tiles(
        control,
        tile_plans=tile_plans,
        full_height=int(full_shape[-2]),
        full_width=int(full_shape[-1]),
    )
    try:
        return control.get_control_orig(
            x_tile,
            t_tile,
            c_tile,
            batched_number,
            transformer_options if isinstance(transformer_options, dict) else {},
        )
    finally:
        restore_control_after_tiles(states)


def prepare_control_for_tiles(
    control: object,
    *,
    tile_plans: Sequence[SpatialTilePlan],
    full_height: int,
    full_width: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    current = control
    while current is not None:
        state = {
            "control": current,
            "cond_hint": getattr(current, "cond_hint", None),
            "cond_hint_original": getattr(current, "cond_hint_original", None),
            "extra_concat_orig": getattr(current, "extra_concat_orig", None),
        }
        cond_hint_original = state["cond_hint_original"]
        if _is_tensor_like(cond_hint_original):
            setattr(
                current,
                "cond_hint_original",
                tile_tensor_batch(
                    cond_hint_original,
                    tile_plans,
                    full_height=full_height,
                    full_width=full_width,
                ),
            )
        extra_concat_orig = state["extra_concat_orig"]
        if isinstance(extra_concat_orig, (list, tuple)) and extra_concat_orig:
            tiled_extra_concat: list[object] = []
            for item in extra_concat_orig:
                if _is_tensor_like(item):
                    tiled_extra_concat.append(
                        tile_tensor_batch(
                            item,
                            tile_plans,
                            full_height=full_height,
                            full_width=full_width,
                        )
                    )
                else:
                    tiled_extra_concat.append(item)
            setattr(current, "extra_concat_orig", tiled_extra_concat)
        setattr(current, "cond_hint", None)
        output.append(state)
        current = getattr(current, "previous_controlnet", None)
    return output


def restore_control_after_tiles(states: Sequence[dict[str, object]]) -> None:
    for state in reversed(list(states)):
        control = state.get("control")
        if control is None:
            continue
        setattr(control, "cond_hint", state.get("cond_hint"))
        setattr(control, "cond_hint_original", state.get("cond_hint_original"))
        setattr(control, "extra_concat_orig", state.get("extra_concat_orig"))


def gaussian_weight_2d(height: int, width: int, *, device: object, dtype: object):
    torch = _import_torch()
    if height <= 0 or width <= 0:
        raise ValueError("tile weight height and width must be positive")

    y = torch.linspace(-1.0, 1.0, steps=int(height), device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, steps=int(width), device=device, dtype=torch.float32)
    y_weight = torch.exp(-(y * y) / 0.5)
    x_weight = torch.exp(-(x * x) / 0.5)
    weight = torch.outer(y_weight, x_weight)
    weight = weight / torch.clamp(weight.max(), min=1e-12)
    weight = torch.clamp(weight, min=1e-4)
    return weight.to(dtype=dtype)


def slice_tensor_spatial(tensor: Any, tile: SpatialTile):
    if not _is_tensor_like(tensor):
        return tensor
    index = [slice(None)] * int(tensor.ndim)
    index[-2] = slice(int(tile.y0), int(tile.y1))
    index[-1] = slice(int(tile.x0), int(tile.x1))
    return tensor[tuple(index)]


def add_weighted_tile(buffer: Any, tile_tensor: Any, tile: SpatialTile, weight: Any | None):
    if not _is_tensor_like(buffer) or not _is_tensor_like(tile_tensor):
        return buffer

    index = [slice(None)] * int(buffer.ndim)
    index[-2] = slice(int(tile.y0), int(tile.y1))
    index[-1] = slice(int(tile.x0), int(tile.x1))
    if weight is None:
        buffer[tuple(index)] = buffer[tuple(index)] + tile_tensor
    else:
        buffer[tuple(index)] = buffer[tuple(index)] + (tile_tensor * weight)
    return buffer


def _round_up(value: int, unit: int) -> int:
    rounded_value = max(1, int(value))
    rounded_unit = max(1, int(unit))
    return int(math.ceil(float(rounded_value) / float(rounded_unit)) * rounded_unit)


def _expand_interval_to_target(start: int, end: int, limit: int, target_size: int) -> tuple[int, int]:
    limit_value = max(1, int(limit))
    start_value = max(0, min(int(start), limit_value - 1))
    end_value = max(start_value + 1, min(int(end), limit_value))
    current_size = max(1, end_value - start_value)
    desired_size = max(current_size, min(max(1, int(target_size)), limit_value))
    if current_size >= desired_size:
        return start_value, end_value

    growth = desired_size - current_size
    grow_left = growth // 2
    grow_right = growth - grow_left
    start_value = max(0, start_value - grow_left)
    end_value = min(limit_value, end_value + grow_right)
    current_size = end_value - start_value
    if current_size >= desired_size:
        return start_value, end_value

    remaining = desired_size - current_size
    extra_left = min(start_value, remaining)
    start_value -= extra_left
    remaining -= extra_left
    if remaining > 0:
        end_value = min(limit_value, end_value + remaining)
    return int(start_value), int(end_value)


def _scaled_length(length: int, full_length: int, scaled_full_length: int, *, minimum: int = 1) -> int:
    if full_length <= 0 or scaled_full_length <= 0:
        return max(1, int(minimum))
    scaled = int(math.ceil(float(length) * float(scaled_full_length) / float(full_length)))
    return max(int(minimum), max(1, scaled))


def _pad_tensor_to_size(tensor: Any, *, target_height: int, target_width: int):
    if not _is_tensor_like(tensor):
        return tensor

    torch = _import_torch()
    current_height = int(tensor.shape[-2])
    current_width = int(tensor.shape[-1])
    wanted_height = max(current_height, int(target_height))
    wanted_width = max(current_width, int(target_width))
    pad_height = max(0, wanted_height - current_height)
    pad_width = max(0, wanted_width - current_width)
    if pad_height <= 0 and pad_width <= 0:
        return tensor

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    pad_tuple = (pad_left, pad_right, pad_top, pad_bottom)
    ndim = int(getattr(tensor, "ndim", 0))
    # Pad spatial axes only. Replicate mode avoids introducing hard zero seams at tile edges.
    if ndim in (4, 5):
        return torch.nn.functional.pad(tensor, pad_tuple, mode="replicate")
    if ndim == 3:
        expanded = tensor.unsqueeze(1)
        padded = torch.nn.functional.pad(expanded, pad_tuple, mode="replicate")
        return padded.squeeze(1)
    return torch.nn.functional.pad(tensor, pad_tuple, mode="constant", value=0.0)


def _partition_axis(length: int, count: int, overlap: int) -> list[tuple[int, int]]:
    if count <= 1 or length <= 1:
        return [(0, length)]

    boundaries = [0]
    for index in range(1, count):
        boundary = int(round(float(index) * float(length) / float(count)))
        if boundary <= boundaries[-1]:
            boundary = boundaries[-1] + 1
        if boundary >= length:
            boundary = length - 1
        boundaries.append(boundary)
    boundaries.append(length)

    left_overlap = overlap // 2
    right_overlap = overlap - left_overlap
    spans: list[tuple[int, int]] = []
    for index in range(count):
        start = boundaries[index]
        end = boundaries[index + 1]
        if index > 0:
            start = max(0, start - left_overlap)
        if index < count - 1:
            end = min(length, end + right_overlap)
        if end <= start:
            end = min(length, start + 1)
        spans.append((start, end))
    return spans


def _scaled_bounds(start: int, end: int, full_length: int, scaled_length: int) -> tuple[int, int]:
    if full_length <= 0 or scaled_length <= 0:
        return 0, max(1, scaled_length)

    scaled_start = int(math.floor(float(start) * float(scaled_length) / float(full_length)))
    scaled_end = int(math.ceil(float(end) * float(scaled_length) / float(full_length)))
    scaled_start = max(0, min(scaled_start, max(0, scaled_length - 1)))
    scaled_end = max(scaled_start + 1, min(scaled_end, scaled_length))
    return scaled_start, scaled_end


def _passthrough_get_control(control: object):
    def _wrapped(*_args, **_kwargs):
        return control

    return _wrapped


def _import_comfy_utils():
    import importlib

    return importlib.import_module("comfy.utils")


def _import_torch():
    import importlib

    return importlib.import_module("torch")


def _is_tensor_like(value: object) -> bool:
    return hasattr(value, "shape") and hasattr(value, "ndim")
