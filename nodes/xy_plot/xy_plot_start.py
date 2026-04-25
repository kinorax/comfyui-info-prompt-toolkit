# Copyright 2026 kinorax
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast
from ...utils.file_hash_cache import normalize_relative_path
from ...utils.model_lora_metadata_pipeline import get_shared_metadata_pipeline
from ...utils.prompt_text import (
    combine_prompt_text,
    normalize_prompt_tokens,
    remove_prompt_comments,
    strip_prompt_weights,
)

MODE_LOOP = "loop"
MODE_LIST = "list"
MODE_BATCH_LEGACY = "batch"
MODE_OPTIONS: tuple[str, ...] = (
    MODE_LOOP,
    MODE_LIST,
)


def _first_value(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) == 1:
            return _first_value(value[0])
        return value
    if isinstance(value, tuple):
        values = list(value)
        if len(values) == 1:
            return _first_value(values[0])
        return values
    return value


def _normalized_mode(mode: Any) -> str:
    text = str(_first_value(mode)) if mode is not None else MODE_LOOP
    if text == MODE_BATCH_LEGACY:
        return MODE_LIST
    if text in MODE_OPTIONS:
        return text
    return MODE_LOOP


def _modifier_items(value: Any) -> list[dict[str, Any]]:
    raw = _first_value(value)
    if raw is None:
        return []

    items: list[Any]
    if isinstance(raw, list):
        items = list(raw)
    elif isinstance(raw, tuple):
        items = list(raw)
    else:
        items = [raw]

    output: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        output.append(
            {
                "label": item.get("label"),
                "changes": dict(item.get("changes")) if isinstance(item.get("changes"), dict) else {},
            }
        )
    return output


def _has_connected_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        # Prompt validation may pass unresolved links as [node_id, output_index].
        if len(value) == 2 and isinstance(value[1], int):
            return True
        for item in value:
            if _has_connected_payload(item):
                return True
        return False
    return True


def _basename(value: str) -> str:
    return normalize_relative_path(value).split("/")[-1]


def _filename_stem(value: str) -> str:
    basename = _basename(value)
    if not basename:
        return ""
    return Path(basename).stem


def _format_float(value: Any) -> str | None:
    try:
        return format(float(value), "g")
    except Exception:
        return None


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _trim_civitai_text_for_label(value: Any) -> str | None:
    text = _text_or_none(value)
    if text is None:
        return None
    if len(text) < 17:
        return text

    separator_indexes = [index for index in (text.find("-"), text.find("/"), text.find("_"), text.find("|")) if index >= 0]
    if len(separator_indexes) > 0:
        trimmed = text[: min(separator_indexes)].rstrip()
        if trimmed and len(trimmed) < len(text):
            return trimmed

    trimmed = text.rstrip()
    if trimmed.endswith(")"):
        opening_index = trimmed.rfind("(")
        if opening_index > 0:
            candidate = trimmed[:opening_index].rstrip()
            if candidate and len(candidate) < len(text):
                return candidate

    return text


def _options_for_folder(folder_name: str) -> tuple[str, ...]:
    if folder_name == Const.MODEL_FOLDER_PATH_CHECKPOINTS:
        return Const.get_CHECKPOINT_OPTIONS()
    if folder_name == Const.MODEL_FOLDER_PATH_DIFFUSION_MODELS:
        return Const.get_DIFFUSION_MODEL_OPTIONS()
    if folder_name == Const.MODEL_FOLDER_PATH_UNET:
        return Const.get_UNET_MODEL_OPTIONS()
    return tuple()


def _match_relative_path_by_name(value: str, options: Sequence[str]) -> str | None:
    normalized = normalize_relative_path(value)
    if not normalized:
        return None

    option_list = [normalize_relative_path(option) for option in options if isinstance(option, str)]
    for option in option_list:
        if option == normalized:
            return option

    basename = _basename(normalized)
    if not basename:
        return None

    if Path(basename).suffix:
        for option in option_list:
            if _basename(option) == basename:
                return option
        return None

    for option in option_list:
        if _filename_stem(option) == basename:
            return option
    return None


def _resolve_model_relative_path(model_value: Mapping[str, Any]) -> tuple[str, str, str] | None:
    folder_name = _text_or_none(model_value.get(Const.MODEL_VALUE_FOLDER_PATHS_KEY))
    name = _text_or_none(model_value.get(Const.MODEL_VALUE_NAME_KEY))
    if folder_name is None or name is None:
        return None

    normalized_name = normalize_relative_path(name)
    if not normalized_name:
        return None

    matched = _match_relative_path_by_name(normalized_name, _options_for_folder(folder_name))
    relative_path = matched or normalized_name
    return folder_name, relative_path, name


def _civitai_model_label(folder_name: str, relative_path: str) -> str | None:
    try:
        pipeline = get_shared_metadata_pipeline(start=False)
    except Exception:
        return None
    if pipeline is None:
        return None

    try:
        info = pipeline.get_model_info_by_relative_path(
            folder_name=folder_name,
            relative_path=relative_path,
        )
    except Exception:
        info = None

    if not isinstance(info, Mapping):
        enqueue = getattr(pipeline, "enqueue_hash_priority", None)
        if callable(enqueue):
            try:
                enqueue(folder_name, relative_path)
            except Exception:
                pass
        return None

    name = _trim_civitai_text_for_label(info.get("civitai_model_name"))
    if name is None:
        return None

    version = _trim_civitai_text_for_label(info.get("civitai_model_version_name"))
    if version:
        return f"{name} {version}"
    return name


def _render_model_value_for_label(value: Any) -> str | None:
    if isinstance(value, Mapping):
        resolved = _resolve_model_relative_path(value)
        if resolved is None:
            name = _text_or_none(value.get(Const.MODEL_VALUE_NAME_KEY))
            if name is None:
                return None
            return _filename_stem(name) or name

        folder_name, relative_path, original_name = resolved
        civitai_label = _civitai_model_label(folder_name, relative_path)
        if civitai_label:
            return civitai_label
        return _filename_stem(relative_path) or _filename_stem(original_name) or original_name

    text = _text_or_none(value)
    if text is None:
        return None
    return _filename_stem(text) or text


def _render_lora_tag(item: Any) -> str | None:
    if not isinstance(item, Mapping):
        return None
    name = _text_or_none(item.get("name"))
    strength = _format_float(item.get("strength"))
    if name is None or strength is None:
        return None
    stem = _filename_stem(name)
    if not stem:
        return None
    return f"<lora:{stem}:{strength}>"


def _render_lora_stack_for_label(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    tags = [tag for tag in (_render_lora_tag(item) for item in value) if tag]
    if len(tags) == 0:
        return None
    return " ".join(tags)


def _render_clip_for_label(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None

    clip_names = value.get(Const.CLIP_VALUE_NAMES_KEY)
    if not isinstance(clip_names, list):
        return None

    rendered = [
        _filename_stem(text)
        for text in (_text_or_none(item) for item in clip_names)
        if text
    ]
    rendered = [item for item in rendered if item]
    if len(rendered) == 0:
        return None
    return f"{Const.IMAGEINFO_CLIP}: {' + '.join(rendered)}"


def _prompt_summary(value: Any) -> str | None:
    text = _text_or_none(value)
    if text is None:
        return None
    normalized = remove_prompt_comments(text)
    normalized = strip_prompt_weights(normalized)
    normalized = normalize_prompt_tokens(normalized)
    tokens = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(tokens) == 0:
        return None
    return ", ".join(tokens[:5])


def _render_prefixed_text(key: str, value: Any) -> str | None:
    text = _text_or_none(value)
    if text is None:
        return None
    return f"{key}: {text}"


def _render_float_field(key: str, value: Any) -> str | None:
    rendered = _format_float(value)
    if rendered is None:
        return None
    return f"{key}: {rendered}"


def _render_int_field(key: str, value: Any) -> str | None:
    try:
        rendered = str(int(value))
    except Exception:
        return None
    return f"{key}: {rendered}"


def _normalized_label_for_modifier(modifier: dict[str, Any]) -> str:
    explicit = modifier.get("label")
    if explicit is not None:
        text = str(explicit).strip()
        if text:
            return text

    changes = modifier.get("changes")
    if not isinstance(changes, dict) or len(changes) == 0:
        return "(no change)"

    first_line_parts: list[str] = []

    model_parts = [
        part
        for part in (
            _render_model_value_for_label(changes.get(Const.IMAGEINFO_MODEL))
            if Const.IMAGEINFO_MODEL in changes
            else None,
            _render_model_value_for_label(changes.get(Const.IMAGEINFO_REFINER_MODEL))
            if Const.IMAGEINFO_REFINER_MODEL in changes
            else None,
            _render_model_value_for_label(changes.get(Const.IMAGEINFO_DETAILER_MODEL))
            if Const.IMAGEINFO_DETAILER_MODEL in changes
            else None,
        )
        if part
    ]
    if len(model_parts) > 0:
        first_line_parts.append(" + ".join(model_parts))

    renderers = (
        (Const.IMAGEINFO_LORA_STACK, _render_lora_stack_for_label),
        (Const.IMAGEINFO_CLIP, _render_clip_for_label),
        (
            Const.IMAGEINFO_VAE,
            lambda value: _render_prefixed_text(
                Const.IMAGEINFO_VAE,
                _filename_stem(text) if (text := _text_or_none(value)) else None,
            ),
        ),
        (Const.IMAGEINFO_STEPS, lambda value: _render_int_field(Const.IMAGEINFO_STEPS, value)),
        (Const.IMAGEINFO_SAMPLER, lambda value: _render_prefixed_text(Const.IMAGEINFO_SAMPLER, value)),
        (Const.IMAGEINFO_SCHEDULER, lambda value: _render_prefixed_text(Const.IMAGEINFO_SCHEDULER, value)),
        (Const.IMAGEINFO_CFG, lambda value: _render_float_field(Const.IMAGEINFO_CFG, value)),
        (Const.IMAGEINFO_SEED, lambda value: _render_int_field(Const.IMAGEINFO_SEED, value)),
    )
    for key, renderer in renderers:
        if key not in changes:
            continue
        rendered = renderer(changes.get(key))
        if rendered:
            first_line_parts.append(rendered)

    lines: list[str] = []
    if len(first_line_parts) > 0:
        lines.append(", ".join(first_line_parts))

    positive_summary = _prompt_summary(changes.get(Const.IMAGEINFO_POSITIVE)) if Const.IMAGEINFO_POSITIVE in changes else None
    if positive_summary:
        lines.append(positive_summary)

    negative_summary = _prompt_summary(changes.get(Const.IMAGEINFO_NEGATIVE)) if Const.IMAGEINFO_NEGATIVE in changes else None
    if negative_summary:
        lines.append(f"{Const.IMAGEINFO_NEGATIVE}: {negative_summary}")

    if len(lines) == 0:
        return "(no change)"
    return "\n".join(lines)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None

    try:
        return str(value)
    except Exception:
        return None


def _combined_prompt_value(base_value: Any, appended_value: Any) -> str | None:
    return combine_prompt_text(_string_or_none(base_value), _string_or_none(appended_value))


def _lora_stack_or_none(value: Any) -> list[Any] | None:
    if not isinstance(value, list):
        return None
    return list(value)


def _combined_lora_stack(base_value: Any, appended_value: Any) -> list[Any] | None:
    base_stack = _lora_stack_or_none(base_value)
    appended_stack = _lora_stack_or_none(appended_value)

    if base_stack is None and appended_stack is None:
        return None
    if base_stack is None:
        return appended_stack
    if appended_stack is None:
        return base_stack

    merged = list(base_stack)
    merged.extend(appended_stack)
    return merged


def _apply_modifier(base_image_info: dict[str, Any], modifier: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(base_image_info)
    changes = modifier.get("changes")
    if not isinstance(changes, dict):
        return output

    for key, value in changes.items():
        if key == Const.IMAGEINFO_EXTRAS:
            merged = dict(output.get(Const.IMAGEINFO_EXTRAS)) if isinstance(output.get(Const.IMAGEINFO_EXTRAS), dict) else {}
            if isinstance(value, dict):
                merged.update(value)
            if len(merged) == 0:
                output.pop(Const.IMAGEINFO_EXTRAS, None)
            else:
                output[Const.IMAGEINFO_EXTRAS] = merged
            continue

        if key in (Const.IMAGEINFO_POSITIVE, Const.IMAGEINFO_NEGATIVE):
            combined_prompt = _combined_prompt_value(output.get(key), value)
            if combined_prompt is None:
                output.pop(key, None)
            else:
                output[key] = combined_prompt
            continue

        if key == Const.IMAGEINFO_LORA_STACK:
            combined_lora_stack = _combined_lora_stack(output.get(key), value)
            if combined_lora_stack is None:
                output.pop(key, None)
            else:
                output[key] = combined_lora_stack
            continue

        output[key] = deepcopy(value)

    return output


def _inject_xy_extras(
    image_info: dict[str, Any],
    x_label: str,
    y_label: str,
    x_index: int,
    y_index: int,
    cell_index: int,
) -> None:
    extras = dict(image_info.get(Const.IMAGEINFO_EXTRAS)) if isinstance(image_info.get(Const.IMAGEINFO_EXTRAS), dict) else {}
    extras["xy.x.index"] = int(x_index)
    extras["xy.y.index"] = int(y_index)
    extras["xy.cell_index"] = int(cell_index)
    extras["xy.x.label"] = x_label
    extras["xy.y.label"] = y_label
    image_info[Const.IMAGEINFO_EXTRAS] = extras


class XYPlotStart(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-XYPlotStart",
            display_name="XY Plot Start",
            category=Const.CATEGORY_XYPLOT,
            inputs=[
                Const.IMAGEINFO_TYPE.Input(
                    Const.IMAGEINFO,
                    tooltip="Base image_info (never modified in-place)",
                ),
                Const.XY_PLOT_MODIFIER_TYPE.Input(
                    "x_modifiers",
                    tooltip="Required modifier array for X axis",
                ),
                Const.XY_PLOT_MODIFIER_TYPE.Input(
                    "y_modifiers",
                    optional=True,
                    tooltip="Optional modifier array for Y axis. If omitted, one-row mode is used.",
                ),
                c_io.Combo.Input(
                    "execution_mode",
                    options=MODE_OPTIONS,
                    default=MODE_LOOP,
                    tooltip="loop: one item per turn, list: all items",
                ),
                c_io.Int.Input(
                    "loop_index",
                    default=0,
                    min=0,
                    max=999999,
                    advanced=True,
                    tooltip="Internal index for recursive loop expansion",
                ),
            ],
            outputs=[
                Const.IMAGEINFO_TYPE.Output(
                    Cast.out_id(Const.IMAGEINFO),
                    display_name=Const.IMAGEINFO,
                    is_output_list=True,
                ),
                Const.XY_PLOT_INFO_TYPE.Output(
                    Cast.out_id("xy_plot_info"),
                    display_name="xy_plot_info",
                ),
                Const.LOOP_CONTROL_TYPE.Output(
                    Cast.out_id("loop_control"),
                    display_name="loop_control",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image_info: Any,
        x_modifiers: Any,
        y_modifiers: Any = None,
        execution_mode: Any = MODE_LOOP,
        loop_index: int = 0,
    ) -> c_io.NodeOutput:
        base_image_info = dict(image_info) if isinstance(image_info, dict) else {}
        x_items = _modifier_items(x_modifiers)
        if len(x_items) == 0:
            raise ValueError("x_modifiers is required")

        y_items = _modifier_items(y_modifiers)
        mode = _normalized_mode(execution_mode)

        cells: list[dict[str, Any]] = []

        if len(y_items) > 0:
            x_labels = [_normalized_label_for_modifier(item) for item in x_items]
            y_labels = [_normalized_label_for_modifier(item) for item in y_items]

            for y_index, y_modifier in enumerate(y_items):
                for x_index, x_modifier in enumerate(x_items):
                    cell_info = _apply_modifier(base_image_info, x_modifier)
                    cell_info = _apply_modifier(cell_info, y_modifier)
                    cell_index = len(cells)
                    _inject_xy_extras(cell_info, x_labels[x_index], y_labels[y_index], x_index, y_index, cell_index)
                    cells.append(cell_info)

            x_count = len(x_items)
            y_count = len(y_items)
        else:
            x_labels = [_normalized_label_for_modifier(item) for item in x_items]
            y_labels = [""]

            for x_index, x_modifier in enumerate(x_items):
                cell_info = _apply_modifier(base_image_info, x_modifier)
                cell_index = len(cells)
                _inject_xy_extras(cell_info, x_labels[x_index], y_labels[0], x_index, 0, cell_index)
                cells.append(cell_info)

            x_count = len(x_items)
            y_count = 1

        xy_plot_info = {
            "x_labels": x_labels,
            "y_labels": y_labels,
            "x_count": x_count,
            "y_count": y_count,
            "cell_count": len(cells),
            "order": "row_major",
        }

        if mode == MODE_LIST:
            return c_io.NodeOutput(cells, xy_plot_info, None)

        index = int(loop_index)
        if index < 0:
            index = 0
        if index >= len(cells):
            index = len(cells) - 1

        flow_control = ("ipt_xy_plot_start", index)
        return c_io.NodeOutput([cells[index]], xy_plot_info, flow_control)
