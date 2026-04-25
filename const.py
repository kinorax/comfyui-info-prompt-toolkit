# Copyright 2026 kinorax
from pathlib import Path
from functools import lru_cache
import comfy.samplers
import folder_paths
from comfy_api.latest import io as c_io
from .utils.model_runtime_settings import (
    MODEL_RUNTIME_SETTING_CLIP_LAST_LAYER_KEY,
    MODEL_RUNTIME_SETTING_SD3_SHIFT_KEY,
)

CATEGORY_IMAGEINFO = "Info-Prompt-Toolkit/ImageInfo"
CATEGORY_PROMPT = "Info-Prompt-Toolkit/Prompt"
CATEGORY_MASK = "Info-Prompt-Toolkit/Mask"
CATEGORY_XYPLOT = "Info-Prompt-Toolkit/XYPlot"
CATEGORY_DEBUG = "Info-Prompt-Toolkit/Debug"

IMAGEINFO_TYPE = c_io.Custom("IPT-IMAGEINFO")
MODEL_TYPE = c_io.Custom("IPT-Model")
LORA_STACK_TYPE = c_io.Custom("IPT-LoraStack")
CLIP_TYPE = c_io.Custom("IPT-Clip")
IMAGEINFO_EXTRAS_TYPE = c_io.Custom("IPT-ImageInfoExtras")
SIZE_TYPE = c_io.Custom("IPT-Size")
SAMPLER_PARAMS_TYPE = c_io.Custom("IPT-SamplerParams")
XY_PLOT_MODIFIER_TYPE = c_io.Custom("IPT-XYPlotModifier")
XY_PLOT_INFO_TYPE = c_io.Custom("IPT-XYPlotInfo")
DETAILER_CONTROL_TYPE = c_io.Custom("IPT-DetailerControl")
LOOP_CONTROL_TYPE = c_io.FlowControl
IMAGEINFO = "image_info"
VIDEOINFO = "video_info"
IMAGEINFO_FALLBACK = "fallback_image_info"
IMAGEINFO_MODEL = "model"
IMAGEINFO_REFINER_MODEL = "refiner"
IMAGEINFO_DETAILER_MODEL = "detailer"
IMAGEINFO_CHECKPOINT = "checkpoint"
IMAGEINFO_DIFFUSION_MODEL = "diffusion_model"
IMAGEINFO_UNET_MODEL = "unet_model"
IMAGEINFO_LORA_STACK = "lora_stack"
IMAGEINFO_CLIP = "clip"
IMAGEINFO_VAE = "vae"
IMAGEINFO_EXTRAS = "extras"
IMAGEINFO_POSITIVE = "positive"
IMAGEINFO_NEGATIVE = "negative"
IMAGEINFO_STEPS = "steps"
IMAGEINFO_SAMPLER = "sampler"
IMAGEINFO_SCHEDULER = "scheduler"
IMAGEINFO_CFG = "cfg"
IMAGEINFO_SEED = "seed"
IMAGEINFO_WIDTH = "width"
IMAGEINFO_HEIGHT = "height"
MODEL_FOLDER_PATH_CHECKPOINTS = "checkpoints"
MODEL_FOLDER_PATH_DIFFUSION_MODELS = "diffusion_models"
MODEL_FOLDER_PATH_UNET = "unet"
MODEL_FOLDER_PATH_LORAS = "loras"
MODEL_FOLDER_PATH_TEXT_ENCODERS = "text_encoders"
MODEL_FOLDER_PATH_VAE = "vae"
CLIP_VALUE_NAMES_KEY = "clip_names"
CLIP_VALUE_STOP_AT_CLIP_LAYER_KEY = MODEL_RUNTIME_SETTING_CLIP_LAST_LAYER_KEY
MODEL_VALUE_NAME_KEY = "name"
MODEL_VALUE_FOLDER_PATHS_KEY = "folder_paths"
MODEL_VALUE_TYPE_KEY = "type"
MODEL_VALUE_DEVICE_KEY = "device"
MODEL_VALUE_WEIGHT_DTYPE_KEY = "weight_dtype"
MODEL_VALUE_SD3_SHIFT_KEY = MODEL_RUNTIME_SETTING_SD3_SHIFT_KEY
MODEL_VALUE_AURAFLOW_SHIFT_KEY = "model_sampling_auraflow_shift"
WEIGHT_DTYPE_OPTIONS_FALLBACK = ("default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2")
CLIP_TYPE_OPTIONS_FALLBACK = (
    "stable_diffusion",
    "stable_cascade",
    "sd3",
    "stable_audio",
    "mochi",
    "ltxv",
    "pixart",
    "cosmos",
    "lumina2",
    "wan",
    "hidream",
    "chroma",
    "ace",
    "omnigen2",
    "qwen_image",
    "hunyuan_image",
    "flux2",
    "ovis",
)
DUAL_CLIP_TYPE_OPTIONS_FALLBACK = (
    "sdxl",
    "sd3",
    "flux",
    "hunyuan_video",
    "hidream",
    "hunyuan_image",
    "hunyuan_video_15",
    "kandinsky5",
    "kandinsky5_image",
    "ltxv",
    "newbie",
    "ace",
)
CLIP_DEVICE_OPTIONS_FALLBACK = ("default", "cpu")
INT64_MIN = -(2 ** 63)
INT64_MAX = (2 ** 63) - 1
MIN_RESOLUTION = 16
MAX_RESOLUTION = 16384


def make_model_value(
    name: str | None,
    folder_path_name: str,
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if name is None:
        return None
    output: dict[str, object] = {
        MODEL_VALUE_NAME_KEY: name,
        MODEL_VALUE_FOLDER_PATHS_KEY: folder_path_name,
    }
    return _apply_extra_fields(output, extra_fields)


def make_clip_value(
    clip_names: list[str] | tuple[str, ...] | None,
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if not isinstance(clip_names, (list, tuple)) or not clip_names:
        return None

    normalized_clip_names: list[str] = []
    for raw_name in clip_names:
        if not isinstance(raw_name, str):
            return None
        normalized_name = raw_name.strip()
        if not normalized_name:
            return None
        normalized_clip_names.append(normalized_name)

    output: dict[str, object] = {
        CLIP_VALUE_NAMES_KEY: normalized_clip_names,
        MODEL_VALUE_FOLDER_PATHS_KEY: MODEL_FOLDER_PATH_TEXT_ENCODERS,
    }
    return _apply_extra_fields(output, extra_fields)


def make_lora_stack_item(name: str | None, strength: float) -> dict[str, str | float] | None:
    if name is None:
        return None
    return {
        "name": name,
        "strength": float(strength),
    }


def _apply_extra_fields(
    output: dict[str, object],
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    if isinstance(extra_fields, dict):
        for key, value in extra_fields.items():
            if not isinstance(key, str) or not key or value is None:
                continue
            output[key] = value
    return output


SAMPLER_OPTIONS: tuple[str, ...] = tuple(comfy.samplers.SAMPLER_NAMES)
SCHEDULER_OPTIONS: tuple[str, ...] = tuple(comfy.samplers.SCHEDULER_NAMES)


def _safe_print(message: str) -> None:
    try:
        print(message)
    except Exception:
        pass


def _normalize_combo_options(options: object) -> tuple[str, ...] | None:
    if not isinstance(options, (list, tuple)):
        return None
    normalized = tuple(option for option in options if isinstance(option, str) and option)
    return normalized or None


def _get_core_loader_combo_options(
    loader_name: str,
    input_name: str,
    *,
    sections: tuple[str, ...] = ("required",),
) -> tuple[tuple[str, ...] | None, str]:
    try:
        import importlib

        comfy_nodes = importlib.import_module("nodes")
        loader_class = getattr(comfy_nodes, loader_name, None)
        if loader_class is None:
            return None, f"ComfyUI nodes.{loader_name} was not found"

        input_types = getattr(loader_class, "INPUT_TYPES", None)
        if not callable(input_types):
            return None, f"ComfyUI nodes.{loader_name}.INPUT_TYPES was not callable"

        input_type_dict = input_types()
        if not isinstance(input_type_dict, dict):
            return None, f"ComfyUI nodes.{loader_name}.INPUT_TYPES() did not return a dict"

        for section_name in sections:
            section = input_type_dict.get(section_name)
            if section is None:
                continue
            if not isinstance(section, dict):
                return None, f"ComfyUI nodes.{loader_name} INPUT_TYPES().{section_name} was missing or malformed"

            entry = section.get(input_name)
            if entry is None:
                continue
            if not isinstance(entry, tuple) or not entry:
                return None, f"ComfyUI nodes.{loader_name} {section_name}.{input_name} was missing or malformed"

            raw_options = entry[0]
            normalized = _normalize_combo_options(raw_options)
            if normalized is not None:
                return normalized, ""
            if isinstance(raw_options, (list, tuple)):
                return None, f"ComfyUI nodes.{loader_name} {input_name} options were empty after normalization"
            return None, f"ComfyUI nodes.{loader_name} {input_name} options were not a list or tuple"

        return None, f"ComfyUI nodes.{loader_name} did not expose {input_name} in INPUT_TYPES()"
    except Exception as exc:
        return None, f"failed to resolve ComfyUI nodes.{loader_name} {input_name} options: {exc}"


@lru_cache(maxsize=1)
def get_WEIGHT_DTYPE_OPTIONS() -> tuple[str, ...]:
    # Prefer the core ComfyUI UNETLoader options so this extension stays aligned.
    options, fallback_reason = _get_core_loader_combo_options("UNETLoader", "weight_dtype")
    if options is not None:
        return options

    _safe_print(f"[IPT][weight_dtype] fallback to bundled options: {fallback_reason}")
    return WEIGHT_DTYPE_OPTIONS_FALLBACK


def get_LORA_OPTIONS() -> tuple[str, ...]:
    return tuple(folder_paths.get_filename_list("loras"))


def get_CLIP_NAME_OPTIONS() -> tuple[str, ...]:
    return tuple(folder_paths.get_filename_list(MODEL_FOLDER_PATH_TEXT_ENCODERS))


@lru_cache(maxsize=1)
def get_CLIP_TYPE_OPTIONS() -> tuple[str, ...]:
    options, fallback_reason = _get_core_loader_combo_options("CLIPLoader", "type")
    if options is not None:
        return options

    _safe_print(f"[IPT][clip.type] fallback to bundled options: {fallback_reason}")
    return CLIP_TYPE_OPTIONS_FALLBACK


@lru_cache(maxsize=1)
def get_CLIP_DEVICE_OPTIONS() -> tuple[str, ...]:
    options, fallback_reason = _get_core_loader_combo_options(
        "CLIPLoader",
        "device",
        sections=("optional", "required"),
    )
    if options is not None:
        return options

    _safe_print(f"[IPT][clip.device] fallback to bundled options: {fallback_reason}")
    return CLIP_DEVICE_OPTIONS_FALLBACK


@lru_cache(maxsize=1)
def get_DUAL_CLIP_TYPE_OPTIONS() -> tuple[str, ...]:
    options, fallback_reason = _get_core_loader_combo_options("DualCLIPLoader", "type")
    if options is not None:
        return options

    _safe_print(f"[IPT][dual_clip.type] fallback to bundled options: {fallback_reason}")
    return DUAL_CLIP_TYPE_OPTIONS_FALLBACK


@lru_cache(maxsize=1)
def get_DUAL_CLIP_DEVICE_OPTIONS() -> tuple[str, ...]:
    options, fallback_reason = _get_core_loader_combo_options(
        "DualCLIPLoader",
        "device",
        sections=("optional", "required"),
    )
    if options is not None:
        return options

    _safe_print(f"[IPT][dual_clip.device] fallback to bundled options: {fallback_reason}")
    return CLIP_DEVICE_OPTIONS_FALLBACK


def get_CHECKPOINT_OPTIONS() -> tuple[str, ...]:
    return tuple(folder_paths.get_filename_list("checkpoints"))


def get_VAE_OPTIONS() -> tuple[str, ...]:
    return tuple(folder_paths.get_filename_list("vae"))


def get_UNET_MODEL_OPTIONS() -> tuple[str, ...]:
    models_dir = getattr(folder_paths, "models_dir", None)
    if not models_dir:
        return tuple()

    unet_root = Path(models_dir) / "unet"
    if not unet_root.exists():
        return tuple()

    extensions = getattr(folder_paths, "supported_pt_extensions", None)
    normalized_extensions = {str(ext).lower() for ext in extensions} if extensions else None
    if normalized_extensions is not None:
        normalized_extensions.add(".gguf")

    files: list[str] = []
    for path in unet_root.rglob("*"):
        if not path.is_file():
            continue

        if normalized_extensions is not None and path.suffix.lower() not in normalized_extensions:
            continue

        files.append(path.relative_to(unet_root).as_posix())

    return tuple(sorted(set(files), key=str.casefold))


def get_DIFFUSION_MODEL_OPTIONS() -> tuple[str, ...]:
    return tuple(folder_paths.get_filename_list("diffusion_models"))
