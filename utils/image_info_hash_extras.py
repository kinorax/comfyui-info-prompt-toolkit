# Copyright 2026 kinorax
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .file_hash_cache import normalize_relative_path
from .model_merge import is_model_merge_value
from .model_lora_metadata_pipeline import ModelLoraMetadataPipeline, get_shared_metadata_pipeline

try:
    import folder_paths  # type: ignore
except Exception:
    folder_paths = None  # type: ignore[assignment]


IMAGEINFO_MODEL = "model"
IMAGEINFO_REFINER_MODEL = "refiner"
IMAGEINFO_DETAILER_MODEL = "detailer"
IMAGEINFO_LORA_STACK = "lora_stack"
IMAGEINFO_CLIP = "clip"
IMAGEINFO_VAE = "vae"
IMAGEINFO_EXTRAS = "extras"

EXTRA_LORA_HASHES = "Lora hashes"
EXTRA_CLIP_HASHES = "Clip hashes"
EXTRA_VAE_HASH = "VAE hash"
EXTRA_MODEL_HASH = "Model hash"
EXTRA_REFINER_HASH = "Refiner hash"
EXTRA_DETAILER_HASH = "Detailer hash"
EXTRA_HASHES = "Hashes"

HASH_ALGO = "sha256"


def clear_representative_hash_extras(image_info: Mapping[str, Any] | None) -> dict[str, Any]:
    output = dict(image_info) if isinstance(image_info, Mapping) else {}
    extras_raw = output.get(IMAGEINFO_EXTRAS)
    extras: dict[str, Any] = dict(extras_raw) if isinstance(extras_raw, Mapping) else {}

    extras.pop(EXTRA_LORA_HASHES, None)
    extras.pop(EXTRA_CLIP_HASHES, None)
    extras.pop(EXTRA_VAE_HASH, None)
    extras.pop(EXTRA_MODEL_HASH, None)
    extras.pop(EXTRA_REFINER_HASH, None)
    extras.pop(EXTRA_DETAILER_HASH, None)
    extras.pop(EXTRA_HASHES, None)

    if extras:
        output[IMAGEINFO_EXTRAS] = extras
    else:
        output.pop(IMAGEINFO_EXTRAS, None)
    return output


def add_civitai_hash_extras(image_info: Mapping[str, Any] | None) -> dict[str, Any]:
    output = dict(image_info) if isinstance(image_info, Mapping) else {}
    extras_raw = output.get(IMAGEINFO_EXTRAS)
    extras: dict[str, Any] = dict(extras_raw) if isinstance(extras_raw, Mapping) else {}

    pipeline = get_shared_metadata_pipeline(start=True)
    option_cache: dict[str, tuple[str, ...]] = {}

    model_hash = _hash_model(output.get(IMAGEINFO_MODEL), pipeline, option_cache)
    _set_extra_or_delete(extras, EXTRA_MODEL_HASH, model_hash)

    refiner_hash = _hash_refiner(
        output.get(IMAGEINFO_REFINER_MODEL),
        output.get(IMAGEINFO_MODEL),
        pipeline,
        option_cache,
    )
    _set_extra_or_delete(extras, EXTRA_REFINER_HASH, refiner_hash)

    detailer_hash = _hash_model_with_optional_folder_fallback(
        output.get(IMAGEINFO_DETAILER_MODEL),
        output.get(IMAGEINFO_MODEL),
        pipeline,
        option_cache,
    )
    _set_extra_or_delete(extras, EXTRA_DETAILER_HASH, detailer_hash)

    vae_hash = _hash_single_file("vae", output.get(IMAGEINFO_VAE), pipeline, option_cache)
    _set_extra_or_delete(extras, EXTRA_VAE_HASH, vae_hash)

    lora_hashes = _hash_loras(output.get(IMAGEINFO_LORA_STACK), pipeline, option_cache)
    _set_extra_or_delete(extras, EXTRA_LORA_HASHES, lora_hashes)

    clip_hashes = _hash_clips(output.get(IMAGEINFO_CLIP), pipeline, option_cache)
    _set_extra_or_delete(extras, EXTRA_CLIP_HASHES, clip_hashes)

    if extras:
        output[IMAGEINFO_EXTRAS] = extras
    else:
        output.pop(IMAGEINFO_EXTRAS, None)
    return output


def _set_extra_or_delete(extras: dict[str, Any], key: str, value: str | None) -> None:
    if value:
        extras[key] = value
    else:
        extras.pop(key, None)


def _hash_model(
    model_value: Any,
    pipeline: ModelLoraMetadataPipeline,
    option_cache: dict[str, tuple[str, ...]],
) -> str | None:
    if not isinstance(model_value, Mapping):
        return None
    if is_model_merge_value(model_value):
        return None

    folder_name = _as_text(model_value.get("folder_paths"))
    model_name = _as_text(model_value.get("name"))
    if not folder_name or not model_name:
        return None

    return _hash_by_folder_and_name(folder_name, model_name, pipeline, option_cache)


def _hash_refiner(
    refiner_value: Any,
    model_value: Any,
    pipeline: ModelLoraMetadataPipeline,
    option_cache: dict[str, tuple[str, ...]],
) -> str | None:
    return _hash_model_with_optional_folder_fallback(
        refiner_value,
        model_value,
        pipeline,
        option_cache,
    )


def _hash_model_with_optional_folder_fallback(
    model_like_value: Any,
    fallback_model_value: Any,
    pipeline: ModelLoraMetadataPipeline,
    option_cache: dict[str, tuple[str, ...]],
) -> str | None:
    if is_model_merge_value(model_like_value):
        return None
    if isinstance(model_like_value, Mapping):
        folder_name = _as_text(model_like_value.get("folder_paths"))
        model_like_name = _as_text(model_like_value.get("name"))
    else:
        folder_name = None
        model_like_name = _as_text(model_like_value)

    if not folder_name and isinstance(fallback_model_value, Mapping):
        folder_name = _as_text(fallback_model_value.get("folder_paths"))

    if not folder_name or not model_like_name:
        return None
    return _hash_by_folder_and_name(folder_name, model_like_name, pipeline, option_cache)


def _hash_single_file(
    folder_name: str,
    value: Any,
    pipeline: ModelLoraMetadataPipeline,
    option_cache: dict[str, tuple[str, ...]],
) -> str | None:
    file_name = _as_text(value)
    if not file_name:
        return None
    return _hash_by_folder_and_name(folder_name, file_name, pipeline, option_cache)


def _hash_loras(
    lora_stack_value: Any,
    pipeline: ModelLoraMetadataPipeline,
    option_cache: dict[str, tuple[str, ...]],
) -> str | None:
    if not isinstance(lora_stack_value, list):
        return None

    rendered: list[str] = []
    for item in lora_stack_value:
        if not isinstance(item, Mapping):
            continue

        lora_name = _as_text(item.get("name"))
        if not lora_name:
            continue

        digest = _hash_by_folder_and_name("loras", lora_name, pipeline, option_cache)
        if not digest:
            continue

        display_name = _filename_stem(lora_name)
        if not display_name:
            continue
        rendered.append(f"{display_name}: {digest}")

    if not rendered:
        return None
    return ", ".join(rendered)


def _hash_clips(
    clip_value: Any,
    pipeline: ModelLoraMetadataPipeline,
    option_cache: dict[str, tuple[str, ...]],
) -> str | None:
    if not isinstance(clip_value, Mapping):
        return None

    clip_names = clip_value.get("clip_names")
    if not isinstance(clip_names, list):
        return None

    rendered: list[str] = []
    for index, raw_name in enumerate(clip_names, start=1):
        clip_name = _as_text(raw_name)
        if not clip_name:
            continue
        digest = _hash_by_folder_and_name("text_encoders", clip_name, pipeline, option_cache)
        if not digest:
            continue
        rendered.append(f"{index}: {digest}")

    if not rendered:
        return None
    return ", ".join(rendered)


def _hash_by_folder_and_name(
    folder_name: str,
    file_name: str,
    pipeline: ModelLoraMetadataPipeline,
    option_cache: dict[str, tuple[str, ...]],
) -> str | None:
    resolved = _resolve_file(folder_name, file_name, option_cache)
    if not resolved:
        return None

    relative_path, _ = resolved
    digest = pipeline.get_hash_by_relative_path(folder_name, relative_path, hash_algo=HASH_ALGO)
    if digest:
        return digest

    pipeline.enqueue_hash_priority(folder_name, relative_path)
    return None


def _resolve_file(
    folder_name: str,
    raw_name: str,
    option_cache: dict[str, tuple[str, ...]],
) -> tuple[str, str] | None:
    if folder_paths is None:
        return None

    relative = normalize_relative_path(raw_name)
    if not relative:
        return None

    full_path = _get_full_path(folder_name, relative)
    if full_path:
        return relative, full_path

    options = _get_folder_options(folder_name, option_cache)
    matched = _match_option_for_name(relative, options)
    if not matched:
        return None

    matched_path = _get_full_path(folder_name, matched)
    if not matched_path:
        return None
    return matched, matched_path


def _get_full_path(folder_name: str, relative: str) -> str | None:
    if folder_paths is None:
        return None
    getter = getattr(folder_paths, "get_full_path", None)
    if not callable(getter):
        return None
    try:
        path = getter(folder_name, relative)
    except Exception:
        return None
    if not isinstance(path, str) or not path:
        return None
    if not os.path.isfile(path):
        return None
    return os.path.abspath(path)


def _get_folder_options(
    folder_name: str,
    option_cache: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    if folder_name in option_cache:
        return option_cache[folder_name]
    if folder_paths is None:
        option_cache[folder_name] = tuple()
        return option_cache[folder_name]

    getter = getattr(folder_paths, "get_filename_list", None)
    if not callable(getter):
        option_cache[folder_name] = tuple()
        return option_cache[folder_name]

    try:
        options = tuple(str(item) for item in getter(folder_name))
    except Exception:
        options = tuple()
    option_cache[folder_name] = options
    return options


def _match_option_for_name(value: str, options: Iterable[str]) -> str | None:
    target = _basename(value)
    if not target:
        return None

    option_list = [normalize_relative_path(option) for option in options if isinstance(option, str)]
    if _has_extension(target):
        for option in option_list:
            if _basename(option) == target:
                return option
        return None

    for option in option_list:
        if _filename_stem(option) == target:
            return option
    return None


def _basename(value: str) -> str:
    return normalize_relative_path(value).split("/")[-1]


def _filename_stem(value: str) -> str:
    basename = _basename(value)
    if not basename:
        return ""
    return Path(basename).stem


def _has_extension(filename: str) -> bool:
    return bool(Path(filename).suffix)


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
