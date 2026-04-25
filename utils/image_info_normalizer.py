# Copyright 2026 kinorax
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .. import const as Const
from .a1111_infotext import INTERNAL_HASH_HINTS_KEY
from .file_hash_cache import normalize_relative_path
from .model_lora_metadata_pipeline import get_shared_metadata_pipeline


_EXTRA_LORA_HASHES = "Lora hashes"
_EXTRA_CLIP_HASHES = "Clip hashes"
_EXTRA_MODEL_HASH = "Model hash"
_EXTRA_REFINER_HASH = "Refiner hash"
_EXTRA_DETAILER_HASH = "Detailer hash"
_EXTRA_VAE_HASH = "VAE hash"

_HASH_TOKEN_RE = re.compile(r"[0-9a-fA-F]{8,64}")
_HASH_TEXT_RE = re.compile(r"^[0-9a-f]{8,64}$")

_HASH_ALGO_SHA256 = "sha256"
_HASH_ALGO_SHA1 = "sha1"
_HASH_ALGO_MD5 = "md5"
_HASH_ALGO_CRC32 = "crc32"
_HASH_ALGO_A1111_LEGACY = "a1111_legacy"

_MODEL_FOLDER_NAMES: tuple[str, ...] = (
    Const.MODEL_FOLDER_PATH_CHECKPOINTS,
    Const.MODEL_FOLDER_PATH_DIFFUSION_MODELS,
    Const.MODEL_FOLDER_PATH_UNET,
)


def _basename(value: str) -> str:
    return value.replace("\\", "/").split("/")[-1]


def _has_extension(filename: str) -> bool:
    return bool(Path(filename).suffix)


def _name_without_extension(filename: str) -> str:
    return Path(filename).stem


def _normalize_hash_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    if text.startswith("sha256:"):
        text = text[7:].strip()

    if _HASH_TEXT_RE.fullmatch(text):
        return text

    token = _HASH_TOKEN_RE.search(text)
    if token is None:
        return None
    return token.group(0).lower()


def _normalize_hash_list(values: Any) -> list[str]:
    if values is None:
        return []

    if isinstance(values, (list, tuple, set)):
        iterable = values
    else:
        iterable = (values,)

    output: list[str] = []
    for raw in iterable:
        normalized = _normalize_hash_text(raw)
        if normalized and normalized not in output:
            output.append(normalized)
    return output


def _merge_hash_lists(*groups: Sequence[str]) -> list[str]:
    output: list[str] = []
    for group in groups:
        for digest in group:
            normalized = _normalize_hash_text(digest)
            if normalized and normalized not in output:
                output.append(normalized)
    return output


def _find_matching_options(value: str | None, options: Iterable[str]) -> list[str]:
    if not value:
        return []

    target = _basename(str(value).strip())
    if not target:
        return []

    option_list = [str(option) for option in options]
    if not option_list:
        return []

    if _has_extension(target):
        matched_with_ext: list[str] = []
        for option in option_list:
            if _basename(option) == target:
                matched_with_ext.append(option)
        return matched_with_ext

    matched_without_ext: list[str] = []
    for option in option_list:
        if _name_without_extension(_basename(option)) == target:
            matched_without_ext.append(option)
    return matched_without_ext


def _candidate_hash_algorithms(expected: str) -> tuple[str, ...]:
    length = len(expected)
    if length <= 8:
        return (
            _HASH_ALGO_A1111_LEGACY,
            _HASH_ALGO_CRC32,
            _HASH_ALGO_SHA256,
            _HASH_ALGO_MD5,
            _HASH_ALGO_SHA1,
        )
    if length <= 32:
        return (
            _HASH_ALGO_MD5,
            _HASH_ALGO_SHA256,
            _HASH_ALGO_SHA1,
            _HASH_ALGO_A1111_LEGACY,
            _HASH_ALGO_CRC32,
        )
    if length <= 40:
        return (
            _HASH_ALGO_SHA1,
            _HASH_ALGO_SHA256,
            _HASH_ALGO_MD5,
            _HASH_ALGO_A1111_LEGACY,
            _HASH_ALGO_CRC32,
        )
    return (
        _HASH_ALGO_SHA256,
        _HASH_ALGO_SHA1,
        _HASH_ALGO_MD5,
        _HASH_ALGO_A1111_LEGACY,
        _HASH_ALGO_CRC32,
    )


def _find_option_by_relative_path(options: Sequence[str], relative_path: str) -> str | None:
    target = normalize_relative_path(relative_path)
    if not target:
        return None
    for option in options:
        normalized = normalize_relative_path(option)
        if normalized == target:
            return option
    return None


def _resolve_option_by_hash(
    options: Iterable[str],
    folder_name: str,
    expected_hashes: Sequence[str] | None,
) -> str | None:
    normalized_hashes = _normalize_hash_list(expected_hashes)
    if not normalized_hashes:
        return None

    option_list = [str(option) for option in options if isinstance(option, str)]
    if not option_list:
        return None

    pipeline = get_shared_metadata_pipeline(start=False)
    if pipeline is None:
        return None
    for expected in normalized_hashes:
        relative_paths = pipeline.find_relative_paths_by_hash(
            folder_name=folder_name,
            hash_prefix=expected,
            preferred_algos=_candidate_hash_algorithms(expected),
        )
        if not relative_paths:
            continue
        matches = [
            matched
            for matched in (_find_option_by_relative_path(option_list, relative_path) for relative_path in relative_paths)
            if matched is not None
        ]
        deduped_matches: list[str] = []
        for match in matches:
            if match not in deduped_matches:
                deduped_matches.append(match)
        if len(deduped_matches) == 1:
            return deduped_matches[0]
    return None


def _resolve_option_with_hash_fallback(
    value: str | None,
    options: Iterable[str],
    folder_name: str,
    expected_hashes: Sequence[str] | None,
) -> str | None:
    matched_by_hash = _resolve_option_by_hash(options, folder_name, expected_hashes)
    if matched_by_hash:
        return matched_by_hash

    matches = _find_matching_options(value, options)
    if not matches:
        return None
    return matches[0]


def _lora_hash_hint_keys(name: str | None) -> list[str]:
    if not name:
        return []

    raw = str(name).strip()
    if not raw:
        return []

    basename = _basename(raw)
    stem = _name_without_extension(basename) if basename else ""

    output: list[str] = []
    for candidate in (raw, basename, stem):
        normalized = candidate.strip().lower()
        if normalized and normalized not in output:
            output.append(normalized)
    return output


def _append_lora_hash_hint(hints: dict[str, list[str]], key: str, digest: str) -> None:
    bucket = hints.setdefault(key, [])
    if digest not in bucket:
        bucket.append(digest)


def _parse_lora_hash_hints(extras: Mapping[str, Any]) -> dict[str, list[str]]:
    raw = extras.get(_EXTRA_LORA_HASHES)
    if raw is None:
        return {}

    text = str(raw).strip()
    if not text:
        return {}

    hints: dict[str, list[str]] = {}
    for part in text.split(","):
        segment = part.strip()
        if not segment:
            continue
        if ":" not in segment:
            continue
        name_part, hash_part = segment.split(":", 1)
        name = name_part.strip()
        digest = _normalize_hash_text(hash_part)
        if not name or not digest:
            continue
        for key in _lora_hash_hint_keys(name):
            _append_lora_hash_hint(hints, key, digest)
    return hints


def _clip_hash_hint_keys(index: int) -> list[str]:
    normalized_index = str(max(1, int(index)))
    return [
        normalized_index,
        f"clip{normalized_index}",
        f"clip {normalized_index}",
        f"clip_{normalized_index}",
    ]


def _parse_clip_hash_hints(extras: Mapping[str, Any]) -> dict[str, list[str]]:
    raw = extras.get(_EXTRA_CLIP_HASHES)
    text = str(raw).strip()
    if not text:
        return {}

    hints: dict[str, list[str]] = {}
    for part in text.split(","):
        segment = part.strip()
        if not segment or ":" not in segment:
            continue
        index_part, hash_part = segment.split(":", 1)
        index_text = str(index_part).strip()
        if not index_text:
            continue
        digest = _normalize_hash_text(hash_part)
        if not digest:
            continue
        try:
            index_int = int(index_text)
        except Exception:
            continue
        if index_int <= 0:
            continue
        for key in _clip_hash_hint_keys(index_int):
            _append_lora_hash_hint(hints, key, digest)
    return hints


def _lora_hash_hint_for_name(name: str | None, hints: Mapping[str, list[str]]) -> list[str]:
    if not hints:
        return []

    output: list[str] = []
    for key in _lora_hash_hint_keys(name):
        for digest in hints.get(key, []):
            if digest not in output:
                output.append(digest)
    return output


def _clip_hash_hint_for_index(index: int, hints: Mapping[str, list[str]]) -> list[str]:
    output: list[str] = []
    for key in _clip_hash_hint_keys(index):
        for digest in hints.get(key, []):
            if digest not in output:
                output.append(digest)
    return output


def _coerce_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _model_options_by_folder(folder_path_name: str) -> tuple[str, ...]:
    if folder_path_name == Const.MODEL_FOLDER_PATH_CHECKPOINTS:
        return Const.get_CHECKPOINT_OPTIONS()
    if folder_path_name == Const.MODEL_FOLDER_PATH_DIFFUSION_MODELS:
        return Const.get_DIFFUSION_MODEL_OPTIONS()
    if folder_path_name == Const.MODEL_FOLDER_PATH_UNET:
        return Const.get_UNET_MODEL_OPTIONS()
    return tuple()


def _available_model_folders() -> list[str]:
    folders: list[str] = []
    for folder_name in _MODEL_FOLDER_NAMES:
        options = _model_options_by_folder(folder_name)
        if options:
            folders.append(folder_name)
    return folders


def _resolve_model_like_option(
    value: Any,
    preferred_folder: str | None,
    expected_hashes: Sequence[str] | None,
) -> tuple[str, str] | None:
    available_folders = _available_model_folders()
    if not available_folders:
        return None

    primary_folder: str | None = None
    if preferred_folder in available_folders:
        primary_folder = preferred_folder

    other_folders = [folder_name for folder_name in available_folders if folder_name != primary_folder]

    if primary_folder is not None:
        primary_options = _model_options_by_folder(primary_folder)
        matched = _resolve_option_by_hash(primary_options, primary_folder, expected_hashes)
        if matched:
            return matched, primary_folder

    for folder_name in other_folders:
        options = _model_options_by_folder(folder_name)
        matched = _resolve_option_by_hash(options, folder_name, expected_hashes)
        if matched:
            return matched, folder_name

    if primary_folder is not None:
        primary_options = _model_options_by_folder(primary_folder)
        matches = _find_matching_options(value, primary_options)
        if matches:
            return matches[0], primary_folder

    for folder_name in other_folders:
        options = _model_options_by_folder(folder_name)
        matches = _find_matching_options(value, options)
        if matches:
            return matches[0], folder_name

    return None


def _folder_name_from_model_value(model_value: Any) -> str | None:
    if not isinstance(model_value, Mapping):
        return None
    folder_name = model_value.get("folder_paths")
    if not isinstance(folder_name, str) or not folder_name:
        return None
    return folder_name


def _model_value_metadata(model_value: Any) -> dict[str, Any]:
    if not isinstance(model_value, Mapping):
        return {}

    output: dict[str, Any] = {}
    for raw_key, raw_value in model_value.items():
        if raw_key in (Const.MODEL_VALUE_NAME_KEY, Const.MODEL_VALUE_FOLDER_PATHS_KEY):
            continue
        if not isinstance(raw_key, str) or not raw_key or raw_value is None:
            continue
        output[raw_key] = raw_value
    return output


def _clip_value_metadata(clip_value: Any) -> dict[str, Any]:
    if not isinstance(clip_value, Mapping):
        return {}

    output: dict[str, Any] = {}
    for raw_key, raw_value in clip_value.items():
        if raw_key in (Const.CLIP_VALUE_NAMES_KEY, Const.MODEL_VALUE_FOLDER_PATHS_KEY):
            continue
        if not isinstance(raw_key, str) or not raw_key or raw_value is None:
            continue
        output[raw_key] = raw_value
    return output


def _coerce_internal_hash_hints(raw: Any) -> dict[str, Any]:
    output: dict[str, Any] = {
        "model": [],
        "refiner": [],
        "detailer": [],
        "vae": [],
        "clips": {},
        "loras": {},
    }

    if not isinstance(raw, Mapping):
        return output

    output["model"] = _normalize_hash_list(raw.get("model"))
    output["refiner"] = _normalize_hash_list(raw.get("refiner"))
    output["detailer"] = _normalize_hash_list(raw.get("detailer"))
    output["vae"] = _normalize_hash_list(raw.get("vae"))

    clip_hints: dict[str, list[str]] = {}
    raw_clips = raw.get("clips")
    if isinstance(raw_clips, Mapping):
        for raw_index, raw_hashes in raw_clips.items():
            try:
                index_int = int(str(raw_index).strip())
            except Exception:
                continue
            if index_int <= 0:
                continue
            hashes = _normalize_hash_list(raw_hashes)
            if not hashes:
                continue
            for key in _clip_hash_hint_keys(index_int):
                for digest in hashes:
                    _append_lora_hash_hint(clip_hints, key, digest)
    output["clips"] = clip_hints

    lora_hints: dict[str, list[str]] = {}
    raw_loras = raw.get("loras")
    if isinstance(raw_loras, Mapping):
        for raw_name, raw_hashes in raw_loras.items():
            name_text = str(raw_name or "").strip()
            if not name_text:
                continue
            hashes = _normalize_hash_list(raw_hashes)
            if not hashes:
                continue
            for key in _lora_hash_hint_keys(name_text):
                for digest in hashes:
                    _append_lora_hash_hint(lora_hints, key, digest)

    output["loras"] = lora_hints
    return output


def _merge_lora_hash_hints(
    primary: Mapping[str, list[str]],
    secondary: Mapping[str, list[str]],
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}

    for source in (primary, secondary):
        for key, digests in source.items():
            if not isinstance(key, str):
                continue
            for digest in _normalize_hash_list(digests):
                _append_lora_hash_hint(output, key, digest)

    return output


def _merge_indexed_hash_hints(
    primary: Mapping[str, list[str]],
    secondary: Mapping[str, list[str]],
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for source in (primary, secondary):
        for key, digests in source.items():
            if not isinstance(key, str):
                continue
            for digest in _normalize_hash_list(digests):
                _append_lora_hash_hint(output, key, digest)
    return output


def _normalize_model(
    image_info: dict[str, Any],
    model_hash_hints: Sequence[str],
) -> None:
    model = image_info.get(Const.IMAGEINFO_MODEL)
    if not isinstance(model, Mapping):
        image_info.pop(Const.IMAGEINFO_MODEL, None)
        return

    preferred_folder = _folder_name_from_model_value(model)
    resolved = _resolve_model_like_option(
        model.get("name"),
        preferred_folder,
        model_hash_hints,
    )
    if resolved is None:
        image_info.pop(Const.IMAGEINFO_MODEL, None)
        return
    matched, resolved_folder = resolved

    normalized = Const.make_model_value(matched, resolved_folder, _model_value_metadata(model))
    if normalized is None:
        image_info.pop(Const.IMAGEINFO_MODEL, None)
        return
    image_info[Const.IMAGEINFO_MODEL] = normalized


def _normalize_refiner(
    image_info: dict[str, Any],
    refiner_hash_hints: Sequence[str],
) -> None:
    refiner = image_info.get(Const.IMAGEINFO_REFINER_MODEL)
    if not isinstance(refiner, Mapping):
        image_info.pop(Const.IMAGEINFO_REFINER_MODEL, None)
        return

    preferred_folder = _folder_name_from_model_value(refiner)
    if not preferred_folder:
        preferred_folder = _folder_name_from_model_value(image_info.get(Const.IMAGEINFO_MODEL))

    resolved = _resolve_model_like_option(
        refiner.get("name"),
        preferred_folder,
        refiner_hash_hints,
    )
    if resolved is None:
        image_info.pop(Const.IMAGEINFO_REFINER_MODEL, None)
        return
    matched, resolved_folder = resolved

    normalized = Const.make_model_value(matched, resolved_folder, _model_value_metadata(refiner))
    if normalized is None:
        image_info.pop(Const.IMAGEINFO_REFINER_MODEL, None)
        return
    image_info[Const.IMAGEINFO_REFINER_MODEL] = normalized


def _normalize_detailer(
    image_info: dict[str, Any],
    detailer_hash_hints: Sequence[str],
) -> None:
    detailer = image_info.get(Const.IMAGEINFO_DETAILER_MODEL)
    if not isinstance(detailer, Mapping):
        image_info.pop(Const.IMAGEINFO_DETAILER_MODEL, None)
        return

    preferred_folder = _folder_name_from_model_value(detailer)
    if not preferred_folder:
        preferred_folder = _folder_name_from_model_value(image_info.get(Const.IMAGEINFO_MODEL))

    resolved = _resolve_model_like_option(
        detailer.get("name"),
        preferred_folder,
        detailer_hash_hints,
    )
    if resolved is None:
        image_info.pop(Const.IMAGEINFO_DETAILER_MODEL, None)
        return
    matched, resolved_folder = resolved

    normalized = Const.make_model_value(matched, resolved_folder, _model_value_metadata(detailer))
    if normalized is None:
        image_info.pop(Const.IMAGEINFO_DETAILER_MODEL, None)
        return
    image_info[Const.IMAGEINFO_DETAILER_MODEL] = normalized


def _normalize_lora_stack(
    image_info: dict[str, Any],
    lora_hash_hints: Mapping[str, list[str]],
) -> None:
    stack = image_info.get(Const.IMAGEINFO_LORA_STACK)
    if not isinstance(stack, list):
        image_info.pop(Const.IMAGEINFO_LORA_STACK, None)
        return

    normalized: list[dict[str, str | float]] = []
    lora_options = Const.get_LORA_OPTIONS()
    for item in stack:
        if not isinstance(item, Mapping):
            continue
        expected_hashes = _lora_hash_hint_for_name(item.get("name"), lora_hash_hints)
        matched = _resolve_option_with_hash_fallback(
            item.get("name"),
            lora_options,
            "loras",
            expected_hashes,
        )
        if not matched:
            continue

        normalized_item = Const.make_lora_stack_item(
            matched,
            _coerce_float(item.get("strength"), 1.0),
        )
        if normalized_item is not None:
            normalized.append(normalized_item)

    if normalized:
        image_info[Const.IMAGEINFO_LORA_STACK] = normalized
    else:
        image_info.pop(Const.IMAGEINFO_LORA_STACK, None)


def _normalize_vae(
    image_info: dict[str, Any],
    vae_hash_hints: Sequence[str],
) -> None:
    matched = _resolve_option_with_hash_fallback(
        image_info.get(Const.IMAGEINFO_VAE),
        Const.get_VAE_OPTIONS(),
        "vae",
        vae_hash_hints,
    )
    if matched:
        image_info[Const.IMAGEINFO_VAE] = matched
    else:
        image_info.pop(Const.IMAGEINFO_VAE, None)


def _normalize_clip(
    image_info: dict[str, Any],
    clip_hash_hints: Mapping[str, list[str]],
) -> None:
    clip_value = image_info.get(Const.IMAGEINFO_CLIP)
    if not isinstance(clip_value, Mapping):
        image_info.pop(Const.IMAGEINFO_CLIP, None)
        return

    clip_names_raw = clip_value.get(Const.CLIP_VALUE_NAMES_KEY)
    if not isinstance(clip_names_raw, list) or len(clip_names_raw) == 0:
        image_info.pop(Const.IMAGEINFO_CLIP, None)
        return

    clip_options = Const.get_CLIP_NAME_OPTIONS()
    normalized_names: list[str] = []
    for index, raw_name in enumerate(clip_names_raw, start=1):
        matched = _resolve_option_with_hash_fallback(
            raw_name if isinstance(raw_name, str) else None,
            clip_options,
            Const.MODEL_FOLDER_PATH_TEXT_ENCODERS,
            _clip_hash_hint_for_index(index, clip_hash_hints),
        )
        if not matched:
            image_info.pop(Const.IMAGEINFO_CLIP, None)
            return
        normalized_names.append(matched)

    normalized_clip = Const.make_clip_value(normalized_names, _clip_value_metadata(clip_value))
    if normalized_clip is None:
        image_info.pop(Const.IMAGEINFO_CLIP, None)
        return
    image_info[Const.IMAGEINFO_CLIP] = normalized_clip


def _normalize_sampler_scheduler(image_info: dict[str, Any]) -> None:
    sampler = image_info.get(Const.IMAGEINFO_SAMPLER)
    if sampler not in Const.SAMPLER_OPTIONS:
        image_info.pop(Const.IMAGEINFO_SAMPLER, None)

    scheduler = image_info.get(Const.IMAGEINFO_SCHEDULER)
    if scheduler not in Const.SCHEDULER_OPTIONS:
        image_info.pop(Const.IMAGEINFO_SCHEDULER, None)


def normalize_image_info_with_comfy_options(image_info: dict[str, Any] | None) -> dict[str, Any]:
    output = dict(image_info) if isinstance(image_info, dict) else {}
    extras = output.get(Const.IMAGEINFO_EXTRAS)
    extras_dict: Mapping[str, Any] = extras if isinstance(extras, Mapping) else {}

    internal_hints = _coerce_internal_hash_hints(output.pop(INTERNAL_HASH_HINTS_KEY, None))

    model_hash_hints = _merge_hash_lists(
        internal_hints["model"],
        _normalize_hash_list(extras_dict.get(_EXTRA_MODEL_HASH)),
    )
    refiner_hash_hints = _merge_hash_lists(
        internal_hints["refiner"],
        _normalize_hash_list(extras_dict.get(_EXTRA_REFINER_HASH)),
    )
    detailer_hash_hints = _merge_hash_lists(
        internal_hints["detailer"],
        _normalize_hash_list(extras_dict.get(_EXTRA_DETAILER_HASH)),
    )
    vae_hash_hints = _merge_hash_lists(
        internal_hints["vae"],
        _normalize_hash_list(extras_dict.get(_EXTRA_VAE_HASH)),
    )
    internal_clip_hints = internal_hints["clips"] if isinstance(internal_hints.get("clips"), Mapping) else {}
    clip_hash_hints = _merge_indexed_hash_hints(
        internal_clip_hints,
        _parse_clip_hash_hints(extras_dict),
    )

    internal_lora_hints = internal_hints["loras"] if isinstance(internal_hints.get("loras"), Mapping) else {}
    lora_hash_hints = _merge_lora_hash_hints(
        internal_lora_hints,
        _parse_lora_hash_hints(extras_dict),
    )

    _normalize_lora_stack(output, lora_hash_hints)
    _normalize_clip(output, clip_hash_hints)
    _normalize_model(output, model_hash_hints)
    _normalize_refiner(output, refiner_hash_hints)
    _normalize_detailer(output, detailer_hash_hints)
    _normalize_vae(output, vae_hash_hints)
    _normalize_sampler_scheduler(output)
    return output


def normalize_lora_stack_with_comfy_options(
    lora_stack: list[dict[str, Any]] | None,
    extras: Mapping[str, Any] | None = None,
) -> list[dict[str, str | float]] | None:
    output: dict[str, Any] = {}
    if isinstance(lora_stack, list):
        output[Const.IMAGEINFO_LORA_STACK] = lora_stack

    extras_dict: Mapping[str, Any] = extras if isinstance(extras, Mapping) else {}
    lora_hash_hints = _parse_lora_hash_hints(extras_dict)

    _normalize_lora_stack(output, lora_hash_hints)
    normalized = output.get(Const.IMAGEINFO_LORA_STACK)
    if isinstance(normalized, list):
        return normalized
    return None
