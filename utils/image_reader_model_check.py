# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any, Mapping, Sequence

try:
    from .. import const as Const
except Exception:  # pragma: no cover - direct test imports
    class _FallbackConst:
        IMAGEINFO_MODEL = "model"
        IMAGEINFO_REFINER_MODEL = "refiner"
        IMAGEINFO_DETAILER_MODEL = "detailer"
        IMAGEINFO_CLIP = "clip"
        IMAGEINFO_VAE = "vae"
        IMAGEINFO_LORA_STACK = "lora_stack"
        IMAGEINFO_EXTRAS = "extras"
        MODEL_FOLDER_PATH_CHECKPOINTS = "checkpoints"
        MODEL_FOLDER_PATH_TEXT_ENCODERS = "text_encoders"
        MODEL_FOLDER_PATH_VAE = "vae"
        MODEL_FOLDER_PATH_LORAS = "loras"

    Const = _FallbackConst()  # type: ignore[assignment]
from .a1111_infotext import INTERNAL_HASH_HINTS_KEY, a1111_infotext_to_image_info
from .model_lora_metadata_pipeline import ModelLoraMetadataPipeline, get_shared_metadata_pipeline
from .model_reference_resolver import resolve_model_reference


_EXTRA_LORA_HASHES = "Lora hashes"
_EXTRA_CLIP_HASHES = "Clip hashes"
_EXTRA_MODEL_HASH = "Model hash"
_EXTRA_REFINER_HASH = "Refiner hash"
_EXTRA_DETAILER_HASH = "Detailer hash"
_EXTRA_VAE_HASH = "VAE hash"

_HASH_ALGO_SHA256 = "sha256"
_HASH_ALGO_AUTOV1 = "autov1"
_HASH_ALGO_AUTOV2 = "autov2"
_HASH_ALGO_AUTOV3 = "autov3"
_HASH_ALGO_CRC32 = "crc32"
_HASH_ALGO_BLAKE3 = "blake3"
_HASH_ALGO_A1111_LEGACY = "a1111_legacy"


def _normalize_hash_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text.startswith("sha256:"):
        text = text[7:].strip()
    if all(char in "0123456789abcdef" for char in text) and 8 <= len(text) <= 64:
        return text
    return None


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
        for digest in _normalize_hash_list(group):
            if digest not in output:
                output.append(digest)
    return output


def _candidate_hash_algorithms(expected: str) -> tuple[str, ...]:
    length = len(expected)
    if length == 64:
        return (_HASH_ALGO_SHA256,)
    if length <= 8:
        return (
            _HASH_ALGO_A1111_LEGACY,
            _HASH_ALGO_CRC32,
            _HASH_ALGO_AUTOV1,
            _HASH_ALGO_AUTOV2,
            _HASH_ALGO_AUTOV3,
            _HASH_ALGO_BLAKE3,
        )
    if length <= 16:
        return (
            _HASH_ALGO_AUTOV1,
            _HASH_ALGO_AUTOV2,
            _HASH_ALGO_AUTOV3,
            _HASH_ALGO_A1111_LEGACY,
            _HASH_ALGO_CRC32,
            _HASH_ALGO_BLAKE3,
        )
    return (
        _HASH_ALGO_BLAKE3,
        _HASH_ALGO_SHA256,
        _HASH_ALGO_AUTOV1,
        _HASH_ALGO_AUTOV2,
        _HASH_ALGO_AUTOV3,
    )


def _build_hash_hint_entries(digests: Sequence[str]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for digest in _normalize_hash_list(digests):
        for algo in _candidate_hash_algorithms(digest):
            key = (algo, digest)
            if key in seen:
                continue
            seen.add(key)
            output.append({"algo": algo, "value": digest})
    return output


def _lora_hash_hint_keys(name: str | None) -> list[str]:
    text = str(name or "").strip()
    if not text:
        return []

    basename = text.replace("\\", "/").split("/")[-1]
    stem = basename.rsplit(".", 1)[0] if "." in basename else basename

    output: list[str] = []
    for candidate in (text, basename, stem):
        normalized = candidate.strip().lower()
        if normalized and normalized not in output:
            output.append(normalized)
    return output


def _append_unique_hash_hint(target: dict[str, list[str]], key: str, digest: str) -> None:
    bucket = target.setdefault(key, [])
    if digest not in bucket:
        bucket.append(digest)


def _coerce_internal_hash_hints(raw: Any) -> dict[str, Any]:
    output: dict[str, Any] = {
        "model": [],
        "refiner": [],
        "detailer": [],
        "clips": {},
        "vae": [],
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
                    _append_unique_hash_hint(clip_hints, key, digest)
    output["clips"] = clip_hints

    lora_hints: dict[str, list[str]] = {}
    raw_loras = raw.get("loras")
    if isinstance(raw_loras, Mapping):
        for raw_name, raw_hashes in raw_loras.items():
            hashes = _normalize_hash_list(raw_hashes)
            if not hashes:
                continue
            for key in _lora_hash_hint_keys(str(raw_name or "")):
                for digest in hashes:
                    _append_unique_hash_hint(lora_hints, key, digest)
    output["loras"] = lora_hints
    return output


def _lora_hash_hints_for_name(name: str | None, hints: Mapping[str, list[str]]) -> list[str]:
    output: list[str] = []
    for key in _lora_hash_hint_keys(name):
        for digest in hints.get(key, []):
            if digest not in output:
                output.append(digest)
    return output


def _merge_lora_hash_hints(*sources: Mapping[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for raw_key, raw_values in source.items():
            key = str(raw_key or "").strip().lower()
            if not key:
                continue
            bucket = merged.setdefault(key, [])
            for digest in _normalize_hash_list(raw_values):
                if digest not in bucket:
                    bucket.append(digest)
    return merged


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
    text = str(raw or "").strip()
    if not text:
        return {}

    hints: dict[str, list[str]] = {}
    for part in text.split(","):
        segment = part.strip()
        if not segment or ":" not in segment:
            continue
        index_part, hash_part = segment.split(":", 1)
        try:
            index_int = int(str(index_part).strip())
        except Exception:
            continue
        if index_int <= 0:
            continue
        digest = _normalize_hash_text(hash_part)
        if not digest:
            continue
        for key in _clip_hash_hint_keys(index_int):
            _append_unique_hash_hint(hints, key, digest)
    return hints


def _clip_hash_hints_for_index(index: int, hints: Mapping[str, list[str]]) -> list[str]:
    output: list[str] = []
    for key in _clip_hash_hint_keys(index):
        for digest in hints.get(key, []):
            if digest not in output:
                output.append(digest)
    return output


def _parse_lora_hash_hints(extras: Mapping[str, Any]) -> dict[str, list[str]]:
    raw = extras.get(_EXTRA_LORA_HASHES)
    text = str(raw or "").strip()
    if not text:
        return {}

    hints: dict[str, list[str]] = {}
    for part in text.split(","):
        segment = part.strip()
        if not segment or ":" not in segment:
            continue
        name_part, hash_part = segment.split(":", 1)
        name = name_part.strip()
        digest = _normalize_hash_text(hash_part)
        if not name or not digest:
            continue
        for key in _lora_hash_hint_keys(name):
            _append_unique_hash_hint(hints, key, digest)
    return hints


def _preferred_sha256(hash_hints: Sequence[str]) -> str | None:
    for digest in _normalize_hash_list(hash_hints):
        if len(digest) == 64:
            return digest
    return None


def _display_name(name_raw: str | None, local_match: Mapping[str, Any] | None) -> str:
    raw = str(name_raw or "").strip()
    if raw:
        return raw
    if isinstance(local_match, Mapping):
        rel = str(local_match.get("relative_path") or "").strip()
        if rel:
            return rel
    return "(unknown)"


def _make_item(
    *,
    pipeline: ModelLoraMetadataPipeline,
    kind: str,
    kind_label: str,
    folder_hint: str,
    name_raw: str,
    strength: float | None,
    hash_hints: Sequence[str],
) -> dict[str, Any]:
    hash_hint_entries = _build_hash_hint_entries(hash_hints)
    resolved = resolve_model_reference(
        pipeline,
        folder_name=folder_hint,
        relative_path=name_raw,
        sha256=_preferred_sha256(hash_hints),
        name_raw=name_raw,
        hash_hints=hash_hint_entries,
        resolve_remote=False,
        include_lora_tags=False,
        enqueue_local_hash=False,
    )
    local_match = resolved.get("local_match")
    return {
        "kind": kind,
        "kind_label": kind_label,
        "display_name": _display_name(name_raw, local_match if isinstance(local_match, Mapping) else None),
        "name_raw": name_raw,
        "folder_hint": folder_hint,
        "strength": strength,
        "hash_hints": hash_hint_entries,
        "sha256": resolved.get("sha256"),
        "local_status": resolved.get("local_status"),
        "local_match": local_match,
        "view_model_info_source": {
            "relative_path": str(local_match.get("relative_path") or "") if isinstance(local_match, Mapping) else "",
            "sha256": resolved.get("sha256"),
            "name_raw": name_raw,
            "hash_hints": hash_hint_entries,
        },
    }


def inspect_infotext_references(
    infotext: str | None,
    *,
    pipeline: ModelLoraMetadataPipeline | None = None,
) -> list[dict[str, Any]]:
    parsed = a1111_infotext_to_image_info(infotext)
    internal_hints = _coerce_internal_hash_hints(parsed.get(INTERNAL_HASH_HINTS_KEY))
    extras = parsed.get(Const.IMAGEINFO_EXTRAS)
    extras_dict: Mapping[str, Any] = extras if isinstance(extras, Mapping) else {}

    model_hash_hints = _merge_hash_lists(
        internal_hints.get("model", []),
        _normalize_hash_list(extras_dict.get(_EXTRA_MODEL_HASH)),
    )
    refiner_hash_hints = _merge_hash_lists(
        internal_hints.get("refiner", []),
        _normalize_hash_list(extras_dict.get(_EXTRA_REFINER_HASH)),
    )
    detailer_hash_hints = _merge_hash_lists(
        internal_hints.get("detailer", []),
        _normalize_hash_list(extras_dict.get(_EXTRA_DETAILER_HASH)),
    )
    vae_hash_hints = _merge_hash_lists(
        internal_hints.get("vae", []),
        _normalize_hash_list(extras_dict.get(_EXTRA_VAE_HASH)),
    )
    clip_hash_hints = _merge_lora_hash_hints(
        internal_hints.get("clips", {}) if isinstance(internal_hints.get("clips"), Mapping) else {},
        _parse_clip_hash_hints(extras_dict),
    )
    lora_hash_hints = _merge_lora_hash_hints(
        internal_hints.get("loras", {}) if isinstance(internal_hints.get("loras"), Mapping) else {},
        _parse_lora_hash_hints(extras_dict),
    )

    metadata_pipeline = pipeline or get_shared_metadata_pipeline(start=False)

    items: list[dict[str, Any]] = []

    model = parsed.get(Const.IMAGEINFO_MODEL)
    if isinstance(model, Mapping):
        name_raw = str(model.get("name") or "").strip()
        folder_hint = str(model.get("folder_paths") or Const.MODEL_FOLDER_PATH_CHECKPOINTS).strip()
        if name_raw and folder_hint:
            items.append(
                _make_item(
                    pipeline=metadata_pipeline,
                    kind="model",
                    kind_label="Model",
                    folder_hint=folder_hint,
                    name_raw=name_raw,
                    strength=None,
                    hash_hints=model_hash_hints,
                )
            )

    refiner = parsed.get(Const.IMAGEINFO_REFINER_MODEL)
    if isinstance(refiner, Mapping):
        name_raw = str(refiner.get("name") or "").strip()
        folder_hint = str(
            refiner.get("folder_paths")
            or (model.get("folder_paths") if isinstance(model, Mapping) else "")
            or Const.MODEL_FOLDER_PATH_CHECKPOINTS
        ).strip()
        if name_raw and folder_hint:
            items.append(
                _make_item(
                    pipeline=metadata_pipeline,
                    kind="refiner",
                    kind_label="Refiner",
                    folder_hint=folder_hint,
                    name_raw=name_raw,
                    strength=None,
                    hash_hints=refiner_hash_hints,
                )
            )

    detailer = parsed.get(Const.IMAGEINFO_DETAILER_MODEL)
    if isinstance(detailer, Mapping):
        name_raw = str(detailer.get("name") or "").strip()
        folder_hint = str(
            detailer.get("folder_paths")
            or (model.get("folder_paths") if isinstance(model, Mapping) else "")
            or Const.MODEL_FOLDER_PATH_CHECKPOINTS
        ).strip()
        if name_raw and folder_hint:
            items.append(
                _make_item(
                    pipeline=metadata_pipeline,
                    kind="detailer",
                    kind_label="Detailer",
                    folder_hint=folder_hint,
                    name_raw=name_raw,
                    strength=None,
                    hash_hints=detailer_hash_hints,
                )
            )

    clip = parsed.get(Const.IMAGEINFO_CLIP)
    if isinstance(clip, Mapping):
        clip_names = clip.get("clip_names")
        if isinstance(clip_names, list):
            for index, raw_name in enumerate(clip_names, start=1):
                name_raw = str(raw_name or "").strip()
                if not name_raw:
                    continue
                items.append(
                    _make_item(
                        pipeline=metadata_pipeline,
                        kind="clip",
                        kind_label="Clip",
                        folder_hint=Const.MODEL_FOLDER_PATH_TEXT_ENCODERS,
                        name_raw=name_raw,
                        strength=None,
                        hash_hints=_clip_hash_hints_for_index(index, clip_hash_hints),
                    )
                )

    vae_name = str(parsed.get(Const.IMAGEINFO_VAE) or "").strip()
    if vae_name:
        items.append(
            _make_item(
                pipeline=metadata_pipeline,
                kind="vae",
                kind_label="VAE",
                folder_hint=Const.MODEL_FOLDER_PATH_VAE,
                name_raw=vae_name,
                strength=None,
                hash_hints=vae_hash_hints,
            )
        )

    lora_stack = parsed.get(Const.IMAGEINFO_LORA_STACK)
    if isinstance(lora_stack, list):
        lora_hints = internal_hints.get("loras", {})
        for entry in lora_stack:
            if not isinstance(entry, Mapping):
                continue
            name_raw = str(entry.get("name") or "").strip()
            if not name_raw:
                continue
            try:
                strength = float(entry.get("strength", 1.0))
            except Exception:
                strength = 1.0
            items.append(
                _make_item(
                    pipeline=metadata_pipeline,
                    kind="lora",
                    kind_label="LoRA",
                    folder_hint=Const.MODEL_FOLDER_PATH_LORAS,
                    name_raw=name_raw,
                    strength=strength,
                    hash_hints=_lora_hash_hints_for_name(name_raw, lora_hash_hints),
                )
            )

    return items
