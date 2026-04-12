from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .file_hash_cache import normalize_relative_path
from .model_runtime_settings import (
    filter_model_runtime_settings_for_folder,
    is_supported_model_runtime_settings_folder,
)

try:
    import folder_paths  # type: ignore
except Exception:
    folder_paths = None  # type: ignore[assignment]


_HASH_ALGO_SHA256 = "sha256"
_COPYABLE_HASH_ALGOS = (
    "sha256",
    "crc32",
    "blake3",
    "autov3",
    "autov1",
    "autov2",
)
_SUPPORTED_HASH_ALGOS = {
    "sha256",
    "autov1",
    "autov2",
    "autov3",
    "crc32",
    "blake3",
    "a1111_legacy",
}
_REMOTE_BY_HASH_ALGOS = {
    "sha256": "SHA256",
    "autov1": "AutoV1",
    "autov2": "AutoV2",
    "autov3": "AutoV3",
    "crc32": "CRC32",
    "blake3": "BLAKE3",
}
_LORA_TAG_STATE_READY = "ready"
_LORA_TAG_STATE_EMPTY = "empty"
_LORA_TAG_STATE_NO_METADATA = "no_metadata"
_LORA_TAG_STATE_UNSUPPORTED = "unsupported"
_LORA_TAG_STATE_PENDING = "pending"
_LORA_TAG_STATE_ERROR = "error"
_DEFAULT_LORA_TAG_ENSURE_TIMEOUT_MS = 2500
_MAX_LORA_TAG_ENSURE_TIMEOUT_MS = 30000
_LORA_TAG_ENSURE_POLL_INTERVAL_SEC = 0.08


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_hash_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def is_sha256_digest(value: Any) -> bool:
    text = normalize_hash_text(value)
    if text is None or len(text) != 64:
        return False
    return all(char in "0123456789abcdef" for char in text)


def is_fetchable_hash_prefix(value: Any, *, min_length: int = 8) -> bool:
    text = normalize_hash_text(value)
    if text is None or len(text) < min_length:
        return False
    return all(char in "0123456789abcdef" for char in text)


def normalize_hash_hints(value: object) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []

    output: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        algo = normalize_hash_text(item.get("algo"))
        digest = normalize_hash_text(item.get("value"))
        if algo == "a1111":
            algo = "a1111_legacy"
        if algo not in _SUPPORTED_HASH_ALGOS or digest is None:
            continue
        output.append({"algo": algo, "value": digest})
    return output


def normalize_civitai_file_hashes(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}

    output: dict[str, str] = {}
    for algo in _COPYABLE_HASH_ALGOS:
        digest = normalize_hash_text(value.get(algo))
        if digest is None:
            continue
        output[algo] = digest
    return output


def extract_remote_civitai_file_hashes(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}

    output: dict[str, str] = {}
    for algo in _COPYABLE_HASH_ALGOS:
        remote_key = _REMOTE_BY_HASH_ALGOS.get(algo)
        if not remote_key:
            continue
        digest = normalize_hash_text(value.get(remote_key))
        if digest is None:
            continue
        output[algo] = digest
    return output


def basename_from_reference(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parts = [part for part in text.replace("\\", "/").split("/") if part]
    if not parts:
        return text
    return parts[-1]


def build_civitai_page_url(model_id: object, model_version_id: object) -> str:
    try:
        normalized_model_id = int(model_id)
        normalized_model_version_id = int(model_version_id)
    except Exception:
        return ""
    if normalized_model_id <= 0 or normalized_model_version_id <= 0:
        return ""
    return f"https://civitai.com/models/{normalized_model_id}?modelVersionId={normalized_model_version_id}"


def build_model_info_payload(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(record, Mapping):
        return None

    model_id = record.get("civitai_model_version_model_id")
    model_version_id = record.get("civitai_model_version_model_version_id")
    model_name = record.get("civitai_model_name")
    version_name = record.get("civitai_model_version_name")
    base_model = record.get("civitai_model_version_base_model")

    if not any(
        value is not None and str(value).strip()
        for value in (model_id, model_version_id, model_name, version_name, base_model)
    ):
        return None

    return {
        "civitai_model_name": model_name,
        "civitai_model_version_model_id": model_id,
        "civitai_model_version_model_version_id": model_version_id,
        "civitai_model_version_name": version_name,
        "civitai_model_version_base_model": base_model,
    }


def build_copyable_hashes_payload(
    record: Mapping[str, Any] | None,
    *,
    fallback_sha256: object | None = None,
) -> dict[str, str]:
    output: dict[str, str] = {}

    record_hashes = normalize_civitai_file_hashes(
        record.get("civitai_file_hashes") if isinstance(record, Mapping) else None
    )

    sha256 = normalize_hash_text(
        record.get("sha256") if isinstance(record, Mapping) else fallback_sha256
    )
    if not is_sha256_digest(sha256):
        sha256 = record_hashes.get("sha256") or normalize_hash_text(fallback_sha256)
    if is_sha256_digest(sha256):
        output["sha256"] = sha256

    for algo in _COPYABLE_HASH_ALGOS:
        if algo == "sha256":
            continue
        digest = record_hashes.get(algo)
        if digest is None:
            continue
        output[algo] = digest

    return output


def build_download_candidate_payload(
    *,
    folder_name: str,
    record: Mapping[str, Any] | None,
    fallback_name: str = "",
) -> dict[str, str] | None:
    if not isinstance(record, Mapping):
        return None

    url = str(record.get("civitai_file_download_url") or "").strip()
    if not url:
        return None

    name = str(record.get("civitai_file_name") or "").strip() or basename_from_reference(fallback_name)
    if not name:
        return None

    directory = str(folder_name or "").strip()
    if not directory:
        return None

    return {
        "name": name,
        "url": url,
        "directory": directory,
    }


def build_page_candidate_payload(record: Mapping[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(record, Mapping):
        return None

    url = build_civitai_page_url(
        record.get("civitai_model_version_model_id"),
        record.get("civitai_model_version_model_version_id"),
    )
    if not url:
        return None
    return {"url": url}


def _reference_stem(value: object) -> str:
    basename = basename_from_reference(value)
    if not basename:
        return ""
    return Path(basename).stem


def build_reference_record_from_civitai_payload(
    payload: Mapping[str, Any] | None,
    *,
    hash_value: str,
    preferred_algos: Sequence[str] | None = None,
    name_hint: str = "",
) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None

    files = payload.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes, bytearray)):
        return None

    requested_hash = normalize_hash_text(hash_value)
    if not is_fetchable_hash_prefix(requested_hash):
        return None

    normalized_algos = [
        str(raw_algo or "").strip().lower()
        for raw_algo in (preferred_algos or ())
        if str(raw_algo or "").strip().lower() in _REMOTE_BY_HASH_ALGOS
    ]
    if not normalized_algos:
        normalized_algos = list(_REMOTE_BY_HASH_ALGOS.keys())

    matching_files: list[Mapping[str, Any]] = []
    for raw_file in files:
        if not isinstance(raw_file, Mapping):
            continue
        hashes = raw_file.get("hashes")
        if not isinstance(hashes, Mapping):
            continue
        if any(
            (normalize_hash_text(hashes.get(_REMOTE_BY_HASH_ALGOS[algo])) or "").startswith(requested_hash)
            for algo in normalized_algos
        ):
            matching_files.append(raw_file)

    if not matching_files:
        return None

    selected_file: Mapping[str, Any] | None = None
    if len(matching_files) == 1:
        selected_file = matching_files[0]
    else:
        hint_basename = basename_from_reference(name_hint)
        if hint_basename:
            basename_matches = [
                item for item in matching_files if basename_from_reference(item.get("name")) == hint_basename
            ]
            if len(basename_matches) == 1:
                selected_file = basename_matches[0]

        if selected_file is None:
            hint_stem = _reference_stem(name_hint)
            if hint_stem:
                stem_matches = [item for item in matching_files if _reference_stem(item.get("name")) == hint_stem]
                if len(stem_matches) == 1:
                    selected_file = stem_matches[0]

    if selected_file is None:
        return None

    selected_hashes = selected_file.get("hashes") if isinstance(selected_file.get("hashes"), Mapping) else {}
    sha256 = normalize_hash_text(selected_hashes.get("SHA256"))
    civitai_file_hashes = extract_remote_civitai_file_hashes(selected_hashes)

    return {
        "content_id": None,
        "sha256": sha256,
        "civitai_model_name": payload.get("model", {}).get("name") if isinstance(payload.get("model"), Mapping) else None,
        "civitai_model_version_model_id": payload.get("modelId"),
        "civitai_model_version_model_version_id": payload.get("id"),
        "civitai_model_version_name": payload.get("name"),
        "civitai_model_version_base_model": payload.get("baseModel"),
        "civitai_model_version_raw_json": None,
        "civitai_file_name": selected_file.get("name"),
        "civitai_file_hashes": civitai_file_hashes,
        "civitai_file_download_url": selected_file.get("downloadUrl"),
        "civitai_version_download_url": payload.get("downloadUrl"),
    }


def _local_model_path(folder_name: str, relative_path: str) -> str | None:
    folder = str(folder_name or "").strip()
    rel = normalize_relative_path(relative_path)
    if not folder or not rel:
        return None

    if folder_paths is not None:
        getter = getattr(folder_paths, "get_full_path", None)
        if callable(getter):
            try:
                path = getter(folder, rel)
            except Exception:
                path = None
            if isinstance(path, str) and path:
                return path

        models_dir = getattr(folder_paths, "models_dir", None)
        if isinstance(models_dir, str) and models_dir:
            candidate = Path(models_dir) / folder / rel
            return str(candidate)

    return None


def local_model_exists(folder_name: str, relative_path: str) -> bool:
    absolute_path = _local_model_path(folder_name, relative_path)
    if not absolute_path:
        return False
    return os.path.isfile(absolute_path)


def _content_id_from_reference_record(record: Mapping[str, Any] | None) -> int | None:
    if not isinstance(record, Mapping):
        return None
    try:
        return int(record.get("content_id"))
    except Exception:
        return None


def _normalize_ensure_timeout_ms(value: object | None) -> int:
    try:
        timeout_ms = int(value) if value is not None else _DEFAULT_LORA_TAG_ENSURE_TIMEOUT_MS
    except Exception:
        timeout_ms = _DEFAULT_LORA_TAG_ENSURE_TIMEOUT_MS
    return max(0, min(timeout_ms, _MAX_LORA_TAG_ENSURE_TIMEOUT_MS))


def _load_lora_tags_for_content(
    pipeline: object,
    content_id: int,
) -> tuple[bool, list[dict[str, Any]]]:
    has_metadata = bool(
        getattr(pipeline, "has_lora_metadata_for_content", lambda *_args, **_kwargs: False)(
            content_id
        )
    )
    if not has_metadata:
        return False, []
    tags = getattr(pipeline, "list_lora_tags_by_content_id", lambda *_args, **_kwargs: [])(
        content_id,
        limit=None,
    )
    if not isinstance(tags, list):
        return True, []
    normalized: list[dict[str, Any]] = []
    for item in tags:
        if not isinstance(item, Mapping):
            continue
        normalized.append(dict(item))
    return True, normalized


def _resolve_content_id_by_path(
    pipeline: object,
    *,
    folder_name: str,
    relative_path: str,
) -> int | None:
    try:
        value = getattr(pipeline, "get_content_id_by_relative_path", lambda **_kwargs: None)(
            folder_name=folder_name,
            relative_path=relative_path,
        )
    except Exception:
        return None
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _refresh_reference_record_by_path(
    pipeline: object,
    *,
    folder_name: str,
    relative_path: str,
    previous: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    try:
        refreshed = getattr(pipeline, "get_model_reference_by_relative_path", lambda **_kwargs: None)(
            folder_name=folder_name,
            relative_path=relative_path,
        )
    except Exception:
        refreshed = None
    if isinstance(refreshed, Mapping):
        return refreshed
    return previous


def _resolve_lora_tags_state(
    pipeline: object,
    *,
    folder_name: str,
    relative_path: str | None,
    local_status: str,
    include_lora_tags: bool,
    ensure_lora_tags: bool,
    ensure_timeout_ms: int,
    reference_record: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], str, Mapping[str, Any] | None, bool]:
    if not include_lora_tags or folder_name != "loras":
        return [], _LORA_TAG_STATE_UNSUPPORTED, reference_record, False
    if local_status != "present":
        return [], _LORA_TAG_STATE_NO_METADATA, reference_record, False

    rel = normalize_relative_path(relative_path)
    if not rel:
        return [], _LORA_TAG_STATE_NO_METADATA, reference_record, False
    if not rel.lower().endswith(".safetensors"):
        return [], _LORA_TAG_STATE_UNSUPPORTED, reference_record, False

    current_record = reference_record if isinstance(reference_record, Mapping) else None
    content_id = _content_id_from_reference_record(current_record)
    if content_id is None:
        content_id = _resolve_content_id_by_path(
            pipeline,
            folder_name=folder_name,
            relative_path=rel,
        )
        if content_id is not None:
            if isinstance(current_record, Mapping):
                next_record = dict(current_record)
            else:
                next_record = {}
            next_record["content_id"] = content_id
            current_record = next_record

    try:
        if content_id is not None:
            has_metadata, tags = _load_lora_tags_for_content(pipeline, content_id)
            if has_metadata:
                state = _LORA_TAG_STATE_READY if tags else _LORA_TAG_STATE_EMPTY
                return tags, state, current_record, False

        pending = bool(
            getattr(pipeline, "is_hash_task_queued", lambda *_args, **_kwargs: False)(
                folder_name,
                rel,
            )
        )
        if not ensure_lora_tags:
            if content_id is not None and not pending:
                return [], _LORA_TAG_STATE_NO_METADATA, current_record, False
            return [], _LORA_TAG_STATE_PENDING, current_record, False

        enqueued = bool(
            getattr(pipeline, "enqueue_hash_priority", lambda *_args, **_kwargs: False)(
                folder_name,
                rel,
            )
        )
        deadline = time.monotonic() + (max(0, ensure_timeout_ms) / 1000.0)

        while True:
            current_record = _refresh_reference_record_by_path(
                pipeline,
                folder_name=folder_name,
                relative_path=rel,
                previous=current_record,
            )
            content_id = _content_id_from_reference_record(current_record)
            if content_id is None:
                content_id = _resolve_content_id_by_path(
                    pipeline,
                    folder_name=folder_name,
                    relative_path=rel,
                )
                if content_id is not None:
                    if isinstance(current_record, Mapping):
                        next_record = dict(current_record)
                    else:
                        next_record = {}
                    next_record["content_id"] = content_id
                    current_record = next_record

            if content_id is not None:
                has_metadata, tags = _load_lora_tags_for_content(pipeline, content_id)
                if has_metadata:
                    state = _LORA_TAG_STATE_READY if tags else _LORA_TAG_STATE_EMPTY
                    return tags, state, current_record, enqueued

            pending = bool(
                getattr(pipeline, "is_hash_task_queued", lambda *_args, **_kwargs: False)(
                    folder_name,
                    rel,
                )
            )
            if time.monotonic() >= deadline:
                if content_id is not None and not pending:
                    return [], _LORA_TAG_STATE_NO_METADATA, current_record, enqueued
                return [], _LORA_TAG_STATE_PENDING, current_record, enqueued

            if content_id is not None and not pending:
                return [], _LORA_TAG_STATE_NO_METADATA, current_record, enqueued

            time.sleep(_LORA_TAG_ENSURE_POLL_INTERVAL_SEC)
    except Exception:
        return [], _LORA_TAG_STATE_ERROR, current_record, False


def resolve_model_reference(
    pipeline: object,
    *,
    folder_name: str,
    relative_path: object | None = None,
    sha256: object | None = None,
    name_raw: object | None = None,
    hash_hints: object | None = None,
    resolve_remote: bool = False,
    include_lora_tags: bool = False,
    ensure_lora_tags: bool = False,
    ensure_timeout_ms: object | None = None,
    enqueue_local_hash: bool = False,
) -> dict[str, Any]:
    folder = str(folder_name or "").strip()
    rel = normalize_relative_path(relative_path)
    normalized_name_raw = str(name_raw or "").strip()
    normalized_hints = normalize_hash_hints(hash_hints)
    normalized_sha256 = normalize_hash_text(sha256) if is_sha256_digest(sha256) else None
    normalized_ensure_timeout_ms = _normalize_ensure_timeout_ms(ensure_timeout_ms)

    fallback_name = basename_from_reference(normalized_name_raw) or basename_from_reference(rel)
    local_status = "missing"
    local_match: dict[str, str] | None = None
    hash_enqueued = False

    candidate_relative_path = rel
    if rel and local_model_exists(folder, rel):
        local_status = "present"
        local_match = {
            "folder_name": folder,
            "relative_path": rel,
            "matched_by": "exact_relative_path",
        }
    else:
        for hint in normalized_hints:
            found = getattr(pipeline, "find_relative_path_by_hash", lambda *args, **kwargs: None)(
                folder,
                hint["value"],
                preferred_algos=[hint["algo"]],
            )
            if not isinstance(found, str) or not found.strip():
                continue
            normalized_found = normalize_relative_path(found)
            if not normalized_found or not local_model_exists(folder, normalized_found):
                continue
            candidate_relative_path = normalized_found
            local_status = "present"
            local_match = {
                "folder_name": folder,
                "relative_path": normalized_found,
                "matched_by": f"hash:{hint['algo']}",
            }
            break

    if candidate_relative_path:
        resolved_hash = getattr(pipeline, "get_hash_by_relative_path", lambda *args, **kwargs: None)(
            folder,
            candidate_relative_path,
            _HASH_ALGO_SHA256,
        )
        if is_sha256_digest(resolved_hash):
            normalized_sha256 = normalize_hash_text(resolved_hash)
        elif enqueue_local_hash and candidate_relative_path:
            hash_enqueued = bool(
                getattr(pipeline, "enqueue_hash_priority", lambda *args, **kwargs: False)(folder, candidate_relative_path)
            )

    reference_record = None
    if candidate_relative_path:
        reference_record = getattr(pipeline, "get_model_reference_by_relative_path", lambda **kwargs: None)(
            folder_name=folder,
            relative_path=candidate_relative_path,
        )

    if reference_record is None and normalized_sha256:
        reference_record = getattr(pipeline, "get_model_reference_by_sha256", lambda **kwargs: None)(
            sha256=normalized_sha256,
        )

    if reference_record is None and normalized_hints:
        for hint in normalized_hints:
            reference_record = getattr(pipeline, "get_model_reference_by_hash_hint", lambda **kwargs: None)(
                hash_value=hint["value"],
                preferred_algos=[hint["algo"]],
                name_hint=fallback_name or normalized_name_raw,
            )
            if reference_record is not None:
                break

    model_info = build_model_info_payload(reference_record)
    download_candidate = build_download_candidate_payload(
        folder_name=folder,
        record=reference_record,
        fallback_name=fallback_name,
    )
    page_candidate = build_page_candidate_payload(reference_record)

    remote_status = "skipped"
    remote_source = "none"

    if resolve_remote:
        if model_info is not None or download_candidate is not None or page_candidate is not None:
            remote_status = "resolved"
            remote_source = "db"
        elif normalized_sha256:
            should_fetch = bool(
                getattr(pipeline, "should_queue_civitai_lookup", lambda *_args, **_kwargs: True)(
                    normalized_sha256,
                    now_iso=utc_now_iso(),
                )
            )
            fetch_result: Mapping[str, Any] | None = None
            if should_fetch:
                fetch_result = getattr(pipeline, "fetch_civitai_by_sha256_now", lambda *_args, **_kwargs: None)(
                    normalized_sha256
                )

            reference_record = getattr(pipeline, "get_model_reference_by_sha256", lambda **kwargs: None)(
                sha256=normalized_sha256,
            )
            model_info = build_model_info_payload(reference_record)
            download_candidate = build_download_candidate_payload(
                folder_name=folder,
                record=reference_record,
                fallback_name=fallback_name,
            )
            page_candidate = build_page_candidate_payload(reference_record)
            if model_info is not None or download_candidate is not None or page_candidate is not None:
                remote_status = "resolved"
                remote_source = "by_hash" if should_fetch else "db"
            elif isinstance(fetch_result, Mapping) and str(fetch_result.get("status") or "") == "not_found":
                remote_status = "not_found"
            elif not should_fetch:
                lookup_state = getattr(pipeline, "get_civitai_lookup_state", lambda *_args, **_kwargs: None)(
                    normalized_sha256
                )
                if isinstance(lookup_state, Mapping) and str(lookup_state.get("status") or "") == "not_found":
                    remote_status = "not_found"
                else:
                    remote_status = "unresolved"
            else:
                remote_status = "unresolved"
        elif normalized_hints:
            for hint in normalized_hints:
                if hint["algo"] not in _REMOTE_BY_HASH_ALGOS or not is_fetchable_hash_prefix(hint["value"]):
                    continue

                fetch_result = getattr(pipeline, "fetch_civitai_by_hash_now", lambda *_args, **_kwargs: None)(
                    hint["value"]
                )
                if not isinstance(fetch_result, Mapping):
                    continue

                reference_record = build_reference_record_from_civitai_payload(
                    fetch_result.get("payload") if isinstance(fetch_result.get("payload"), Mapping) else None,
                    hash_value=hint["value"],
                    preferred_algos=[hint["algo"]],
                    name_hint=fallback_name or normalized_name_raw,
                )
                if reference_record is None:
                    continue

                if is_sha256_digest(reference_record.get("sha256")):
                    normalized_sha256 = normalize_hash_text(reference_record.get("sha256"))

                model_info = build_model_info_payload(reference_record)
                download_candidate = build_download_candidate_payload(
                    folder_name=folder,
                    record=reference_record,
                    fallback_name=fallback_name,
                )
                page_candidate = build_page_candidate_payload(reference_record)
                if model_info is not None or download_candidate is not None or page_candidate is not None:
                    remote_status = "resolved"
                    remote_source = "by_hash_hint"
                    break
            else:
                remote_status = "unresolved"
        else:
            remote_status = "unresolved"

    lora_tags: list[dict[str, Any]] = []
    lora_tags_state = _LORA_TAG_STATE_UNSUPPORTED
    if include_lora_tags and folder == "loras":
        (
            lora_tags,
            lora_tags_state,
            reference_record,
            lora_hash_enqueued,
        ) = _resolve_lora_tags_state(
            pipeline,
            folder_name=folder,
            relative_path=candidate_relative_path or rel,
            local_status=local_status,
            include_lora_tags=include_lora_tags,
            ensure_lora_tags=ensure_lora_tags,
            ensure_timeout_ms=normalized_ensure_timeout_ms,
            reference_record=reference_record if isinstance(reference_record, Mapping) else None,
        )
        hash_enqueued = hash_enqueued or lora_hash_enqueued

    resolved_hash_hints = list(normalized_hints)
    runtime_settings: dict[str, int | float] = {}
    runtime_settings_editable = False
    if isinstance(reference_record, Mapping):
        try:
            content_id = int(reference_record.get("content_id"))
        except Exception:
            content_id = None
        if content_id is not None:
            stored_hash_hints = getattr(pipeline, "list_hash_hints_by_content_id", lambda *_args, **_kwargs: [])(
                content_id,
                include_sha256=False,
            )
            normalized_stored_hash_hints = normalize_hash_hints(stored_hash_hints)
            if normalized_stored_hash_hints:
                resolved_hash_hints = normalized_stored_hash_hints

            if is_supported_model_runtime_settings_folder(folder):
                runtime_settings = filter_model_runtime_settings_for_folder(
                    folder,
                    reference_record.get("runtime_settings"),
                )
                runtime_settings_editable = local_status == "present"

    return {
        "ok": True,
        "folder_name": folder,
        "relative_path": rel or "",
        "name_raw": normalized_name_raw,
        "sha256": normalized_sha256,
        "hash_hints": resolved_hash_hints,
        "copyable_hashes": build_copyable_hashes_payload(
            reference_record,
            fallback_sha256=normalized_sha256,
        ),
        "hash_enqueued": hash_enqueued,
        "local_status": local_status,
        "local_match": local_match,
        "model_info": model_info,
        "download_candidate": download_candidate,
        "page_candidate": page_candidate,
        "lora_tags": lora_tags,
        "lora_tags_state": lora_tags_state,
        "runtime_settings": runtime_settings,
        "runtime_settings_supported": is_supported_model_runtime_settings_folder(folder),
        "runtime_settings_editable": runtime_settings_editable,
        "remote_status": remote_status,
        "remote_source": remote_source,
    }
