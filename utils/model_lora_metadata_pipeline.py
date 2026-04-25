# Copyright 2026 kinorax
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
from typing import Any, Callable, Mapping, Sequence
from urllib import error as url_error
from urllib import request as url_request

from .file_hash_cache import DEFAULT_CHUNK_SIZE, normalize_relative_path
from .model_runtime_settings import (
    filter_model_runtime_settings_for_folder,
    is_supported_model_runtime_settings_folder,
)
from .model_lora_metadata_db import MetadataDatabase, resolve_db_path

try:
    import folder_paths  # type: ignore
except Exception:
    folder_paths = None  # type: ignore[assignment]


_CIVITAI_HASH_URL = "https://civitai.com/api/v1/model-versions/by-hash/"
_CIVITAI_DEFAULT_HEADERS = {
    "accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}
_STARTUP_BUDGET_BYTES = 40 * 1024 * 1024 * 1024
_STARTUP_CIVITAI_REQUEUE_LIMIT = 2000
_STARTUP_GC_RETENTION_DAYS = 30
_STARTUP_GC_MIN_DELETED_PATHS = 200
_STARTUP_GC_MIN_ORPHAN_IDENTITIES = 200
_STARTUP_GC_MAX_DELETE_PATHS = 5000
_STARTUP_GC_MAX_DELETE_ORPHAN_IDENTITIES = 5000
_HASH_WORKER_NAME = "IPT-HashLoraWorker"
_CIVITAI_WORKER_NAME = "IPT-CivitaiWorker"
_DB_WORKER_NAME = "IPT-DbWriter"
_SEED_WORKER_NAME = "IPT-SeedWorker"

_HASH_ALGO_SHA256 = "sha256"
_CIVITAI_STATUS_FOUND = "found"
_CIVITAI_STATUS_NOT_FOUND = "not_found"
_CIVITAI_STATUS_AUTH_BLOCKED = "unknown_auth_or_blocked"
_CIVITAI_STATUS_TEMP_ERROR = "temporary_error"
_CIVITAI_NOT_FOUND_RETRY_SECONDS = 7 * 24 * 60 * 60
_CIVITAI_AUTH_BLOCKED_RETRY_SECONDS = 24 * 60 * 60
_CIVITAI_TEMP_ERROR_RETRY_SECONDS = 30 * 60


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_hash_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return text


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _safe_print(message: str) -> None:
    try:
        print(message)
    except Exception:
        pass


def _short_sha256(value: str | None, prefix_len: int = 12) -> str:
    text = str(value or "").strip().lower()
    if len(text) <= prefix_len:
        return text
    return f"{text[:prefix_len]}..."


def _log_civitai_fetch_anomaly(
    sha256: str | None,
    *,
    error: str,
    http_status: int | None = None,
) -> None:
    message = f"[IPT][civitai] fetch anomaly sha256={_short_sha256(sha256)} error={error}"
    if http_status is not None:
        message += f" http_status={http_status}"
    _safe_print(message)


@dataclass(frozen=True)
class _HashTask:
    folder_name: str
    relative_path: str


@dataclass(frozen=True)
class _CivitaiTask:
    sha256: str


@dataclass(frozen=True)
class _CivitaiFetchResult:
    status: str
    http_status: int | None = None
    payload: Mapping[str, Any] | None = None
    error: str | None = None


class ModelLoraMetadataPipeline:
    def __init__(self, db: MetadataDatabase | None = None) -> None:
        self._db = db or MetadataDatabase(resolve_db_path())
        self._db_queue: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._hash_priority_queue: queue.Queue[_HashTask] = queue.Queue()
        self._hash_normal_queue: queue.Queue[_HashTask] = queue.Queue()
        self._civitai_queue: queue.Queue[_CivitaiTask] = queue.Queue()

        self._hash_dedupe_lock = threading.Lock()
        self._queued_hash_tasks: set[tuple[str, str]] = set()
        self._queued_civitai_lock = threading.Lock()
        self._queued_civitai_hashes: set[str] = set()

        self._start_lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            self._db.initialize()

            self._start_thread(_DB_WORKER_NAME, self._db_writer_loop)
            self._start_thread(_HASH_WORKER_NAME, self._hash_worker_loop)
            self._start_thread(_CIVITAI_WORKER_NAME, self._civitai_worker_loop)
            self._start_thread(_SEED_WORKER_NAME, self._seed_normal_queue_loop)

            self._started = True

    def _start_thread(self, name: str, target: Callable[[], None]) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()

    def enqueue_hash_priority(self, folder_name: str, relative_path: str) -> bool:
        return self._enqueue_hash_task(folder_name, relative_path, priority=True)

    def enqueue_hash_normal(self, folder_name: str, relative_path: str) -> bool:
        return self._enqueue_hash_task(folder_name, relative_path, priority=False)

    def _enqueue_hash_task(self, folder_name: str, relative_path: str, *, priority: bool) -> bool:
        folder = str(folder_name or "").strip()
        rel = normalize_relative_path(relative_path)
        if not folder or not rel:
            return False

        task = _HashTask(folder_name=folder, relative_path=rel)
        dedupe_key = (task.folder_name, task.relative_path)
        with self._hash_dedupe_lock:
            if dedupe_key in self._queued_hash_tasks:
                return False
            self._queued_hash_tasks.add(dedupe_key)

        if priority:
            self._hash_priority_queue.put(task)
        else:
            self._hash_normal_queue.put(task)
        return True

    def find_relative_path_by_hash(
        self,
        folder_name: str,
        hash_prefix: str,
        preferred_algos: Sequence[str] | None = None,
    ) -> str | None:
        return self._db.find_relative_path_by_hash(
            folder_name=folder_name,
            hash_prefix=hash_prefix,
            preferred_algos=preferred_algos,
        )

    def find_relative_paths_by_hash(
        self,
        folder_name: str,
        hash_prefix: str,
        preferred_algos: Sequence[str] | None = None,
    ) -> list[str]:
        return self._db.find_relative_paths_by_hash(
            folder_name=folder_name,
            hash_prefix=hash_prefix,
            preferred_algos=preferred_algos,
        )

    def get_hash_by_relative_path(
        self,
        folder_name: str,
        relative_path: str,
        hash_algo: str = _HASH_ALGO_SHA256,
    ) -> str | None:
        return self._db.get_hash_by_relative_path(
            folder_name=folder_name,
            relative_path=relative_path,
            hash_algo=hash_algo,
        )

    def should_queue_civitai_lookup(self, sha256: str, *, now_iso: str) -> bool:
        return self._db.should_queue_civitai_lookup(sha256, now_iso=now_iso)

    def get_civitai_lookup_state(self, sha256: str) -> dict[str, Any] | None:
        return self._db.get_civitai_lookup_state(sha256)

    def get_model_info_by_relative_path(
        self,
        *,
        folder_name: str,
        relative_path: str,
    ) -> dict[str, Any] | None:
        return self._db.get_model_info_by_relative_path(
            folder_name=folder_name,
            relative_path=relative_path,
        )

    def get_model_reference_by_relative_path(
        self,
        *,
        folder_name: str,
        relative_path: str,
    ) -> dict[str, Any] | None:
        return self._db.get_model_reference_by_relative_path(
            folder_name=folder_name,
            relative_path=relative_path,
        )

    def get_model_reference_by_sha256(
        self,
        *,
        sha256: str,
    ) -> dict[str, Any] | None:
        return self._db.get_model_reference_by_sha256(sha256=sha256)

    def get_model_reference_by_hash_hint(
        self,
        *,
        hash_value: str,
        preferred_algos: Sequence[str] | None = None,
        name_hint: str | None = None,
    ) -> dict[str, Any] | None:
        return self._db.get_model_reference_by_hash_hint(
            hash_value=hash_value,
            preferred_algos=preferred_algos,
            name_hint=name_hint,
        )

    def get_civitai_version_payload_by_relative_path(
        self,
        *,
        folder_name: str,
        relative_path: str,
    ) -> dict[str, Any] | None:
        return self._db.get_civitai_version_payload_by_relative_path(
            folder_name=folder_name,
            relative_path=relative_path,
        )

    def get_model_runtime_settings_by_relative_path(
        self,
        *,
        folder_name: str,
        relative_path: str,
    ) -> dict[str, int | float]:
        return self._db.get_model_runtime_settings_by_relative_path(
            folder_name=folder_name,
            relative_path=relative_path,
        )

    def upsert_model_runtime_settings_by_relative_path(
        self,
        *,
        folder_name: str,
        relative_path: str,
        settings: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        folder = str(folder_name or "").strip()
        rel = normalize_relative_path(relative_path)
        if not folder or not rel or not is_supported_model_runtime_settings_folder(folder):
            return None

        content_id = self._db.get_content_id_by_relative_path(
            folder_name=folder,
            relative_path=rel,
        )
        if content_id is None:
            self.enqueue_hash_priority(folder, rel)
            return None

        updated_at = _utc_now_iso()
        con = self._db.open_writer_connection()
        try:
            con.execute("BEGIN")
            normalized_settings = self._db.replace_model_runtime_settings(
                con,
                content_id=content_id,
                folder_name=folder,
                settings=settings,
                updated_at=updated_at,
            )
            con.execute("COMMIT")
        except Exception:
            try:
                con.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            con.close()

        return {
            "content_id": content_id,
            "folder_name": folder,
            "relative_path": rel,
            "runtime_settings": filter_model_runtime_settings_for_folder(folder, normalized_settings),
            "updated_at": updated_at,
        }

    def list_hash_hints_by_content_id(
        self,
        content_id: int,
        *,
        include_sha256: bool = False,
    ) -> list[dict[str, str]]:
        return self._db.list_hash_hints_by_content_id(content_id, include_sha256=include_sha256)

    def list_lora_tags_by_content_id(
        self,
        content_id: int,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self._db.list_lora_tags_by_content_id(content_id, limit=limit)

    def get_content_id_by_relative_path(
        self,
        *,
        folder_name: str,
        relative_path: str,
    ) -> int | None:
        return self._db.get_content_id_by_relative_path(
            folder_name=folder_name,
            relative_path=relative_path,
        )

    def has_lora_metadata_for_content(self, content_id: int) -> bool:
        return self._db.has_lora_metadata_for_content(content_id)

    def is_hash_task_queued(self, folder_name: str, relative_path: str) -> bool:
        folder = str(folder_name or "").strip()
        rel = normalize_relative_path(relative_path)
        if not folder or not rel:
            return False
        with self._hash_dedupe_lock:
            return (folder, rel) in self._queued_hash_tasks

    def _submit_db_job(self, fn: Callable[[Any], None]) -> None:
        def wrapped() -> None:
            con = self._db.open_writer_connection()
            try:
                con.execute("BEGIN")
                fn(con)
                con.execute("COMMIT")
            except Exception:
                try:
                    con.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                con.close()

        self._db_queue.put(wrapped)

    def _db_writer_loop(self) -> None:
        while True:
            task = self._db_queue.get()
            try:
                if task is None:
                    return
                try:
                    task()
                except Exception as exc:
                    _safe_print(f"[IPT][db] writer error: {exc}")
            finally:
                self._db_queue.task_done()

    def _hash_worker_loop(self) -> None:
        while True:
            task, from_priority = self._next_hash_task()
            if task is None:
                continue

            dedupe_key = (task.folder_name, task.relative_path)
            try:
                self._process_hash_task(task)
            except Exception as exc:
                _safe_print(f"[IPT][hash] worker error: {exc}")
            finally:
                with self._hash_dedupe_lock:
                    self._queued_hash_tasks.discard(dedupe_key)
                if from_priority:
                    self._hash_priority_queue.task_done()
                else:
                    self._hash_normal_queue.task_done()

    def _next_hash_task(self) -> tuple[_HashTask | None, bool]:
        try:
            return self._hash_priority_queue.get_nowait(), True
        except queue.Empty:
            pass

        try:
            return self._hash_normal_queue.get(timeout=1.0), False
        except queue.Empty:
            return None, False

    def _process_hash_task(self, task: _HashTask) -> None:
        resolved = self._resolve_absolute_path(task.folder_name, task.relative_path)
        if resolved is None:
            return
        relative_path, absolute_path = resolved
        if not os.path.isfile(absolute_path):
            return

        stat_result = os.stat(absolute_path)
        file_size = int(stat_result.st_size)
        mtime_ns = int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)))
        ctime_ns = int(getattr(stat_result, "st_ctime_ns", int(stat_result.st_ctime * 1_000_000_000)))
        identity_kind, identity_key, volume_hint = self._extract_file_identity(stat_result)

        resolved_content_id: int | None = None
        try:
            resolved_content_id = self._db.resolve_content_id_for_observation(
                folder_name=task.folder_name,
                relative_path=relative_path,
                file_size=file_size,
                mtime_ns=mtime_ns,
                identity_kind=identity_kind,
                identity_key=identity_key,
            )
        except Exception as exc:
            _safe_print(
                f"[IPT][hash] content-resolve failed folder={task.folder_name} path={relative_path} error={exc}"
            )

        observed_at = _utc_now_iso()
        self._submit_db_job(
            lambda con, folder=task.folder_name, rel=relative_path, size=file_size, mt=mtime_ns, ct=ctime_ns, seen=observed_at, kind=identity_kind, key=identity_key, volume=volume_hint: self._db.upsert_observed_path_and_identity(
                con,
                folder_name=folder,
                relative_path=rel,
                file_size=size,
                mtime_ns=mt,
                seen_at=seen,
                identity_kind=kind,
                identity_key=key,
                volume_hint=volume,
                ctime_ns=ct,
            )
        )

        sha256: str | None = None
        if resolved_content_id is not None:
            sha256 = self._db.get_sha256_by_content_id(resolved_content_id)

        if not sha256:
            sha256 = self._compute_sha256(absolute_path)
            if not sha256:
                return
            seen_at = _utc_now_iso()
            self._submit_db_job(
                lambda con, folder=task.folder_name, rel=relative_path, size=file_size, mt=mtime_ns, ct=ctime_ns, digest=sha256, seen=seen_at, kind=identity_kind, key=identity_key, volume=volume_hint: self._db.upsert_local_asset_and_sha256(
                    con,
                    folder_name=folder,
                    relative_path=rel,
                    file_size=size,
                    mtime_ns=mt,
                    sha256=digest,
                    seen_at=seen,
                    identity_kind=kind,
                    identity_key=key,
                    volume_hint=volume,
                    ctime_ns=ct,
                )
            )

        if task.folder_name == "loras":
            metadata_is_current = False
            if resolved_content_id is not None:
                try:
                    metadata_is_current = self._db.has_lora_metadata_for_content(resolved_content_id)
                except Exception as exc:
                    _safe_print(
                        f"[IPT][hash] lora-content-metadata-check failed path={relative_path} error={exc}"
                    )
                    metadata_is_current = False

            if not metadata_is_current:
                try:
                    metadata_is_current = self._db.has_lora_metadata_current(
                        relative_path=relative_path,
                        file_size=file_size,
                        mtime_ns=mtime_ns,
                    )
                except Exception as exc:
                    _safe_print(
                        f"[IPT][hash] lora-path-metadata-check failed path={relative_path} error={exc}"
                    )
                    metadata_is_current = False

            if not metadata_is_current:
                metadata = self._read_safetensors_metadata(absolute_path)
                if metadata:
                    parsed_at = _utc_now_iso()
                    self._submit_db_job(
                        lambda con, folder=task.folder_name, rel=relative_path, parsed=parsed_at, payload=metadata: self._db.upsert_lora_metadata(
                            con,
                            folder_name=folder,
                            relative_path=rel,
                            metadata=payload,
                            parsed_at=parsed,
                        )
                    )

        if sha256:
            self._enqueue_civitai_lookup(sha256)

    def _compute_sha256(self, absolute_path: str) -> str | None:
        try:
            hasher = hashlib.sha256()
            with open(absolute_path, "rb") as f:
                while True:
                    chunk = f.read(DEFAULT_CHUNK_SIZE)
                    if not chunk:
                        break
                    hasher.update(chunk)
            return hasher.hexdigest().lower()
        except Exception:
            return None

    def _read_safetensors_metadata(self, absolute_path: str) -> Mapping[str, Any] | None:
        if not str(absolute_path).lower().endswith(".safetensors"):
            return None
        try:
            with open(absolute_path, "rb") as f:
                header_size = int.from_bytes(f.read(8), "little", signed=False)
                if header_size <= 0:
                    return None
                header = f.read(header_size)
            payload = json.loads(header)
            metadata = payload.get("__metadata__")
            if isinstance(metadata, Mapping):
                return dict(metadata)
            return None
        except Exception:
            return None

    def _resolve_absolute_path(self, folder_name: str, relative_path: str) -> tuple[str, str] | None:
        if folder_paths is None:
            return None
        getter = getattr(folder_paths, "get_full_path", None)
        if not callable(getter):
            return None

        normalized = normalize_relative_path(relative_path)
        if not normalized:
            return None

        try:
            absolute = getter(folder_name, normalized)
        except Exception:
            absolute = None
        if isinstance(absolute, str) and absolute and os.path.isfile(absolute):
            return normalized, os.path.abspath(absolute)
        return None

    def _extract_file_identity(self, stat_result: os.stat_result) -> tuple[str | None, str | None, str | None]:
        inode = _as_int(getattr(stat_result, "st_ino", None))
        if inode is None or inode <= 0:
            return None, None, None

        dev = _as_int(getattr(stat_result, "st_dev", None))
        volume_hint = str(dev) if dev is not None else None
        if os.name == "nt":
            if dev is None:
                return "win_file_id", str(inode), volume_hint
            return "win_file_id", f"{dev}:{inode}", volume_hint

        if dev is None:
            return "posix_inode", str(inode), volume_hint
        return "posix_inode", f"{dev}:{inode}", volume_hint

    def _enqueue_civitai_lookup(self, sha256: str) -> bool:
        digest = _normalize_hash_text(sha256)
        if not digest:
            return False
        now_iso = _utc_now_iso()
        try:
            should_queue = self._db.should_queue_civitai_lookup(digest, now_iso=now_iso)
        except Exception as exc:
            _safe_print(f"[IPT][civitai] enqueue check failed sha256={_short_sha256(digest)} error={exc}")
            should_queue = True
        if not should_queue:
            return False

        with self._queued_civitai_lock:
            if digest in self._queued_civitai_hashes:
                return False
            self._queued_civitai_hashes.add(digest)
        self._civitai_queue.put(_CivitaiTask(sha256=digest))
        return True

    def _civitai_worker_loop(self) -> None:
        while True:
            try:
                task = self._civitai_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                fetch_result = self._fetch_civitai_by_hash(task.sha256)
                checked_at = _utc_now_iso()
                next_retry_at = self._next_retry_at_for_status(fetch_result.status, checked_at=checked_at)

                self._submit_db_job(
                    lambda con: self._apply_civitai_fetch_result(
                        con,
                        sha256=task.sha256,
                        fetch_result=fetch_result,
                        checked_at=checked_at,
                        next_retry_at=next_retry_at,
                    )
                )
            except Exception as exc:
                _safe_print(f"[IPT][civitai] worker error: {exc}")
            finally:
                with self._queued_civitai_lock:
                    self._queued_civitai_hashes.discard(task.sha256)
                self._civitai_queue.task_done()

    def _apply_civitai_fetch_result(
        self,
        con: Any,
        *,
        sha256: str,
        fetch_result: _CivitaiFetchResult,
        checked_at: str,
        next_retry_at: str | None,
    ) -> None:
        if fetch_result.payload:
            self._db.upsert_civitai_payload(
                con,
                requested_sha256=sha256,
                payload=fetch_result.payload,
                fetched_at=checked_at,
            )

        self._db.upsert_civitai_lookup_state(
            con,
            sha256=sha256,
            status=fetch_result.status,
            checked_at=checked_at,
            http_status=fetch_result.http_status,
            next_retry_at=next_retry_at,
            last_error=fetch_result.error,
        )

    def fetch_civitai_by_sha256_now(self, sha256: str) -> dict[str, Any]:
        digest = _normalize_hash_text(sha256)
        if not digest:
            return {
                "status": _CIVITAI_STATUS_TEMP_ERROR,
                "http_status": None,
                "error": "invalid_sha256",
            }

        fetch_result = self._fetch_civitai_by_hash(digest)
        checked_at = _utc_now_iso()
        next_retry_at = self._next_retry_at_for_status(fetch_result.status, checked_at=checked_at)

        con = self._db.open_writer_connection()
        try:
            con.execute("BEGIN")
            self._apply_civitai_fetch_result(
                con,
                sha256=digest,
                fetch_result=fetch_result,
                checked_at=checked_at,
                next_retry_at=next_retry_at,
            )
            con.execute("COMMIT")
        except Exception:
            try:
                con.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            con.close()

        return {
            "status": fetch_result.status,
            "http_status": fetch_result.http_status,
            "error": fetch_result.error,
        }

    def fetch_civitai_by_hash_now(self, hash_value: str) -> dict[str, Any]:
        digest = _normalize_hash_text(hash_value)
        if not digest:
            return {
                "status": _CIVITAI_STATUS_TEMP_ERROR,
                "http_status": None,
                "error": "invalid_hash",
            }

        fetch_result = self._fetch_civitai_by_hash(digest)
        response: dict[str, Any] = {
            "status": fetch_result.status,
            "http_status": fetch_result.http_status,
            "error": fetch_result.error,
        }
        if isinstance(fetch_result.payload, Mapping):
            response["payload"] = dict(fetch_result.payload)
        return response

    def _next_retry_at_for_status(self, status: str, *, checked_at: str) -> str | None:
        checked = datetime.fromisoformat(str(checked_at).replace("Z", "+00:00"))
        if status == _CIVITAI_STATUS_FOUND:
            return None
        if status == _CIVITAI_STATUS_NOT_FOUND:
            retry_at = checked + timedelta(seconds=_CIVITAI_NOT_FOUND_RETRY_SECONDS)
            return retry_at.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        if status == _CIVITAI_STATUS_AUTH_BLOCKED:
            retry_at = checked + timedelta(seconds=_CIVITAI_AUTH_BLOCKED_RETRY_SECONDS)
            return retry_at.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        retry_at = checked + timedelta(seconds=_CIVITAI_TEMP_ERROR_RETRY_SECONDS)
        return retry_at.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _startup_gc_cutoff_iso(self) -> str:
        cutoff = datetime.now(timezone.utc) - timedelta(days=_STARTUP_GC_RETENTION_DAYS)
        return cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _enqueue_startup_gc(self) -> None:
        cutoff_iso = self._startup_gc_cutoff_iso()

        def _job(con: Any) -> None:
            self._db.run_startup_gc_if_needed(
                con,
                older_than_iso=cutoff_iso,
                min_deleted_paths=_STARTUP_GC_MIN_DELETED_PATHS,
                min_orphan_identities=_STARTUP_GC_MIN_ORPHAN_IDENTITIES,
                max_deleted_paths=_STARTUP_GC_MAX_DELETE_PATHS,
                max_orphan_identities=_STARTUP_GC_MAX_DELETE_ORPHAN_IDENTITIES,
            )

        self._submit_db_job(_job)

    def _fetch_civitai_by_hash(self, sha256: str) -> _CivitaiFetchResult:
        digest = _normalize_hash_text(sha256)
        if not digest:
            _log_civitai_fetch_anomaly(sha256, error="invalid_sha256")
            return _CivitaiFetchResult(
                status=_CIVITAI_STATUS_TEMP_ERROR,
                error="invalid_sha256",
            )

        request = url_request.Request(
            _CIVITAI_HASH_URL + digest.upper(),
            headers=dict(_CIVITAI_DEFAULT_HEADERS),
            method="GET",
        )
        try:
            with url_request.urlopen(request, timeout=20) as response:
                status = int(getattr(response, "status", 200))
                if status != 200:
                    if status == 404:
                        return _CivitaiFetchResult(
                            status=_CIVITAI_STATUS_NOT_FOUND,
                            http_status=status,
                            error="http_404",
                        )
                    if status in (401, 403):
                        _log_civitai_fetch_anomaly(digest, error=f"http_{status}", http_status=status)
                        return _CivitaiFetchResult(
                            status=_CIVITAI_STATUS_AUTH_BLOCKED,
                            http_status=status,
                            error=f"http_{status}",
                        )
                    if 400 <= status < 500:
                        _log_civitai_fetch_anomaly(digest, error=f"http_{status}", http_status=status)
                    return _CivitaiFetchResult(
                        status=_CIVITAI_STATUS_TEMP_ERROR,
                        http_status=status,
                        error=f"http_{status}",
                    )
                payload = json.loads(response.read().decode("utf-8"))
        except url_error.HTTPError as exc:
            if exc.code == 404:
                return _CivitaiFetchResult(
                    status=_CIVITAI_STATUS_NOT_FOUND,
                    http_status=int(exc.code),
                    error="http_404",
                )
            if exc.code in (401, 403):
                _log_civitai_fetch_anomaly(digest, error=f"http_{int(exc.code)}", http_status=int(exc.code))
                return _CivitaiFetchResult(
                    status=_CIVITAI_STATUS_AUTH_BLOCKED,
                    http_status=int(exc.code),
                    error=f"http_{int(exc.code)}",
                )
            if 400 <= int(exc.code) < 500:
                _log_civitai_fetch_anomaly(digest, error=f"http_{int(exc.code)}", http_status=int(exc.code))
            return _CivitaiFetchResult(
                status=_CIVITAI_STATUS_TEMP_ERROR,
                http_status=int(exc.code),
                error=f"http_{int(exc.code)}",
            )
        except url_error.URLError as exc:
            return _CivitaiFetchResult(
                status=_CIVITAI_STATUS_TEMP_ERROR,
                error=f"url_error:{exc.reason}",
            )
        except Exception:
            _log_civitai_fetch_anomaly(digest, error="unknown_exception")
            return _CivitaiFetchResult(
                status=_CIVITAI_STATUS_TEMP_ERROR,
                error="unknown_exception",
            )

        if isinstance(payload, Mapping):
            try:
                int(payload.get("id"))
                int(payload.get("modelId"))
            except Exception:
                _log_civitai_fetch_anomaly(digest, error="missing_required_ids", http_status=200)
                return _CivitaiFetchResult(
                    status=_CIVITAI_STATUS_TEMP_ERROR,
                    http_status=200,
                    error="missing_required_ids",
                )
            return _CivitaiFetchResult(
                status=_CIVITAI_STATUS_FOUND,
                http_status=200,
                payload=dict(payload),
            )
        _log_civitai_fetch_anomaly(digest, error="invalid_payload_type", http_status=200)
        return _CivitaiFetchResult(
            status=_CIVITAI_STATUS_TEMP_ERROR,
            http_status=200,
            error="invalid_payload_type",
        )

    def _seed_normal_queue_loop(self) -> None:
        if folder_paths is None:
            return

        getter_list = getattr(folder_paths, "get_filename_list", None)
        getter_path = getattr(folder_paths, "get_full_path", None)
        if not callable(getter_list) or not callable(getter_path):
            return

        budget = _STARTUP_BUDGET_BYTES
        queued_bytes = 0
        # Keep VAE last so seed-enqueued hash tasks stay lowest among model-family folders.
        target_folders = ("checkpoints", "diffusion_models", "unet", "text_encoders", "loras", "vae")
        hash_enqueue_budget_exhausted = False
        scan_started_at = _utc_now_iso()

        for folder in target_folders:
            try:
                options = tuple(str(item) for item in getter_list(folder))
            except Exception:
                options = tuple()

            for option in options:
                normalized = normalize_relative_path(option)
                if not normalized:
                    continue

                try:
                    absolute_path = getter_path(folder, normalized)
                except Exception:
                    absolute_path = None
                if not isinstance(absolute_path, str) or not absolute_path or not os.path.isfile(absolute_path):
                    continue

                try:
                    stat_result = os.stat(absolute_path)
                except OSError:
                    continue

                file_size = int(stat_result.st_size)
                mtime_ns = int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)))
                ctime_ns = int(getattr(stat_result, "st_ctime_ns", int(stat_result.st_ctime * 1_000_000_000)))
                identity_kind, identity_key, volume_hint = self._extract_file_identity(stat_result)

                resolved_content_id: int | None = None
                try:
                    resolved_content_id = self._db.resolve_content_id_for_observation(
                        folder_name=folder,
                        relative_path=normalized,
                        file_size=file_size,
                        mtime_ns=mtime_ns,
                        identity_kind=identity_kind,
                        identity_key=identity_key,
                    )
                except Exception as exc:
                    _safe_print(
                        f"[IPT][seed] content-resolve failed folder={folder} path={normalized} error={exc}"
                    )

                self._submit_db_job(
                    lambda con, folder_name=folder, relative=normalized, size=file_size, mt=mtime_ns, ct=ctime_ns, seen=scan_started_at, kind=identity_kind, key=identity_key, volume=volume_hint: self._db.upsert_observed_path_and_identity(
                        con,
                        folder_name=folder_name,
                        relative_path=relative,
                        file_size=size,
                        mtime_ns=mt,
                        seen_at=seen,
                        identity_kind=kind,
                        identity_key=key,
                        volume_hint=volume,
                        ctime_ns=ct,
                    )
                )

                if resolved_content_id is not None:
                    continue

                if hash_enqueue_budget_exhausted:
                    continue

                if queued_bytes >= budget:
                    hash_enqueue_budget_exhausted = True
                    continue

                if queued_bytes + file_size > budget and queued_bytes > 0:
                    hash_enqueue_budget_exhausted = True
                    continue

                enqueued = self.enqueue_hash_normal(folder, normalized)
                if enqueued:
                    queued_bytes += file_size

        marked_at = _utc_now_iso()
        self._submit_db_job(
            lambda con, folders=target_folders, scan_at=scan_started_at, mark_at=marked_at: self._db.mark_missing_paths_deleted(
                con,
                folder_names=folders,
                scan_started_at=scan_at,
                marked_at=mark_at,
            )
        )
        self._enqueue_startup_gc()

        try:
            pending_hashes = self._db.list_sha256_without_active_civitai_match(
                as_of_iso=_utc_now_iso(),
                limit=_STARTUP_CIVITAI_REQUEUE_LIMIT
            )
        except Exception as exc:
            _safe_print(f"[IPT][civitai] startup requeue scan failed: {exc}")
            return

        for digest in pending_hashes:
            self._enqueue_civitai_lookup(digest)


_SHARED_PIPELINE: ModelLoraMetadataPipeline | None = None
_SHARED_PIPELINE_LOCK = threading.Lock()


def get_shared_metadata_pipeline(*, start: bool = True) -> ModelLoraMetadataPipeline:
    global _SHARED_PIPELINE
    with _SHARED_PIPELINE_LOCK:
        if _SHARED_PIPELINE is None:
            _SHARED_PIPELINE = ModelLoraMetadataPipeline()
        pipeline = _SHARED_PIPELINE
    if start:
        pipeline.start()
    return pipeline

