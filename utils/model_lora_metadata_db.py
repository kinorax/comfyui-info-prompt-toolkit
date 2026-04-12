from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Mapping, Sequence

from .file_hash_cache import normalize_relative_path
from .model_runtime_settings import (
    filter_model_runtime_settings_for_folder,
    is_supported_model_runtime_settings_folder,
    normalize_model_runtime_settings,
)

try:
    import folder_paths  # type: ignore
except Exception:
    folder_paths = None  # type: ignore[assignment]


CACHE_DIR_NAME = "info_prompt_toolkit"
DB_FILE_NAME = "model_lora_metadata.sqlite3"
_DDL_RELATIVE_PATH = "resources/sql/model-and-lora-metadata-rdb-initial.sql"

_HASH_ALGO_SHA256 = "sha256"
_HASH_KEY_TO_ALGO = {
    "SHA256": "sha256",
    "AutoV1": "autov1",
    "AutoV2": "autov2",
    "AutoV3": "autov3",
    "CRC32": "crc32",
    "BLAKE3": "blake3",
}

_CIVITAI_LOOKUP_STATUSES = (
    "found",
    "not_found",
    "unknown_auth_or_blocked",
    "temporary_error",
)


def _normalize_hash_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _normalize_identity_kind(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in ("win_file_id", "posix_inode"):
        return text
    return None


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_json_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except Exception:
        return str(value)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _as_bool_int(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) != 0 else 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return 1
        if lowered in ("0", "false", "no", "off"):
            return 0
    return default


def _safe_rowcount(cursor: sqlite3.Cursor) -> int:
    try:
        count = int(cursor.rowcount)
    except Exception:
        return 0
    return count if count >= 0 else 0


def _parse_tag_frequency(value: Any) -> dict[str, Mapping[str, Any]]:
    if value is None:
        return {}

    parsed: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}

    if not isinstance(parsed, Mapping):
        return {}

    output: dict[str, Mapping[str, Any]] = {}
    for raw_dataset, raw_tags in parsed.items():
        dataset_name = str(raw_dataset or "").strip()
        if not isinstance(raw_tags, Mapping):
            continue
        output[dataset_name] = raw_tags
    return output


def resolve_db_path() -> str:
    if folder_paths is not None:
        getter = getattr(folder_paths, "get_user_directory", None)
        if callable(getter):
            try:
                user_dir = getter()
            except Exception:
                user_dir = None
            if isinstance(user_dir, str) and user_dir:
                return str(Path(user_dir) / CACHE_DIR_NAME / DB_FILE_NAME)

    return str(Path(__file__).resolve().parents[1] / ".cache" / CACHE_DIR_NAME / DB_FILE_NAME)


def _load_schema_sql() -> str:
    root = Path(__file__).resolve().parents[1]
    ddl_path = root / _DDL_RELATIVE_PATH
    if not ddl_path.is_file():
        raise FileNotFoundError(f"DDL file not found: {ddl_path}")
    return ddl_path.read_text(encoding="utf-8")


class MetadataDatabase:
    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self._db_path = Path(db_path)
        self._init_lock = threading.Lock()
        self._initialized = False
        self._reader_local = threading.local()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def initialize(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            con = self._connect()
            try:
                con.executescript(_load_schema_sql())
            finally:
                con.close()
            self._initialized = True

    def open_writer_connection(self) -> sqlite3.Connection:
        self.initialize()
        return self._connect()

    def _reader_connection(self) -> sqlite3.Connection:
        self.initialize()
        con = getattr(self._reader_local, "connection", None)
        if isinstance(con, sqlite3.Connection):
            return con
        con = self._connect()
        self._reader_local.connection = con
        return con

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._db_path), timeout=5.0, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=5000")
        return con

    def is_sha256_current(
        self,
        folder_name: str,
        relative_path: str,
        file_size: int,
        mtime_ns: int,
    ) -> bool:
        folder = str(folder_name or "").strip()
        rel = normalize_relative_path(relative_path)
        if not folder or not rel:
            return False

        con = self._reader_connection()
        row = con.execute(
            """
            SELECT 1
            FROM local_asset_path AS p
            JOIN file_identity AS fi ON fi.file_id = p.file_id
            JOIN content AS c ON c.content_id = fi.content_id
            WHERE p.folder_name = ?
              AND p.relative_path = ?
              AND p.is_deleted = 0
              AND fi.last_file_size = ?
              AND fi.last_mtime_ns = ?
              AND c.sha256 IS NOT NULL
              AND LENGTH(c.sha256) = 64
            LIMIT 1
            """,
            (folder, rel, int(file_size), int(mtime_ns)),
        ).fetchone()
        return row is not None

    def resolve_content_id_for_observation(
        self,
        *,
        folder_name: str,
        relative_path: str,
        file_size: int,
        mtime_ns: int,
        identity_kind: str | None = None,
        identity_key: str | None = None,
    ) -> int | None:
        folder = str(folder_name or "").strip()
        rel = normalize_relative_path(relative_path)
        if not folder or not rel:
            return None

        kind = _normalize_identity_kind(identity_kind)
        key = str(identity_key or "").strip()

        con = self._reader_connection()

        if kind and key:
            row = con.execute(
                """
                SELECT content_id
                FROM file_identity
                WHERE identity_kind = ?
                  AND identity_key = ?
                  AND content_id IS NOT NULL
                  AND last_file_size = ?
                  AND last_mtime_ns = ?
                LIMIT 1
                """,
                (kind, key, int(file_size), int(mtime_ns)),
            ).fetchone()
            if row is not None and row[0] is not None:
                return int(row[0])

        row = con.execute(
            """
            SELECT fi.content_id
            FROM local_asset_path AS p
            JOIN file_identity AS fi ON fi.file_id = p.file_id
            WHERE p.folder_name = ?
              AND p.relative_path = ?
              AND p.is_deleted = 0
              AND fi.content_id IS NOT NULL
              AND fi.last_file_size = ?
              AND fi.last_mtime_ns = ?
            LIMIT 1
            """,
            (folder, rel, int(file_size), int(mtime_ns)),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    def get_sha256_by_content_id(self, content_id: int) -> str | None:
        cid = _as_int(content_id)
        if cid is None:
            return None
        con = self._reader_connection()
        row = con.execute(
            "SELECT sha256 FROM content WHERE content_id = ? LIMIT 1",
            (cid,),
        ).fetchone()
        if row is None:
            return None
        return _normalize_hash_text(row[0])

    def get_hash_by_relative_path(
        self,
        folder_name: str,
        relative_path: str,
        hash_algo: str = _HASH_ALGO_SHA256,
    ) -> str | None:
        folder = str(folder_name or "").strip()
        rel = normalize_relative_path(relative_path)
        algo = str(hash_algo or "").strip().lower()
        if not folder or not rel or not algo:
            return None

        con = self._reader_connection()
        if algo == _HASH_ALGO_SHA256:
            row = con.execute(
                """
                SELECT c.sha256
                FROM local_asset_path AS p
                JOIN file_identity AS fi ON fi.file_id = p.file_id
                JOIN content AS c ON c.content_id = fi.content_id
                WHERE p.folder_name = ?
                  AND p.relative_path = ?
                  AND p.is_deleted = 0
                LIMIT 1
                """,
                (folder, rel),
            ).fetchone()
            if row is None:
                return None
            return _normalize_hash_text(row[0])

        row = con.execute(
            """
            SELECT ch.hash_value
            FROM local_asset_path AS p
            JOIN file_identity AS fi ON fi.file_id = p.file_id
            JOIN content_hash AS ch ON ch.content_id = fi.content_id
            WHERE p.folder_name = ?
              AND p.relative_path = ?
              AND p.is_deleted = 0
              AND ch.hash_algo = ?
            LIMIT 1
            """,
            (folder, rel, algo),
        ).fetchone()
        if row is None:
            return None
        return _normalize_hash_text(row[0])

    def find_relative_path_by_hash(
        self,
        folder_name: str,
        hash_prefix: str,
        preferred_algos: Sequence[str] | None = None,
    ) -> str | None:
        matches = self.find_relative_paths_by_hash(
            folder_name=folder_name,
            hash_prefix=hash_prefix,
            preferred_algos=preferred_algos,
        )
        if not matches:
            return None
        return matches[0]

    def find_relative_paths_by_hash(
        self,
        folder_name: str,
        hash_prefix: str,
        preferred_algos: Sequence[str] | None = None,
    ) -> list[str]:
        folder = str(folder_name or "").strip()
        prefix = _normalize_hash_text(hash_prefix)
        if not folder or not prefix:
            return []

        con = self._reader_connection()

        def _query_sha256() -> list[str]:
            rows = con.execute(
                """
                SELECT DISTINCT p.relative_path
                FROM content AS c
                JOIN file_identity AS fi ON fi.content_id = c.content_id
                JOIN local_asset_path AS p ON p.file_id = fi.file_id
                WHERE p.folder_name = ?
                  AND p.is_deleted = 0
                  AND LOWER(c.sha256) LIKE ?
                ORDER BY p.relative_path COLLATE NOCASE ASC
                """,
                (folder, f"{prefix}%"),
            ).fetchall()
            output: list[str] = []
            for row in rows:
                value = row[0]
                if not isinstance(value, str) or not value:
                    continue
                normalized = normalize_relative_path(value)
                if normalized and normalized not in output:
                    output.append(normalized)
            return output

        def _query_content_hash(algo: str | None = None) -> list[str]:
            if algo:
                rows = con.execute(
                    """
                    SELECT DISTINCT p.relative_path
                    FROM content_hash AS ch
                    JOIN file_identity AS fi ON fi.content_id = ch.content_id
                    JOIN local_asset_path AS p ON p.file_id = fi.file_id
                    WHERE p.folder_name = ?
                      AND p.is_deleted = 0
                      AND ch.hash_algo = ?
                      AND LOWER(ch.hash_value) LIKE ?
                    ORDER BY p.relative_path COLLATE NOCASE ASC
                    """,
                    (folder, algo, f"{prefix}%"),
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT DISTINCT p.relative_path
                    FROM content_hash AS ch
                    JOIN file_identity AS fi ON fi.content_id = ch.content_id
                    JOIN local_asset_path AS p ON p.file_id = fi.file_id
                    WHERE p.folder_name = ?
                      AND p.is_deleted = 0
                      AND LOWER(ch.hash_value) LIKE ?
                    ORDER BY p.relative_path COLLATE NOCASE ASC
                    """,
                    (folder, f"{prefix}%"),
                ).fetchall()
            output: list[str] = []
            for row in rows:
                value = row[0]
                if not isinstance(value, str) or not value:
                    continue
                normalized = normalize_relative_path(value)
                if normalized and normalized not in output:
                    output.append(normalized)
            return output

        def _dedupe_extend(output: list[str], values: Sequence[str]) -> None:
            for value in values:
                if value not in output:
                    output.append(value)

        matches: list[str] = []
        if preferred_algos:
            for raw_algo in preferred_algos:
                algo = str(raw_algo or "").strip().lower()
                if not algo:
                    continue
                if algo == _HASH_ALGO_SHA256:
                    found = _query_sha256()
                else:
                    found = _query_content_hash(algo)
                _dedupe_extend(matches, found)
            if matches:
                return matches

        _dedupe_extend(matches, _query_sha256())
        _dedupe_extend(matches, _query_content_hash())
        return matches

    def should_queue_civitai_lookup(self, sha256: str, *, now_iso: str) -> bool:
        digest = _normalize_hash_text(sha256)
        now_text = str(now_iso or "").strip()
        if not digest or not now_text:
            return False

        con = self._reader_connection()
        matched_row = con.execute(
            """
            SELECT 1
            FROM content AS c
            JOIN content_civitai_match AS m
              ON m.content_id = c.content_id
             AND m.is_active = 1
            WHERE c.sha256 = ?
            LIMIT 1
            """,
            (digest,),
        ).fetchone()
        if matched_row is not None:
            return False

        row = con.execute(
            """
            SELECT status, next_retry_at
            FROM civitai_lookup_state
            WHERE sha256 = ?
            LIMIT 1
            """,
            (digest,),
        ).fetchone()
        if row is None:
            return True

        status = _as_text(row[0]) or ""
        next_retry_at = _as_text(row[1])

        if status == "found":
            return False
        if not next_retry_at:
            return True
        return next_retry_at <= now_text

    def get_civitai_lookup_state(self, sha256: str) -> dict[str, Any] | None:
        digest = _normalize_hash_text(sha256)
        if not digest:
            return None

        con = self._reader_connection()
        row = con.execute(
            """
            SELECT status, http_status, next_retry_at, last_error, checked_at
            FROM civitai_lookup_state
            WHERE sha256 = ?
            LIMIT 1
            """,
            (digest,),
        ).fetchone()
        if row is None:
            return None

        return {
            "status": _as_text(row["status"]),
            "http_status": _as_int(row["http_status"]),
            "next_retry_at": _as_text(row["next_retry_at"]),
            "last_error": _as_text(row["last_error"]),
            "checked_at": _as_text(row["checked_at"]),
        }

    def list_sha256_without_active_civitai_match(
        self,
        *,
        as_of_iso: str,
        limit: int | None = None,
    ) -> list[str]:
        now_text = str(as_of_iso or "").strip()
        if not now_text:
            return []

        sql = """
        SELECT c.sha256
        FROM content AS c
        WHERE EXISTS (
                SELECT 1
                FROM file_identity AS fi
                JOIN local_asset_path AS p ON p.file_id = fi.file_id
                WHERE fi.content_id = c.content_id
                  AND p.is_deleted = 0
              )
          AND NOT EXISTS (
                SELECT 1
                FROM content_civitai_match AS m
                WHERE m.content_id = c.content_id
                  AND m.is_active = 1
              )
          AND (
                NOT EXISTS (
                    SELECT 1
                    FROM civitai_lookup_state AS s
                    WHERE s.sha256 = c.sha256
                )
                OR EXISTS (
                    SELECT 1
                    FROM civitai_lookup_state AS s
                    WHERE s.sha256 = c.sha256
                      AND s.status <> 'found'
                      AND (s.next_retry_at IS NULL OR s.next_retry_at <= ?)
                )
              )
        ORDER BY c.last_seen_at DESC, c.content_id DESC
        """

        params: tuple[Any, ...] = (now_text,)
        if limit is not None:
            safe_limit = max(1, int(limit))
            sql = f"{sql}\nLIMIT ?"
            params = (now_text, safe_limit)

        con = self._reader_connection()
        rows = con.execute(sql, params).fetchall()
        output: list[str] = []
        for row in rows:
            digest = _normalize_hash_text(row[0])
            if digest and digest not in output:
                output.append(digest)
        return output

    def has_lora_metadata_current(
        self,
        *,
        relative_path: str,
        file_size: int,
        mtime_ns: int,
    ) -> bool:
        rel = normalize_relative_path(relative_path)
        if not rel:
            return False

        con = self._reader_connection()
        row = con.execute(
            """
            SELECT 1
            FROM local_asset_path AS p
            JOIN file_identity AS fi ON fi.file_id = p.file_id
            JOIN lora_metadata AS lm ON lm.content_id = fi.content_id
            WHERE p.folder_name = 'loras'
              AND p.relative_path = ?
              AND p.is_deleted = 0
              AND fi.last_file_size = ?
              AND fi.last_mtime_ns = ?
            LIMIT 1
            """,
            (rel, int(file_size), int(mtime_ns)),
        ).fetchone()
        return row is not None

    def has_lora_metadata_for_content(self, content_id: int) -> bool:
        cid = _as_int(content_id)
        if cid is None:
            return False
        con = self._reader_connection()
        row = con.execute(
            "SELECT 1 FROM lora_metadata WHERE content_id = ? LIMIT 1",
            (cid,),
        ).fetchone()
        return row is not None

    def get_model_info_by_relative_path(
        self,
        *,
        folder_name: str,
        relative_path: str,
    ) -> dict[str, Any] | None:
        folder = str(folder_name or "").strip()
        rel = normalize_relative_path(relative_path)
        if not folder or not rel:
            return None

        con = self._reader_connection()
        row = con.execute(
            """
            SELECT
                c.content_id AS content_id,
                cm.name AS civitai_model_name,
                cv.model_id AS civitai_model_version_model_id,
                cv.model_version_id AS civitai_model_version_model_version_id,
                cv.name AS civitai_model_version_name,
                cv.base_model AS civitai_model_version_base_model
            FROM local_asset_path AS p
            JOIN file_identity AS fi ON fi.file_id = p.file_id
            JOIN content AS c ON c.content_id = fi.content_id
            LEFT JOIN content_civitai_match AS ccm
                ON ccm.content_id = c.content_id
               AND ccm.is_active = 1
            LEFT JOIN civitai_file AS cf ON cf.civitai_file_id = ccm.civitai_file_id
            LEFT JOIN civitai_model_version AS cv ON cv.model_version_id = cf.model_version_id
            LEFT JOIN civitai_model AS cm ON cm.model_id = cv.model_id
            WHERE p.folder_name = ?
              AND p.relative_path = ?
              AND p.is_deleted = 0
            ORDER BY
                COALESCE(cf.is_primary, 0) DESC,
                ccm.last_confirmed_at DESC,
                ccm.civitai_file_id DESC
            LIMIT 1
            """,
            (folder, rel),
        ).fetchone()
        if row is None:
            return None

        content_id = _as_int(row["content_id"])
        if content_id is None:
            return None

        return {
            "content_id": content_id,
            "civitai_model_name": _as_text(row["civitai_model_name"]),
            "civitai_model_version_model_id": _as_int(row["civitai_model_version_model_id"]),
            "civitai_model_version_model_version_id": _as_int(row["civitai_model_version_model_version_id"]),
            "civitai_model_version_name": _as_text(row["civitai_model_version_name"]),
            "civitai_model_version_base_model": _as_text(row["civitai_model_version_base_model"]),
        }

    def get_content_id_by_relative_path(
        self,
        *,
        folder_name: str,
        relative_path: str,
    ) -> int | None:
        folder = str(folder_name or "").strip()
        rel = normalize_relative_path(relative_path)
        if not folder or not rel:
            return None

        con = self._reader_connection()
        row = con.execute(
            """
            SELECT fi.content_id
            FROM local_asset_path AS p
            JOIN file_identity AS fi ON fi.file_id = p.file_id
            WHERE p.folder_name = ?
              AND p.relative_path = ?
              AND p.is_deleted = 0
              AND fi.content_id IS NOT NULL
            LIMIT 1
            """,
            (folder, rel),
        ).fetchone()
        if row is None:
            return None
        return _as_int(row[0])

    def get_model_runtime_settings_by_content_id(
        self,
        content_id: int,
    ) -> dict[str, int | float]:
        cid = _as_int(content_id)
        if cid is None:
            return {}

        con = self._reader_connection()
        row = con.execute(
            """
            SELECT settings_json
            FROM model_runtime_settings
            WHERE content_id = ?
            LIMIT 1
            """,
            (cid,),
        ).fetchone()
        if row is None:
            return {}
        return normalize_model_runtime_settings(row["settings_json"])

    def _get_civitai_file_hashes_by_file_id(
        self,
        civitai_file_id: int | None,
    ) -> dict[str, str]:
        file_id = _as_int(civitai_file_id)
        if file_id is None:
            return {}

        con = self._reader_connection()
        rows = con.execute(
            """
            SELECT hash_algo, hash_value
            FROM civitai_file_hash
            WHERE civitai_file_id = ?
            """,
            (file_id,),
        ).fetchall()

        output: dict[str, str] = {}
        for row in rows:
            algo = _as_text(row["hash_algo"])
            digest = _normalize_hash_text(row["hash_value"])
            if not algo or not digest:
                continue
            output[algo] = digest
        return output

    def get_model_runtime_settings_by_relative_path(
        self,
        *,
        folder_name: str,
        relative_path: str,
    ) -> dict[str, int | float]:
        if not is_supported_model_runtime_settings_folder(folder_name):
            return {}

        content_id = self.get_content_id_by_relative_path(
            folder_name=folder_name,
            relative_path=relative_path,
        )
        if content_id is None:
            return {}

        settings = self.get_model_runtime_settings_by_content_id(content_id)
        return filter_model_runtime_settings_for_folder(folder_name, settings)

    def get_model_reference_by_relative_path(
        self,
        *,
        folder_name: str,
        relative_path: str,
    ) -> dict[str, Any] | None:
        folder = str(folder_name or "").strip()
        rel = normalize_relative_path(relative_path)
        if not folder or not rel:
            return None

        con = self._reader_connection()
        row = con.execute(
            """
            SELECT
                c.content_id AS content_id,
                c.sha256 AS sha256,
                cf.civitai_file_id AS civitai_file_id,
                cm.name AS civitai_model_name,
                cv.model_id AS civitai_model_version_model_id,
                cv.model_version_id AS civitai_model_version_model_version_id,
                cv.name AS civitai_model_version_name,
                cv.base_model AS civitai_model_version_base_model,
                cv.raw_json AS civitai_model_version_raw_json,
                cf.name AS civitai_file_name,
                cf.download_url AS civitai_file_download_url,
                cv.version_download_url AS civitai_version_download_url
            FROM local_asset_path AS p
            JOIN file_identity AS fi ON fi.file_id = p.file_id
            JOIN content AS c ON c.content_id = fi.content_id
            LEFT JOIN content_civitai_match AS ccm
                ON ccm.content_id = c.content_id
               AND ccm.is_active = 1
            LEFT JOIN civitai_file AS cf ON cf.civitai_file_id = ccm.civitai_file_id
            LEFT JOIN civitai_model_version AS cv ON cv.model_version_id = cf.model_version_id
            LEFT JOIN civitai_model AS cm ON cm.model_id = cv.model_id
            WHERE p.folder_name = ?
              AND p.relative_path = ?
              AND p.is_deleted = 0
            ORDER BY
                COALESCE(cf.is_primary, 0) DESC,
                ccm.last_confirmed_at DESC,
                ccm.civitai_file_id DESC
            LIMIT 1
            """,
            (folder, rel),
        ).fetchone()
        if row is None:
            return None

        civitai_file_id = _as_int(row["civitai_file_id"])
        return {
            "content_id": _as_int(row["content_id"]),
            "sha256": _normalize_hash_text(row["sha256"]),
            "civitai_file_hashes": self._get_civitai_file_hashes_by_file_id(civitai_file_id),
            "civitai_model_name": _as_text(row["civitai_model_name"]),
            "civitai_model_version_model_id": _as_int(row["civitai_model_version_model_id"]),
            "civitai_model_version_model_version_id": _as_int(row["civitai_model_version_model_version_id"]),
            "civitai_model_version_name": _as_text(row["civitai_model_version_name"]),
            "civitai_model_version_base_model": _as_text(row["civitai_model_version_base_model"]),
            "civitai_model_version_raw_json": _as_text(row["civitai_model_version_raw_json"]),
            "civitai_file_name": _as_text(row["civitai_file_name"]),
            "civitai_file_download_url": _as_text(row["civitai_file_download_url"]),
            "civitai_version_download_url": _as_text(row["civitai_version_download_url"]),
            "runtime_settings": self.get_model_runtime_settings_by_content_id(_as_int(row["content_id"]) or 0),
        }

    def get_model_reference_by_sha256(
        self,
        *,
        sha256: str,
    ) -> dict[str, Any] | None:
        digest = _normalize_hash_text(sha256)
        if digest is None:
            return None

        con = self._reader_connection()
        row = con.execute(
            """
            SELECT
                c.content_id AS content_id,
                fh.hash_value AS sha256,
                cf.civitai_file_id AS civitai_file_id,
                cm.name AS civitai_model_name,
                cv.model_id AS civitai_model_version_model_id,
                cv.model_version_id AS civitai_model_version_model_version_id,
                cv.name AS civitai_model_version_name,
                cv.base_model AS civitai_model_version_base_model,
                cv.raw_json AS civitai_model_version_raw_json,
                cf.name AS civitai_file_name,
                cf.download_url AS civitai_file_download_url,
                cv.version_download_url AS civitai_version_download_url
            FROM civitai_file_hash AS fh
            JOIN civitai_file AS cf ON cf.civitai_file_id = fh.civitai_file_id
            LEFT JOIN civitai_model_version AS cv ON cv.model_version_id = cf.model_version_id
            LEFT JOIN civitai_model AS cm ON cm.model_id = cv.model_id
            LEFT JOIN content AS c ON c.sha256 = fh.hash_value
            WHERE fh.hash_algo = 'sha256'
              AND fh.hash_value = ?
            ORDER BY
                COALESCE(cf.is_primary, 0) DESC,
                cf.civitai_file_id DESC
            LIMIT 1
            """,
            (digest,),
        ).fetchone()
        if row is None:
            return None

        civitai_file_id = _as_int(row["civitai_file_id"])
        return {
            "content_id": _as_int(row["content_id"]),
            "sha256": _normalize_hash_text(row["sha256"]),
            "civitai_file_hashes": self._get_civitai_file_hashes_by_file_id(civitai_file_id),
            "civitai_model_name": _as_text(row["civitai_model_name"]),
            "civitai_model_version_model_id": _as_int(row["civitai_model_version_model_id"]),
            "civitai_model_version_model_version_id": _as_int(row["civitai_model_version_model_version_id"]),
            "civitai_model_version_name": _as_text(row["civitai_model_version_name"]),
            "civitai_model_version_base_model": _as_text(row["civitai_model_version_base_model"]),
            "civitai_model_version_raw_json": _as_text(row["civitai_model_version_raw_json"]),
            "civitai_file_name": _as_text(row["civitai_file_name"]),
            "civitai_file_download_url": _as_text(row["civitai_file_download_url"]),
            "civitai_version_download_url": _as_text(row["civitai_version_download_url"]),
            "runtime_settings": self.get_model_runtime_settings_by_content_id(_as_int(row["content_id"]) or 0),
        }

    def get_model_reference_by_hash_hint(
        self,
        *,
        hash_value: str,
        preferred_algos: Sequence[str] | None = None,
        name_hint: str | None = None,
    ) -> dict[str, Any] | None:
        digest = _normalize_hash_text(hash_value)
        if digest is None:
            return None

        con = self._reader_connection()

        def _to_payload(row: sqlite3.Row) -> dict[str, Any]:
            return {
                "content_id": _as_int(row["content_id"]),
                "sha256": _normalize_hash_text(row["sha256"]),
                "civitai_file_hashes": self._get_civitai_file_hashes_by_file_id(_as_int(row["civitai_file_id"])),
                "civitai_model_name": _as_text(row["civitai_model_name"]),
                "civitai_model_version_model_id": _as_int(row["civitai_model_version_model_id"]),
                "civitai_model_version_model_version_id": _as_int(row["civitai_model_version_model_version_id"]),
                "civitai_model_version_name": _as_text(row["civitai_model_version_name"]),
                "civitai_model_version_base_model": _as_text(row["civitai_model_version_base_model"]),
                "civitai_model_version_raw_json": _as_text(row["civitai_model_version_raw_json"]),
                "civitai_file_name": _as_text(row["civitai_file_name"]),
                "civitai_file_download_url": _as_text(row["civitai_file_download_url"]),
                "civitai_version_download_url": _as_text(row["civitai_version_download_url"]),
                "runtime_settings": self.get_model_runtime_settings_by_content_id(_as_int(row["content_id"]) or 0),
            }

        def _basename(value: str | None) -> str:
            text = str(value or "").strip()
            if not text:
                return ""
            return text.replace("\\", "/").split("/")[-1]

        def _stem(value: str | None) -> str:
            basename = _basename(value)
            if not basename:
                return ""
            return Path(basename).stem

        def _select_candidate(rows: list[sqlite3.Row]) -> dict[str, Any] | None:
            if not rows:
                return None
            if len(rows) == 1:
                return _to_payload(rows[0])

            hint_basename = _basename(name_hint)
            if hint_basename:
                basename_matches = [
                    row for row in rows if _basename(_as_text(row["civitai_file_name"])) == hint_basename
                ]
                if len(basename_matches) == 1:
                    return _to_payload(basename_matches[0])

            hint_stem = _stem(name_hint)
            if hint_stem:
                stem_matches = [
                    row for row in rows if _stem(_as_text(row["civitai_file_name"])) == hint_stem
                ]
                if len(stem_matches) == 1:
                    return _to_payload(stem_matches[0])

            return None

        def _query_single(algo: str) -> dict[str, Any] | None:
            rows = con.execute(
                """
                SELECT
                    c.content_id AS content_id,
                    c.sha256 AS sha256,
                    cf.civitai_file_id AS civitai_file_id,
                    cm.name AS civitai_model_name,
                    cv.model_id AS civitai_model_version_model_id,
                    cv.model_version_id AS civitai_model_version_model_version_id,
                    cv.name AS civitai_model_version_name,
                    cv.base_model AS civitai_model_version_base_model,
                    cv.raw_json AS civitai_model_version_raw_json,
                    cf.name AS civitai_file_name,
                    cf.download_url AS civitai_file_download_url,
                    cv.version_download_url AS civitai_version_download_url
                FROM civitai_file_hash AS fh
                JOIN civitai_file AS cf ON cf.civitai_file_id = fh.civitai_file_id
                LEFT JOIN civitai_model_version AS cv ON cv.model_version_id = cf.model_version_id
                LEFT JOIN civitai_model AS cm ON cm.model_id = cv.model_id
                LEFT JOIN content_civitai_match AS ccm
                    ON ccm.civitai_file_id = cf.civitai_file_id
                   AND ccm.is_active = 1
                LEFT JOIN content AS c ON c.content_id = ccm.content_id
                WHERE fh.hash_algo = ?
                  AND LOWER(fh.hash_value) LIKE ?
                ORDER BY
                    COALESCE(cf.is_primary, 0) DESC,
                    cf.civitai_file_id DESC
                LIMIT 25
                """,
                (algo, f"{digest}%"),
            ).fetchall()
            return _select_candidate(list(rows))

        if preferred_algos:
            for raw_algo in preferred_algos:
                algo = str(raw_algo or "").strip().lower()
                if not algo:
                    continue
                found = _query_single(algo)
                if found is not None:
                    return found

        fallback_algos = ("autov1", "autov2", "autov3", "crc32", "blake3", "sha256")
        for algo in fallback_algos:
            found = _query_single(algo)
            if found is not None:
                return found
        return None

    def get_civitai_version_payload_by_relative_path(
        self,
        *,
        folder_name: str,
        relative_path: str,
    ) -> dict[str, Any] | None:
        folder = str(folder_name or "").strip()
        rel = normalize_relative_path(relative_path)
        if not folder or not rel:
            return None

        con = self._reader_connection()
        row = con.execute(
            """
            SELECT
                c.content_id AS content_id,
                c.sha256 AS sha256,
                cv.model_version_id AS civitai_model_version_model_version_id,
                cv.raw_json AS civitai_model_version_raw_json
            FROM local_asset_path AS p
            JOIN file_identity AS fi ON fi.file_id = p.file_id
            JOIN content AS c ON c.content_id = fi.content_id
            LEFT JOIN content_civitai_match AS ccm
                ON ccm.content_id = c.content_id
               AND ccm.is_active = 1
            LEFT JOIN civitai_file AS cf ON cf.civitai_file_id = ccm.civitai_file_id
            LEFT JOIN civitai_model_version AS cv ON cv.model_version_id = cf.model_version_id
            WHERE p.folder_name = ?
              AND p.relative_path = ?
              AND p.is_deleted = 0
            ORDER BY
                COALESCE(cf.is_primary, 0) DESC,
                ccm.last_confirmed_at DESC,
                ccm.civitai_file_id DESC
            LIMIT 1
            """,
            (folder, rel),
        ).fetchone()
        if row is None:
            return None

        content_id = _as_int(row["content_id"])
        if content_id is None:
            return None

        model_version_id = _as_int(row["civitai_model_version_model_version_id"])
        raw_json = _as_text(row["civitai_model_version_raw_json"])
        if model_version_id is None or raw_json is None:
            fallback_sha256 = _normalize_hash_text(row["sha256"])
            if fallback_sha256:
                fallback = self.get_model_reference_by_sha256(sha256=fallback_sha256)
                if isinstance(fallback, Mapping):
                    fallback_model_version_id = _as_int(
                        fallback.get("civitai_model_version_model_version_id")
                    )
                    fallback_raw_json = _as_text(fallback.get("civitai_model_version_raw_json"))
                    if fallback_model_version_id is not None or fallback_raw_json is not None:
                        return {
                            "content_id": content_id,
                            "civitai_model_version_model_version_id": fallback_model_version_id,
                            "civitai_model_version_raw_json": fallback_raw_json,
                        }

        return {
            "content_id": content_id,
            "civitai_model_version_model_version_id": model_version_id,
            "civitai_model_version_raw_json": raw_json,
        }

    def list_hash_hints_by_content_id(
        self,
        content_id: int,
        *,
        include_sha256: bool = False,
    ) -> list[dict[str, str]]:
        cid = _as_int(content_id)
        if cid is None:
            return []

        con = self._reader_connection()
        rows = con.execute(
            """
            SELECT hash_algo, hash_value
            FROM content_hash
            WHERE content_id = ?
            """,
            (cid,),
        ).fetchall()

        order = {
            "autov3": 0,
            "blake3": 1,
            "autov2": 2,
            "crc32": 3,
            "autov1": 4,
            "a1111_legacy": 5,
            "sha256": 6,
        }
        output: list[dict[str, str]] = []
        for row in rows:
            algo = _as_text(row["hash_algo"])
            digest = _normalize_hash_text(row["hash_value"])
            if not algo or not digest:
                continue
            if algo == "sha256" and not include_sha256:
                continue
            output.append({"algo": algo, "value": digest})

        output.sort(key=lambda item: (order.get(item["algo"], 99), item["algo"], item["value"]))
        return output

    def list_lora_tags_by_content_id(
        self,
        content_id: int,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        cid = _as_int(content_id)
        if cid is None:
            return []

        sql = """
        SELECT
            tag,
            SUM(frequency) AS total_frequency
        FROM lora_tag_frequency
        WHERE content_id = ?
        GROUP BY tag
        ORDER BY total_frequency DESC, tag COLLATE NOCASE ASC
        """
        params: tuple[Any, ...] = (cid,)
        if limit is not None:
            safe_limit = max(1, int(limit))
            sql = f"{sql}\nLIMIT ?"
            params = (cid, safe_limit)

        con = self._reader_connection()
        rows = con.execute(sql, params).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            tag = _as_text(row["tag"])
            freq = _as_int(row["total_frequency"])
            if not tag or freq is None:
                continue
            output.append({"tag": tag, "frequency": freq})
        return output

    def replace_model_runtime_settings(
        self,
        con: sqlite3.Connection,
        *,
        content_id: int,
        folder_name: str,
        settings: Mapping[str, Any] | None,
        updated_at: str,
    ) -> dict[str, int | float]:
        cid = _as_int(content_id)
        updated_at_text = str(updated_at or "").strip()
        if cid is None or not updated_at_text:
            return {}

        normalized_settings = filter_model_runtime_settings_for_folder(folder_name, settings)
        if normalized_settings:
            con.execute(
                """
                INSERT INTO model_runtime_settings (
                    content_id,
                    settings_json,
                    updated_at
                )
                VALUES (?, ?, ?)
                ON CONFLICT(content_id) DO UPDATE SET
                    settings_json = excluded.settings_json,
                    updated_at = excluded.updated_at
                """,
                (
                    cid,
                    json.dumps(normalized_settings, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                    updated_at_text,
                ),
            )
        else:
            con.execute("DELETE FROM model_runtime_settings WHERE content_id = ?", (cid,))

        return normalized_settings

    def upsert_civitai_lookup_state(
        self,
        con: sqlite3.Connection,
        *,
        sha256: str,
        status: str,
        checked_at: str,
        http_status: int | None = None,
        next_retry_at: str | None = None,
        last_error: str | None = None,
    ) -> None:
        digest = _normalize_hash_text(sha256)
        status_text = str(status or "").strip()
        checked_at_text = str(checked_at or "").strip()
        if not digest or not status_text or not checked_at_text:
            return

        if status_text not in _CIVITAI_LOOKUP_STATUSES:
            return

        fail_increment = 0 if status_text == "found" else 1
        fail_seed = 0 if status_text == "found" else 1

        con.execute(
            """
            INSERT INTO civitai_lookup_state (
                sha256,
                status,
                http_status,
                checked_at,
                next_retry_at,
                fail_count,
                last_error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sha256) DO UPDATE SET
                status = excluded.status,
                http_status = excluded.http_status,
                checked_at = excluded.checked_at,
                next_retry_at = excluded.next_retry_at,
                fail_count = CASE
                    WHEN excluded.status = 'found' THEN 0
                    ELSE civitai_lookup_state.fail_count + ?
                END,
                last_error = excluded.last_error
            """,
            (
                digest,
                status_text,
                _as_int(http_status),
                checked_at_text,
                _as_text(next_retry_at),
                fail_seed,
                _as_text(last_error),
                fail_increment,
            ),
        )

    def upsert_observed_path_and_identity(
        self,
        con: sqlite3.Connection,
        *,
        folder_name: str,
        relative_path: str,
        file_size: int,
        mtime_ns: int,
        seen_at: str,
        identity_kind: str | None = None,
        identity_key: str | None = None,
        volume_hint: str | None = None,
        ctime_ns: int | None = None,
    ) -> int | None:
        folder = str(folder_name or "").strip()
        rel = normalize_relative_path(relative_path)
        seen_at_text = str(seen_at or "").strip()
        if not folder or not rel or not seen_at_text:
            return None

        basename = Path(rel).name
        stem = Path(rel).stem

        file_id: int | None = None
        content_id: int | None = None

        kind = _normalize_identity_kind(identity_kind)
        key = str(identity_key or "").strip()

        if kind and key:
            con.execute(
                """
                INSERT INTO file_identity (
                    identity_kind,
                    identity_key,
                    volume_hint,
                    last_file_size,
                    last_mtime_ns,
                    last_ctime_ns,
                    first_seen_at,
                    last_seen_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(identity_kind, identity_key) DO UPDATE SET
                    volume_hint = COALESCE(excluded.volume_hint, file_identity.volume_hint),
                    last_file_size = excluded.last_file_size,
                    last_mtime_ns = excluded.last_mtime_ns,
                    last_ctime_ns = COALESCE(excluded.last_ctime_ns, file_identity.last_ctime_ns),
                    last_seen_at = excluded.last_seen_at
                """,
                (
                    kind,
                    key,
                    _as_text(volume_hint),
                    int(file_size),
                    int(mtime_ns),
                    _as_int(ctime_ns),
                    seen_at_text,
                    seen_at_text,
                ),
            )
            row = con.execute(
                """
                SELECT file_id, content_id
                FROM file_identity
                WHERE identity_kind = ?
                  AND identity_key = ?
                LIMIT 1
                """,
                (kind, key),
            ).fetchone()
            if row is not None:
                file_id = int(row[0])
                if row[1] is not None:
                    content_id = int(row[1])

        con.execute(
            """
            INSERT INTO local_asset_path (
                folder_name,
                relative_path,
                basename,
                stem,
                file_id,
                is_deleted,
                first_seen_at,
                last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            ON CONFLICT(folder_name, relative_path) DO UPDATE SET
                basename = excluded.basename,
                stem = excluded.stem,
                file_id = COALESCE(excluded.file_id, local_asset_path.file_id),
                is_deleted = 0,
                last_seen_at = excluded.last_seen_at
            """,
            (
                folder,
                rel,
                basename,
                stem,
                _as_int(file_id),
                seen_at_text,
                seen_at_text,
            ),
        )

        if content_id is not None:
            return content_id

        row = con.execute(
            """
            SELECT fi.content_id
            FROM local_asset_path AS p
            JOIN file_identity AS fi ON fi.file_id = p.file_id
            WHERE p.folder_name = ?
              AND p.relative_path = ?
              AND p.is_deleted = 0
              AND fi.content_id IS NOT NULL
              AND fi.last_file_size = ?
              AND fi.last_mtime_ns = ?
            LIMIT 1
            """,
            (folder, rel, int(file_size), int(mtime_ns)),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    def mark_missing_paths_deleted(
        self,
        con: sqlite3.Connection,
        *,
        folder_names: Sequence[str],
        scan_started_at: str,
        marked_at: str,
    ) -> int:
        folders = tuple(
            sorted({str(folder or "").strip() for folder in folder_names if str(folder or "").strip()})
        )
        scan_started_at_text = str(scan_started_at or "").strip()
        marked_at_text = str(marked_at or "").strip()
        if not folders or not scan_started_at_text or not marked_at_text:
            return 0

        placeholders = ", ".join("?" for _ in folders)
        sql = f"""
        UPDATE local_asset_path
        SET is_deleted = 1,
            last_seen_at = ?
        WHERE folder_name IN ({placeholders})
          AND is_deleted = 0
          AND last_seen_at < ?
        """
        params: tuple[Any, ...] = (marked_at_text, *folders, scan_started_at_text)
        cur = con.execute(sql, params)
        return int(cur.rowcount)

    def get_startup_gc_candidate_counts(self, *, older_than_iso: str) -> tuple[int, int]:
        cutoff = str(older_than_iso or "").strip()
        if not cutoff:
            return 0, 0

        con = self._reader_connection()
        deleted_path_row = con.execute(
            """
            SELECT COUNT(*)
            FROM local_asset_path
            WHERE is_deleted = 1
              AND last_seen_at <= ?
            """,
            (cutoff,),
        ).fetchone()
        deleted_path_count = int(deleted_path_row[0]) if deleted_path_row else 0

        orphan_identity_row = con.execute(
            """
            SELECT COUNT(*)
            FROM file_identity AS fi
            WHERE fi.last_seen_at <= ?
              AND NOT EXISTS (
                    SELECT 1
                    FROM local_asset_path AS p
                    WHERE p.file_id = fi.file_id
              )
            """,
            (cutoff,),
        ).fetchone()
        orphan_identity_count = int(orphan_identity_row[0]) if orphan_identity_row else 0
        return deleted_path_count, orphan_identity_count

    def gc_stale_local_tracking(
        self,
        con: sqlite3.Connection,
        *,
        older_than_iso: str,
        max_deleted_paths: int,
        max_orphan_identities: int,
    ) -> tuple[int, int]:
        cutoff = str(older_than_iso or "").strip()
        if not cutoff:
            return 0, 0

        max_paths = max(1, int(max_deleted_paths))
        max_identities = max(1, int(max_orphan_identities))

        deleted_path_cur = con.execute(
            """
            DELETE FROM local_asset_path
            WHERE asset_path_id IN (
                SELECT asset_path_id
                FROM local_asset_path
                WHERE is_deleted = 1
                  AND last_seen_at <= ?
                ORDER BY last_seen_at ASC, asset_path_id ASC
                LIMIT ?
            )
            """,
            (cutoff, max_paths),
        )
        deleted_path_count = _safe_rowcount(deleted_path_cur)

        orphan_identity_cur = con.execute(
            """
            DELETE FROM file_identity
            WHERE file_id IN (
                SELECT fi.file_id
                FROM file_identity AS fi
                WHERE fi.last_seen_at <= ?
                  AND NOT EXISTS (
                        SELECT 1
                        FROM local_asset_path AS p
                        WHERE p.file_id = fi.file_id
                  )
                ORDER BY fi.last_seen_at ASC, fi.file_id ASC
                LIMIT ?
            )
            """,
            (cutoff, max_identities),
        )
        orphan_identity_count = _safe_rowcount(orphan_identity_cur)
        return deleted_path_count, orphan_identity_count

    def run_startup_gc_if_needed(
        self,
        con: sqlite3.Connection,
        *,
        older_than_iso: str,
        min_deleted_paths: int,
        min_orphan_identities: int,
        max_deleted_paths: int,
        max_orphan_identities: int,
    ) -> tuple[int, int, int, int]:
        cutoff = str(older_than_iso or "").strip()
        if not cutoff:
            return 0, 0, 0, 0

        deleted_path_row = con.execute(
            """
            SELECT COUNT(*)
            FROM local_asset_path
            WHERE is_deleted = 1
              AND last_seen_at <= ?
            """,
            (cutoff,),
        ).fetchone()
        deleted_path_candidates = int(deleted_path_row[0]) if deleted_path_row else 0

        orphan_identity_row = con.execute(
            """
            SELECT COUNT(*)
            FROM file_identity AS fi
            WHERE fi.last_seen_at <= ?
              AND NOT EXISTS (
                    SELECT 1
                    FROM local_asset_path AS p
                    WHERE p.file_id = fi.file_id
              )
            """,
            (cutoff,),
        ).fetchone()
        orphan_identity_candidates = int(orphan_identity_row[0]) if orphan_identity_row else 0

        if (
            deleted_path_candidates < max(1, int(min_deleted_paths))
            and orphan_identity_candidates < max(1, int(min_orphan_identities))
        ):
            return deleted_path_candidates, orphan_identity_candidates, 0, 0

        deleted_path_count, orphan_identity_count = self.gc_stale_local_tracking(
            con,
            older_than_iso=cutoff,
            max_deleted_paths=max_deleted_paths,
            max_orphan_identities=max_orphan_identities,
        )
        return (
            deleted_path_candidates,
            orphan_identity_candidates,
            deleted_path_count,
            orphan_identity_count,
        )

    def upsert_local_asset_and_sha256(
        self,
        con: sqlite3.Connection,
        *,
        folder_name: str,
        relative_path: str,
        file_size: int,
        mtime_ns: int,
        sha256: str,
        seen_at: str,
        identity_kind: str | None = None,
        identity_key: str | None = None,
        volume_hint: str | None = None,
        ctime_ns: int | None = None,
    ) -> int | None:
        folder = str(folder_name or "").strip()
        rel = normalize_relative_path(relative_path)
        digest = _normalize_hash_text(sha256)
        seen_at_text = str(seen_at or "").strip()
        if not folder or not rel or not digest or not seen_at_text:
            return None

        basename = Path(rel).name
        stem = Path(rel).stem

        con.execute(
            """
            INSERT INTO content (
                sha256,
                first_seen_at,
                last_seen_at
            )
            VALUES (?, ?, ?)
            ON CONFLICT(sha256) DO UPDATE SET
                last_seen_at = excluded.last_seen_at
            """,
            (digest, seen_at_text, seen_at_text),
        )

        row = con.execute(
            "SELECT content_id FROM content WHERE sha256 = ? LIMIT 1",
            (digest,),
        ).fetchone()
        if row is None:
            return None
        content_id = int(row[0])

        file_id: int | None = None
        kind = _normalize_identity_kind(identity_kind)
        key = str(identity_key or "").strip()

        if kind and key:
            con.execute(
                """
                INSERT INTO file_identity (
                    identity_kind,
                    identity_key,
                    volume_hint,
                    last_file_size,
                    last_mtime_ns,
                    last_ctime_ns,
                    content_id,
                    content_link_kind,
                    content_linked_at,
                    first_seen_at,
                    last_seen_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'sha256', ?, ?, ?)
                ON CONFLICT(identity_kind, identity_key) DO UPDATE SET
                    volume_hint = COALESCE(excluded.volume_hint, file_identity.volume_hint),
                    last_file_size = excluded.last_file_size,
                    last_mtime_ns = excluded.last_mtime_ns,
                    last_ctime_ns = COALESCE(excluded.last_ctime_ns, file_identity.last_ctime_ns),
                    content_id = excluded.content_id,
                    content_link_kind = excluded.content_link_kind,
                    content_linked_at = excluded.content_linked_at,
                    last_seen_at = excluded.last_seen_at
                """,
                (
                    kind,
                    key,
                    _as_text(volume_hint),
                    int(file_size),
                    int(mtime_ns),
                    _as_int(ctime_ns),
                    content_id,
                    seen_at_text,
                    seen_at_text,
                    seen_at_text,
                ),
            )
            row = con.execute(
                """
                SELECT file_id
                FROM file_identity
                WHERE identity_kind = ?
                  AND identity_key = ?
                LIMIT 1
                """,
                (kind, key),
            ).fetchone()
            if row is not None:
                file_id = int(row[0])
        else:
            row = con.execute(
                """
                SELECT file_id
                FROM local_asset_path
                WHERE folder_name = ?
                  AND relative_path = ?
                LIMIT 1
                """,
                (folder, rel),
            ).fetchone()
            if row is not None and row[0] is not None:
                file_id = int(row[0])

            if file_id is not None:
                con.execute(
                    """
                    UPDATE file_identity
                    SET last_file_size = ?,
                        last_mtime_ns = ?,
                        last_ctime_ns = COALESCE(?, last_ctime_ns),
                        content_id = ?,
                        content_link_kind = 'sha256',
                        content_linked_at = ?,
                        last_seen_at = ?
                    WHERE file_id = ?
                    """,
                    (
                        int(file_size),
                        int(mtime_ns),
                        _as_int(ctime_ns),
                        content_id,
                        seen_at_text,
                        seen_at_text,
                        file_id,
                    ),
                )

        con.execute(
            """
            INSERT INTO local_asset_path (
                folder_name,
                relative_path,
                basename,
                stem,
                file_id,
                is_deleted,
                first_seen_at,
                last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            ON CONFLICT(folder_name, relative_path) DO UPDATE SET
                basename = excluded.basename,
                stem = excluded.stem,
                file_id = COALESCE(excluded.file_id, local_asset_path.file_id),
                is_deleted = 0,
                last_seen_at = excluded.last_seen_at
            """,
            (
                folder,
                rel,
                basename,
                stem,
                _as_int(file_id),
                seen_at_text,
                seen_at_text,
            ),
        )

        return content_id

    def upsert_lora_metadata(
        self,
        con: sqlite3.Connection,
        *,
        folder_name: str,
        relative_path: str,
        metadata: Mapping[str, Any],
        parsed_at: str,
    ) -> None:
        folder = str(folder_name or "").strip()
        rel = normalize_relative_path(relative_path)
        if folder != "loras" or not rel:
            return

        row = con.execute(
            """
            SELECT fi.content_id
            FROM local_asset_path AS p
            JOIN file_identity AS fi ON fi.file_id = p.file_id
            WHERE p.folder_name = ?
              AND p.relative_path = ?
              AND p.is_deleted = 0
              AND fi.content_id IS NOT NULL
            LIMIT 1
            """,
            (folder, rel),
        ).fetchone()
        if row is None or row[0] is None:
            return
        content_id = int(row[0])

        raw_json = json.dumps(dict(metadata), ensure_ascii=False, separators=(",", ":"), sort_keys=True)

        con.execute(
            """
            INSERT INTO lora_metadata (
                content_id,
                ss_output_name,
                ss_sd_model_name,
                ss_clip_skip,
                ss_resolution,
                ss_bucket_info_json,
                ss_tag_frequency_json,
                modelspec_trigger_phrase,
                modelspec_description,
                modelspec_tags,
                modelspec_usage_hint,
                raw_metadata_json,
                parsed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(content_id) DO UPDATE SET
                ss_output_name = excluded.ss_output_name,
                ss_sd_model_name = excluded.ss_sd_model_name,
                ss_clip_skip = excluded.ss_clip_skip,
                ss_resolution = excluded.ss_resolution,
                ss_bucket_info_json = excluded.ss_bucket_info_json,
                ss_tag_frequency_json = excluded.ss_tag_frequency_json,
                modelspec_trigger_phrase = excluded.modelspec_trigger_phrase,
                modelspec_description = excluded.modelspec_description,
                modelspec_tags = excluded.modelspec_tags,
                modelspec_usage_hint = excluded.modelspec_usage_hint,
                raw_metadata_json = excluded.raw_metadata_json,
                parsed_at = excluded.parsed_at
            """,
            (
                content_id,
                _as_text(metadata.get("ss_output_name")),
                _as_text(metadata.get("ss_sd_model_name")),
                _as_text(metadata.get("ss_clip_skip")),
                _as_text(metadata.get("ss_resolution")),
                _as_json_text(metadata.get("ss_bucket_info")),
                _as_json_text(metadata.get("ss_tag_frequency")),
                _as_text(metadata.get("modelspec.trigger_phrase")),
                _as_text(metadata.get("modelspec.description")),
                _as_text(metadata.get("modelspec.tags")),
                _as_text(metadata.get("modelspec.usage_hint")),
                raw_json,
                parsed_at,
            ),
        )

        con.execute("DELETE FROM lora_tag_frequency WHERE content_id = ?", (content_id,))
        tag_freq = _parse_tag_frequency(metadata.get("ss_tag_frequency"))
        for dataset_name, tags in tag_freq.items():
            if not isinstance(tags, Mapping):
                continue
            for tag, raw_count in tags.items():
                tag_text = str(tag or "").strip()
                if not tag_text:
                    continue
                try:
                    count = int(raw_count)
                except Exception:
                    continue
                if count < 0:
                    continue
                con.execute(
                    """
                    INSERT INTO lora_tag_frequency (content_id, dataset_name, tag, frequency)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(content_id, dataset_name, tag) DO UPDATE SET
                        frequency = excluded.frequency
                    """,
                    (content_id, dataset_name, tag_text, count),
                )
    def upsert_civitai_payload(
        self,
        con: sqlite3.Connection,
        *,
        requested_sha256: str,
        payload: Mapping[str, Any],
        fetched_at: str,
    ) -> None:
        requested_hash = _normalize_hash_text(requested_sha256)
        if not requested_hash:
            return

        model_id = _as_int(payload.get("modelId"))
        version_id = _as_int(payload.get("id"))
        model = payload.get("model") if isinstance(payload.get("model"), Mapping) else {}
        if model_id is None or version_id is None:
            return

        con.execute(
            """
            INSERT INTO civitai_model (
                model_id,
                name,
                type,
                nsfw,
                poi,
                first_seen_at,
                last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                name = excluded.name,
                type = excluded.type,
                nsfw = excluded.nsfw,
                poi = excluded.poi,
                last_seen_at = excluded.last_seen_at
            """,
            (
                model_id,
                _as_text(model.get("name")) or "",
                _as_text(model.get("type")) or "",
                _as_bool_int(model.get("nsfw")),
                _as_bool_int(model.get("poi")),
                fetched_at,
                fetched_at,
            ),
        )

        trained_words = payload.get("trainedWords")
        trained_words_json = None
        if isinstance(trained_words, list):
            trained_words_json = json.dumps(trained_words, ensure_ascii=False, separators=(",", ":"))

        con.execute(
            """
            INSERT INTO civitai_model_version (
                model_version_id,
                model_id,
                name,
                air,
                base_model,
                base_model_type,
                status,
                published_at,
                created_at,
                updated_at,
                early_access_ends_at,
                usage_control,
                upload_type,
                description_html,
                trained_words_json,
                version_download_url,
                raw_json,
                fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_version_id) DO UPDATE SET
                model_id = excluded.model_id,
                name = excluded.name,
                air = excluded.air,
                base_model = excluded.base_model,
                base_model_type = excluded.base_model_type,
                status = excluded.status,
                published_at = excluded.published_at,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                early_access_ends_at = excluded.early_access_ends_at,
                usage_control = excluded.usage_control,
                upload_type = excluded.upload_type,
                description_html = excluded.description_html,
                trained_words_json = excluded.trained_words_json,
                version_download_url = excluded.version_download_url,
                raw_json = excluded.raw_json,
                fetched_at = excluded.fetched_at
            """,
            (
                version_id,
                model_id,
                _as_text(payload.get("name")),
                _as_text(payload.get("air")),
                _as_text(payload.get("baseModel")),
                _as_text(payload.get("baseModelType")),
                _as_text(payload.get("status")),
                _as_text(payload.get("publishedAt")),
                _as_text(payload.get("createdAt")),
                _as_text(payload.get("updatedAt")),
                _as_text(payload.get("earlyAccessEndsAt")),
                _as_text(payload.get("usageControl")),
                _as_text(payload.get("uploadType")),
                _as_text(payload.get("description")),
                trained_words_json,
                _as_text(payload.get("downloadUrl")),
                json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                fetched_at,
            ),
        )

        files = payload.get("files") if isinstance(payload.get("files"), list) else []
        matched_file_id: int | None = None

        for raw_file in files:
            if not isinstance(raw_file, Mapping):
                continue
            civitai_file_id = _as_int(raw_file.get("id"))
            if civitai_file_id is None:
                continue

            file_json = json.dumps(dict(raw_file), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            file_metadata = raw_file.get("metadata") if isinstance(raw_file.get("metadata"), Mapping) else {}

            con.execute(
                """
                INSERT INTO civitai_file (
                    civitai_file_id,
                    model_version_id,
                    name,
                    type,
                    size_kb,
                    is_primary,
                    download_url,
                    pickle_scan_result,
                    pickle_scan_message,
                    virus_scan_result,
                    virus_scan_message,
                    scanned_at,
                    metadata_format,
                    metadata_size,
                    metadata_fp,
                    raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(civitai_file_id) DO UPDATE SET
                    model_version_id = excluded.model_version_id,
                    name = excluded.name,
                    type = excluded.type,
                    size_kb = excluded.size_kb,
                    is_primary = excluded.is_primary,
                    download_url = excluded.download_url,
                    pickle_scan_result = excluded.pickle_scan_result,
                    pickle_scan_message = excluded.pickle_scan_message,
                    virus_scan_result = excluded.virus_scan_result,
                    virus_scan_message = excluded.virus_scan_message,
                    scanned_at = excluded.scanned_at,
                    metadata_format = excluded.metadata_format,
                    metadata_size = excluded.metadata_size,
                    metadata_fp = excluded.metadata_fp,
                    raw_json = excluded.raw_json
                """,
                (
                    civitai_file_id,
                    version_id,
                    _as_text(raw_file.get("name")) or "",
                    _as_text(raw_file.get("type")),
                    _as_float(raw_file.get("sizeKB")),
                    _as_bool_int(raw_file.get("primary"), default=0),
                    _as_text(raw_file.get("downloadUrl")),
                    _as_text(raw_file.get("pickleScanResult")),
                    _as_text(raw_file.get("pickleScanMessage")),
                    _as_text(raw_file.get("virusScanResult")),
                    _as_text(raw_file.get("virusScanMessage")),
                    _as_text(raw_file.get("scannedAt")),
                    _as_text(file_metadata.get("format")),
                    _as_text(file_metadata.get("size")),
                    _as_text(file_metadata.get("fp")),
                    file_json,
                ),
            )

            hashes = raw_file.get("hashes") if isinstance(raw_file.get("hashes"), Mapping) else {}
            for hash_key, hash_value in hashes.items():
                algo = _HASH_KEY_TO_ALGO.get(str(hash_key))
                digest = _normalize_hash_text(hash_value)
                if not algo or not digest:
                    continue
                con.execute(
                    """
                    INSERT INTO civitai_file_hash (civitai_file_id, hash_algo, hash_value)
                    VALUES (?, ?, ?)
                    ON CONFLICT(civitai_file_id, hash_algo) DO UPDATE SET
                        hash_value = excluded.hash_value
                    """,
                    (civitai_file_id, algo, digest),
                )
                if algo == "sha256" and digest == requested_hash:
                    matched_file_id = civitai_file_id

        if matched_file_id is None:
            return

        row = con.execute(
            "SELECT content_id FROM content WHERE sha256 = ? LIMIT 1",
            (requested_hash,),
        ).fetchone()
        if row is None:
            return
        content_id = int(row[0])

        con.execute(
            """
            INSERT INTO content_civitai_match (
                content_id,
                civitai_file_id,
                match_algo,
                match_hash_value,
                is_active,
                matched_at,
                last_confirmed_at
            )
            VALUES (?, ?, 'sha256', ?, 1, ?, ?)
            ON CONFLICT(content_id, civitai_file_id) DO UPDATE SET
                match_algo = excluded.match_algo,
                match_hash_value = excluded.match_hash_value,
                is_active = excluded.is_active,
                last_confirmed_at = excluded.last_confirmed_at
            """,
            (content_id, matched_file_id, requested_hash, fetched_at, fetched_at),
        )

        hash_rows = con.execute(
            "SELECT hash_algo, hash_value FROM civitai_file_hash WHERE civitai_file_id = ?",
            (matched_file_id,),
        ).fetchall()

        for hash_row in hash_rows:
            algo = _as_text(hash_row[0])
            digest = _normalize_hash_text(hash_row[1])
            if not algo or not digest or algo == "sha256":
                continue
            con.execute(
                """
                INSERT INTO content_hash (
                    content_id,
                    hash_algo,
                    hash_value,
                    source,
                    computed_at,
                    verified_at
                )
                VALUES (?, ?, ?, 'civitai_api', ?, NULL)
                ON CONFLICT(content_id, hash_algo) DO UPDATE SET
                    hash_value = excluded.hash_value,
                    source = excluded.source,
                    computed_at = excluded.computed_at,
                    verified_at = excluded.verified_at
                """,
                (content_id, algo, digest, fetched_at),
            )
