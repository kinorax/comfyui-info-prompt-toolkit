from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Mapping

DEFAULT_MAX_ENTRIES = 5000
DEFAULT_CLEANUP_INTERVAL = 64
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
_A1111_LEGACY_OFFSET = 0x100000
_A1111_LEGACY_LENGTH = 0x10000
SUPPORTED_HASH_ALGORITHMS = {
    "sha256",
    "sha1",
    "md5",
    "crc32",
    "a1111_legacy",
}

HashComputer = Callable[[str, str], str | None]


def normalize_relative_path(relative_path: str) -> str:
    text = str(relative_path or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def build_cache_key(
    relative_path: str,
    file_size: int,
    mtime: int,
    hash_algo: str,
) -> str:
    key_data = {
        "hash_algo": str(hash_algo).strip().lower(),
        "mtime": int(mtime),
        "relative_path": normalize_relative_path(relative_path),
        "size": int(file_size),
    }
    return json.dumps(key_data, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def compute_file_hash(absolute_path: str, hash_algo: str = "sha256") -> str | None:
    algo = str(hash_algo or "").strip().lower()
    if algo not in SUPPORTED_HASH_ALGORITHMS:
        return None

    if algo == "a1111_legacy":
        with open(absolute_path, "rb") as f:
            f.seek(_A1111_LEGACY_OFFSET)
            payload = f.read(_A1111_LEGACY_LENGTH)
        return hashlib.sha256(payload).hexdigest()[:8]

    if algo == "crc32":
        checksum = 0
        with open(absolute_path, "rb") as f:
            while True:
                chunk = f.read(DEFAULT_CHUNK_SIZE)
                if not chunk:
                    break
                checksum = zlib.crc32(chunk, checksum)
        return f"{checksum & 0xFFFFFFFF:08x}"

    try:
        hasher = hashlib.new(algo)
    except ValueError:
        return None

    with open(absolute_path, "rb") as f:
        while True:
            chunk = f.read(DEFAULT_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_file_stat(absolute_path: str) -> tuple[int, int] | None:
    try:
        stat_result = os.stat(absolute_path)
    except OSError:
        return None
    size = int(stat_result.st_size)
    mtime_ns = int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)))
    return size, mtime_ns


def _is_hex_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    for ch in value:
        if ch not in "0123456789abcdef":
            return False
    return True


class PersistentFileHashCache:
    def __init__(
        self,
        cache_file_path: str | os.PathLike[str],
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        cleanup_interval: int = DEFAULT_CLEANUP_INTERVAL,
        hash_computer: HashComputer | None = None,
    ) -> None:
        self._cache_file_path = Path(cache_file_path)
        self._max_entries = max(1, int(max_entries))
        self._cleanup_interval = max(1, int(cleanup_interval))
        self._hash_computer = hash_computer or compute_file_hash
        self._lock = threading.Lock()
        self._loaded = False
        self._write_counter = 0
        self._entries: dict[str, dict[str, Any]] = {}

    def get_or_compute(
        self,
        *,
        absolute_path: str,
        relative_path: str,
        hash_algo: str = "sha256",
    ) -> str | None:
        abs_path = os.path.abspath(absolute_path)
        rel_path = normalize_relative_path(relative_path)
        algo = str(hash_algo or "").strip().lower()
        if not abs_path or not rel_path or not algo:
            return None

        stat_info = _safe_file_stat(abs_path)
        if stat_info is None:
            return None
        file_size, mtime_ns = stat_info
        key = build_cache_key(rel_path, file_size, mtime_ns, algo)

        with self._lock:
            self._load_locked()

            cached = self._entries.get(key)
            if self._is_valid_cached_value(cached, abs_path, algo):
                return str(cached["hash"])

            digest = self._hash_computer(abs_path, algo)
            if not isinstance(digest, str):
                return None
            digest = digest.strip().lower()
            if not digest:
                return None
            if algo == "sha256" and not _is_hex_sha256(digest):
                return None

            self._entries[key] = {
                "hash": digest,
                "hash_algo": algo,
                "mtime": mtime_ns,
                "path": abs_path,
                "relative_path": rel_path,
                "size": file_size,
                "updated_at": float(time.time()),
            }

            self._write_counter += 1
            full_cleanup = (self._write_counter % self._cleanup_interval) == 0
            self._prune_locked(force_full=full_cleanup)
            self._save_locked()
            return digest

    def force_cleanup(self) -> None:
        with self._lock:
            self._load_locked()
            changed = self._prune_locked(force_full=True)
            if changed:
                self._save_locked()

    def entry_count(self) -> int:
        with self._lock:
            self._load_locked()
            return len(self._entries)

    def _is_valid_cached_value(
        self,
        entry: Mapping[str, Any] | None,
        absolute_path: str,
        hash_algo: str,
    ) -> bool:
        if not isinstance(entry, Mapping):
            return False

        digest = entry.get("hash")
        if not isinstance(digest, str) or not digest:
            return False

        stored_path = entry.get("path")
        if not isinstance(stored_path, str) or os.path.abspath(stored_path) != absolute_path:
            return False

        stored_algo = entry.get("hash_algo")
        if not isinstance(stored_algo, str) or stored_algo.strip().lower() != hash_algo:
            return False

        if hash_algo == "sha256" and not _is_hex_sha256(digest):
            return False
        return True

    def _load_locked(self) -> None:
        if self._loaded:
            return

        loaded_entries: dict[str, dict[str, Any]] = {}
        if self._cache_file_path.is_file():
            try:
                with self._cache_file_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}

            entries_raw = payload.get("entries") if isinstance(payload, Mapping) else None
            if isinstance(entries_raw, Mapping):
                for raw_key, raw_value in entries_raw.items():
                    if not isinstance(raw_key, str) or not isinstance(raw_value, Mapping):
                        continue
                    loaded_entries[raw_key] = dict(raw_value)

        self._entries = loaded_entries
        self._loaded = True

        changed = self._prune_locked(force_full=True)
        if changed:
            self._save_locked()

    def _prune_locked(self, *, force_full: bool) -> bool:
        changed = False

        if force_full:
            remove_keys = [
                key
                for key, entry in self._entries.items()
                if not self._entry_points_existing_file(entry)
            ]
            if remove_keys:
                changed = True
                for key in remove_keys:
                    self._entries.pop(key, None)

        if len(self._entries) > self._max_entries:
            overflow = len(self._entries) - self._max_entries
            sorted_keys = sorted(
                self._entries.keys(),
                key=lambda k: float(self._entries.get(k, {}).get("updated_at", 0.0)),
            )
            for key in sorted_keys[:overflow]:
                self._entries.pop(key, None)
            changed = True

        return changed

    def _entry_points_existing_file(self, entry: Mapping[str, Any]) -> bool:
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value:
            return False
        return os.path.isfile(path_value)

    def _save_locked(self) -> None:
        self._cache_file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "entries": self._entries,
            "version": 1,
        }
        temp_path = self._cache_file_path.with_suffix(self._cache_file_path.suffix + ".tmp")

        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            os.replace(temp_path, self._cache_file_path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
