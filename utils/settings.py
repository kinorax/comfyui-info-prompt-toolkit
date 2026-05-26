# Copyright 2026 kinorax
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any

try:
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover - unavailable outside ComfyUI runtime
    folder_paths = None  # type: ignore[assignment]


SETTINGS_DIR_NAME = "info_prompt_toolkit"
SETTINGS_FILE_NAME = "settings.json"
SETTINGS_VERSION = 1

USE_LOADED_MODEL_CACHE_SECTION = "use_loaded_model_cache"

CACHE_RETENTION_AUTO = "auto"
CACHE_RETENTION_FIXED = "fixed"
CACHE_RETENTION_OPTIONS = (CACHE_RETENTION_AUTO, CACHE_RETENTION_FIXED)

DEFAULT_CACHE_RETENTION = CACHE_RETENTION_AUTO
DEFAULT_MAX_CACHE_ENTRIES = 1
MIN_CACHE_ENTRIES = 1
MAX_CACHE_ENTRIES = 9
DEFAULT_MEMORY_BUDGET_RATIO = 0.72
MIN_MEMORY_BUDGET_RATIO = 0.01
MAX_MEMORY_BUDGET_RATIO = 1.0
DEFAULT_CACHE_LOG_ENABLED = True

_SETTINGS_LOCK = threading.Lock()
_USE_LOADED_MODEL_CACHE_SETTINGS: UseLoadedModelCacheSettings | None = None


@dataclass(frozen=True)
class UseLoadedModelCacheSettings:
    cache_retention: str = DEFAULT_CACHE_RETENTION
    max_cache_entries: int = DEFAULT_MAX_CACHE_ENTRIES
    memory_budget_ratio: float = DEFAULT_MEMORY_BUDGET_RATIO
    cache_log_enabled: bool = DEFAULT_CACHE_LOG_ENABLED

    def as_dict(self) -> dict[str, Any]:
        return {
            "cache_retention": self.cache_retention,
            "max_cache_entries": self.max_cache_entries,
            "memory_budget_ratio": self.memory_budget_ratio,
            "cache_log_enabled": self.cache_log_enabled,
        }


def resolve_settings_path() -> Path:
    if folder_paths is not None:
        getter = getattr(folder_paths, "get_user_directory", None)
        if callable(getter):
            try:
                user_dir = getter()
            except Exception:
                user_dir = None
            if isinstance(user_dir, str) and user_dir:
                return Path(user_dir) / SETTINGS_DIR_NAME / SETTINGS_FILE_NAME

    return Path(__file__).resolve().parents[1] / ".cache" / SETTINGS_DIR_NAME / SETTINGS_FILE_NAME


def get_use_loaded_model_cache_settings() -> UseLoadedModelCacheSettings:
    with _SETTINGS_LOCK:
        return _get_use_loaded_model_cache_settings_unlocked()


def update_use_loaded_model_cache_settings(payload: Mapping[str, Any] | None) -> UseLoadedModelCacheSettings:
    with _SETTINGS_LOCK:
        current = _get_use_loaded_model_cache_settings_unlocked()
        merged = current.as_dict()
        if isinstance(payload, Mapping):
            merged.update(dict(payload))
        settings = normalize_use_loaded_model_cache_settings(merged)
        _set_use_loaded_model_cache_settings_unlocked(settings)
        return settings


def get_settings_payload() -> dict[str, Any]:
    settings = get_use_loaded_model_cache_settings()
    return {
        "version": SETTINGS_VERSION,
        USE_LOADED_MODEL_CACHE_SECTION: settings.as_dict(),
    }


def normalize_use_loaded_model_cache_settings(value: Mapping[str, Any] | None) -> UseLoadedModelCacheSettings:
    payload = value if isinstance(value, Mapping) else {}

    cache_retention = _normalized_option(
        payload.get("cache_retention"),
        CACHE_RETENTION_OPTIONS,
        DEFAULT_CACHE_RETENTION,
    )
    max_cache_entries = _clamp_int(
        payload.get("max_cache_entries"),
        DEFAULT_MAX_CACHE_ENTRIES,
        MIN_CACHE_ENTRIES,
        MAX_CACHE_ENTRIES,
    )
    memory_budget_ratio = _clamp_float(
        payload.get("memory_budget_ratio"),
        DEFAULT_MEMORY_BUDGET_RATIO,
        MIN_MEMORY_BUDGET_RATIO,
        MAX_MEMORY_BUDGET_RATIO,
    )
    cache_log_enabled = _normalized_log_enabled(payload)

    return UseLoadedModelCacheSettings(
        cache_retention=cache_retention,
        max_cache_entries=max_cache_entries,
        memory_budget_ratio=memory_budget_ratio,
        cache_log_enabled=cache_log_enabled,
    )


def _get_use_loaded_model_cache_settings_unlocked() -> UseLoadedModelCacheSettings:
    global _USE_LOADED_MODEL_CACHE_SETTINGS

    if _USE_LOADED_MODEL_CACHE_SETTINGS is None:
        _USE_LOADED_MODEL_CACHE_SETTINGS = _load_use_loaded_model_cache_settings()
    return _USE_LOADED_MODEL_CACHE_SETTINGS


def _set_use_loaded_model_cache_settings_unlocked(settings: UseLoadedModelCacheSettings) -> None:
    global _USE_LOADED_MODEL_CACHE_SETTINGS

    _USE_LOADED_MODEL_CACHE_SETTINGS = settings
    _write_settings(settings)


def _load_use_loaded_model_cache_settings() -> UseLoadedModelCacheSettings:
    path = resolve_settings_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raw = {}

    if not isinstance(raw, Mapping):
        raw = {}

    section = raw.get(USE_LOADED_MODEL_CACHE_SECTION)
    if isinstance(section, Mapping):
        return normalize_use_loaded_model_cache_settings(section)

    return normalize_use_loaded_model_cache_settings(raw)


def _write_settings(settings: UseLoadedModelCacheSettings) -> None:
    payload = {
        "version": SETTINGS_VERSION,
        USE_LOADED_MODEL_CACHE_SECTION: settings.as_dict(),
    }
    path = resolve_settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def _normalized_option(value: Any, options: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in options else default


def _normalized_log_enabled(payload: Mapping[str, Any]) -> bool:
    if "cache_log_enabled" in payload:
        return _bool_or_default(payload.get("cache_log_enabled"), DEFAULT_CACHE_LOG_ENABLED)

    legacy_level = str(payload.get("cache_log_level") or "").strip().lower()
    if legacy_level:
        return legacy_level != "off"

    return DEFAULT_CACHE_LOG_ENABLED


def _bool_or_default(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except Exception:
        number = default
    return max(minimum, min(maximum, number))


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = default
    if not number == number:
        number = default
    return max(minimum, min(maximum, number))
