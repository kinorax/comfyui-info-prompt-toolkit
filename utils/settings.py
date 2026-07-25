# Copyright 2026 kinorax
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import copy
import json
import os
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
PROTECTED_SETTINGS_SECTION = "protected_settings"

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

_SETTINGS_LOCK = threading.RLock()
_USE_LOADED_MODEL_CACHE_SETTINGS: UseLoadedModelCacheSettings | None = None


class SettingsStorageError(RuntimeError):
    """The IPT settings document could not be read or written safely."""


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
        PROTECTED_SETTINGS_SECTION: get_protected_settings_payload(),
    }


def get_protected_settings_payload(*, ensure_defaults: bool = True) -> dict[str, Any]:
    from .protected_settings import list_protected_setting_ids

    with _SETTINGS_LOCK:
        result: dict[str, Any] = {}
        for setting_id in list_protected_setting_ids():
            envelope = _get_protected_setting_envelope_unlocked(
                setting_id,
                ensure_default=ensure_defaults,
            )
            if envelope is not None:
                result[setting_id] = copy.deepcopy(envelope)
        return result


def get_protected_setting_envelope(
    setting_id: str,
    *,
    ensure_default: bool = True,
) -> dict[str, Any] | None:
    with _SETTINGS_LOCK:
        envelope = _get_protected_setting_envelope_unlocked(
            setting_id,
            ensure_default=ensure_default,
        )
        return copy.deepcopy(envelope) if envelope is not None else None


def get_protected_setting_value(setting_id: str) -> Any:
    from .protected_settings import unprotect_setting

    envelope = get_protected_setting_envelope(setting_id, ensure_default=True)
    if envelope is None:
        raise SettingsStorageError("The protected setting is unavailable.")
    return unprotect_setting(setting_id, envelope)


def get_metadata_encryption_settings() -> dict[str, Any]:
    from .protected_settings import METADATA_ENCRYPTION_SETTING_ID

    value = get_protected_setting_value(METADATA_ENCRYPTION_SETTING_ID)
    if not isinstance(value, dict):
        raise SettingsStorageError("The metadata encryption setting is invalid.")
    return value


def update_protected_setting_envelope(setting_id: str, envelope: Any) -> dict[str, Any]:
    from .protected_settings import normalize_envelope, unprotect_setting

    normalized = normalize_envelope(envelope)
    unprotect_setting(setting_id, normalized)

    with _SETTINGS_LOCK:
        document = _read_settings_document(strict=True)
        protected = document.get(PROTECTED_SETTINGS_SECTION)
        if protected is None:
            protected_document: dict[str, Any] = {}
        elif isinstance(protected, Mapping):
            protected_document = dict(protected)
        else:
            raise SettingsStorageError("The protected settings section is invalid.")

        protected_document[setting_id] = normalized
        document["version"] = SETTINGS_VERSION
        document[PROTECTED_SETTINGS_SECTION] = protected_document
        _write_settings_document(document)
        return copy.deepcopy(normalized)


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
    try:
        raw = _read_settings_document(strict=False)
    except SettingsStorageError:
        raw = {}

    section = raw.get(USE_LOADED_MODEL_CACHE_SECTION)
    if isinstance(section, Mapping):
        return normalize_use_loaded_model_cache_settings(section)

    return normalize_use_loaded_model_cache_settings(raw)


def _write_settings(settings: UseLoadedModelCacheSettings) -> None:
    try:
        payload = _read_settings_document(strict=True)
        payload["version"] = SETTINGS_VERSION
        payload[USE_LOADED_MODEL_CACHE_SECTION] = settings.as_dict()
        _write_settings_document(payload)
    except SettingsStorageError:
        pass


def _get_protected_setting_envelope_unlocked(
    setting_id: str,
    *,
    ensure_default: bool,
) -> dict[str, Any] | None:
    from .protected_settings import (
        METADATA_ENCRYPTION_SETTING_ID,
        ProtectedSettingError,
        create_default_value,
        normalize_envelope,
        protect_setting,
        unprotect_setting,
    )

    document = _read_settings_document(strict=True)
    protected = document.get(PROTECTED_SETTINGS_SECTION)
    if protected is not None and not isinstance(protected, Mapping):
        raise SettingsStorageError("The protected settings section is invalid.")

    stored = protected.get(setting_id) if isinstance(protected, Mapping) else None
    if stored is not None:
        try:
            normalized = normalize_envelope(stored)
        except ProtectedSettingError:
            # A pre-release implementation may have stored the logical value directly.
            try:
                from .protected_settings import get_protected_setting_registration

                logical_value = get_protected_setting_registration(setting_id).normalizer(stored)
            except ProtectedSettingError as error:
                raise SettingsStorageError("The protected setting entry is invalid.") from error
            normalized = protect_setting(setting_id, logical_value, 1)
            _store_migrated_envelope(document, setting_id, normalized)
        unprotect_setting(setting_id, normalized)
        return normalized

    legacy = document.get("metadata_encryption")
    if legacy is not None and setting_id == METADATA_ENCRYPTION_SETTING_ID:
        try:
            from .protected_settings import get_protected_setting_registration

            logical_value = get_protected_setting_registration(setting_id).normalizer(legacy)
            normalized = protect_setting(setting_id, logical_value, 1)
        except ProtectedSettingError as error:
            raise SettingsStorageError("The legacy metadata encryption setting is invalid.") from error
        document.pop("metadata_encryption", None)
        _store_migrated_envelope(document, setting_id, normalized)
        return normalized

    if not ensure_default:
        return None

    logical_value = create_default_value(setting_id)
    normalized = protect_setting(setting_id, logical_value, 1)
    _store_migrated_envelope(document, setting_id, normalized)
    return normalized


def _store_migrated_envelope(
    document: dict[str, Any],
    setting_id: str,
    envelope: Mapping[str, Any],
) -> None:
    protected = document.get(PROTECTED_SETTINGS_SECTION)
    protected_document = dict(protected) if isinstance(protected, Mapping) else {}
    protected_document[setting_id] = dict(envelope)
    document["version"] = SETTINGS_VERSION
    document[PROTECTED_SETTINGS_SECTION] = protected_document
    _write_settings_document(document)


def _read_settings_document(*, strict: bool) -> dict[str, Any]:
    path = resolve_settings_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as error:
        if strict:
            raise SettingsStorageError("The IPT settings file cannot be read.") from error
        return {}

    if not isinstance(raw, Mapping):
        if strict:
            raise SettingsStorageError("The IPT settings file must contain an object.")
        return {}
    return dict(raw)


def _write_settings_document(payload: Mapping[str, Any]) -> None:
    path = resolve_settings_path()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        raise SettingsStorageError("The IPT settings file cannot be written.") from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
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
