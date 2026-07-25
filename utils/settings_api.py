# Copyright 2026 kinorax
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - unavailable outside ComfyUI runtime
    web = None  # type: ignore[assignment]
    PromptServer = None  # type: ignore[assignment]

from .settings import (
    PROTECTED_SETTINGS_SECTION,
    SettingsStorageError,
    USE_LOADED_MODEL_CACHE_SECTION,
    get_protected_settings_payload,
    get_settings_payload,
    get_use_loaded_model_cache_settings,
    update_protected_setting_envelope,
    update_use_loaded_model_cache_settings,
)
from .protected_settings import (
    METADATA_ENCRYPTION_SETTING_ID,
    ProtectedSettingError,
    generate_device_default_metadata_key,
    generate_random_metadata_key,
    get_protected_setting_registration,
    normalize_envelope,
    protect_setting,
    unprotect_setting,
)

_ROUTES_REGISTERED = False


def _settings_from_payload(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}

    section = payload.get(USE_LOADED_MODEL_CACHE_SECTION)
    if isinstance(section, Mapping):
        return section

    return payload


def _json_response(payload: Mapping[str, Any], *, status: int = 200):
    response = web.json_response(dict(payload), status=status)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return response


async def _request_payload(request) -> Mapping[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _protected_setting_error_response(error: Exception):
    if isinstance(error, SettingsStorageError):
        message = "The protected settings storage is unavailable."
        status = 409
    else:
        message = "The protected setting is invalid or cannot be decrypted."
        status = 400
    return _json_response({"ok": False, "error": message}, status=status)


def register_routes() -> None:
    global _ROUTES_REGISTERED

    if _ROUTES_REGISTERED:
        return
    if PromptServer is None or web is None:
        return
    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        return

    @PromptServer.instance.routes.get("/ipt/settings")
    async def ipt_settings_get(_request):
        try:
            payload = get_settings_payload()
        except (ProtectedSettingError, SettingsStorageError) as error:
            return _protected_setting_error_response(error)
        return _json_response({"ok": True, **payload})

    @PromptServer.instance.routes.post("/ipt/settings")
    async def ipt_settings_post(request):
        payload = await _request_payload(request)
        try:
            if USE_LOADED_MODEL_CACHE_SECTION in payload:
                settings = update_use_loaded_model_cache_settings(_settings_from_payload(payload))
            elif PROTECTED_SETTINGS_SECTION not in payload:
                settings = update_use_loaded_model_cache_settings(_settings_from_payload(payload))
            else:
                settings = get_use_loaded_model_cache_settings()

            protected_payload = payload.get(PROTECTED_SETTINGS_SECTION)
            if protected_payload is not None:
                if not isinstance(protected_payload, Mapping):
                    raise ProtectedSettingError("The protected settings payload must be an object.")
                for setting_id, envelope in protected_payload.items():
                    update_protected_setting_envelope(str(setting_id), envelope)

            protected = get_protected_settings_payload()
        except (ProtectedSettingError, SettingsStorageError) as error:
            return _protected_setting_error_response(error)

        return _json_response(
            {
                "ok": True,
                USE_LOADED_MODEL_CACHE_SECTION: settings.as_dict(),
                PROTECTED_SETTINGS_SECTION: protected,
            }
        )

    @PromptServer.instance.routes.post("/ipt/protected-settings/protect")
    async def ipt_protected_settings_protect(request):
        payload = await _request_payload(request)
        try:
            setting_id = str(payload.get("setting_id") or "")
            envelope = protect_setting(
                setting_id,
                payload.get("value"),
                payload.get("revision"),
                key_source=payload.get("key_source"),
            )
        except (ProtectedSettingError, SettingsStorageError) as error:
            return _protected_setting_error_response(error)
        return _json_response({"ok": True, "envelope": envelope})

    @PromptServer.instance.routes.post("/ipt/protected-settings/unprotect")
    async def ipt_protected_settings_unprotect(request):
        payload = await _request_payload(request)
        try:
            setting_id = str(payload.get("setting_id") or "")
            registration = get_protected_setting_registration(setting_id)
            if not registration.frontend_readable:
                raise ProtectedSettingError("The protected setting is not frontend-readable.")
            envelope = normalize_envelope(payload.get("envelope"))
            value = unprotect_setting(setting_id, envelope)
        except (ProtectedSettingError, SettingsStorageError) as error:
            return _protected_setting_error_response(error)
        return _json_response({"ok": True, "value": value})

    @PromptServer.instance.routes.post("/ipt/metadata-encryption/random-key")
    async def ipt_metadata_encryption_random_key(_request):
        return _json_response({"ok": True, "key": generate_random_metadata_key()})

    @PromptServer.instance.routes.post("/ipt/metadata-encryption/device-default")
    async def ipt_metadata_encryption_device_default(_request):
        key, reproducible = generate_device_default_metadata_key()
        return _json_response(
            {
                "ok": True,
                "key": key,
                "reproducible": reproducible,
                "setting_id": METADATA_ENCRYPTION_SETTING_ID,
            }
        )

    _ROUTES_REGISTERED = True


register_routes()
