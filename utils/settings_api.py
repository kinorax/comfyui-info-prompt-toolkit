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
    USE_LOADED_MODEL_CACHE_SECTION,
    get_settings_payload,
    update_use_loaded_model_cache_settings,
)

_ROUTES_REGISTERED = False


def _settings_from_payload(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}

    section = payload.get(USE_LOADED_MODEL_CACHE_SECTION)
    if isinstance(section, Mapping):
        return section

    return payload


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
        return web.json_response({"ok": True, **get_settings_payload()})

    @PromptServer.instance.routes.post("/ipt/settings")
    async def ipt_settings_post(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        settings = update_use_loaded_model_cache_settings(_settings_from_payload(payload))
        return web.json_response(
            {
                "ok": True,
                USE_LOADED_MODEL_CACHE_SECTION: settings.as_dict(),
            }
        )

    _ROUTES_REGISTERED = True


register_routes()
