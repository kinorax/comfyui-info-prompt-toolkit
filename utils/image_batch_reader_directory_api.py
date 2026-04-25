# Copyright 2026 kinorax
from __future__ import annotations

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - unavailable outside ComfyUI runtime
    web = None  # type: ignore[assignment]
    PromptServer = None  # type: ignore[assignment]

from ..nodes.batch_image_reader import _list_directory_options

_ROUTES_REGISTERED = False


def register_routes() -> None:
    global _ROUTES_REGISTERED

    if _ROUTES_REGISTERED:
        return
    if PromptServer is None or web is None:
        return
    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        return

    @PromptServer.instance.routes.get("/iis/batch-image-reader/directories")
    async def iis_batch_image_reader_directories(request):
        path_source = str(request.query.get("path_source", "input") or "input")
        try:
            directories = _list_directory_options(path_source)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

        return web.json_response(
            {
                "ok": True,
                "path_source": path_source,
                "directories": directories,
            }
        )

    _ROUTES_REGISTERED = True


register_routes()
