# Copyright 2026 kinorax
from __future__ import annotations

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover
    web = None  # type: ignore[assignment]
    PromptServer = None  # type: ignore[assignment]

import folder_paths

from .image_reader_metadata import read_a1111_text_from_image_selection
from .image_reader_model_check import inspect_infotext_references
from .model_lora_metadata_pipeline import get_shared_metadata_pipeline

_ROUTES_REGISTERED = False


def register_routes() -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    if PromptServer is None or web is None:
        return
    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        return

    pipeline = get_shared_metadata_pipeline(start=True)

    @PromptServer.instance.routes.post("/ipt/image-reader/model-check")
    async def ipt_image_reader_model_check(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        image = str(payload.get("image", "") or "").strip()
        if not image:
            return web.json_response({"ok": False, "error": "image is required"}, status=400)
        if not folder_paths.exists_annotated_filepath(image):
            return web.json_response({"ok": False, "error": "invalid image"}, status=400)

        try:
            image_path, infotext = read_a1111_text_from_image_selection(image)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

        items = inspect_infotext_references(infotext, pipeline=pipeline) if infotext else []
        return web.json_response(
            {
                "ok": True,
                "image": image,
                "image_path": image_path,
                "infotext_found": bool(infotext),
                "items": items,
            }
        )

    _ROUTES_REGISTERED = True


register_routes()
