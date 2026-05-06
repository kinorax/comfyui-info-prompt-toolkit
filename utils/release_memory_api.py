# Copyright 2026 kinorax
from __future__ import annotations

import asyncio
from typing import Any

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - unavailable outside ComfyUI runtime
    web = None  # type: ignore[assignment]
    PromptServer = None  # type: ignore[assignment]

from .release_memory import bool_or_default, release_memory

_ROUTES_REGISTERED = False


def _is_prompt_running() -> bool:
    if PromptServer is None:
        return False
    instance = getattr(PromptServer, "instance", None)
    prompt_queue = getattr(instance, "prompt_queue", None)
    if prompt_queue is None:
        return False

    mutex = getattr(prompt_queue, "mutex", None)
    running = getattr(prompt_queue, "currently_running", None)
    if not isinstance(running, dict):
        return False

    if mutex is None:
        return len(running) > 0
    with mutex:
        return len(running) > 0


def _release_options_from_payload(payload: dict[str, Any]) -> dict[str, bool]:
    return {
        "generation_runtime": bool_or_default(payload.get("generation_runtime"), True),
        "sam3_runtime": bool_or_default(payload.get("sam3_runtime"), True),
        "pixai_tagger_runtime": bool_or_default(payload.get("pixai_tagger_runtime"), True),
        "gc_cuda_cleanup": bool_or_default(payload.get("gc_cuda_cleanup"), True),
    }


def register_routes() -> None:
    global _ROUTES_REGISTERED

    if _ROUTES_REGISTERED:
        return
    if PromptServer is None or web is None:
        return
    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        return

    @PromptServer.instance.routes.post("/ipt/release-memory")
    async def ipt_release_memory(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        if _is_prompt_running():
            return web.json_response(
                {
                    "ok": False,
                    "error": "busy",
                    "message": "A prompt is currently running. Use the node in the workflow or try again after execution finishes.",
                },
                status=409,
            )

        options = _release_options_from_payload(payload)
        result = await asyncio.to_thread(release_memory, **options)
        return web.json_response(result, status=200 if result.get("ok") else 500)

    _ROUTES_REGISTERED = True


register_routes()
