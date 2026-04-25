# Copyright 2026 kinorax
from __future__ import annotations

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover
    web = None  # type: ignore[assignment]
    PromptServer = None  # type: ignore[assignment]

from .prompt_wildcards import (
    get_wildcards_directory,
    list_prompt_wildcard_items,
    list_prompt_wildcards,
)

_ROUTES_REGISTERED = False
_DEFAULT_LIMIT = 100
_MAX_LIMIT = 300


def register_routes() -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    if PromptServer is None or web is None:
        return
    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        return

    @PromptServer.instance.routes.get("/ipt/prompt-template/wildcards")
    async def ipt_prompt_template_wildcards(request):
        mode = str(request.query.get("mode", "wildcards") or "wildcards").strip().lower()
        query = str(request.query.get("q", "") or "")
        token = str(request.query.get("token", "") or "")
        limit = _coerce_limit(request.query.get("limit"))

        if mode == "items":
            items = list_prompt_wildcard_items(token=token, query=query, limit=limit)
            return web.json_response(
                {
                    "ok": True,
                    "mode": "items",
                    "query": query,
                    "token": token,
                    "wildcards_dir": _string_or_empty(get_wildcards_directory()),
                    "entries": [
                        {
                            "index": item.index,
                            "value": item.value,
                            "display_text": item.display_text,
                            "description": item.description,
                            "weight": item.weight,
                            "random_enabled": item.weight > 0,
                            "insert_text": str(item.index),
                            "selector_token": f"__{token}__#{item.index}",
                        }
                        for item in items
                    ],
                }
            )

        entries = list_prompt_wildcards(query=query, limit=limit)
        return web.json_response(
            {
                "ok": True,
                "mode": "wildcards",
                "query": query,
                "wildcards_dir": _string_or_empty(get_wildcards_directory()),
                "entries": [
                    {
                        "path": wildcard_token,
                        "token": f"__{wildcard_token}__",
                    }
                    for wildcard_token in entries
                ],
            }
        )

    _ROUTES_REGISTERED = True


def _coerce_limit(value: object) -> int:
    try:
        parsed = int(value)
    except Exception:
        return _DEFAULT_LIMIT

    if parsed <= 0:
        return _DEFAULT_LIMIT
    return min(parsed, _MAX_LIMIT)


def _string_or_empty(value: object) -> str:
    return "" if value is None else str(value)


register_routes()
