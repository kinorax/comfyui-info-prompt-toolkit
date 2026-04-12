from __future__ import annotations

import re
import threading
from typing import Iterable, Optional

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - unavailable outside ComfyUI runtime
    web = None  # type: ignore[assignment]
    PromptServer = None  # type: ignore[assignment]

_LOCK = threading.Lock()
_MAX_MAP_SIZE = 512
_MASK_TO_SOURCE: dict[str, str] = {}
_ROUTES_REGISTERED = False
_PAINTED_RE = re.compile(r"(?:^|/)clipspace-painted-(\d+)\.png$")


def _normalize_ref(value: Optional[str]) -> str:
    if not value:
        return ""
    s = str(value).strip().replace("\\", "/")
    if s.endswith("]") and " [" in s:
        s = s[: s.rfind(" [")]
    return s.lstrip("./").lower()


def _resolve_painted_source_fallback_locked(normalized_key: str) -> Optional[str]:
    match = _PAINTED_RE.search(normalized_key)
    if not match:
        return None
    stamp = match.group(1)
    fallback_keys = (
        f"clipspace/clipspace-mask-{stamp}.png",
        f"clipspace-mask-{stamp}.png",
    )
    for fallback_key in fallback_keys:
        source = _MASK_TO_SOURCE.get(fallback_key)
        if source:
            return source
    return None


def resolve_source_annotated(source_ref: Optional[str], max_hops: int = 8) -> Optional[str]:
    current = (source_ref or "").strip()
    if not current:
        return None

    visited: set[str] = set()
    hops = 0

    while hops < max_hops:
        key = _normalize_ref(current)
        if not key or key in visited:
            break
        visited.add(key)

        with _LOCK:
            nxt = _MASK_TO_SOURCE.get(key)
            if not nxt:
                nxt = _resolve_painted_source_fallback_locked(key)

        if not nxt:
            return current

        current = nxt
        hops += 1

    return current


def set_mask_source_mapping(masked_candidates: Iterable[str], source_annotated: str) -> int:
    source = resolve_source_annotated(source_annotated) or (source_annotated or "").strip()
    if not source:
        return 0

    keys = []
    for candidate in masked_candidates:
        key = _normalize_ref(candidate)
        if key:
            keys.append(key)

    if not keys:
        return 0

    with _LOCK:
        for key in keys:
            _MASK_TO_SOURCE[key] = source
        while len(_MASK_TO_SOURCE) > _MAX_MAP_SIZE:
            _MASK_TO_SOURCE.pop(next(iter(_MASK_TO_SOURCE)))
    return len(keys)


def get_source_annotated(masked_ref: Optional[str]) -> Optional[str]:
    key = _normalize_ref(masked_ref)
    if not key:
        return None
    with _LOCK:
        found = _MASK_TO_SOURCE.get(key)
    resolved = resolve_source_annotated(found) if found else None
    return resolved


def register_routes() -> None:
    global _ROUTES_REGISTERED

    if _ROUTES_REGISTERED:
        return
    if PromptServer is None or web is None:
        return
    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        return

    @PromptServer.instance.routes.post("/iis/clipspace-source")
    async def iis_clipspace_source(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid_json"}, status=400)

        source_annotated = str(payload.get("source_annotated") or "").strip()
        masked_candidates = payload.get("masked_candidates") or []
        if isinstance(masked_candidates, str):
            masked_candidates = [masked_candidates]
        if not isinstance(masked_candidates, list):
            masked_candidates = []

        updated = set_mask_source_mapping(masked_candidates, source_annotated)
        if updated == 0:
            return web.json_response({"ok": False, "error": "invalid_payload"}, status=400)
        return web.json_response({"ok": True, "updated": updated})

    _ROUTES_REGISTERED = True


register_routes()
