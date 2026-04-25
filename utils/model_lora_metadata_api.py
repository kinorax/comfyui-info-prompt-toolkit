# Copyright 2026 kinorax
from __future__ import annotations
import asyncio

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover
    web = None  # type: ignore[assignment]
    PromptServer = None  # type: ignore[assignment]

from .model_lora_metadata_pipeline import get_shared_metadata_pipeline
from .model_preview_thumbnail_pipeline import get_shared_thumbnail_pipeline
from .model_runtime_settings import (
    filter_model_runtime_settings_for_folder,
    is_supported_model_runtime_settings_folder,
    normalize_model_runtime_settings,
)
from .model_reference_resolver import resolve_model_reference

_ROUTES_REGISTERED = False
_ALLOWED_FOLDERS = {"checkpoints", "diffusion_models", "unet", "loras", "text_encoders", "vae"}
_LORA_TAG_LIMIT = 1000


def register_routes() -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    if PromptServer is None or web is None:
        return
    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        return

    pipeline = get_shared_metadata_pipeline(start=True)
    thumbnail_pipeline = get_shared_thumbnail_pipeline(start=True, metadata_pipeline=pipeline)

    @PromptServer.instance.routes.post("/ipt/model-metadata/queue-priority")
    async def ipt_model_metadata_queue_priority(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        folder_name = str(payload.get("folder_name", "") or "").strip()
        relative_path = str(payload.get("relative_path", "") or "").strip()
        if folder_name not in _ALLOWED_FOLDERS:
            return web.json_response({"ok": False, "error": "invalid folder_name"}, status=400)
        if not relative_path:
            return web.json_response({"ok": False, "error": "relative_path is required"}, status=400)

        enqueued = pipeline.enqueue_hash_priority(folder_name, relative_path)
        return web.json_response(
            {
                "ok": True,
                "enqueued": bool(enqueued),
                "folder_name": folder_name,
                "relative_path": relative_path,
            }
        )

    @PromptServer.instance.routes.post("/ipt/model-reference/resolve")
    async def ipt_model_reference_resolve(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        folder_name = str(payload.get("folder_name", "") or "").strip()
        if folder_name not in _ALLOWED_FOLDERS:
            return web.json_response({"ok": False, "error": "invalid folder_name"}, status=400)

        result = await asyncio.to_thread(
            resolve_model_reference,
            pipeline,
            folder_name=folder_name,
            relative_path=payload.get("relative_path"),
            sha256=payload.get("sha256"),
            name_raw=payload.get("name_raw"),
            hash_hints=payload.get("hash_hints"),
            resolve_remote=bool(payload.get("resolve_remote")),
            include_lora_tags=bool(payload.get("include_lora_tags")),
            ensure_lora_tags=bool(payload.get("ensure_lora_tags")),
            ensure_timeout_ms=payload.get("ensure_timeout_ms"),
            enqueue_local_hash=bool(payload.get("enqueue_local_hash")),
        )
        return web.json_response(result)

    @PromptServer.instance.routes.post("/ipt/model-runtime-settings/upsert")
    async def ipt_model_runtime_settings_upsert(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        folder_name = str(payload.get("folder_name", "") or "").strip()
        relative_path = str(payload.get("relative_path", "") or "").strip()
        if not is_supported_model_runtime_settings_folder(folder_name):
            return web.json_response({"ok": False, "error": "invalid folder_name"}, status=400)
        if not relative_path:
            return web.json_response({"ok": False, "error": "relative_path is required"}, status=400)

        normalized_settings = normalize_model_runtime_settings(payload.get("runtime_settings"))
        result = await asyncio.to_thread(
            pipeline.upsert_model_runtime_settings_by_relative_path,
            folder_name=folder_name,
            relative_path=relative_path,
            settings=normalized_settings,
        )
        if result is None:
            return web.json_response(
                {
                    "ok": False,
                    "error": "runtime settings target is not ready yet",
                },
                status=409,
            )

        return web.json_response(
            {
                "ok": True,
                "folder_name": folder_name,
                "relative_path": relative_path,
                "runtime_settings": filter_model_runtime_settings_for_folder(
                    folder_name,
                    result.get("runtime_settings"),
                ),
                "updated_at": result.get("updated_at"),
            }
        )

    @PromptServer.instance.routes.get("/ipt/model-metadata/model-info")
    async def ipt_model_metadata_model_info(request):
        folder_name = str(request.query.get("folder_name", "") or "").strip()
        relative_path = str(request.query.get("relative_path", "") or "").strip()
        if folder_name not in _ALLOWED_FOLDERS:
            return web.json_response({"ok": False, "error": "invalid folder_name"}, status=400)
        if not relative_path:
            return web.json_response({"ok": False, "error": "relative_path is required"}, status=400)

        info = pipeline.get_model_info_by_relative_path(
            folder_name=folder_name,
            relative_path=relative_path,
        )
        if info is None:
            return web.json_response(
                {
                    "ok": True,
                    "found": False,
                    "folder_name": folder_name,
                    "relative_path": relative_path,
                }
            )

        content_id_raw = info.get("content_id")
        content_id = int(content_id_raw) if isinstance(content_id_raw, int) else None
        lora_tags = []
        if folder_name == "loras" and content_id is not None:
            lora_tags = pipeline.list_lora_tags_by_content_id(content_id, limit=_LORA_TAG_LIMIT)

        return web.json_response(
            {
                "ok": True,
                "found": True,
                "folder_name": folder_name,
                "relative_path": relative_path,
                "model_info": {
                    "civitai_model_name": info.get("civitai_model_name"),
                    "civitai_model_version_model_id": info.get("civitai_model_version_model_id"),
                    "civitai_model_version_model_version_id": info.get("civitai_model_version_model_version_id"),
                    "civitai_model_version_name": info.get("civitai_model_version_name"),
                    "civitai_model_version_base_model": info.get("civitai_model_version_base_model"),
                },
                "runtime_settings": pipeline.get_model_runtime_settings_by_relative_path(
                    folder_name=folder_name,
                    relative_path=relative_path,
                ) if is_supported_model_runtime_settings_folder(folder_name) else {},
                "lora_tags": lora_tags,
            }
        )

    @PromptServer.instance.routes.get("/ipt/model-metadata/model-thumbnail")
    async def ipt_model_metadata_model_thumbnail(request):
        folder_name = str(request.query.get("folder_name", "") or "").strip()
        relative_path = str(request.query.get("relative_path", "") or "").strip()
        if folder_name not in _ALLOWED_FOLDERS:
            return web.json_response({"ok": False, "error": "invalid folder_name"}, status=400)
        if not relative_path:
            return web.json_response({"ok": False, "error": "relative_path is required"}, status=400)

        pipeline.enqueue_hash_priority(folder_name, relative_path)
        result = thumbnail_pipeline.get_thumbnail_by_relative_path(
            folder_name=folder_name,
            relative_path=relative_path,
        )
        state = str(result.get("state") or "")
        if state == "ready":
            image_blob = result.get("image_blob")
            if isinstance(image_blob, (bytes, bytearray)) and image_blob:
                return web.Response(
                    body=bytes(image_blob),
                    content_type=str(result.get("image_mime") or "image/webp"),
                    headers={
                        "Cache-Control": "private, max-age=86400",
                    },
                )

        if state == "no_source":
            payload = {
                "ok": True,
                "ready": False,
                "state": state,
                "folder_name": folder_name,
                "relative_path": relative_path,
            }
            for key in ("content_id", "source_url", "media_type", "detail", "thumbnail_state"):
                if key in result:
                    payload[key] = result[key]
            return web.json_response(payload, status=404)

        payload = {
            "ok": True,
            "ready": False,
            "state": state or "pending",
            "folder_name": folder_name,
            "relative_path": relative_path,
        }
        for key in ("content_id", "source_url", "media_type", "detail", "thumbnail_state"):
            if key in result:
                payload[key] = result[key]
        return web.json_response(payload, status=202)

    _ROUTES_REGISTERED = True


register_routes()
