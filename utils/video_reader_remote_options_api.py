# Copyright 2026 kinorax
from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlencode

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - unavailable outside ComfyUI runtime
    web = None  # type: ignore[assignment]
    PromptServer = None  # type: ignore[assignment]

from ..nodes.video_reader import (
    _list_video_directories,
    _list_video_files,
    _resolve_video_path,
)

_ROUTES_REGISTERED = False
_PREVIEW_MODE_SOURCE = "source"
_PREVIEW_REASON_DIRECT_SOURCE_ONLY = "preview_does_not_reencode"


def _parse_int_query(raw: Any, *, key: str, minimum: int) -> int:
    try:
        value = int(str(raw))
    except Exception as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


def _build_source_view_url(*, path_source: str, directory: str, file_name: str) -> str:
    subfolder = directory.replace("\\", "/").strip()
    if subfolder in ("", "."):
        query = urlencode({"filename": file_name, "type": path_source})
    else:
        query = urlencode({"filename": file_name, "subfolder": subfolder, "type": path_source})
    return f"/view?{query}"


def _prepare_preview_file(
    *,
    path_source: str,
    directory: str,
    file_name: str,
    frame_load_cap: int,
    skip_first_frames: int,
    select_every_nth: int,
) -> dict[str, Any]:
    # Preview intentionally avoids server-side re-encoding and only returns
    # the selected source file URL. Saving/codec availability stays delegated
    # to the user's environment.
    _ = frame_load_cap
    _ = skip_first_frames
    _ = select_every_nth

    _resolve_video_path(directory, file_name, path_source)
    return {
        "ok": True,
        "mode": _PREVIEW_MODE_SOURCE,
        "reason": _PREVIEW_REASON_DIRECT_SOURCE_ONLY,
        "cache_key": None,
        "url": _build_source_view_url(
            path_source=path_source,
            directory=directory,
            file_name=file_name,
        ),
        "source_fps": None,
        "preview_fps": None,
        "selected_frames": None,
        "threshold_frames": None,
        "threshold_pixel_frames": None,
    }


def register_routes() -> None:
    global _ROUTES_REGISTERED

    if _ROUTES_REGISTERED:
        return
    if PromptServer is None or web is None:
        return
    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        return

    @PromptServer.instance.routes.get("/iis/video-reader/directories")
    async def iis_video_reader_directories(request):
        path_source = str(request.query.get("path_source", "input") or "input")
        try:
            directories = _list_video_directories(path_source)
        except Exception:
            directories = ["."]
        return web.json_response(directories)

    @PromptServer.instance.routes.get("/iis/video-reader/files")
    async def iis_video_reader_files(request):
        path_source = str(request.query.get("path_source", "input") or "input")
        directory = str(request.query.get("directory", ".") or ".")
        try:
            files = _list_video_files(directory=directory, path_source=path_source)
        except Exception:
            files = []
        return web.json_response(files)

    @PromptServer.instance.routes.get("/iis/video-reader/preview")
    async def iis_video_reader_preview(request):
        path_source = str(request.query.get("path_source", "input") or "input")
        directory = str(request.query.get("directory", ".") or ".")
        file_name = str(request.query.get("file", "") or "").strip()
        if not file_name:
            return web.json_response({"ok": False, "error": "file is required"}, status=400)

        try:
            frame_load_cap = _parse_int_query(
                request.query.get("frame_load_cap", 0),
                key="frame_load_cap",
                minimum=0,
            )
            skip_first_frames = _parse_int_query(
                request.query.get("skip_first_frames", 0),
                key="skip_first_frames",
                minimum=0,
            )
            select_every_nth = _parse_int_query(
                request.query.get("select_every_nth", 1),
                key="select_every_nth",
                minimum=1,
            )
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

        try:
            result = await asyncio.to_thread(
                _prepare_preview_file,
                path_source=path_source,
                directory=directory,
                file_name=file_name,
                frame_load_cap=frame_load_cap,
                skip_first_frames=skip_first_frames,
                select_every_nth=select_every_nth,
            )
            return web.json_response(result)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    _ROUTES_REGISTERED = True


register_routes()
