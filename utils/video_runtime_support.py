# Copyright 2026 kinorax
from __future__ import annotations

from typing import Any, Callable, Iterator


VIDEO_READER_BACKEND_UNAVAILABLE_MESSAGE = (
    "Video reading is unavailable in this ComfyUI environment. Install `av` first."
)
VIDEO_SAVER_PYAV_REQUIRED_MESSAGE = "PyAV is required to save videos. Install `av` first."


def _iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    current: BaseException | None = exc
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current

        next_exc = current.__cause__
        if next_exc is None or id(next_exc) in seen:
            next_exc = current.__context__
        current = next_exc


def get_video_from_file_loader(input_impl: Any) -> Callable[[str], Any]:
    loader = getattr(input_impl, "VideoFromFile", None)
    if callable(loader):
        return loader
    raise RuntimeError(VIDEO_READER_BACKEND_UNAVAILABLE_MESSAGE)


def is_video_backend_unavailable_error(exc: BaseException) -> bool:
    for current in _iter_exception_chain(exc):
        if isinstance(current, ModuleNotFoundError):
            if str(getattr(current, "name", "") or "") == "av":
                return True

        message = str(current or "").strip().lower()
        if not message:
            continue

        if "no module named" in message and ("'av'" in message or '"av"' in message):
            return True
        if "pyav" in message and ("missing" in message or "required" in message or "not installed" in message):
            return True
        if "videofromfile" in message and ("unavailable" in message or "not available" in message):
            return True

    return False
