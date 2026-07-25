# Copyright 2026 kinorax
from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import dataclass
from threading import Lock

from ._runtime_loader import cache_descriptor


@dataclass(frozen=True)
class ClipRuntimeState:
    cache_key: str
    lora_applied: bool
    reusable: bool


_CLIP_RUNTIME_CACHE_LOCK = Lock()
_REUSABLE_CLIPS: dict[str, weakref.ReferenceType[object]] = {}
_RUNTIME_STATES: dict[int, tuple[weakref.ReferenceType[object], ClipRuntimeState]] = {}


def clip_runtime_cache_key(clip_descriptor: object) -> str:
    payload = cache_descriptor(clip_descriptor)
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def reusable_clip_runtime(clip_descriptor: object) -> object | None:
    cache_key = clip_runtime_cache_key(clip_descriptor)
    with _CLIP_RUNTIME_CACHE_LOCK:
        runtime_ref = _REUSABLE_CLIPS.get(cache_key)
        if runtime_ref is None:
            return None
        runtime = runtime_ref()
        if runtime is None:
            _REUSABLE_CLIPS.pop(cache_key, None)
            return None
        return runtime


def register_reusable_clip_runtime(clip_descriptor: object, runtime: object | None) -> None:
    if runtime is None:
        return

    cache_key = clip_runtime_cache_key(clip_descriptor)
    runtime_id = id(runtime)

    def remove_dead_runtime(runtime_ref: weakref.ReferenceType[object]) -> None:
        with _CLIP_RUNTIME_CACHE_LOCK:
            reusable_ref = _REUSABLE_CLIPS.get(cache_key)
            if reusable_ref is runtime_ref:
                _REUSABLE_CLIPS.pop(cache_key, None)
            state_entry = _RUNTIME_STATES.get(runtime_id)
            if state_entry is not None and state_entry[0] is runtime_ref:
                _RUNTIME_STATES.pop(runtime_id, None)

    try:
        runtime_ref = weakref.ref(runtime, remove_dead_runtime)
    except TypeError:
        return

    state = ClipRuntimeState(cache_key=cache_key, lora_applied=False, reusable=True)
    with _CLIP_RUNTIME_CACHE_LOCK:
        _REUSABLE_CLIPS[cache_key] = runtime_ref
        _RUNTIME_STATES[runtime_id] = (runtime_ref, state)


def mark_clip_lora_applied(input_clip: object | None, output_clip: object | None) -> None:
    if input_clip is None:
        return

    input_state = clip_runtime_state(input_clip)
    if input_state is None:
        return

    with _CLIP_RUNTIME_CACHE_LOCK:
        reusable_ref = _REUSABLE_CLIPS.get(input_state.cache_key)
        if reusable_ref is not None and reusable_ref() is input_clip:
            _REUSABLE_CLIPS.pop(input_state.cache_key, None)

    if output_clip is None:
        return

    runtime_id = id(output_clip)

    def remove_dead_runtime(runtime_ref: weakref.ReferenceType[object]) -> None:
        with _CLIP_RUNTIME_CACHE_LOCK:
            state_entry = _RUNTIME_STATES.get(runtime_id)
            if state_entry is not None and state_entry[0] is runtime_ref:
                _RUNTIME_STATES.pop(runtime_id, None)

    try:
        runtime_ref = weakref.ref(output_clip, remove_dead_runtime)
    except TypeError:
        return

    state = ClipRuntimeState(
        cache_key=input_state.cache_key,
        lora_applied=True,
        reusable=False,
    )
    with _CLIP_RUNTIME_CACHE_LOCK:
        _RUNTIME_STATES[runtime_id] = (runtime_ref, state)


def clip_runtime_state(runtime: object | None) -> ClipRuntimeState | None:
    if runtime is None:
        return None

    runtime_id = id(runtime)
    with _CLIP_RUNTIME_CACHE_LOCK:
        entry = _RUNTIME_STATES.get(runtime_id)
        if entry is None:
            return None
        runtime_ref, state = entry
        if runtime_ref() is runtime:
            return state
        _RUNTIME_STATES.pop(runtime_id, None)
        return None


def clear_clip_runtime_cache() -> int:
    with _CLIP_RUNTIME_CACHE_LOCK:
        cleared_entries = len(_REUSABLE_CLIPS)
        _REUSABLE_CLIPS.clear()
        _RUNTIME_STATES.clear()
        return cleared_entries
