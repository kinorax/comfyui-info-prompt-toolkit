# Copyright 2026 kinorax
from __future__ import annotations

import gc
import importlib
from typing import Any


def bool_or_default(value: Any, default: bool) -> bool:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return bool_or_default(value[0], default)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _append_step(result: dict[str, Any], name: str, **payload: Any) -> None:
    result["steps"].append({"name": name, **payload})


def _append_error(result: dict[str, Any], name: str, exc: BaseException) -> None:
    result["ok"] = False
    result["errors"].append(
        {
            "name": name,
            "error": f"{type(exc).__name__}: {exc}",
        }
    )


def _clear_use_loaded_model_cache(result: dict[str, Any]) -> None:
    from ..nodes import use_loaded_model

    with use_loaded_model._CACHE_LOCK:
        entry_count = len(use_loaded_model._LAST_CACHE)
        use_loaded_model._LAST_CACHE.clear()
        use_loaded_model._LAST_CACHE_BYTES.clear()

    _append_step(result, "use_loaded_model_cache", cleared_entries=entry_count)


def _clear_load_new_model_vae_cache(result: dict[str, Any]) -> None:
    from ..nodes import load_new_model

    with load_new_model._VAE_CACHE_LOCK:
        had_runtime = load_new_model._LAST_VAE_RUNTIME is not None
        load_new_model._LAST_VAE_NAME = None
        load_new_model._LAST_VAE_RUNTIME = None

    _append_step(result, "load_new_model_vae_cache", cleared_entries=1 if had_runtime else 0)


def _unload_comfyui_runtime_models(result: dict[str, Any]) -> None:
    try:
        comfy_model_management = importlib.import_module("comfy.model_management")
    except Exception as exc:
        _append_step(
            result,
            "comfyui_runtime_models",
            skipped=True,
            reason=f"unavailable:{type(exc).__name__}",
        )
        return

    unload_all_models = getattr(comfy_model_management, "unload_all_models", None)
    cleanup_models = getattr(comfy_model_management, "cleanup_models", None)

    called: list[str] = []
    if callable(unload_all_models):
        unload_all_models()
        called.append("unload_all_models")
    if callable(cleanup_models):
        cleanup_models()
        called.append("cleanup_models")

    _append_step(result, "comfyui_runtime_models", called=called)


def _release_generation_runtime(result: dict[str, Any]) -> None:
    for name, fn in (
        ("use_loaded_model_cache", _clear_use_loaded_model_cache),
        ("load_new_model_vae_cache", _clear_load_new_model_vae_cache),
        ("comfyui_runtime_models", _unload_comfyui_runtime_models),
    ):
        try:
            fn(result)
        except Exception as exc:
            _append_error(result, name, exc)


def _release_sam3_runtime(result: dict[str, Any]) -> None:
    from ..nodes.mask import sam3_text_mask

    with sam3_text_mask._PROCESSOR_CACHE_LOCK:
        entry_count = len(sam3_text_mask._PROCESSOR_CACHE)
        sam3_text_mask._PROCESSOR_CACHE.clear()

    _append_step(result, "sam3_runtime", cleared_entries=entry_count)


def _release_pixai_tagger_runtime(result: dict[str, Any]) -> None:
    from ..nodes.prompt import pixai_tagger

    with pixai_tagger._MODEL_CACHE_LOCK:
        entry_count = len(pixai_tagger._MODEL_CACHE)
        pixai_tagger._MODEL_CACHE.clear()

    _append_step(result, "pixai_tagger_runtime", cleared_entries=entry_count)


def _run_python_gc(result: dict[str, Any]) -> None:
    collected = gc.collect()
    _append_step(result, "python_gc", collected_objects=int(collected))


def _clear_cuda_cache(result: dict[str, Any]) -> None:
    called: list[str] = []

    try:
        comfy_model_management = importlib.import_module("comfy.model_management")
        soft_empty_cache = getattr(comfy_model_management, "soft_empty_cache", None)
        if callable(soft_empty_cache):
            try:
                soft_empty_cache(force=True)
            except TypeError:
                soft_empty_cache()
            called.append("comfy.model_management.soft_empty_cache")
    except Exception as exc:
        _append_step(
            result,
            "comfy_cuda_cache",
            skipped=True,
            reason=f"unavailable:{type(exc).__name__}",
        )

    try:
        torch = importlib.import_module("torch")
        cuda = getattr(torch, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        if cuda is not None and callable(is_available) and bool(is_available()):
            empty_cache = getattr(cuda, "empty_cache", None)
            ipc_collect = getattr(cuda, "ipc_collect", None)
            if callable(empty_cache):
                empty_cache()
                called.append("torch.cuda.empty_cache")
            if callable(ipc_collect):
                ipc_collect()
                called.append("torch.cuda.ipc_collect")
        else:
            called.append("torch.cuda.unavailable")
    except Exception as exc:
        _append_step(
            result,
            "torch_cuda_cache",
            skipped=True,
            reason=f"unavailable:{type(exc).__name__}",
        )

    _append_step(result, "cuda_cache", called=called)


def release_memory(
    *,
    generation_runtime: Any = True,
    sam3_runtime: Any = True,
    pixai_tagger_runtime: Any = True,
    gc_cuda_cleanup: Any = True,
) -> dict[str, Any]:
    options = {
        "generation_runtime": bool_or_default(generation_runtime, True),
        "sam3_runtime": bool_or_default(sam3_runtime, True),
        "pixai_tagger_runtime": bool_or_default(pixai_tagger_runtime, True),
        "gc_cuda_cleanup": bool_or_default(gc_cuda_cleanup, True),
    }
    result: dict[str, Any] = {
        "ok": True,
        "requested": dict(options),
        "steps": [],
        "errors": [],
    }

    if options["generation_runtime"]:
        _release_generation_runtime(result)
    if options["sam3_runtime"]:
        try:
            _release_sam3_runtime(result)
        except Exception as exc:
            _append_error(result, "sam3_runtime", exc)
    if options["pixai_tagger_runtime"]:
        try:
            _release_pixai_tagger_runtime(result)
        except Exception as exc:
            _append_error(result, "pixai_tagger_runtime", exc)
    if options["gc_cuda_cleanup"]:
        try:
            _run_python_gc(result)
        except Exception as exc:
            _append_error(result, "python_gc", exc)
        try:
            _clear_cuda_cache(result)
        except Exception as exc:
            _append_error(result, "cuda_cache", exc)

    return result
