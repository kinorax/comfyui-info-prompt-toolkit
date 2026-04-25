# Copyright 2026 kinorax
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from threading import Lock

import comfy.model_management as comfy_model_management
from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.model_lora_metadata_pipeline import get_shared_metadata_pipeline
from ._runtime_loader import (
    cache_descriptor,
    is_checkpoint_model,
    is_diffusion_model,
    normalized_clip_names,
    normalized_model_name_or_none,
)
from .lora_stack_lorader import LoraStackLorader

LORA_STACK_ITEM_NAME_KEY = "name"
LORA_STACK_ITEM_STRENGTH_KEY = "strength"
MODEL_RUNTIME_TYPE = c_io.Custom("MODEL")
CLIP_RUNTIME_TYPE = c_io.Custom("CLIP")
VAE_RUNTIME_TYPE = c_io.Custom("VAE")

_CACHE_LOCK = Lock()
_LAST_CACHE_LIMIT = 2
_LAST_CACHE: OrderedDict[str, tuple[object, object | None, object | None]] = OrderedDict()
_LAST_CACHE_BYTES: OrderedDict[str, int] = OrderedDict()
_CACHE_TWO_ENTRY_BUDGET_RATIO = 0.72

_MISSING = object()


def _bool_or_default(value: object, default: bool) -> bool:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _bool_or_default(value[0], default)
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


def _to_lora_stack_items(value: object) -> list[tuple[str, float]]:
    if not isinstance(value, list):
        return []

    normalized: list[tuple[str, float]] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        name = str(item.get(LORA_STACK_ITEM_NAME_KEY, "")).strip()
        if not name:
            continue

        strength_raw = item.get(LORA_STACK_ITEM_STRENGTH_KEY, 1.0)
        try:
            strength = float(strength_raw)
        except Exception:
            strength = 1.0

        normalized.append((name, strength))

    return normalized


def _to_lora_stack_payload(lora_stack_items: list[tuple[str, float]]) -> list[dict[str, str | float]]:
    return [
        {
            LORA_STACK_ITEM_NAME_KEY: name,
            LORA_STACK_ITEM_STRENGTH_KEY: strength,
        }
        for name, strength in lora_stack_items
    ]


def _runtime_settings_for_model(model: object) -> dict[str, int | float]:
    model_name = normalized_model_name_or_none(model)
    if model_name is None:
        return {}

    if is_checkpoint_model(model):
        folder_name = Const.MODEL_FOLDER_PATH_CHECKPOINTS
    elif is_diffusion_model(model):
        folder_name = Const.MODEL_FOLDER_PATH_DIFFUSION_MODELS
    else:
        return {}

    pipeline = get_shared_metadata_pipeline(start=True)
    return pipeline.get_model_runtime_settings_by_relative_path(
        folder_name=folder_name,
        relative_path=model_name,
    )


def _cache_key(
    model: object,
    lora_stack_items: list[tuple[str, float]],
    clip: object | None,
    vae: object | None,
) -> str:
    payload = {
        "model": cache_descriptor(model),
        "lora_stack": _to_lora_stack_payload(lora_stack_items),
        "runtime_settings": _runtime_settings_for_model(model),
    }

    if is_checkpoint_model(model):
        if vae is not None:
            payload["vae"] = cache_descriptor(vae)
    elif is_diffusion_model(model):
        payload["clip"] = cache_descriptor(clip)
        payload["vae"] = cache_descriptor(vae)
    else:
        raise RuntimeError("Only checkpoint and diffusion_models are supported")

    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _runtime_size_bytes(value: object) -> int:
    if value is None:
        return 0

    get_ram_usage = getattr(value, "get_ram_usage", None)
    if callable(get_ram_usage):
        try:
            return max(0, int(get_ram_usage()))
        except Exception:
            return 0

    model_size = getattr(value, "model_size", None)
    if callable(model_size):
        try:
            return max(0, int(model_size()))
        except Exception:
            return 0

    return 0


def _bundle_size_bytes(model: object, clip: object | None, vae: object | None) -> tuple[int, int, int, int]:
    model_bytes = _runtime_size_bytes(model)
    clip_bytes = _runtime_size_bytes(clip)
    vae_bytes = _runtime_size_bytes(vae)
    return model_bytes, clip_bytes, vae_bytes, model_bytes + clip_bytes + vae_bytes


def _total_vram_bytes_or_zero() -> int:
    try:
        return max(0, int(comfy_model_management.get_total_memory(comfy_model_management.get_torch_device())))
    except Exception:
        return 0


def _load_from_last_cache(cache_key: str) -> tuple[object, object | None, object | None] | None:
    with _CACHE_LOCK:
        value = _LAST_CACHE.get(cache_key)
        if value is None:
            return None
        _LAST_CACHE.move_to_end(cache_key)
        return value[0], value[1], value[2]


def _store_last_cache(cache_key: str, model: object, clip: object | None, vae: object | None) -> None:
    _, _, _, bundle_bytes = _bundle_size_bytes(model, clip, vae)
    total_vram_bytes = _total_vram_bytes_or_zero()
    budget_two_bytes = int(total_vram_bytes * _CACHE_TWO_ENTRY_BUDGET_RATIO)

    with _CACHE_LOCK:
        _LAST_CACHE[cache_key] = (model, clip, vae)
        _LAST_CACHE_BYTES[cache_key] = bundle_bytes
        _LAST_CACHE.move_to_end(cache_key)
        _LAST_CACHE_BYTES.move_to_end(cache_key)

        recent_keys = list(_LAST_CACHE.keys())[-_LAST_CACHE_LIMIT:]
        recent_total_bytes = sum(_LAST_CACHE_BYTES.get(key, 0) for key in recent_keys)

        if total_vram_bytes <= 0:
            target_limit = 1
        elif recent_total_bytes <= budget_two_bytes:
            target_limit = 2
        else:
            target_limit = 1

        while len(_LAST_CACHE) > target_limit:
            old_key, _ = _LAST_CACHE.popitem(last=False)
            _LAST_CACHE_BYTES.pop(old_key, None)


def _load_from_cache(
    cache_key: str,
) -> tuple[tuple[object, object | None, object | None], str] | None:
    cached = _load_from_last_cache(cache_key)
    if cached is not None:
        return cached, "last"

    return None


def _store_cache(
    cache_key: str,
    model: object,
    clip: object | None,
    vae: object | None,
) -> bool:
    _store_last_cache(cache_key, model, clip, vae)
    return True


def _apply_lora_stack_with_project_node(
    model: object,
    clip: object | None,
    lora_stack: list[dict[str, str | float]] | None,
) -> tuple[object, object | None]:
    result = LoraStackLorader.execute(
        model=model,
        clip=clip,
        lora_stack=lora_stack,
    )
    return result[0], result[1]


def _requires_clip_runtime(model: object, clip: object | None) -> bool:
    if is_checkpoint_model(model):
        return True
    if is_diffusion_model(model):
        return len(normalized_clip_names(clip)) > 0
    raise RuntimeError("Only checkpoint and diffusion_models are supported")


class UseLoadedModel(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        socket_force_input = {"forceInput": True}
        return c_io.Schema(
            node_id="IPT-UseLoadedModel",
            display_name="Use Loaded Model",
            category=Const.CATEGORY_IMAGEINFO,
            not_idempotent=True,
            hidden=[
                c_io.Hidden.unique_id,
            ],
            inputs=[
                Const.MODEL_TYPE.Input(
                    "model",
                    tooltip="Model selector value used to compute runtime reuse key",
                ),
                Const.LORA_STACK_TYPE.Input(
                    Const.IMAGEINFO_LORA_STACK,
                    optional=True,
                ),
                Const.CLIP_TYPE.Input(
                    "clip",
                    optional=True,
                    extra_dict=socket_force_input,
                    tooltip="Optional CLIP reference used for diffusion model cache keys",
                ),
                c_io.AnyType.Input(
                    "vae",
                    optional=True,
                    extra_dict=socket_force_input,
                    tooltip="Optional VAE override used for cache keys",
                ),
                MODEL_RUNTIME_TYPE.Input(
                    "loaded_model",
                    optional=True,
                    lazy=True,
                    tooltip="Raw or patched runtime model for cache-miss path",
                ),
                CLIP_RUNTIME_TYPE.Input(
                    "loaded_clip",
                    optional=True,
                    lazy=True,
                    tooltip="Raw or patched runtime CLIP for cache-miss path",
                ),
                VAE_RUNTIME_TYPE.Input(
                    "loaded_vae",
                    optional=True,
                    lazy=True,
                    tooltip="Raw runtime VAE for cache-miss path",
                ),
                c_io.Boolean.Input(
                    "apply_lora_stack",
                    default=True,
                    tooltip=(
                        "If false, skip internal lora_stack application. "
                        "lora_stack is still used for the cache key."
                    ),
                ),
            ],
            outputs=[
                MODEL_RUNTIME_TYPE.Output(
                    Cast.out_id("model"),
                    display_name="model",
                ),
                CLIP_RUNTIME_TYPE.Output(
                    Cast.out_id("clip"),
                    display_name="clip",
                ),
                VAE_RUNTIME_TYPE.Output(
                    Cast.out_id("vae"),
                    display_name="vae",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        model: object | None = None,
        lora_stack: object | None = None,
        clip: object | None = None,
        vae: object | None = None,
        loaded_model: object | None = None,
        loaded_clip: object | None = None,
        loaded_vae: object | None = None,
        apply_lora_stack: object | None = True,
    ) -> bool | str:
        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        model: object | None = None,
        lora_stack: object | None = None,
        clip: object | None = None,
        vae: object | None = None,
        loaded_model: object | None = None,
        loaded_clip: object | None = None,
        loaded_vae: object | None = None,
        apply_lora_stack: object | None = True,
    ) -> int:
        return time.time_ns()

    @classmethod
    def check_lazy_status(
        cls,
        model: object,
        lora_stack: object | None = None,
        clip: object | None = None,
        vae: object | None = None,
        loaded_model: object = _MISSING,
        loaded_clip: object = _MISSING,
        loaded_vae: object = _MISSING,
        apply_lora_stack: object | None = True,
    ) -> list[str]:
        if normalized_model_name_or_none(model) is None:
            return []

        lora_stack_items = _to_lora_stack_items(lora_stack)
        cache_key = _cache_key(model, lora_stack_items, clip, vae)
        cached = _load_from_cache(cache_key)
        if cached is not None:
            return []

        required: list[str] = ["loaded_model"]
        if _requires_clip_runtime(model, clip):
            required.append("loaded_clip")
        required.append("loaded_vae")
        return required

    @classmethod
    def execute(
        cls,
        model: object,
        lora_stack: list[dict[str, str | float]] | None = None,
        clip: object | None = None,
        vae: object | None = None,
        loaded_model: object = _MISSING,
        loaded_clip: object = _MISSING,
        loaded_vae: object = _MISSING,
        apply_lora_stack: object | None = True,
    ) -> c_io.NodeOutput:
        if normalized_model_name_or_none(model) is None:
            raise RuntimeError("model is required")

        lora_stack_items = _to_lora_stack_items(lora_stack)
        cache_key = _cache_key(model, lora_stack_items, clip, vae)

        cached = _load_from_cache(cache_key)
        if cached is not None:
            return c_io.NodeOutput(cached[0][0], cached[0][1], cached[0][2])

        if loaded_model is _MISSING or loaded_model is None:
            raise RuntimeError("loaded_model input is required on cache miss")

        requires_clip_runtime = _requires_clip_runtime(model, clip)
        if requires_clip_runtime and (loaded_clip is _MISSING or loaded_clip is None):
            raise RuntimeError("loaded_clip input is required on cache miss")

        normalized_lora_stack = _to_lora_stack_payload(lora_stack_items)
        runtime_model = loaded_model
        runtime_clip = None if loaded_clip is _MISSING else loaded_clip
        should_apply_lora_stack = _bool_or_default(apply_lora_stack, True)
        if should_apply_lora_stack and normalized_lora_stack:
            runtime_model, runtime_clip = _apply_lora_stack_with_project_node(
                runtime_model,
                runtime_clip,
                normalized_lora_stack,
            )

        runtime_vae = None if loaded_vae is _MISSING else loaded_vae
        _store_cache(cache_key, runtime_model, runtime_clip, runtime_vae)
        return c_io.NodeOutput(runtime_model, runtime_clip, runtime_vae)
