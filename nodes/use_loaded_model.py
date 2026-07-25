# Copyright 2026 kinorax
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

import comfy.model_management as comfy_model_management
from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ..utils.model_lora_metadata_pipeline import get_shared_metadata_pipeline
from ..utils.model_merge import model_runtime_settings_tree
from ..utils.model_runtime_settings import effective_model_runtime_settings
from ..utils.settings import (
    CACHE_RETENTION_FIXED,
    get_use_loaded_model_cache_settings,
)
from ._runtime_loader import (
    cache_descriptor,
    is_checkpoint_model,
    is_diffusion_model,
    normalized_clip_names,
    normalized_model_folder_or_none,
    normalized_model_name_or_none,
)
from .lora_stack_lorader import LoraStackLorader

LORA_STACK_ITEM_NAME_KEY = "name"
LORA_STACK_ITEM_STRENGTH_KEY = "strength"
MODEL_RUNTIME_TYPE = c_io.Custom("MODEL")
CLIP_RUNTIME_TYPE = c_io.Custom("CLIP")
VAE_RUNTIME_TYPE = c_io.Custom("VAE")

_CACHE_LOCK = Lock()
_LAST_CACHE: OrderedDict[str, tuple[object, object | None, object | None]] = OrderedDict()
_LAST_CACHE_BYTES: OrderedDict[str, int] = OrderedDict()
_LAST_CACHE_UNKNOWN: OrderedDict[str, tuple[str, ...]] = OrderedDict()

_MISSING = object()


@dataclass(frozen=True)
class _RuntimeSize:
    bytes: int
    known: bool
    source: str


@dataclass(frozen=True)
class _BundleSize:
    model_bytes: int
    clip_bytes: int
    vae_bytes: int
    total_bytes: int
    unknown_targets: tuple[str, ...]


@dataclass(frozen=True)
class _StoreCacheResult:
    bundle_size: _BundleSize
    retention: str
    max_entries: int
    target_limit: int
    total_memory_bytes: int
    budget_bytes: int
    budget_ratio: float
    recent_total_bytes: int
    final_total_bytes: int
    entries: int
    evicted_entries: tuple[tuple[str, int], ...]
    reason: str


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

    if not is_checkpoint_model(model):
        return {}

    pipeline = get_shared_metadata_pipeline(start=True)
    return effective_model_runtime_settings(
        pipeline.get_model_runtime_settings_by_relative_path(
            folder_name=Const.MODEL_FOLDER_PATH_CHECKPOINTS,
            relative_path=model_name,
        )
    )


def _runtime_settings_tree_for_model(model: object) -> dict[str, object]:
    return model_runtime_settings_tree(
        model,
        lambda payload: _runtime_settings_for_model(payload),
    )


def _cache_key(
    model: object,
    lora_stack_items: list[tuple[str, float]],
    clip: object | None,
    vae: object | None,
    apply_lora_to_clip: object | None,
) -> str:
    payload = {
        "model": cache_descriptor(model),
        "lora_stack": _to_lora_stack_payload(lora_stack_items),
        "runtime_settings": _runtime_settings_tree_for_model(model),
        "apply_lora_to_clip": _bool_or_default(apply_lora_to_clip, True),
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


def _cache_key_label(cache_key: str) -> str:
    return str(cache_key)[:12]


def _model_label(model: object) -> str:
    name = normalized_model_name_or_none(model) or "unknown"
    folder = normalized_model_folder_or_none(model) or "unknown"
    return f"{folder}/{name}"


def _format_bytes(value: int) -> str:
    number = max(0, int(value))
    gib = 1024 ** 3
    mib = 1024 ** 2
    if number >= gib:
        return f"{number / gib:.2f}GiB"
    if number >= mib:
        return f"{number / mib:.1f}MiB"
    return f"{number}B"


def _format_percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{(max(0, int(numerator)) / denominator) * 100:.1f}%"


def _format_ratio_percent(value: float) -> str:
    return f"{max(0.0, float(value)) * 100:.0f}%"


def _cache_log(message: str) -> None:
    settings = get_use_loaded_model_cache_settings()
    if not settings.cache_log_enabled:
        return

    try:
        print(f"[IPT][UseLoadedModel][cache] {message}")
    except Exception:
        pass


def _runtime_size(value: object) -> _RuntimeSize:
    if value is None:
        return _RuntimeSize(0, True, "none")

    get_ram_usage = getattr(value, "get_ram_usage", None)
    if callable(get_ram_usage):
        try:
            return _RuntimeSize(max(0, int(get_ram_usage())), True, "get_ram_usage")
        except Exception:
            return _RuntimeSize(0, False, "get_ram_usage_error")

    model_size = getattr(value, "model_size", None)
    if callable(model_size):
        try:
            return _RuntimeSize(max(0, int(model_size())), True, "model_size")
        except Exception:
            return _RuntimeSize(0, False, "model_size_error")

    return _RuntimeSize(0, False, "unavailable")


def _bundle_size_bytes(model: object, clip: object | None, vae: object | None) -> _BundleSize:
    model_size = _runtime_size(model)
    clip_size = _runtime_size(clip)
    vae_size = _runtime_size(vae)
    unknown_targets = tuple(
        name
        for name, size in (
            ("model", model_size),
            ("clip", clip_size),
            ("vae", vae_size),
        )
        if not size.known
    )
    return _BundleSize(
        model_size.bytes,
        clip_size.bytes,
        vae_size.bytes,
        model_size.bytes + clip_size.bytes + vae_size.bytes,
        unknown_targets,
    )


def _force_single_unknown_targets(unknown_targets: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(target for target in unknown_targets if target == "model")


def _cache_total_bytes(cache_keys: list[str] | None = None) -> int:
    keys = list(_LAST_CACHE.keys()) if cache_keys is None else cache_keys
    seen_runtime_ids: set[int] = set()
    total_bytes = 0
    for key in keys:
        bundle = _LAST_CACHE.get(key)
        if bundle is None:
            continue
        for runtime in bundle:
            if runtime is None:
                continue
            runtime_id = id(runtime)
            if runtime_id in seen_runtime_ids:
                continue
            seen_runtime_ids.add(runtime_id)
            total_bytes += _runtime_size(runtime).bytes
    return total_bytes


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


def _store_last_cache(cache_key: str, model: object, clip: object | None, vae: object | None) -> _StoreCacheResult:
    bundle_size = _bundle_size_bytes(model, clip, vae)
    settings = get_use_loaded_model_cache_settings()
    max_entries = max(1, min(9, int(settings.max_cache_entries)))
    retention = settings.cache_retention
    total_vram_bytes = _total_vram_bytes_or_zero()
    budget_bytes = int(total_vram_bytes * settings.memory_budget_ratio) if total_vram_bytes > 0 else 0
    recent_total_bytes = 0
    reason = "fixed_count"

    evicted_entries: list[tuple[str, int]] = []
    with _CACHE_LOCK:
        _LAST_CACHE[cache_key] = (model, clip, vae)
        _LAST_CACHE_BYTES[cache_key] = bundle_size.total_bytes
        _LAST_CACHE_UNKNOWN[cache_key] = bundle_size.unknown_targets
        _LAST_CACHE.move_to_end(cache_key)
        _LAST_CACHE_BYTES.move_to_end(cache_key)
        _LAST_CACHE_UNKNOWN.move_to_end(cache_key)

        recent_keys = list(_LAST_CACHE.keys())[-max_entries:]
        recent_total_bytes = _cache_total_bytes(recent_keys)
        recent_force_unknown = any(
            _force_single_unknown_targets(_LAST_CACHE_UNKNOWN.get(key, ()))
            for key in recent_keys
        )
        force_unknown_targets = _force_single_unknown_targets(bundle_size.unknown_targets)

        if force_unknown_targets or recent_force_unknown:
            target_limit = 1
            reason = "size_unknown"
        elif retention == CACHE_RETENTION_FIXED:
            target_limit = max_entries
            reason = "fixed_count"
        else:
            if total_vram_bytes <= 0:
                target_limit = 1
                reason = "memory_unavailable"
            else:
                target_limit = _auto_cache_limit(recent_keys, budget_bytes)
                reason = "budget" if target_limit < max_entries else "budget_ok"

        while len(_LAST_CACHE) > target_limit:
            cache_total_before = _cache_total_bytes()
            old_key, _ = _LAST_CACHE.popitem(last=False)
            _LAST_CACHE_BYTES.pop(old_key, None)
            _LAST_CACHE_UNKNOWN.pop(old_key, None)
            evicted_entries.append((old_key, max(0, cache_total_before - _cache_total_bytes())))

        final_total_bytes = _cache_total_bytes()

        return _StoreCacheResult(
            bundle_size=bundle_size,
            retention=retention,
            max_entries=max_entries,
            target_limit=target_limit,
            total_memory_bytes=total_vram_bytes,
            budget_bytes=budget_bytes,
            budget_ratio=settings.memory_budget_ratio,
            recent_total_bytes=recent_total_bytes,
            final_total_bytes=final_total_bytes,
            entries=len(_LAST_CACHE),
            evicted_entries=tuple(evicted_entries),
            reason=reason,
        )


def _auto_cache_limit(recent_keys: list[str], budget_bytes: int) -> int:
    if not recent_keys:
        return 1

    target_limit = 1
    for count, key in enumerate(reversed(recent_keys), start=1):
        selected_keys = recent_keys[-count:]
        next_total = _cache_total_bytes(selected_keys)
        if count > 1 and next_total > budget_bytes:
            break
        target_limit = count
    return max(1, target_limit)


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
    result = _store_last_cache(cache_key, model, clip, vae)
    _log_store_result(cache_key, result)
    return True


def _log_store_result(cache_key: str, result: _StoreCacheResult) -> None:
    size = result.bundle_size
    budget_ratio_text = _format_ratio_percent(result.budget_ratio)
    budget_text = (
        f"{_format_bytes(result.budget_bytes)} ({budget_ratio_text})"
        if result.retention != CACHE_RETENTION_FIXED and result.budget_bytes > 0
        else f"n/a ({budget_ratio_text})"
    )
    total_memory_text = _format_bytes(result.total_memory_bytes) if result.total_memory_bytes > 0 else "n/a"
    size_percent = _format_percent(size.total_bytes, result.total_memory_bytes)
    final_total_percent = _format_percent(result.final_total_bytes, result.total_memory_bytes)
    _cache_log(
        (
            f"store key={_cache_key_label(cache_key)} size={_format_bytes(size.total_bytes)} ({size_percent}) "
            f"cache_total={_format_bytes(result.final_total_bytes)} ({final_total_percent}) "
            f"entries={result.entries} limit={result.target_limit} max_entries={result.max_entries} "
            f"retention={result.retention} reason={result.reason} budget={budget_text} total_memory={total_memory_text}"
        )
    )

    force_unknown_targets = _force_single_unknown_targets(size.unknown_targets)
    non_forcing_unknown_targets = tuple(
        target for target in size.unknown_targets if target not in force_unknown_targets
    )
    if force_unknown_targets:
        _cache_log(
            (
                f"size_unknown key={_cache_key_label(cache_key)} targets={','.join(force_unknown_targets)} "
                "action=force_single_entry"
            )
        )
    elif result.reason == "size_unknown" and result.target_limit == 1 and result.max_entries > 1:
        _cache_log(
            (
                f"force_single_entry key={_cache_key_label(cache_key)} reason=size_unknown "
                f"source=recent_cache max_entries={result.max_entries} retention={result.retention}"
            )
        )
    if non_forcing_unknown_targets:
        _cache_log(
            (
                f"size_unknown key={_cache_key_label(cache_key)} targets={','.join(non_forcing_unknown_targets)} "
                "action=ignored_for_entry_limit"
            )
        )

    if result.target_limit == 1 and result.max_entries > 1 and result.reason != "size_unknown":
        _cache_log(
            (
                f"force_single_entry key={_cache_key_label(cache_key)} reason={result.reason} "
                f"max_entries={result.max_entries} retention={result.retention}"
            )
        )

    for evicted_key, evicted_bytes in result.evicted_entries:
        _cache_log(
            (
                f"evict key={_cache_key_label(evicted_key)} freed={_format_bytes(evicted_bytes)} "
                f"cache_total={_format_bytes(result.final_total_bytes)} ({final_total_percent}) "
                f"reason={result.reason} entries={result.entries} limit={result.target_limit} "
                f"budget={budget_text} total_memory={total_memory_text}"
            )
        )


def _apply_lora_stack_with_project_node(
    model: object,
    clip: object | None,
    lora_stack: list[dict[str, str | float]] | None,
    apply_lora_to_clip: object | None,
) -> tuple[object, object | None]:
    result = LoraStackLorader.execute(
        model=model,
        clip=clip,
        lora_stack=lora_stack,
        apply_lora_to_clip=apply_lora_to_clip,
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
                c_io.Boolean.Input(
                    "apply_lora_to_clip",
                    default=True,
                    tooltip=(
                        "Controls CLIP LoRA application and always participates in the cache key. "
                        "When apply_lora_stack is false, set this to match the externally prepared CLIP."
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
        apply_lora_to_clip: object | None = True,
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
        apply_lora_to_clip: object | None = True,
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
        apply_lora_to_clip: object | None = True,
    ) -> list[str]:
        if normalized_model_name_or_none(model) is None:
            return []

        lora_stack_items = _to_lora_stack_items(lora_stack)
        cache_key = _cache_key(model, lora_stack_items, clip, vae, apply_lora_to_clip)
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
        apply_lora_to_clip: object | None = True,
    ) -> c_io.NodeOutput:
        if normalized_model_name_or_none(model) is None:
            raise RuntimeError("model is required")

        lora_stack_items = _to_lora_stack_items(lora_stack)
        cache_key = _cache_key(model, lora_stack_items, clip, vae, apply_lora_to_clip)

        cached = _load_from_cache(cache_key)
        if cached is not None:
            _cache_log(
                (
                    f"hit key={_cache_key_label(cache_key)} source={cached[1]} "
                    f"model={_model_label(model)} entries={len(_LAST_CACHE)}"
                )
            )
            return c_io.NodeOutput(cached[0][0], cached[0][1], cached[0][2])

        required_inputs = ["loaded_model"]
        if _requires_clip_runtime(model, clip):
            required_inputs.append("loaded_clip")
        required_inputs.append("loaded_vae")
        _cache_log(
            (
                f"miss key={_cache_key_label(cache_key)} model={_model_label(model)} "
                f"required={','.join(required_inputs)} entries={len(_LAST_CACHE)}"
            )
        )

        if loaded_model is _MISSING or loaded_model is None:
            raise RuntimeError("loaded_model input is required on cache miss")

        if "loaded_clip" in required_inputs and (loaded_clip is _MISSING or loaded_clip is None):
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
                apply_lora_to_clip,
            )

        runtime_vae = None if loaded_vae is _MISSING else loaded_vae
        _store_cache(cache_key, runtime_model, runtime_clip, runtime_vae)
        return c_io.NodeOutput(runtime_model, runtime_clip, runtime_vae)
