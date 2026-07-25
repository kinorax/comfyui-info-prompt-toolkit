# Copyright 2026 kinorax
from __future__ import annotations

import importlib
import sys
from typing import Callable

from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast
from ._clip_runtime_cache import mark_clip_lora_applied

LORA_STACK_ITEM_NAME_KEY = "name"
LORA_STACK_ITEM_STRENGTH_KEY = "strength"
MODEL_RUNTIME_TYPE = c_io.Custom("MODEL")
CLIP_RUNTIME_TYPE = c_io.Custom("CLIP")

MODEL_AND_CLIP_LOADER_KEYS: tuple[str, ...] = ("LoraLoader",)
MODEL_ONLY_LOADER_KEYS: tuple[str, ...] = (
    "LoraLoaderModelOnly",
    "LoraLoaderOnlyModel",
)


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


def _core_nodes_module_or_none() -> object | None:
    module = sys.modules.get("nodes")
    if module is not None and isinstance(getattr(module, "NODE_CLASS_MAPPINGS", None), dict):
        return module

    try:
        module = importlib.import_module("nodes")
    except Exception:
        return None

    if not isinstance(getattr(module, "NODE_CLASS_MAPPINGS", None), dict):
        return None
    return module


def _loader_class_or_none(module: object, keys: tuple[str, ...]) -> type | None:
    mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
    if isinstance(mappings, dict):
        for key in keys:
            value = mappings.get(key)
            if isinstance(value, type):
                return value

    for key in keys:
        value = getattr(module, key, None)
        if isinstance(value, type):
            return value
    return None


def _loader_method_or_none(loader_instance: object) -> Callable[..., object] | None:
    function_name = getattr(loader_instance, "FUNCTION", None)
    if isinstance(function_name, str):
        fn = getattr(loader_instance, function_name, None)
        if callable(fn):
            return fn

    for name in ("load_lora", "load_lora_model_only"):
        fn = getattr(loader_instance, name, None)
        if callable(fn):
            return fn
    return None


def _apply_with_model_and_clip_loader(
    loader_instance: object,
    model: object,
    clip: object | None,
    lora_name: str,
    strength: float,
) -> tuple[object, object | None]:
    method = _loader_method_or_none(loader_instance)
    if method is None:
        raise RuntimeError("Failed to resolve Load LoRA (Model and CLIP) function")

    try:
        result = method(model, clip, lora_name, strength, strength)
    except TypeError:
        result = method(
            model=model,
            clip=clip,
            lora_name=lora_name,
            strength_model=strength,
            strength_clip=strength,
        )

    if isinstance(result, (list, tuple)):
        if len(result) >= 2:
            return result[0], result[1]
        if len(result) == 1:
            return result[0], clip
    return result, clip


def _apply_with_model_only_loader(
    loader_instance: object,
    model: object,
    lora_name: str,
    strength: float,
) -> object:
    method = _loader_method_or_none(loader_instance)
    if method is None:
        raise RuntimeError("Failed to resolve Load LoRA function")

    try:
        result = method(model, lora_name, strength)
    except TypeError:
        try:
            result = method(
                model=model,
                lora_name=lora_name,
                strength_model=strength,
            )
        except TypeError:
            result = method(
                model=model,
                lora_name=lora_name,
                strength=strength,
            )

    if isinstance(result, (list, tuple)):
        return result[0] if result else model
    return model if result is None else result


class LoraStackLorader(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        socket_force_input = {"forceInput": True}
        return c_io.Schema(
            node_id="IPT-LoraStackLorader",
            display_name="Lora Stack Lorader",
            category=Const.CATEGORY_IMAGEINFO,
            inputs=[
                MODEL_RUNTIME_TYPE.Input(
                    "model",
                    tooltip="Loaded model to apply lora_stack",
                    extra_dict=socket_force_input,
                ),
                CLIP_RUNTIME_TYPE.Input(
                    "clip",
                    optional=True,
                    tooltip="Loaded CLIP (optional)",
                    extra_dict=socket_force_input,
                ),
                Const.LORA_STACK_TYPE.Input(
                    Const.IMAGEINFO_LORA_STACK,
                    optional=True,
                ),
                c_io.Boolean.Input(
                    "apply_lora_to_clip",
                    default=True,
                    tooltip="If false, apply every LoRA to the model only and keep CLIP unchanged",
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
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        model: object | None = None,
        clip: object | None = None,
        lora_stack: object | None = None,
        apply_lora_to_clip: object | None = True,
    ) -> bool | str:
        # ComfyUI can invoke custom validation with placeholder/None values
        # during graph validation. Hard-failing here causes false negatives.
        return True

    @classmethod
    def execute(
        cls,
        model: object,
        clip: object | None = None,
        lora_stack: list[dict[str, str | float]] | None = None,
        apply_lora_to_clip: object | None = True,
    ) -> c_io.NodeOutput:
        stack_items = _to_lora_stack_items(lora_stack)
        if not stack_items:
            return c_io.NodeOutput(model, clip)

        core_nodes_module = _core_nodes_module_or_none()
        if core_nodes_module is None:
            raise RuntimeError("ComfyUI core nodes module is unavailable")

        model_and_clip_loader_class = _loader_class_or_none(core_nodes_module, MODEL_AND_CLIP_LOADER_KEYS)
        model_only_loader_class = _loader_class_or_none(core_nodes_module, MODEL_ONLY_LOADER_KEYS)

        model_and_clip_loader = model_and_clip_loader_class() if model_and_clip_loader_class else None
        model_only_loader = model_only_loader_class() if model_only_loader_class else None

        current_model = model
        current_clip = clip
        should_apply_lora_to_clip = _bool_or_default(apply_lora_to_clip, True)

        for lora_name, strength in stack_items:
            if current_clip is None or not should_apply_lora_to_clip:
                if model_only_loader is not None:
                    current_model = _apply_with_model_only_loader(
                        model_only_loader,
                        current_model,
                        lora_name,
                        strength,
                    )
                    continue

                if model_and_clip_loader is not None:
                    current_model, _ = _apply_with_model_and_clip_loader(
                        model_and_clip_loader,
                        current_model,
                        None,
                        lora_name,
                        strength,
                    )
                    continue

                raise RuntimeError("Load LoRA node is unavailable")

            if model_and_clip_loader is None:
                raise RuntimeError("Load LoRA (Model and CLIP) node is unavailable")

            input_clip = current_clip
            current_model, current_clip = _apply_with_model_and_clip_loader(
                model_and_clip_loader,
                current_model,
                current_clip,
                lora_name,
                strength,
            )
            if strength != 0.0:
                mark_clip_lora_applied(input_clip, current_clip)

        return c_io.NodeOutput(current_model, current_clip)
