from __future__ import annotations

import importlib
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

from .. import const as Const

CHECKPOINT_LOADER_KEYS: tuple[str, ...] = ("CheckpointLoaderSimple",)
DIFFUSION_MODEL_LOADER_KEYS: tuple[str, ...] = ("UNETLoader",)
VAE_LOADER_KEYS: tuple[str, ...] = ("VAELoader",)
CLIP_SET_LAST_LAYER_KEYS: tuple[str, ...] = ("CLIPSetLastLayer",)
MODEL_SAMPLING_SD3_KEYS: tuple[str, ...] = ("ModelSamplingSD3",)
CLIP_LOADER_KEYS_BY_COUNT: dict[int, tuple[str, ...]] = {
    1: ("CLIPLoader",),
    2: ("DualCLIPLoader",),
    3: ("TripleCLIPLoader",),
    4: ("QuadrupleCLIPLoader",),
}


def normalized_model_payload_or_none(value: object) -> dict[str, object] | None:
    if isinstance(value, dict) and "__value__" in value:
        return normalized_model_payload_or_none(value.get("__value__"))
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return normalized_model_payload_or_none(value[0])
    return None


def normalized_model_name_or_none(value: object) -> str | None:
    payload = normalized_model_payload_or_none(value)
    if payload is not None:
        name = payload.get(Const.MODEL_VALUE_NAME_KEY)
    else:
        name = value
    return _normalized_optional_text(name)


def normalized_model_folder_or_none(value: object) -> str | None:
    payload = normalized_model_payload_or_none(value)
    if payload is None:
        return None
    return _normalized_optional_text(payload.get(Const.MODEL_VALUE_FOLDER_PATHS_KEY))


def normalized_model_weight_dtype_or_none(value: object) -> str | None:
    payload = normalized_model_payload_or_none(value)
    if payload is None:
        return None
    return _normalized_optional_text(payload.get(Const.MODEL_VALUE_WEIGHT_DTYPE_KEY))


def is_checkpoint_model(value: object) -> bool:
    return normalized_model_folder_or_none(value) == Const.MODEL_FOLDER_PATH_CHECKPOINTS


def is_diffusion_model(value: object) -> bool:
    return normalized_model_folder_or_none(value) == Const.MODEL_FOLDER_PATH_DIFFUSION_MODELS


def normalized_clip_payload_or_none(value: object) -> dict[str, object] | None:
    if isinstance(value, dict) and "__value__" in value:
        return normalized_clip_payload_or_none(value.get("__value__"))
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return normalized_clip_payload_or_none(value[0])
    return None


def normalized_clip_names(value: object) -> list[str]:
    payload = normalized_clip_payload_or_none(value)
    if payload is None:
        return []
    raw_names = payload.get(Const.CLIP_VALUE_NAMES_KEY)
    if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes, bytearray)):
        return []
    output: list[str] = []
    for raw_name in raw_names:
        normalized_name = _normalized_optional_text(raw_name)
        if normalized_name is None:
            return []
        output.append(normalized_name)
    return output


def normalized_clip_type_or_none(value: object) -> str | None:
    payload = normalized_clip_payload_or_none(value)
    if payload is None:
        return None
    return _normalized_optional_text(payload.get(Const.MODEL_VALUE_TYPE_KEY))


def normalized_clip_device_or_none(value: object) -> str | None:
    payload = normalized_clip_payload_or_none(value)
    if payload is None:
        return None
    return _normalized_optional_text(payload.get(Const.MODEL_VALUE_DEVICE_KEY))


def normalized_clip_last_layer_or_none(value: object) -> int | None:
    payload = normalized_clip_payload_or_none(value)
    if payload is None:
        return None
    return _coerce_int(payload.get(Const.CLIP_VALUE_STOP_AT_CLIP_LAYER_KEY))


def normalized_vae_name_or_none(value: object) -> str | None:
    if isinstance(value, dict) and "__value__" in value:
        return normalized_vae_name_or_none(value.get("__value__"))
    return _normalized_optional_text(value)


def core_nodes_module_or_none() -> object | None:
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


def loader_class_or_none(module: object, keys: tuple[str, ...]) -> type | None:
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


def loader_method_or_none(
    loader_instance: object,
    fallback_names: Iterable[str] = (),
) -> Callable[..., object] | None:
    function_name = getattr(loader_instance, "FUNCTION", None)
    if isinstance(function_name, str):
        fn = getattr(loader_instance, function_name, None)
        if callable(fn):
            return fn

    for name in fallback_names:
        fn = getattr(loader_instance, name, None)
        if callable(fn):
            return fn
    return None


def load_checkpoint_with_core_loader(loader_instance: object, checkpoint_name: str) -> tuple[object, object, object]:
    method = loader_method_or_none(loader_instance, ("load_checkpoint",))
    if method is None:
        raise RuntimeError("Failed to resolve Load Checkpoint function")

    result = _call_with_variants(
        method,
        [
            ((checkpoint_name,), {}),
            ((), {"ckpt_name": checkpoint_name}),
            ((), {"checkpoint_name": checkpoint_name}),
        ],
    )
    if not isinstance(result, (list, tuple)) or len(result) < 3:
        raise RuntimeError("Load Checkpoint returned unexpected outputs")
    return result[0], result[1], result[2]


def load_diffusion_model_with_core_loader(
    loader_instance: object,
    model_name: str,
    weight_dtype: str | None = None,
) -> object:
    method = loader_method_or_none(loader_instance, ("load_unet", "load_model", "load_diffusion_model"))
    if method is None:
        raise RuntimeError("Failed to resolve Load Diffusion Model function")

    variants: list[tuple[tuple[object, ...], dict[str, object]]] = []
    if weight_dtype is not None:
        variants.extend(
            [
                ((model_name, weight_dtype), {}),
                ((model_name,), {"weight_dtype": weight_dtype}),
                ((), {"unet_name": model_name, "weight_dtype": weight_dtype}),
                ((), {"model_name": model_name, "weight_dtype": weight_dtype}),
            ]
        )
    variants.extend(
        [
            ((model_name,), {}),
            ((), {"unet_name": model_name}),
            ((), {"model_name": model_name}),
        ]
    )
    result = _call_with_variants(method, variants)
    return _first_result_value(result)


def load_clip_with_core_loader(
    loader_instance: object,
    clip_names: Sequence[str],
    clip_type: str | None = None,
    device: str | None = None,
) -> object:
    method = loader_method_or_none(loader_instance, ("load_clip", "load"))
    if method is None:
        raise RuntimeError("Failed to resolve Load CLIP function")

    count = len(clip_names)
    if count < 1 or count > 4:
        raise RuntimeError(f"Unsupported CLIP count: {count}")

    variants: list[tuple[tuple[object, ...], dict[str, object]]] = []
    if count == 1:
        name = clip_names[0]
        if clip_type is not None and device is not None:
            variants.append(((name, clip_type, device), {}))
            variants.append(((), {"clip_name": name, "type": clip_type, "device": device}))
        if clip_type is not None:
            variants.append(((name, clip_type), {}))
            variants.append(((), {"clip_name": name, "type": clip_type}))
        if device is not None:
            variants.append(((), {"clip_name": name, "device": device}))
        variants.append(((name,), {}))
        variants.append(((), {"clip_name": name}))
    elif count == 2:
        name1, name2 = clip_names
        if clip_type is not None and device is not None:
            variants.append(((name1, name2, clip_type, device), {}))
            variants.append(((), {"clip_name1": name1, "clip_name2": name2, "type": clip_type, "device": device}))
        if clip_type is not None:
            variants.append(((name1, name2, clip_type), {}))
            variants.append(((), {"clip_name1": name1, "clip_name2": name2, "type": clip_type}))
        variants.append(((name1, name2), {}))
        variants.append(((), {"clip_name1": name1, "clip_name2": name2}))
    elif count == 3:
        name1, name2, name3 = clip_names
        variants.extend(
            [
                ((name1, name2, name3), {}),
                ((), {"clip_name1": name1, "clip_name2": name2, "clip_name3": name3}),
            ]
        )
    else:
        name1, name2, name3, name4 = clip_names
        variants.extend(
            [
                ((name1, name2, name3, name4), {}),
                (
                    (),
                    {
                        "clip_name1": name1,
                        "clip_name2": name2,
                        "clip_name3": name3,
                        "clip_name4": name4,
                    },
                ),
            ]
        )

    result = _call_with_variants(method, variants)
    return _first_result_value(result)


def load_vae_with_core_loader(loader_instance: object, vae_name: str) -> object:
    method = loader_method_or_none(loader_instance, ("load_vae",))
    if method is None:
        raise RuntimeError("Failed to resolve Load VAE function")

    result = _call_with_variants(
        method,
        [
            ((vae_name,), {}),
            ((), {"vae_name": vae_name}),
        ],
    )
    return _first_result_value(result)


def apply_clip_last_layer_with_core_node(loader_instance: object, clip: object, stop_at_clip_layer: int) -> object:
    method = loader_method_or_none(loader_instance, ("set_last_layer", "patch"))
    if method is None:
        raise RuntimeError("Failed to resolve CLIP Set Last Layer function")

    result = _call_with_variants(
        method,
        [
            ((clip, stop_at_clip_layer), {}),
            ((), {"clip": clip, "stop_at_clip_layer": stop_at_clip_layer}),
        ],
    )
    return _first_result_value(result)


def apply_model_sampling_sd3_with_core_node(loader_instance: object, model: object, shift: float) -> object:
    method = loader_method_or_none(loader_instance, ("patch", "apply", "model_sampling"))
    if method is None:
        raise RuntimeError("Failed to resolve ModelSamplingSD3 function")

    result = _call_with_variants(
        method,
        [
            ((model, shift), {}),
            ((), {"model": model, "shift": shift}),
        ],
    )
    return _first_result_value(result)


def cache_descriptor(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str):
            normalized = _normalized_optional_text(value)
            return normalized
        return value
    if isinstance(value, Mapping):
        normalized_map: dict[str, object] = {}
        for key in sorted(str(item) for item in value.keys()):
            normalized_map[key] = cache_descriptor(value.get(key))
        return normalized_map
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [cache_descriptor(item) for item in value]
    return {"runtime_id": id(value)}


def _call_with_variants(
    method: Callable[..., object],
    variants: Sequence[tuple[tuple[object, ...], dict[str, object]]],
) -> object:
    last_error: Exception | None = None
    for args, kwargs in variants:
        try:
            return method(*args, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("No loader invocation variants were provided")


def _first_result_value(result: object) -> object:
    if isinstance(result, (list, tuple)):
        return result[0] if result else None
    return result


def _normalized_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "None":
        return None
    return text


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)
    except Exception:
        return None
