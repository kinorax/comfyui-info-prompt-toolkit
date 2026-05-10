# Copyright 2026 kinorax
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import folder_paths
import torch
import torch.nn.functional as F
from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast

_OPPAI_DIR_NAME = "oppai_oracle"
_WD14_OPPAI_DIR_PARTS = ("wd14_tagger", "oppai_oracle")
_WD14_OPPAI_REPO_DIR_PARTS = ("wd14_tagger", "oppai_oracle", "Grio43_OppaiOracle")
_MODEL_REPO = "Grio43/OppaiOracle"
_VARIANT_OPTIONS = ("auto", "V1.1", "V1.0")
_VARIANT_SEARCH_ORDER = ("V1.1", "V1.0")
_WEIGHTS_FILE_NAME = "model.safetensors"
_CONFIG_FILE_NAME = "config.json"
_PREPROCESSING_FILE_NAME = "preprocessing.json"
_VOCABULARY_FILE_NAME = "vocabulary.json"
_THRESHOLDS_FILE_NAME = "pr_thresholds.json"
_SORT_ORDER_OPTIONS = ("score", "tag_id")
_DEVICE_OPTIONS = ("auto", "cuda", "cpu")
_RUNTIME_WEIGHT_DTYPE = "float32"
_MODEL_CACHE_LOCK = Lock()
_MODEL_CACHE: dict[tuple[str, str, str], tuple[torch.nn.Module, str]] = {}
_METADATA_CACHE_LOCK = Lock()
_METADATA_CACHE: dict[str, "_BundleMetadata"] = {}


@dataclass(frozen=True)
class _BundleMetadata:
    bundle_root: Path
    weights_path: Path
    config_path: Path
    preprocessing_path: Path
    vocabulary_path: Path
    thresholds_path: Path | None
    config: dict[str, Any]
    preprocessing: dict[str, Any]
    index_to_tag_map: dict[int, str]
    rating_indices: tuple[int, ...]
    pad_index: int
    unk_index: int
    image_size: int
    patch_size: int
    num_labels: int
    model_label: str


class _VitBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        layer_norm_eps: float,
        attention_bias: bool,
        hidden_dropout_prob: float,
    ) -> None:
        super().__init__()
        self.num_attention_heads = int(num_attention_heads)
        self.head_dim = int(hidden_size) // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != int(hidden_size):
            raise RuntimeError("hidden_size must be divisible by num_attention_heads")

        self.norm1 = torch.nn.LayerNorm(int(hidden_size), eps=float(layer_norm_eps))
        self.qkv = torch.nn.Linear(int(hidden_size), int(hidden_size) * 3, bias=bool(attention_bias))
        self.proj = torch.nn.Linear(int(hidden_size), int(hidden_size), bias=True)
        self.norm2 = torch.nn.LayerNorm(int(hidden_size), eps=float(layer_norm_eps))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(int(hidden_size), int(intermediate_size)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(hidden_dropout_prob)),
            torch.nn.Linear(int(intermediate_size), int(hidden_size)),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        batch, token_count, channels = x.shape
        normalized = self.norm1(x)
        qkv = self.qkv(normalized)
        qkv = qkv.reshape(batch, token_count, 3, self.num_attention_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn.float(), dim=-1).to(dtype=query.dtype)
        context = torch.matmul(attn, value)
        context = context.transpose(1, 2).reshape(batch, token_count, channels)
        x = x + self.proj(context)
        x = x + self.mlp(self.norm2(x))
        return x


class _OppaiOracleVit(torch.nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        image_size = int(config.get("image_size", 448))
        patch_size = int(config.get("patch_size", 16))
        num_channels = int(config.get("num_channels", 3))
        hidden_size = int(config.get("hidden_size", 1024))
        num_layers = int(config.get("num_hidden_layers", 18))
        num_heads = int(config.get("num_attention_heads", 16))
        intermediate_size = int(config.get("intermediate_size", 4096))
        num_labels = int(config.get("num_labels", 19294))
        layer_norm_eps = float(config.get("layer_norm_eps", 1e-6))
        attention_bias = bool(config.get("attention_bias", True))
        hidden_dropout_prob = float(config.get("hidden_dropout_prob", 0.0))
        pos_dropout = float(config.get("pos_dropout", 0.0))

        if image_size % patch_size != 0:
            raise RuntimeError("image_size must be divisible by patch_size")

        patch_count = (image_size // patch_size) ** 2
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_embed = torch.nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, patch_count + 1, hidden_size))
        self.pos_drop = torch.nn.Dropout(pos_dropout)
        self.blocks = torch.nn.ModuleList(
            [
                _VitBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    intermediate_size=intermediate_size,
                    layer_norm_eps=layer_norm_eps,
                    attention_bias=attention_bias,
                    hidden_dropout_prob=hidden_dropout_prob,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.tag_head = torch.nn.Linear(hidden_size, num_labels)

    def _token_padding_mask(self, padding_mask: torch.Tensor | None) -> torch.Tensor | None:
        if padding_mask is None:
            return None
        if padding_mask.ndim != 3:
            raise RuntimeError("padding_mask must have shape [B,H,W]")

        patch_mask = F.max_pool2d(
            padding_mask.float().unsqueeze(1),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        patch_mask = patch_mask.squeeze(1).flatten(1).bool()
        if not bool(patch_mask.any().item()):
            return None

        cls_mask = torch.zeros(
            (patch_mask.shape[0], 1),
            dtype=torch.bool,
            device=patch_mask.device,
        )
        return torch.cat((cls_mask, patch_mask), dim=1)

    def forward(self, pixel_values: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        key_padding_mask = self._token_padding_mask(padding_mask)
        for block in self.blocks:
            x = block(x, key_padding_mask)

        x = self.norm(x)
        return self.tag_head(x[:, 0])


def _candidate_bundle_roots(variant: object = "auto") -> tuple[Path, ...]:
    roots: list[Path] = []
    models_dir = getattr(folder_paths, "models_dir", None)
    if isinstance(models_dir, str) and models_dir.strip():
        models_root = Path(models_dir).resolve()
        base_roots = (
            models_root / _OPPAI_DIR_NAME,
            models_root.joinpath(*_WD14_OPPAI_DIR_PARTS),
            models_root.joinpath(*_WD14_OPPAI_REPO_DIR_PARTS),
        )
        variants = _variant_search_order(variant)
        for base_root in base_roots:
            for variant_name in variants:
                roots.append(base_root / variant_name)
            if _normalized_variant(variant) == "auto":
                roots.append(base_root)

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        normalized = str(root).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(root)
    return tuple(deduped)


def _bundle_files(bundle_root: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        bundle_root / _WEIGHTS_FILE_NAME,
        bundle_root / _CONFIG_FILE_NAME,
        bundle_root / _PREPROCESSING_FILE_NAME,
        bundle_root / _VOCABULARY_FILE_NAME,
        bundle_root / _THRESHOLDS_FILE_NAME,
    )


def _find_bundle_root(variant: object = "auto") -> Path | None:
    for bundle_root in _candidate_bundle_roots(variant):
        weights_path, config_path, preprocessing_path, vocabulary_path, _ = _bundle_files(bundle_root)
        if (
            weights_path.is_file()
            and config_path.is_file()
            and preprocessing_path.is_file()
            and vocabulary_path.is_file()
        ):
            return bundle_root.resolve()
    return None


def _describe_search_roots(variant: object = "auto") -> str:
    roots = _candidate_bundle_roots(variant)
    if not roots:
        return "ComfyUI models directory is unavailable"
    return ", ".join(root.as_posix() for root in roots)


def _missing_bundle_message(variant: object = "auto") -> str:
    return (
        "OppaiOracle tagger bundle was not found. "
        f"Search roots: {_describe_search_roots(variant)}. "
        "Required files per directory: "
        f"{_WEIGHTS_FILE_NAME}, {_CONFIG_FILE_NAME}, {_PREPROCESSING_FILE_NAME}, {_VOCABULARY_FILE_NAME}. "
        f"Download the safetensors variant from {_MODEL_REPO}."
    )


def _normalized_variant(value: object) -> str:
    text = str(value).strip() if value is not None else "auto"
    for option in _VARIANT_OPTIONS:
        if text.casefold() == option.casefold():
            return option
    return "auto"


def _variant_search_order(value: object) -> tuple[str, ...]:
    variant = _normalized_variant(value)
    if variant == "auto":
        return _VARIANT_SEARCH_ORDER
    return (variant,)


def _normalized_sort_order(value: object) -> str:
    text = str(value).strip().lower() if value is not None else _SORT_ORDER_OPTIONS[0]
    if text == "tag_id":
        return "tag_id"
    return "score"


def _is_valid_sort_order(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in _SORT_ORDER_OPTIONS or text in {"threshold", "topk"}


def _is_valid_variant(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip().casefold()
    return any(text == option.casefold() for option in _VARIANT_OPTIONS)


def _normalized_device(value: object) -> str:
    text = str(value).strip().lower() if value is not None else "auto"
    if text == "cpu":
        return "cpu"
    if text.startswith("cuda"):
        return "cuda" if _cuda_usable() else "cpu"
    return "cuda" if _cuda_usable() else "cpu"


def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _ = torch.zeros(1, device="cuda").cpu()
    except Exception:
        return False
    return True


def _resolved_weight_dtype(device: str, value: object) -> tuple[torch.dtype, str]:
    text = str(value).strip().lower() if value is not None else "auto"
    if text == "float32":
        return torch.float32, "float32"
    if text == "bfloat16":
        return torch.bfloat16, "bfloat16"

    if device == "cuda":
        is_supported = getattr(torch.cuda, "is_bf16_supported", None)
        try:
            if callable(is_supported) and bool(is_supported()):
                return torch.bfloat16, "bfloat16"
        except Exception:
            pass
    return torch.float32, "float32"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON payload must be an object: {path.as_posix()}")
    return payload


def _int_config(config: dict[str, Any], key: str, default: int) -> int:
    try:
        value = int(config.get(key, default))
    except Exception as exc:
        raise RuntimeError(f"config.{key} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"config.{key} must be positive")
    return value


def _load_bundle_metadata(bundle_root: Path) -> _BundleMetadata:
    cache_key = str(bundle_root.resolve())
    with _METADATA_CACHE_LOCK:
        cached = _METADATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    weights_path, config_path, preprocessing_path, vocabulary_path, thresholds_path = _bundle_files(bundle_root)
    config = _load_json(config_path)
    preprocessing = _load_json(preprocessing_path)
    vocabulary = _load_json(vocabulary_path)

    tag_to_index = vocabulary.get("tag_to_index")
    if not isinstance(tag_to_index, dict) or not tag_to_index:
        raise RuntimeError(f"tag_to_index is missing or malformed: {vocabulary_path.as_posix()}")

    index_to_tag_map: dict[int, str] = {}
    for raw_tag, raw_index in tag_to_index.items():
        if not isinstance(raw_tag, str) or not raw_tag:
            continue
        try:
            index_value = int(raw_index)
        except Exception:
            continue
        index_to_tag_map[index_value] = raw_tag

    if not index_to_tag_map:
        raise RuntimeError(f"tag_to_index did not contain any valid tags: {vocabulary_path.as_posix()}")

    num_labels = _int_config(config, "num_labels", len(index_to_tag_map))
    if len(index_to_tag_map) != num_labels:
        raise RuntimeError(
            "vocabulary size does not match config.num_labels. "
            f"vocabulary={len(index_to_tag_map)}, num_labels={num_labels}"
        )

    image_size = _int_config(config, "image_size", int(preprocessing.get("image_size", 448) or 448))
    patch_size = _int_config(config, "patch_size", int(preprocessing.get("patch_size", 16) or 16))
    pad_index = int(tag_to_index.get("<PAD>", 0))
    unk_index = int(tag_to_index.get("<UNK>", 1))
    rating_indices = tuple(
        sorted(
            index
            for index, tag in index_to_tag_map.items()
            if tag.startswith("rating:")
        )
    )

    metadata = _BundleMetadata(
        bundle_root=bundle_root,
        weights_path=weights_path,
        config_path=config_path,
        preprocessing_path=preprocessing_path,
        vocabulary_path=vocabulary_path,
        thresholds_path=thresholds_path if thresholds_path.is_file() else None,
        config=config,
        preprocessing=preprocessing,
        index_to_tag_map=index_to_tag_map,
        rating_indices=rating_indices,
        pad_index=pad_index,
        unk_index=unk_index,
        image_size=image_size,
        patch_size=patch_size,
        num_labels=num_labels,
        model_label=bundle_root.name,
    )
    with _METADATA_CACHE_LOCK:
        _METADATA_CACHE[cache_key] = metadata
    return metadata


def _load_safetensors(weights_path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OppaiOracle Tagger runtime is unavailable because `safetensors` is missing. "
            "ComfyUI normally includes it; reinstall ComfyUI requirements and restart."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"OppaiOracle Tagger runtime failed to import safetensors: {exc}") from exc

    try:
        state_dict = load_file(str(weights_path), device="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load OppaiOracle safetensors: {exc}") from exc
    return dict(state_dict)


def _build_model(metadata: _BundleMetadata, device: str, weight_dtype: str) -> tuple[torch.nn.Module, str]:
    dtype, dtype_label = _resolved_weight_dtype(device, weight_dtype)
    model = _OppaiOracleVit(metadata.config)
    model = model.to(dtype=dtype)
    state_dict = _load_safetensors(metadata.weights_path)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load OppaiOracle tagger weights: {exc}") from exc

    model = model.to(device)
    model.eval()
    return model, dtype_label


def _load_model(metadata: _BundleMetadata, device: str, weight_dtype: str) -> tuple[torch.nn.Module, str]:
    cache_key = (str(metadata.bundle_root.resolve()), device, str(weight_dtype))
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    loaded = _build_model(metadata, device, weight_dtype)
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[cache_key] = loaded
    return loaded


def _flatten_image_inputs(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_image_inputs(item))
        return flattened
    return [value]


def _normalized_image_batches(image: Any) -> list[torch.Tensor]:
    values = _flatten_image_inputs(image)
    if len(values) == 0:
        raise RuntimeError("image input must be a ComfyUI IMAGE tensor")

    batches: list[torch.Tensor] = []
    for value in values:
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("image input must be a ComfyUI IMAGE tensor")
        if value.ndim == 3:
            batches.append(value.unsqueeze(0))
            continue
        if value.ndim == 4:
            batches.append(value)
            continue
        raise RuntimeError("image input must have shape [B,H,W,C] or [H,W,C]")
    return batches


def _interpolate_nchw(image_nchw: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    try:
        return F.interpolate(
            image_nchw,
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    except TypeError:
        return F.interpolate(
            image_nchw,
            size=size,
            mode="bilinear",
            align_corners=False,
        )


def _metadata_vector(
    metadata: _BundleMetadata,
    key: str,
    default: tuple[float, float, float],
    *,
    device: torch.device,
) -> torch.Tensor:
    value = metadata.preprocessing.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        value = default
    return torch.tensor([float(v) for v in value], dtype=torch.float32, device=device).view(3, 1, 1)


def _pad_color_tensor(metadata: _BundleMetadata, *, device: torch.device) -> torch.Tensor:
    value = metadata.preprocessing.get("pad_color_rgb", (114, 114, 114))
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        value = (114, 114, 114)
    return torch.tensor([float(v) / 255.0 for v in value], dtype=torch.float32, device=device).view(3, 1, 1)


def _preprocess_single_image(image_hwc: torch.Tensor, metadata: _BundleMetadata) -> tuple[torch.Tensor, torch.Tensor]:
    if image_hwc.ndim != 3 or image_hwc.shape[-1] < 3:
        raise RuntimeError("image input must have shape [H,W,C] with at least 3 channels")

    source = image_hwc.detach().float()
    device = source.device
    image_size = int(metadata.image_size)
    height = int(source.shape[0])
    width = int(source.shape[1])
    if height <= 0 or width <= 0:
        raise RuntimeError("image input must have positive height and width")

    pad_color = _pad_color_tensor(metadata, device=device)
    rgb = source[..., :3].clamp(0.0, 1.0).permute(2, 0, 1).contiguous()
    if source.shape[-1] >= 4:
        alpha = source[..., 3:4].clamp(0.0, 1.0).permute(2, 0, 1).contiguous()
        rgb = rgb * alpha + pad_color * (1.0 - alpha)

    scale = min(image_size / float(width), image_size / float(height), 1.0)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = _interpolate_nchw(rgb.unsqueeze(0), (new_height, new_width)).squeeze(0)

    canvas = pad_color.expand(3, image_size, image_size).clone()
    padding_mask = torch.ones((image_size, image_size), dtype=torch.bool, device=device)
    top = (image_size - new_height) // 2
    left = (image_size - new_width) // 2
    canvas[:, top : top + new_height, left : left + new_width] = resized
    padding_mask[top : top + new_height, left : left + new_width] = False

    mean = _metadata_vector(metadata, "normalize_mean", (0.5, 0.5, 0.5), device=device)
    std = _metadata_vector(metadata, "normalize_std", (0.5, 0.5, 0.5), device=device)
    pixel_values = canvas.sub(mean).div(std)
    return pixel_values, padding_mask


def _preprocess_image_batch(image_batch: torch.Tensor, metadata: _BundleMetadata) -> tuple[torch.Tensor, torch.Tensor]:
    if image_batch.shape[-1] < 3:
        raise RuntimeError("image input must have at least 3 channels")

    pixel_values: list[torch.Tensor] = []
    padding_masks: list[torch.Tensor] = []
    for item in image_batch:
        pixel_value, padding_mask = _preprocess_single_image(item, metadata)
        pixel_values.append(pixel_value)
        padding_masks.append(padding_mask)

    return torch.stack(pixel_values, dim=0), torch.stack(padding_masks, dim=0)


def _preprocess_image_input(image: Any, metadata: _BundleMetadata) -> tuple[torch.Tensor, torch.Tensor]:
    batches = _normalized_image_batches(image)
    pixel_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    for batch in batches:
        pixel_values, padding_mask = _preprocess_image_batch(batch, metadata)
        pixel_batches.append(pixel_values)
        mask_batches.append(padding_mask)

    if len(pixel_batches) == 1:
        return pixel_batches[0], mask_batches[0]
    return torch.cat(pixel_batches, dim=0), torch.cat(mask_batches, dim=0)


def _argsort_descending(values: torch.Tensor) -> torch.Tensor:
    try:
        return torch.argsort(values, descending=True, stable=True)
    except TypeError:
        return torch.argsort(values, descending=True)


def _format_output_tag(tag: str) -> str:
    output = tag.replace("_", " ")
    output = output.replace("\\", "\\\\")
    output = output.replace("(", r"\(")
    output = output.replace(")", r"\)")
    return output


def _rating_from_probs(probs: torch.Tensor, metadata: _BundleMetadata) -> tuple[str, float] | tuple[None, None]:
    if not metadata.rating_indices:
        return None, None
    best_tag: str | None = None
    best_score = -1.0
    for index in metadata.rating_indices:
        score = float(probs[index].detach().float().cpu())
        if score > best_score:
            raw_tag = metadata.index_to_tag_map.get(index)
            best_tag = raw_tag.split(":", 1)[1] if isinstance(raw_tag, str) and ":" in raw_tag else raw_tag
            best_score = score
    if best_tag is None:
        return None, None
    return best_tag, best_score


def _selected_indices_and_scores(
    probs: torch.Tensor,
    metadata: _BundleMetadata,
    *,
    threshold: float,
    max_tags: int,
) -> tuple[list[int], list[float]]:
    probs = probs.reshape(-1)
    score_values = probs.detach()
    tag_limit = max(0, int(max_tags))
    if tag_limit == 0:
        return [], []

    skip_indices = {metadata.pad_index, metadata.unk_index, *metadata.rating_indices}

    selected_mask = score_values >= float(threshold)
    for index in skip_indices:
        if 0 <= index < selected_mask.numel():
            selected_mask[index] = False
    indices = selected_mask.nonzero(as_tuple=True)[0]
    scores = score_values.index_select(0, indices)
    if scores.numel() > 0:
        order = _argsort_descending(scores)
        indices = indices.index_select(0, order)
        scores = scores.index_select(0, order)
    indices = indices[:tag_limit]
    scores = scores[:tag_limit]

    return (
        [int(index) for index in indices.detach().cpu().tolist()],
        [float(score) for score in scores.detach().float().cpu().tolist()],
    )


def _ordered_index_score_pairs(
    indices: list[int],
    scores: list[float],
    *,
    sort_order: str,
) -> list[tuple[int, float]]:
    pairs = list(zip(indices, scores))
    if _normalized_sort_order(sort_order) == "tag_id":
        pairs.sort(key=lambda item: item[0])
    else:
        pairs.sort(key=lambda item: (-item[1], item[0]))
    return pairs


def _result_item(
    probs: torch.Tensor,
    metadata: _BundleMetadata,
    *,
    sort_order: str,
    threshold: float,
    max_tags: int,
) -> dict[str, Any]:
    indices, scores = _selected_indices_and_scores(
        probs,
        metadata,
        threshold=threshold,
        max_tags=max_tags,
    )

    tags: list[str] = []
    raw_tags: list[str] = []
    tag_scores: dict[str, float] = {}
    raw_tag_scores: dict[str, float] = {}
    for index, score in _ordered_index_score_pairs(indices, scores, sort_order=sort_order):
        raw_tag = metadata.index_to_tag_map.get(index)
        if raw_tag is None:
            continue
        formatted_tag = _format_output_tag(raw_tag)
        raw_tags.append(raw_tag)
        tags.append(formatted_tag)
        tag_scores[formatted_tag] = score
        raw_tag_scores[raw_tag] = score

    rating, rating_score = _rating_from_probs(probs, metadata)
    return {
        "tags": tags,
        "raw_tags": raw_tags,
        "tag_scores": tag_scores,
        "raw_tag_scores": raw_tag_scores,
        "rating": rating,
        "rating_score": rating_score,
    }


def _prompt_text_from_tag_batches(tag_batches: list[list[str]]) -> str:
    lines = [", ".join(tags) for tags in tag_batches]
    if len(lines) == 1:
        return lines[0]
    return "\n".join(lines)


def _rating_text_from_items(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items:
        rating = item.get("rating")
        lines.append(str(rating) if rating is not None else "")
    if len(lines) == 1:
        return lines[0]
    return "\n".join(lines)


class OppaiOracleTagger(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-OppaiOracleTagger",
            display_name="OppaiOracle Tagger",
            category=Const.CATEGORY_PROMPT,
            description=(
                "Runs the local OppaiOracle safetensors bundle from ComfyUI models/oppai_oracle/ "
                "without onnxruntime and returns prompt-ready anime tags."
            ),
            inputs=[
                c_io.Image.Input(
                    "image",
                    tooltip="Input image batch to classify with the local OppaiOracle tagger bundle.",
                ),
                c_io.Combo.Input(
                    "variant",
                    options=_VARIANT_OPTIONS,
                    default=_VARIANT_OPTIONS[0],
                    tooltip="Model variant to load. auto prefers V1.1, then V1.0.",
                ),
                c_io.Combo.Input(
                    "sort_order",
                    options=_SORT_ORDER_OPTIONS,
                    default=_SORT_ORDER_OPTIONS[0],
                    tooltip="Order for selected tags. score keeps highest-confidence tags first; tag_id uses vocabulary order.",
                ),
                c_io.Float.Input(
                    "threshold",
                    default=0.75,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Only tags with this score or higher are considered.",
                ),
                c_io.Int.Input(
                    "max_tags",
                    default=45,
                    min=0,
                    max=4096,
                    tooltip="Maximum tags returned per image. Selection is score-based before sort_order is applied.",
                ),
                c_io.Combo.Input(
                    "device",
                    options=_DEVICE_OPTIONS,
                    default=_DEVICE_OPTIONS[0],
                    tooltip="Execution device for the local OppaiOracle runtime.",
                    advanced=True,
                ),
            ],
            outputs=[
                c_io.String.Output(
                    Cast.out_id("tags"),
                    display_name="tags",
                ),
                c_io.String.Output(
                    Cast.out_id("rating"),
                    display_name="rating",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        image: Any = None,
        variant: object = "auto",
        sort_order: object = "score",
        threshold: object = 0.75,
        max_tags: object = 45,
        device: object = "auto",
        mode: object | None = None,
        weight_dtype: object | None = None,
    ) -> bool | str:
        try:
            threshold_value = float(threshold)
            max_tags_value = int(max_tags)
        except Exception:
            return "threshold and max_tags must be numeric"

        if not 0.0 <= threshold_value <= 1.0:
            return "threshold must be between 0.0 and 1.0"
        if max_tags_value < 0:
            return "max_tags must be 0 or greater"
        if not _is_valid_sort_order(sort_order):
            return f"sort_order must be one of: {', '.join(_SORT_ORDER_OPTIONS)}"
        if not _is_valid_variant(variant):
            return f"variant must be one of: {', '.join(_VARIANT_OPTIONS)}"

        if _find_bundle_root(variant) is not None:
            return True
        return _missing_bundle_message(variant)

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        variant: str,
        sort_order: str,
        threshold: float,
        max_tags: int,
        device: str,
        mode: str | None = None,
        weight_dtype: str | None = None,
    ) -> c_io.NodeOutput:
        bundle_root = _find_bundle_root(variant)
        if bundle_root is None:
            raise RuntimeError(_missing_bundle_message(variant))

        metadata = _load_bundle_metadata(bundle_root)
        runtime_device = _normalized_device(device)
        model, dtype_label = _load_model(metadata, runtime_device, _RUNTIME_WEIGHT_DTYPE)

        pixel_values, padding_mask = _preprocess_image_input(image, metadata)
        model_dtype = next(model.parameters()).dtype
        pixel_values = pixel_values.to(device=runtime_device, dtype=model_dtype)
        padding_mask = padding_mask.to(device=runtime_device)

        with torch.inference_mode():
            logits_batch = model(pixel_values, padding_mask=padding_mask)
            probs_batch = torch.sigmoid(logits_batch)

        sort_order_value = _normalized_sort_order(sort_order)
        items: list[dict[str, Any]] = []
        tag_batches: list[list[str]] = []
        for probs in probs_batch:
            item = _result_item(
                probs,
                metadata,
                sort_order=sort_order_value,
                threshold=float(threshold),
                max_tags=int(max_tags),
            )
            items.append(item)
            tag_batches.append(list(item.get("tags", ())))

        result_payload = {
            "items": items,
            "_params": {
                "variant": _normalized_variant(variant),
                "sort_order": sort_order_value,
                "threshold": float(threshold),
                "max_tags": int(max_tags),
                "device": runtime_device,
                "weight_dtype": dtype_label,
                "model_root": metadata.bundle_root.as_posix(),
                "model_label": metadata.model_label,
                "image_size": metadata.image_size,
            },
        }
        # Uncomment to inspect the full OppaiOracle tagger score payload in the console.
        # print(json.dumps(result_payload, ensure_ascii=True))

        return c_io.NodeOutput(
            _prompt_text_from_tag_batches(tag_batches),
            _rating_text_from_items(items),
        )
