# Copyright 2026 kinorax
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import folder_paths
import torch
import torch.nn.functional as F
from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast

_PIXAI_DIR_NAME = "pixai_tagger"
_WD14_PIXAI_DIR_PARTS = ("wd14_tagger", "pixai", "pixai-labs_pixai-tagger-v0.9")
_WEIGHTS_FILE_NAME = "model_v0.9.pth"
_TAGS_FILE_NAME = "tags_v0.9_13k.json"
_MAPPING_FILE_NAME = "char_ip_map.json"
_IMAGE_SIZE = 448
_MODE_OPTIONS = ("threshold", "topk")
_DEVICE_OPTIONS = ("auto", "cuda", "cpu")
_MODEL_NAME_CANDIDATES = (
    "eva02_large_patch14_448",
    "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
    "eva02_large_patch14_448.mim_in22k_ft_in22k_in1k",
    "eva02_large_patch14_448.mim_in22k_ft_in1k",
    "eva02_large_patch14_448.mim_m38m_ft_in1k",
)
_MODEL_CACHE_LOCK = Lock()
_MODEL_CACHE: dict[tuple[str, str], tuple[torch.nn.Module, str]] = {}
_METADATA_CACHE_LOCK = Lock()
_METADATA_CACHE: dict[str, "_BundleMetadata"] = {}


@dataclass(frozen=True)
class _BundleMetadata:
    bundle_root: Path
    weights_path: Path
    tags_path: Path
    mapping_path: Path
    index_to_tag_map: dict[int, str]
    gen_tag_count: int
    character_tag_count: int
    character_ip_mapping: dict[str, tuple[str, ...]]
    tag_count: int


class _TaggingHead(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.head = torch.nn.Sequential(torch.nn.Linear(int(input_dim), int(num_classes)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(x)
        return torch.sigmoid(logits)


def _candidate_bundle_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    models_dir = getattr(folder_paths, "models_dir", None)
    if isinstance(models_dir, str) and models_dir.strip():
        models_root = Path(models_dir).resolve()
        roots.append(models_root / _PIXAI_DIR_NAME)
        roots.append(models_root.joinpath(*_WD14_PIXAI_DIR_PARTS))

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        normalized = str(root).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(root)
    return tuple(deduped)


def _bundle_files(bundle_root: Path) -> tuple[Path, Path, Path]:
    return (
        bundle_root / _WEIGHTS_FILE_NAME,
        bundle_root / _TAGS_FILE_NAME,
        bundle_root / _MAPPING_FILE_NAME,
    )


def _find_bundle_root() -> Path | None:
    for bundle_root in _candidate_bundle_roots():
        weights_path, tags_path, mapping_path = _bundle_files(bundle_root)
        if weights_path.is_file() and tags_path.is_file() and mapping_path.is_file():
            return bundle_root
    return None


def _describe_search_roots() -> str:
    roots = _candidate_bundle_roots()
    if not roots:
        return "ComfyUI models directory is unavailable"
    return ", ".join(root.as_posix() for root in roots)


def _missing_bundle_message() -> str:
    return (
        "PixAI tagger bundle was not found. "
        f"Search roots: {_describe_search_roots()}. "
        "Required files per directory: "
        f"{_WEIGHTS_FILE_NAME}, {_TAGS_FILE_NAME}, {_MAPPING_FILE_NAME}"
    )


def _normalized_mode(value: object) -> str:
    text = str(value).strip().lower() if value is not None else _MODE_OPTIONS[0]
    if text == "topk":
        return "topk"
    return "threshold"


def _is_valid_mode(value: object) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in _MODE_OPTIONS


@lru_cache(maxsize=1)
def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _ = torch.zeros(1, device="cuda").cpu()
    except Exception:
        return False
    return True


def _normalized_device(value: object) -> str:
    text = str(value).strip().lower() if value is not None else "auto"
    if text == "cpu":
        return "cpu"
    if text.startswith("cuda"):
        return "cuda" if _cuda_usable() else "cpu"
    return "cuda" if _cuda_usable() else "cpu"


def _load_bundle_metadata(bundle_root: Path) -> _BundleMetadata:
    cache_key = str(bundle_root.resolve())
    with _METADATA_CACHE_LOCK:
        cached = _METADATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    weights_path, tags_path, mapping_path = _bundle_files(bundle_root)
    with tags_path.open("r", encoding="utf-8") as handle:
        tags_payload = json.load(handle)
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping_payload = json.load(handle)

    tag_map = tags_payload.get("tag_map")
    if not isinstance(tag_map, dict) or not tag_map:
        raise RuntimeError(f"tag_map is missing or malformed: {tags_path.as_posix()}")

    tag_split = tags_payload.get("tag_split")
    if not isinstance(tag_split, dict):
        raise RuntimeError(f"tag_split is missing or malformed: {tags_path.as_posix()}")

    try:
        gen_tag_count = int(tag_split.get("gen_tag_count"))
        character_tag_count = int(tag_split.get("character_tag_count"))
    except Exception as exc:
        raise RuntimeError(f"tag_split counts are malformed: {tags_path.as_posix()}") from exc

    index_to_tag_map: dict[int, str] = {}
    for raw_tag, raw_index in tag_map.items():
        if not isinstance(raw_tag, str) or not raw_tag:
            continue
        try:
            index_value = int(raw_index)
        except Exception:
            continue
        index_to_tag_map[index_value] = raw_tag

    if not index_to_tag_map:
        raise RuntimeError(f"tag_map did not contain any valid tags: {tags_path.as_posix()}")

    tag_count = max(index_to_tag_map.keys()) + 1
    if gen_tag_count < 0 or character_tag_count < 0:
        raise RuntimeError(f"tag_split counts must be non-negative: {tags_path.as_posix()}")
    if gen_tag_count + character_tag_count != tag_count:
        raise RuntimeError(
            "tag_split counts do not match tag_map size. "
            f"gen={gen_tag_count}, character={character_tag_count}, tag_count={tag_count}"
        )

    character_ip_mapping: dict[str, tuple[str, ...]] = {}
    if isinstance(mapping_payload, dict):
        for raw_character, raw_ip_tags in mapping_payload.items():
            if not isinstance(raw_character, str) or not raw_character:
                continue
            if not isinstance(raw_ip_tags, list):
                continue
            normalized_ip_tags = tuple(
                ip_tag
                for ip_tag in raw_ip_tags
                if isinstance(ip_tag, str) and ip_tag
            )
            character_ip_mapping[raw_character] = normalized_ip_tags

    metadata = _BundleMetadata(
        bundle_root=bundle_root,
        weights_path=weights_path,
        tags_path=tags_path,
        mapping_path=mapping_path,
        index_to_tag_map=index_to_tag_map,
        gen_tag_count=gen_tag_count,
        character_tag_count=character_tag_count,
        character_ip_mapping=character_ip_mapping,
        tag_count=tag_count,
    )
    with _METADATA_CACHE_LOCK:
        _METADATA_CACHE[cache_key] = metadata
    return metadata


def _import_timm() -> object:
    try:
        import timm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PixAI Tagger runtime is unavailable because `timm` is missing. "
            "Install the extension requirements and restart ComfyUI."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"PixAI Tagger runtime failed to import timm: {exc}") from exc
    return timm


def _create_encoder(timm_module: object) -> tuple[torch.nn.Module, str]:
    errors: list[str] = []
    create_model = getattr(timm_module, "create_model", None)
    if not callable(create_model):
        raise RuntimeError("timm.create_model was not found")

    for model_name in _MODEL_NAME_CANDIDATES:
        try:
            encoder = create_model(model_name, pretrained=False)
            reset_classifier = getattr(encoder, "reset_classifier", None)
            if callable(reset_classifier):
                reset_classifier(0)
            return encoder, model_name
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

    raise RuntimeError(
        "Failed to construct a local EVA02 encoder via timm. "
        f"Tried: {', '.join(_MODEL_NAME_CANDIDATES)}. "
        f"Errors: {' | '.join(errors)}"
    )


def _load_state_dict(weights_path: Path) -> dict[str, Any]:
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location="cpu")

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected checkpoint payload: {weights_path.as_posix()}")
    return dict(state_dict)


def _build_model(metadata: _BundleMetadata, device: str) -> tuple[torch.nn.Module, str]:
    timm_module = _import_timm()
    encoder, model_name = _create_encoder(timm_module)

    feature_dim = int(getattr(encoder, "num_features", 0) or 0)
    if feature_dim <= 0:
        raise RuntimeError(f"Failed to resolve encoder.num_features for {model_name}")

    model = torch.nn.Sequential(
        encoder,
        _TaggingHead(feature_dim, metadata.tag_count),
    )
    state_dict = _load_state_dict(metadata.weights_path)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load PixAI tagger weights: {exc}") from exc

    model = model.to(device)
    model.eval()
    return model, model_name


def _load_model(metadata: _BundleMetadata, device: str) -> tuple[torch.nn.Module, str]:
    cache_key = (str(metadata.bundle_root.resolve()), device)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    loaded = _build_model(metadata, device)
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


def _normalized_image_batch(image: Any) -> torch.Tensor:
    batches = _normalized_image_batches(image)
    if len(batches) == 1:
        return batches[0]
    try:
        return torch.cat(batches, dim=0)
    except Exception as exc:
        raise RuntimeError("all image items must have matching height, width, and channels") from exc


def _resize_nchw(image_nchw: torch.Tensor) -> torch.Tensor:
    try:
        return F.interpolate(
            image_nchw,
            size=(_IMAGE_SIZE, _IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    except TypeError:
        return F.interpolate(
            image_nchw,
            size=(_IMAGE_SIZE, _IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )


def _preprocess_image_batch(image_batch: torch.Tensor) -> torch.Tensor:
    if image_batch.shape[-1] < 3:
        raise RuntimeError("image input must have at least 3 channels")

    tensor = image_batch.detach().float()[..., :3]
    tensor = tensor.permute(0, 3, 1, 2).contiguous()
    tensor = _resize_nchw(tensor)
    tensor = tensor.clamp(0.0, 1.0)
    return tensor.sub(0.5).div(0.5)


def _preprocess_image_input(image: Any) -> torch.Tensor:
    batches = _normalized_image_batches(image)
    preprocessed: list[torch.Tensor] = []
    for batch in batches:
        preprocessed.append(_preprocess_image_batch(batch))

    if len(preprocessed) == 1:
        return preprocessed[0]
    return torch.cat(preprocessed, dim=0)


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


def _formatted_tags_and_scores(
    items: list[tuple[int, str, float]],
) -> tuple[list[str], dict[str, float]]:
    ordered_tags: list[str] = []
    score_map: dict[str, float] = {}
    for _, raw_tag, score in items:
        tag = _format_output_tag(raw_tag)
        current = score_map.get(tag)
        if current is None:
            ordered_tags.append(tag)
            score_map[tag] = score
            continue
        if score > current:
            score_map[tag] = score
    return ordered_tags, score_map


def _selected_indices_and_scores(
    probs: torch.Tensor,
    metadata: _BundleMetadata,
    *,
    mode: str,
    threshold_general: float,
    threshold_character: float,
    topk_general: int,
    topk_character: int,
) -> tuple[list[int], list[float]]:
    probs = probs.reshape(-1)

    if mode == "topk":
        gen_slice = probs[: metadata.gen_tag_count]
        char_slice = probs[metadata.gen_tag_count : metadata.gen_tag_count + metadata.character_tag_count]

        k_gen = max(0, min(int(topk_general), metadata.gen_tag_count))
        k_char = max(0, min(int(topk_character), metadata.character_tag_count))

        gen_indices = torch.empty(0, dtype=torch.long, device=probs.device)
        gen_scores = torch.empty(0, dtype=probs.dtype, device=probs.device)
        char_indices = torch.empty(0, dtype=torch.long, device=probs.device)
        char_scores = torch.empty(0, dtype=probs.dtype, device=probs.device)

        if k_gen > 0:
            gen_scores, gen_indices = torch.topk(gen_slice, k_gen)
        if k_char > 0:
            char_scores, char_indices = torch.topk(char_slice, k_char)
            char_indices = char_indices + metadata.gen_tag_count

        combined_indices = torch.cat((gen_indices, char_indices), dim=0)
        combined_scores = torch.cat((gen_scores, char_scores), dim=0)
    else:
        general_mask = probs[: metadata.gen_tag_count] > float(threshold_general)
        character_mask = (
            probs[metadata.gen_tag_count : metadata.gen_tag_count + metadata.character_tag_count]
            > float(threshold_character)
        )
        general_indices = general_mask.nonzero(as_tuple=True)[0]
        character_indices = character_mask.nonzero(as_tuple=True)[0] + metadata.gen_tag_count
        general_scores = probs.index_select(0, general_indices)
        character_scores = probs.index_select(0, character_indices)

        if general_scores.numel() > 0:
            general_order = _argsort_descending(general_scores)
            general_indices = general_indices.index_select(0, general_order)
            general_scores = general_scores.index_select(0, general_order)
        if character_scores.numel() > 0:
            character_order = _argsort_descending(character_scores)
            character_indices = character_indices.index_select(0, character_order)
            character_scores = character_scores.index_select(0, character_order)

        combined_indices = torch.cat((general_indices, character_indices), dim=0)
        combined_scores = torch.cat((general_scores, character_scores), dim=0)

    indices = [int(index) for index in combined_indices.detach().cpu().tolist()]
    scores = [float(score) for score in combined_scores.detach().float().cpu().tolist()]
    return indices, scores


def _result_item(
    probs: torch.Tensor,
    metadata: _BundleMetadata,
    *,
    mode: str,
    threshold_general: float,
    threshold_character: float,
    topk_general: int,
    topk_character: int,
) -> dict[str, Any]:
    indices, scores = _selected_indices_and_scores(
        probs,
        metadata,
        mode=mode,
        threshold_general=threshold_general,
        threshold_character=threshold_character,
        topk_general=topk_general,
        topk_character=topk_character,
    )

    general_items: list[tuple[int, str, float]] = []
    character_items: list[tuple[int, str, float]] = []

    for index, score in zip(indices, scores):
        tag = metadata.index_to_tag_map.get(index)
        if tag is None:
            continue
        if index < metadata.gen_tag_count:
            general_items.append((index, tag, score))
            continue
        character_items.append((index, tag, score))

    general_items.sort(key=lambda item: item[0])
    character_items.sort(key=lambda item: (-item[2], item[0]))

    general_tags, general_scores = _formatted_tags_and_scores(general_items)
    character_tags, character_scores = _formatted_tags_and_scores(character_items)

    ip_score_map: dict[str, float] = {}
    for _, character_tag, score in character_items:
        for ip_tag in metadata.character_ip_mapping.get(character_tag, ()):
            current = ip_score_map.get(ip_tag)
            if current is None or score > current:
                ip_score_map[ip_tag] = score
    formatted_ip_score_map: dict[str, float] = {}
    for raw_ip_tag, score in ip_score_map.items():
        formatted_ip_tag = _format_output_tag(raw_ip_tag)
        current = formatted_ip_score_map.get(formatted_ip_tag)
        if current is None or score > current:
            formatted_ip_score_map[formatted_ip_tag] = score
    ip_tags = [
        ip_tag
        for ip_tag, _ in sorted(
            formatted_ip_score_map.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    output: dict[str, Any] = {
        "general": general_tags,
        "character": character_tags,
        "ip": ip_tags,
        "general_scores": general_scores,
        "character_scores": character_scores,
    }
    return output


def _prompt_text_from_tag_batches(tag_batches: list[list[str]]) -> str:
    lines = [", ".join(tags) for tags in tag_batches]
    if len(lines) == 1:
        return lines[0]
    return "\n".join(lines)


class PixAITagger(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-PixAITagger",
            display_name="PixAI Tagger",
            category=Const.CATEGORY_PROMPT,
            description=(
                "Runs the local PixAI Tagger v0.9 bundle from "
                "ComfyUI models/pixai_tagger/ or models/wd14_tagger/pixai/pixai-labs_pixai-tagger-v0.9/ "
                "and returns prompt-ready tags."
            ),
            inputs=[
                c_io.Image.Input(
                    "image",
                    tooltip="Input image batch to classify with the local PixAI tagger bundle.",
                ),
                c_io.Combo.Input(
                    "mode",
                    options=_MODE_OPTIONS,
                    default=_MODE_OPTIONS[0],
                    tooltip="threshold keeps tags above category thresholds in score order. topk keeps the top-K tags per category.",
                ),
                c_io.Float.Input(
                    "threshold_general",
                    default=0.3,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Threshold for general tags when mode=threshold.",
                ),
                c_io.Float.Input(
                    "threshold_character",
                    default=0.85,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Threshold for character tags when mode=threshold.",
                ),
                c_io.Int.Input(
                    "topk_general",
                    default=25,
                    min=0,
                    max=2048,
                    tooltip="Number of general tags kept when mode=topk.",
                ),
                c_io.Int.Input(
                    "topk_character",
                    default=10,
                    min=0,
                    max=2048,
                    tooltip="Number of character tags kept when mode=topk.",
                ),
                c_io.Combo.Input(
                    "device",
                    options=_DEVICE_OPTIONS,
                    default=_DEVICE_OPTIONS[0],
                    tooltip="Execution device for the local PixAI tagger runtime.",
                    advanced=True,
                ),
            ],
            outputs=[
                c_io.String.Output(
                    Cast.out_id("general"),
                    display_name="general",
                ),
                c_io.String.Output(
                    Cast.out_id("character"),
                    display_name="character",
                ),
                c_io.String.Output(
                    Cast.out_id("ip"),
                    display_name="ip",
                ),
                c_io.String.Output(
                    Cast.out_id("result_json"),
                    display_name="result_json",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        image: Any = None,
        threshold_general: object = 0.3,
        threshold_character: object = 0.85,
        mode: object = "threshold",
        topk_general: object = 25,
        topk_character: object = 10,
        device: object = "auto",
        general_threshold: object | None = None,
        character_threshold: object | None = None,
        feature_threshold: object | None = None,
        feature_topk: object | None = None,
        character_topk: object | None = None,
    ) -> bool | str:
        if general_threshold is not None:
            threshold_general = general_threshold
        if feature_threshold is not None:
            threshold_general = feature_threshold
        if character_threshold is not None:
            threshold_character = character_threshold
        if feature_topk is not None:
            topk_general = feature_topk
        if character_topk is not None:
            topk_character = character_topk

        try:
            threshold_general_value = float(threshold_general)
            threshold_character_value = float(threshold_character)
            topk_general_value = int(topk_general)
            topk_character_value = int(topk_character)
        except Exception:
            return "threshold_general, threshold_character, topk_general, and topk_character must be numeric"

        if not 0.0 <= threshold_general_value <= 1.0:
            return "threshold_general must be between 0.0 and 1.0"
        if not 0.0 <= threshold_character_value <= 1.0:
            return "threshold_character must be between 0.0 and 1.0"
        if topk_general_value < 0:
            return "topk_general must be 0 or greater"
        if topk_character_value < 0:
            return "topk_character must be 0 or greater"
        if not _is_valid_mode(mode):
            return f"mode must be one of: {', '.join(_MODE_OPTIONS)}"

        if _find_bundle_root() is not None:
            return True
        return _missing_bundle_message()

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        threshold_general: float,
        threshold_character: float,
        mode: str,
        topk_general: int,
        topk_character: int,
        device: str,
        general_threshold: float | None = None,
        character_threshold: float | None = None,
        feature_threshold: float | None = None,
        feature_topk: int | None = None,
        character_topk: int | None = None,
    ) -> c_io.NodeOutput:
        if general_threshold is not None:
            threshold_general = float(general_threshold)
        if feature_threshold is not None:
            threshold_general = float(feature_threshold)
        if character_threshold is not None:
            threshold_character = float(character_threshold)
        if feature_topk is not None:
            topk_general = int(feature_topk)
        if character_topk is not None:
            topk_character = int(character_topk)

        bundle_root = _find_bundle_root()
        if bundle_root is None:
            raise RuntimeError(_missing_bundle_message())

        metadata = _load_bundle_metadata(bundle_root)
        runtime_device = _normalized_device(device)
        model, model_name = _load_model(metadata, runtime_device)

        image_tensor = _preprocess_image_input(image)
        image_tensor = image_tensor.to(runtime_device)

        with torch.inference_mode():
            probs_batch = model(image_tensor)

        mode_value = _normalized_mode(mode)
        items: list[dict[str, Any]] = []
        general_batches: list[list[str]] = []
        character_batches: list[list[str]] = []
        ip_batches: list[list[str]] = []

        for probs in probs_batch:
            item = _result_item(
                probs,
                metadata,
                mode=mode_value,
                threshold_general=float(threshold_general),
                threshold_character=float(threshold_character),
                topk_general=int(topk_general),
                topk_character=int(topk_character),
            )
            items.append(item)
            general_batches.append(list(item.get("general", ())))
            character_batches.append(list(item.get("character", ())))
            ip_batches.append(list(item.get("ip", ())))

        result_payload = {
            "items": items,
            "_params": {
                "mode": mode_value,
                "threshold_general": float(threshold_general),
                "threshold_character": float(threshold_character),
                "topk_general": int(topk_general),
                "topk_character": int(topk_character),
                "device": runtime_device,
                "model_name": model_name,
            },
        }

        return c_io.NodeOutput(
            _prompt_text_from_tag_batches(general_batches),
            _prompt_text_from_tag_batches(character_batches),
            _prompt_text_from_tag_batches(ip_batches),
            json.dumps(result_payload, ensure_ascii=True),
        )
