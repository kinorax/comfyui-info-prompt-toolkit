# Copyright 2026 kinorax
from __future__ import annotations

import json
from pathlib import Path
import sys
from threading import Lock
from typing import Any

import folder_paths
import numpy as np
import torch
from PIL import Image
from comfy_api.latest import io as c_io

from ... import const as Const
from ...utils import cast as Cast

_SAM3_DIR_NAME = "sam3"
_CHECKPOINT_NAME = "sam3.pt"
_COMBINE_MODE_OPTIONS = ("union", "top1")
_DEVICE_OPTIONS = ("auto", "cuda", "cpu")
_PROCESSOR_CACHE_LOCK = Lock()
_PROCESSOR_CACHE: dict[tuple[str, str], object] = {}
_SAM3_RUNTIME_DEPENDENCIES = (
    "timm",
    "ftfy",
    "regex",
    "iopath",
    "typing_extensions",
)
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_VENDOR_ROOT = _PACKAGE_ROOT / "vendor"
_VENDORED_SAM3_ROOT = _VENDOR_ROOT / "sam3"


def _candidate_model_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    models_dir = getattr(folder_paths, "models_dir", None)
    if models_dir:
        roots.append(Path(models_dir).resolve() / _SAM3_DIR_NAME)
    roots.append((_PACKAGE_ROOT / _SAM3_DIR_NAME).resolve())

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        normalized = str(root).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(root)
    return tuple(deduped)


def _find_checkpoint_path() -> Path | None:
    for root in _candidate_model_roots():
        candidate = root / _CHECKPOINT_NAME
        if candidate.is_file():
            return candidate.resolve()
    return None


def _describe_search_roots() -> str:
    roots = _candidate_model_roots()
    if not roots:
        return "<none>"
    return ", ".join(root.as_posix() for root in roots)


def _normalized_device(value: object) -> str:
    text = str(value).strip().lower() if value is not None else "auto"
    if text.startswith("cuda"):
        return text if torch.cuda.is_available() else "cpu"
    if text == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _result_to_combined_mask(
    result: dict[str, Any] | None,
    image_size: tuple[int, int],
    *,
    score_threshold: float,
    combine_mode: object = "union",
) -> tuple[torch.Tensor, list[float], list[list[float]], int]:
    width, height = image_size
    empty_mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")

    if not isinstance(result, dict):
        return empty_mask, [], [], 0

    mask_tensor = _normalize_mask_tensor(result.get("masks_logits"))
    if mask_tensor is None:
        mask_tensor = _normalize_mask_tensor(result.get("masks"))
    if mask_tensor is None or mask_tensor.numel() == 0:
        return empty_mask, [], _normalize_boxes(result.get("boxes")), 0

    scores = _normalize_float_list(result.get("scores"))
    boxes = _normalize_boxes(result.get("boxes"))
    keep_indices = _filtered_indices(mask_tensor.shape[0], scores, float(score_threshold))
    if not keep_indices:
        return empty_mask, [], [], 0

    selected_masks = mask_tensor[keep_indices]
    selected_scores = [scores[index] for index in keep_indices if index < len(scores)]
    selected_boxes = [boxes[index] for index in keep_indices if index < len(boxes)]

    if _normalize_combine_mode(combine_mode) == "top1":
        best_position = 0
        if selected_scores:
            best_position = max(range(len(selected_scores)), key=selected_scores.__getitem__)
        selected_masks = selected_masks[best_position : best_position + 1]
        if selected_scores:
            selected_scores = [selected_scores[best_position]]
        if selected_boxes:
            selected_boxes = [selected_boxes[best_position]]

    combined = selected_masks.max(dim=0).values.clamp(0.0, 1.0)
    return combined, selected_scores, selected_boxes, selected_masks.shape[0]


def _load_processor(checkpoint_path: Path, device: str) -> object:
    cache_key = (str(checkpoint_path.resolve()), device)
    with _PROCESSOR_CACHE_LOCK:
        cached = _PROCESSOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    build_sam3_image_model, sam3_processor_class = _import_backend()
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
    )
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model = model.eval()
    processor = sam3_processor_class(model)

    with _PROCESSOR_CACHE_LOCK:
        _PROCESSOR_CACHE[cache_key] = processor
    return processor


def _set_image_state(processor: object, image: Image.Image) -> dict[str, Any]:
    state = processor.set_image(image)
    if isinstance(state, dict):
        return state
    raise RuntimeError("SAM3 runtime returned an unexpected image state payload")


def _run_inference_with_state(
    processor: object,
    state: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    result = processor.set_text_prompt(state=state, prompt=prompt)
    if isinstance(result, dict):
        return dict(result)
    raise RuntimeError("SAM3 runtime returned an unexpected result payload")


def _ensure_vendored_sam3_import_path() -> None:
    if not _VENDORED_SAM3_ROOT.exists():
        raise RuntimeError(
            "Vendored SAM3 package was not found. "
            f"Expected: {_VENDORED_SAM3_ROOT.as_posix()}. "
            "This extension expects vendor/sam3 to be bundled in the distribution."
        )

    vendor_root_text = str(_VENDOR_ROOT.resolve())
    if vendor_root_text not in sys.path:
        sys.path.insert(0, vendor_root_text)


def _import_backend() -> tuple[object, object]:
    _ensure_vendored_sam3_import_path()
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "sam3"
        if missing_name == "sam3":
            raise RuntimeError(
                "Vendored SAM3 package failed to import. "
                f"Expected: {_VENDORED_SAM3_ROOT.as_posix()}"
            ) from exc
        raise RuntimeError(
            "SAM3 runtime is unavailable because "
            f"`{missing_name}` is missing. Install the vendored SAM3 runtime deps "
            f"({', '.join(_SAM3_RUNTIME_DEPENDENCIES)})."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"SAM3 runtime failed to import: {exc}") from exc
    return build_sam3_image_model, Sam3Processor


def _to_pil_rgb(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().numpy()
    array_uint8 = np.clip(array[..., :3] * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array_uint8, mode="RGB")


def _normalized_prompt_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalized_prompt_list_or_none(value: object) -> list[str] | None:
    text = _normalized_prompt_or_none(value)
    if text is None:
        return None

    prompts: list[str] = []
    seen: set[str] = set()
    for part in text.split(","):
        prompt = _normalized_prompt_or_none(part)
        if prompt is None or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
    return prompts or None


def _combine_prompt_results(
    prompt_results: list[tuple[torch.Tensor, list[float], list[list[float]], int]],
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, list[float], list[list[float]], int]:
    width, height = image_size
    combined_mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")
    combined_scores: list[float] = []
    combined_boxes: list[list[float]] = []
    total_match_count = 0

    for mask, scores, boxes, match_count in prompt_results:
        combined_mask = torch.maximum(combined_mask, mask)
        combined_scores.extend(scores)
        combined_boxes.extend(boxes)
        total_match_count += int(match_count)

    return combined_mask, combined_scores, combined_boxes, total_match_count


def _normalized_image_batch(image: Any) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise RuntimeError("image input must be a ComfyUI IMAGE tensor")
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim != 4:
        raise RuntimeError("image input must have shape [B,H,W,C]")
    return image


def _normalize_mask_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().cpu()
    else:
        try:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        except Exception:
            return None

    tensor = tensor.squeeze()
    if tensor.ndim < 2:
        return None
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim > 3:
        height = int(tensor.shape[-2])
        width = int(tensor.shape[-1])
        tensor = tensor.reshape(-1, height, width)
    return tensor.float().cpu()


def _normalize_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().float().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        output: list[float] = []
        for item in value:
            try:
                output.append(float(item))
            except Exception:
                continue
        return output
    try:
        return [float(value)]
    except Exception:
        return []


def _normalize_boxes(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        data = value.detach().float().cpu().tolist()
    elif isinstance(value, (list, tuple)):
        data = list(value)
    else:
        return []

    output: list[list[float]] = []
    for item in data:
        if isinstance(item, torch.Tensor):
            item = item.detach().float().cpu().tolist()
        if not isinstance(item, (list, tuple)):
            continue
        box: list[float] = []
        for number in item:
            try:
                box.append(float(number))
            except Exception:
                break
        if box:
            output.append(box)
    return output


def _filtered_indices(mask_count: int, scores: list[float], score_threshold: float) -> list[int]:
    if mask_count <= 0:
        return []
    if not scores:
        return list(range(mask_count))

    output: list[int] = []
    for index in range(mask_count):
        score = scores[index] if index < len(scores) else None
        if score is None or score >= score_threshold:
            output.append(index)
    return output


def _normalize_combine_mode(value: object) -> str:
    text = str(value).strip().lower() if value is not None else "union"
    if text == "top1":
        return "top1"
    return "union"


class Sam3PromptToMask(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-Sam3PromptToMask",
            display_name="SAM3 Prompt To Mask",
            category=Const.CATEGORY_MASK,
            inputs=[
                c_io.Image.Input(
                    "image",
                    tooltip="Input image batch processed with the same SAM3 text prompt",
                ),
                c_io.String.Input(
                    "prompt",
                    default="person",
                    tooltip="Text prompt used to select target objects. Comma-separated prompts are treated as OR",
                ),
                c_io.Float.Input(
                    "score_threshold",
                    default=0.15,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Detections below this score are discarded when scores are available",
                ),
                c_io.Combo.Input(
                    "combine_mode",
                    options=_COMBINE_MODE_OPTIONS,
                    default=_COMBINE_MODE_OPTIONS[0],
                    tooltip="union merges all matched masks. top1 keeps only the top-scoring one for each comma-separated prompt",
                ),
                c_io.Combo.Input(
                    "device",
                    options=_DEVICE_OPTIONS,
                    default=_DEVICE_OPTIONS[0],
                    tooltip="Execution device for the SAM3 runtime",
                    advanced=True,
                ),
            ],
            outputs=[
                c_io.Mask.Output(
                    Cast.out_id("mask"),
                    display_name="mask",
                ),
                c_io.Int.Output(
                    Cast.out_id("match_count"),
                    display_name="match_count",
                ),
                c_io.String.Output(
                    Cast.out_id("scores_json"),
                    display_name="scores_json",
                ),
                c_io.String.Output(
                    Cast.out_id("boxes_json"),
                    display_name="boxes_json",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        image: Any = None,
        prompt: object = "",
        score_threshold: object = 0.15,
        combine_mode: object = "union",
        device: object = "auto",
    ) -> bool | str:
        prompt_list = _normalized_prompt_list_or_none(prompt)
        if prompt is not None and prompt_list is None:
            return "prompt is required"

        if _find_checkpoint_path() is not None:
            return True
        return (
            "sam3.pt was not found. "
            f"Search roots: {_describe_search_roots()}"
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        prompt: str,
        score_threshold: float,
        combine_mode: str,
        device: str,
    ) -> c_io.NodeOutput:
        prompt_list = _normalized_prompt_list_or_none(prompt)
        if prompt_list is None:
            raise RuntimeError("prompt is required")

        checkpoint_path = _find_checkpoint_path()
        if checkpoint_path is None:
            raise RuntimeError(
                "sam3.pt was not found. "
                f"Search roots: {_describe_search_roots()}"
            )

        image_batch = _normalized_image_batch(image)
        runtime_device = _normalized_device(device)
        processor = _load_processor(checkpoint_path, runtime_device)

        masks: list[torch.Tensor] = []
        score_payload: list[list[float]] = []
        box_payload: list[list[list[float]]] = []
        total_match_count = 0

        for frame in image_batch:
            pil_image = _to_pil_rgb(frame)
            state = _set_image_state(processor, pil_image)
            prompt_results: list[tuple[torch.Tensor, list[float], list[list[float]], int]] = []

            for prompt_text in prompt_list:
                result = _run_inference_with_state(processor, state, prompt_text)
                prompt_results.append(
                    _result_to_combined_mask(
                        result,
                        pil_image.size,
                        score_threshold=float(score_threshold),
                        combine_mode=combine_mode,
                    )
                )

            combined_mask, scores, boxes, match_count = _combine_prompt_results(
                prompt_results,
                pil_image.size,
            )
            masks.append(combined_mask.unsqueeze(0))
            score_payload.append(scores)
            box_payload.append(boxes)
            total_match_count += int(match_count)

        output_mask = torch.cat(masks, dim=0) if len(masks) > 1 else masks[0]
        return c_io.NodeOutput(
            output_mask,
            total_match_count,
            json.dumps(score_payload, ensure_ascii=True),
            json.dumps(box_payload, ensure_ascii=True),
        )
