# Copyright 2026 kinorax
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .model_merge import is_model_merge_value, model_merge_json_or_none, model_value_from_merge_json

IMAGEINFO_POSITIVE = "positive"
IMAGEINFO_NEGATIVE = "negative"
IMAGEINFO_STEPS = "steps"
IMAGEINFO_SAMPLER = "sampler"
IMAGEINFO_SCHEDULER = "scheduler"
IMAGEINFO_CFG = "cfg"
IMAGEINFO_SEED = "seed"
IMAGEINFO_WIDTH = "width"
IMAGEINFO_HEIGHT = "height"
IMAGEINFO_MODEL = "model"
IMAGEINFO_REFINER_MODEL = "refiner"
IMAGEINFO_DETAILER_MODEL = "detailer"
IMAGEINFO_LORA_STACK = "lora_stack"
IMAGEINFO_CLIP = "clip"
IMAGEINFO_VAE = "vae"
IMAGEINFO_EXTRAS = "extras"
# Legacy field kept for backward-compatibility on write path only.
IMAGEINFO_EXTRA_INFO_RAW = "extra_info_raw"

_MODEL_FOLDER_CHECKPOINTS = "checkpoints"
_MODEL_FOLDER_TEXT_ENCODERS = "text_encoders"

_PARAM_STEPS = "Steps"
_PARAM_SAMPLER = "Sampler"
_PARAM_SCHEDULE_TYPE = "Schedule type"
_PARAM_CFG_SCALE = "CFG scale"
_PARAM_SEED = "Seed"
_PARAM_SIZE = "Size"
_PARAM_MODEL = "Model"
_PARAM_REFINER = "Refiner"
_PARAM_DETAILER = "Detailer"
_PARAM_MODEL_MERGE = "Model Merge"
_PARAM_REFINER_MERGE = "Refiner Merge"
_PARAM_DETAILER_MERGE = "Detailer Merge"
_PARAM_MODEL_FOLDER_PATHS = "Model folder paths"
_PARAM_CLIP_TYPE = "Clip type"
_PARAM_CLIP_DEVICE = "Clip device"
_PARAM_VAE = "VAE"
_PARAM_HASHES = "Hashes"
_PARAM_EXTRA_INFO = "Extra info"
_MODEL_WEIGHT_DTYPE_KEY = "weight_dtype"
_MODEL_AURAFLOW_SHIFT_KEY = "model_sampling_auraflow_shift"
_PARAM_MODEL_WEIGHT_DTYPE = "Model weight dtype"
_PARAM_MODEL_AURAFLOW_SHIFT = "Model ModelSamplingAuraFlow shift"
_PARAM_REFINER_WEIGHT_DTYPE = "Refiner weight dtype"
_PARAM_REFINER_AURAFLOW_SHIFT = "Refiner ModelSamplingAuraFlow shift"
_PARAM_DETAILER_WEIGHT_DTYPE = "Detailer weight dtype"
_PARAM_DETAILER_AURAFLOW_SHIFT = "Detailer ModelSamplingAuraFlow shift"

_STEPS_LINE_RE = re.compile(r"^Steps:\s*\d+\b")
_NEGATIVE_PROMPT_RE = re.compile(r"^Negative prompt:[ \t]*", re.MULTILINE)
_SIZE_RE = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*$")
_LORA_TAG_RE = re.compile(
    r"<lora:([^:>]+):([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)>"
)
_JSON_DECODER = json.JSONDecoder()
_HASH_TOKEN_RE = re.compile(r"[0-9a-fA-F]{8,64}")
_HASH_TEXT_RE = re.compile(r"^[0-9a-f]{8,64}$")
INTERNAL_HASH_HINTS_KEY = "__iis_hashes_hints"
_KNOWN_PARAMETER_KEYS = {
    _PARAM_STEPS,
    _PARAM_SAMPLER,
    _PARAM_SCHEDULE_TYPE,
    _PARAM_CFG_SCALE,
    _PARAM_SEED,
    _PARAM_SIZE,
    _PARAM_MODEL,
    _PARAM_REFINER,
    _PARAM_DETAILER,
    _PARAM_MODEL_MERGE,
    _PARAM_REFINER_MERGE,
    _PARAM_DETAILER_MERGE,
    _PARAM_MODEL_FOLDER_PATHS,
    _PARAM_CLIP_TYPE,
    _PARAM_CLIP_DEVICE,
    _PARAM_VAE,
    _PARAM_HASHES,
    _PARAM_MODEL_WEIGHT_DTYPE,
    _PARAM_MODEL_AURAFLOW_SHIFT,
    _PARAM_REFINER_WEIGHT_DTYPE,
    _PARAM_REFINER_AURAFLOW_SHIFT,
    _PARAM_DETAILER_WEIGHT_DTYPE,
    _PARAM_DETAILER_AURAFLOW_SHIFT,
}


def image_info_to_a1111_infotext(image_info: Mapping[str, Any] | None) -> str:
    data = image_info if isinstance(image_info, Mapping) else {}
    positive = _coerce_prompt_text(data.get(IMAGEINFO_POSITIVE))
    lora_tags = _render_lora_tags(data.get(IMAGEINFO_LORA_STACK))
    if lora_tags:
        if positive:
            positive = f"{positive} {lora_tags}"
        else:
            positive = lora_tags
    negative = _coerce_prompt_text(data.get(IMAGEINFO_NEGATIVE))

    lines: list[str] = []
    if positive:
        lines.append(positive)
    if negative:
        lines.append(f"Negative prompt: {negative}")

    parameter_line = _build_parameter_line(data)
    if parameter_line:
        lines.append(parameter_line)

    extra_info_raw = _extract_extra_info_text(data)
    if extra_info_raw:
        extra_info_text = f"{_PARAM_EXTRA_INFO}: {extra_info_raw}"
        if parameter_line:
            lines[-1] = f"{lines[-1]}, {extra_info_text}"
        else:
            lines.append(extra_info_text)
    return "\n".join(lines)


def a1111_infotext_to_image_info(infotext: str | None) -> dict[str, Any]:
    text = _normalize_newlines(infotext or "")
    base_text, extra_info_raw = _split_extra_info_block(text)
    prompt_block, parameter_line = _split_prompt_and_parameters(base_text)
    parameter_line, hashes_hints = _extract_and_strip_hashes_parameter_line(parameter_line)
    positive, negative = _split_positive_negative(prompt_block)

    positive, lora_stack = extract_lora_stack_from_prompt(positive)

    output: dict[str, Any] = {}
    if positive:
        output[IMAGEINFO_POSITIVE] = positive
        # Keep negative as empty string when positive exists but marker is absent.
        output[IMAGEINFO_NEGATIVE] = negative
    elif negative:
        output[IMAGEINFO_NEGATIVE] = negative

    if lora_stack:
        output[IMAGEINFO_LORA_STACK] = lora_stack

    extras: dict[str, str] = {}
    for key, value in _parse_parameter_line(parameter_line):
        if not _assign_known_parameter(output, key, value):
            extras[key] = value

    if extra_info_raw is not None and extra_info_raw != "":
        extras[_PARAM_EXTRA_INFO] = extra_info_raw
    if extras:
        output[IMAGEINFO_EXTRAS] = extras
    _finalize_clip_value(output)
    if hashes_hints:
        output[INTERNAL_HASH_HINTS_KEY] = hashes_hints
    return output


def extract_lora_stack_from_prompt(prompt: str | None) -> tuple[str, list[dict[str, Any]]]:
    text = _normalize_newlines(prompt or "")
    return _extract_lora_stack_from_positive(text)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _coerce_prompt_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).rstrip("\n")


def _extract_extra_info_text(data: Mapping[str, Any]) -> str | None:
    extras = data.get(IMAGEINFO_EXTRAS)
    if isinstance(extras, Mapping):
        if _PARAM_EXTRA_INFO in extras:
            value = _stringify_value(extras.get(_PARAM_EXTRA_INFO))
            if value:
                return value

    legacy = data.get(IMAGEINFO_EXTRA_INFO_RAW)
    if isinstance(legacy, str) and legacy:
        return legacy
    return None


def _build_parameter_line(data: Mapping[str, Any]) -> str:
    pairs: list[tuple[str, str]] = []
    used_keys: set[str] = set()

    def add_pair(key: str, value: Any) -> None:
        if value is None:
            return
        value_text = _stringify_value(value)
        if not value_text:
            return
        pairs.append((key, value_text))
        used_keys.add(key)

    add_pair(_PARAM_STEPS, _coerce_int(data.get(IMAGEINFO_STEPS)))
    add_pair(_PARAM_SAMPLER, _coerce_string(data.get(IMAGEINFO_SAMPLER)))
    add_pair(_PARAM_SCHEDULE_TYPE, _coerce_string(data.get(IMAGEINFO_SCHEDULER)))
    add_pair(_PARAM_CFG_SCALE, _coerce_float(data.get(IMAGEINFO_CFG)))
    add_pair(_PARAM_SEED, _coerce_int(data.get(IMAGEINFO_SEED)))

    width = _coerce_int(data.get(IMAGEINFO_WIDTH))
    height = _coerce_int(data.get(IMAGEINFO_HEIGHT))
    if width is not None and height is not None:
        add_pair(_PARAM_SIZE, f"{width}x{height}")

    model_value = data.get(IMAGEINFO_MODEL)
    refiner_value = data.get(IMAGEINFO_REFINER_MODEL)
    detailer_value = data.get(IMAGEINFO_DETAILER_MODEL)

    model_name, _ = _extract_model_fields(model_value)
    if is_model_merge_value(model_value):
        add_pair(_PARAM_MODEL_MERGE, model_merge_json_or_none(model_value))
    else:
        add_pair(_PARAM_MODEL, model_name)

    refiner_name, _ = _extract_model_fields(refiner_value)
    if is_model_merge_value(refiner_value):
        add_pair(_PARAM_REFINER_MERGE, model_merge_json_or_none(refiner_value))
    else:
        add_pair(_PARAM_REFINER, refiner_name)

    detailer_name, _ = _extract_model_fields(detailer_value)
    if is_model_merge_value(detailer_value):
        add_pair(_PARAM_DETAILER_MERGE, model_merge_json_or_none(detailer_value))
    else:
        add_pair(_PARAM_DETAILER, detailer_name)

    add_pair(_PARAM_MODEL_FOLDER_PATHS, _shared_non_merge_model_folder_paths(data))
    for key, value in _model_metadata_parameter_pairs(data):
        add_pair(key, value)
    for key, value in _clip_parameter_pairs(data):
        add_pair(key, value)

    vae_name = _extract_filename_only(_coerce_string(data.get(IMAGEINFO_VAE)))
    add_pair(_PARAM_VAE, vae_name)

    extras = data.get(IMAGEINFO_EXTRAS)
    if isinstance(extras, Mapping):
        for raw_key, raw_value in extras.items():
            if not isinstance(raw_key, str):
                continue
            key = raw_key.strip()
            if not key or key in used_keys or key == _PARAM_EXTRA_INFO:
                continue
            add_pair(key, raw_value)

    if not pairs:
        return ""
    return ", ".join(f"{key}: {_render_parameter_value(value)}" for key, value in pairs)


def _extract_model_fields(model_value: Any) -> tuple[str | None, str | None]:
    folder_paths: str | None = None
    if isinstance(model_value, Mapping):
        name = _coerce_string(model_value.get("name"))
        folder_paths = _coerce_string(model_value.get("folder_paths"))
    else:
        name = _coerce_string(model_value)
    return _extract_filename_only(name), folder_paths


def _model_metadata_parameter_pairs(data: Mapping[str, Any]) -> list[tuple[str, Any]]:
    output: list[tuple[str, Any]] = []
    output.extend(_single_model_metadata_parameter_pairs(data.get(IMAGEINFO_MODEL), _PARAM_MODEL))
    output.extend(_single_model_metadata_parameter_pairs(data.get(IMAGEINFO_REFINER_MODEL), _PARAM_REFINER))
    output.extend(_single_model_metadata_parameter_pairs(data.get(IMAGEINFO_DETAILER_MODEL), _PARAM_DETAILER))
    return output


def _shared_non_merge_model_folder_paths(data: Mapping[str, Any]) -> str | None:
    for key in (IMAGEINFO_MODEL, IMAGEINFO_REFINER_MODEL, IMAGEINFO_DETAILER_MODEL):
        value = data.get(key)
        if not isinstance(value, Mapping) or is_model_merge_value(value):
            continue
        folder_paths = _coerce_string(value.get("folder_paths"))
        if folder_paths:
            return folder_paths
    return None


def _clip_parameter_pairs(data: Mapping[str, Any]) -> list[tuple[str, Any]]:
    clip_value = data.get(IMAGEINFO_CLIP)
    if not isinstance(clip_value, Mapping):
        return []

    output: list[tuple[str, Any]] = []
    clip_names = clip_value.get("clip_names")
    if isinstance(clip_names, (list, tuple)):
        for index, raw_name in enumerate(clip_names, start=1):
            name = _extract_filename_only(_coerce_string(raw_name))
            if name:
                output.append((f"Clip {index}", name))

    clip_type = _coerce_string(clip_value.get("type"))
    if clip_type:
        output.append((_PARAM_CLIP_TYPE, clip_type))

    clip_device = _coerce_string(clip_value.get("device"))
    if clip_device:
        output.append((_PARAM_CLIP_DEVICE, clip_device))

    return output


def _single_model_metadata_parameter_pairs(model_value: Any, label: str) -> list[tuple[str, Any]]:
    if not isinstance(model_value, Mapping):
        return []
    if is_model_merge_value(model_value):
        return []

    output: list[tuple[str, Any]] = []
    weight_dtype = _coerce_string(model_value.get(_MODEL_WEIGHT_DTYPE_KEY))
    if weight_dtype:
        output.append((f"{label} weight dtype", weight_dtype))

    auraflow_shift = _coerce_float(model_value.get(_MODEL_AURAFLOW_SHIFT_KEY))
    if auraflow_shift is not None:
        output.append((f"{label} ModelSamplingAuraFlow shift", auraflow_shift))

    return output


def _extract_filename_only(value: str | None) -> str | None:
    if not value:
        return None
    basename = value.replace("\\", "/").split("/")[-1]
    if not basename:
        return None
    return basename


def _extract_filename_without_extension(value: str | None) -> str | None:
    basename = _extract_filename_only(value)
    if not basename:
        return None
    return Path(basename).stem


def _render_lora_tags(value: Any) -> str:
    if not isinstance(value, (list, tuple)):
        return ""

    tags: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = _extract_filename_without_extension(_coerce_string(item.get("name")))
        strength = _coerce_float(item.get("strength"))
        if not name or strength is None:
            continue
        tags.append(f"<lora:{name}:{format(strength, 'g')}>")

    return " ".join(tags)


def _extract_lora_stack_from_positive(positive: str) -> tuple[str, list[dict[str, Any]]]:
    if not positive:
        return "", []

    lora_stack: list[dict[str, Any]] = []

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        strength_text = match.group(2)
        try:
            strength = float(strength_text)
        except Exception:
            return match.group(0)
        lora_stack.append({"name": name, "strength": strength})
        return ""

    cleaned = _LORA_TAG_RE.sub(repl, positive).strip()
    return cleaned, lora_stack


def _render_parameter_value(value: str) -> str:
    if value != value.strip() or any(ch in value for ch in [",", ":", "\n", '"', "\\"]):
        return json.dumps(value, ensure_ascii=False)
    return value


def _split_extra_info_block(text: str) -> tuple[str, str | None]:
    if not text:
        return "", None

    candidates: list[tuple[int, int]] = []
    header = f"{_PARAM_EXTRA_INFO}:"

    if text.startswith(header):
        candidates.append((0, 0))

    line_marker = f"\n{_PARAM_EXTRA_INFO}:"
    line_idx = text.rfind(line_marker)
    if line_idx >= 0:
        # base は改行手前、header は改行直後から始まる。
        candidates.append((line_idx, line_idx + 1))

    inline_marker = f", {_PARAM_EXTRA_INFO}:"
    inline_idx = text.rfind(inline_marker)
    if inline_idx >= 0:
        # base は ", " 手前、header は "E" から始まる。
        candidates.append((inline_idx, inline_idx + 2))

    if not candidates:
        return text, None

    base_end, header_start = max(candidates, key=lambda item: item[1])
    base_text = text[:base_end]
    raw = text[header_start + len(header) :]
    if raw.startswith(" "):
        raw = raw[1:]
    return base_text, raw


def _split_prompt_and_parameters(base_text: str) -> tuple[str, str]:
    if not base_text:
        return "", ""

    lines = base_text.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if _looks_like_parameter_line(line):
            pre = "\n".join(lines[:i]).rstrip("\n")
            return pre, line

    return base_text.rstrip("\n"), ""


def _looks_like_parameter_line(line: str) -> bool:
    if not line:
        return False
    if _STEPS_LINE_RE.match(line):
        return True

    for key, value in _parse_parameter_line(line):
        if key not in _KNOWN_PARAMETER_KEYS and _parse_clip_parameter_index(key) is None:
            continue
        if key == _PARAM_HASHES and value.strip():
            return True
        probe: dict[str, Any] = {}
        if _assign_known_parameter(probe, key, value):
            return True
    return False


def _split_positive_negative(prompt_block: str) -> tuple[str, str]:
    if not prompt_block:
        return "", ""

    matches = list(_NEGATIVE_PROMPT_RE.finditer(prompt_block))
    if not matches:
        return prompt_block.rstrip("\n"), ""

    marker = matches[-1]
    positive = prompt_block[: marker.start()].rstrip("\n")
    negative = prompt_block[marker.end() :].rstrip("\n")
    return positive, negative


def _extract_and_strip_hashes_parameter_line(parameter_line: str) -> tuple[str, dict[str, Any] | None]:
    line = parameter_line.strip()
    if not line:
        return line, None

    extracted_hashes: list[str] = []

    while True:
        idx = _find_hashes_pair_start(line)
        if idx < 0:
            break

        value_start = idx + len(_PARAM_HASHES) + 1
        while value_start < len(line) and line[value_start] == " ":
            value_start += 1

        if value_start >= len(line):
            line = line[:idx].rstrip(", ")
            break

        if line[value_start] == "{":
            value_end = _find_json_object_end(line, value_start)
            if value_end is None:
                line = line[:idx].rstrip(", ")
                break
        else:
            value_end = _find_parameter_separator(line, value_start)

        raw_value = line[value_start:value_end].strip()
        if raw_value:
            extracted_hashes.append(raw_value)

        line = f"{line[:idx].rstrip(', ')}{line[value_end:]}"
        line = line.strip()
        if line.startswith(","):
            line = line[1:].lstrip()

    return line, _parse_hashes_hint_payloads(extracted_hashes)


def _strip_hashes_parameter_line(parameter_line: str) -> str:
    stripped, _ = _extract_and_strip_hashes_parameter_line(parameter_line)
    return stripped


def _parse_hashes_hint_payloads(payloads: list[str]) -> dict[str, Any] | None:
    output: dict[str, Any] = {
        "model": [],
        "refiner": [],
        "detailer": [],
        "vae": [],
        "loras": {},
    }

    for payload in payloads:
        payload_text = str(payload or "").strip()
        if not payload_text:
            continue

        parsed: Any | None = None
        if payload_text.startswith("{"):
            try:
                parsed = json.loads(payload_text)
            except Exception:
                parsed = None

        if isinstance(parsed, Mapping):
            _apply_hashes_hint_mapping(output, parsed)

    has_simple = bool(output["model"] or output["refiner"] or output["detailer"] or output["vae"])
    has_lora = bool(output["loras"])
    if not has_simple and not has_lora:
        return None
    return output


def _apply_hashes_hint_mapping(output: dict[str, Any], parsed: Mapping[str, Any]) -> None:
    loras_map = output.setdefault("loras", {})
    if not isinstance(loras_map, dict):
        loras_map = {}
        output["loras"] = loras_map

    for raw_key, raw_value in parsed.items():
        key_text = str(raw_key or "").strip()
        if not key_text:
            continue

        digest = _normalize_hash_hint(raw_value)
        if not digest:
            continue

        key_lower = key_text.lower()
        if key_lower in ("model", "model hash", "checkpoint", "ckpt"):
            _append_unique_hash_hint(output["model"], digest)
            continue
        if key_lower in ("refiner", "refiner hash"):
            _append_unique_hash_hint(output["refiner"], digest)
            continue
        if key_lower in ("detailer", "detailer hash"):
            _append_unique_hash_hint(output["detailer"], digest)
            continue
        if key_lower in ("vae", "vae hash"):
            _append_unique_hash_hint(output["vae"], digest)
            continue

        lora_name: str | None = None
        if key_lower.startswith("lora:"):
            lora_name = key_text.split(":", 1)[1].strip()
        elif key_lower.startswith("lora/"):
            lora_name = key_text.split("/", 1)[1].strip()
        elif key_lower.startswith("lora "):
            lora_name = key_text.split(" ", 1)[1].strip()

        if not lora_name:
            continue

        lora_hashes = loras_map.setdefault(lora_name, [])
        if isinstance(lora_hashes, list):
            _append_unique_hash_hint(lora_hashes, digest)


def _append_unique_hash_hint(target: list[str], digest: str) -> None:
    if digest not in target:
        target.append(digest)


def _normalize_hash_hint(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    if text.startswith("sha256:"):
        text = text[7:].strip()

    if _HASH_TEXT_RE.fullmatch(text):
        return text

    token = _HASH_TOKEN_RE.search(text)
    if token is None:
        return None
    return token.group(0).lower()


def _find_hashes_pair_start(line: str) -> int:
    search_from = 0
    token = f"{_PARAM_HASHES}:"
    while True:
        idx = line.find(token, search_from)
        if idx < 0:
            return -1

        probe = idx - 1
        while probe >= 0 and line[probe] == " ":
            probe -= 1
        if probe < 0 or line[probe] == ",":
            return idx

        search_from = idx + len(token)


def _find_json_object_end(text: str, start: int) -> int | None:
    if start < 0 or start >= len(text) or text[start] != "{":
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1

    return None


def _parse_parameter_line(parameter_line: str) -> list[tuple[str, str]]:
    line = parameter_line.strip()
    if not line:
        return []

    pairs: list[tuple[str, str]] = []
    i = 0
    n = len(line)
    while i < n:
        while i < n and line[i] in (" ", ","):
            i += 1
        if i >= n:
            break

        colon = line.find(":", i)
        if colon < 0:
            break

        key = line[i:colon].strip()
        if not key:
            break

        i = colon + 1
        if i < n and line[i] == " ":
            i += 1

        if i < n and line[i] == '"':
            decoded, consumed = _decode_json_quoted_string(line[i:])
            if consumed is not None:
                value = decoded
                i += consumed
            else:
                end = _find_parameter_separator(line, i)
                value = line[i:end].strip()
                i = end
        else:
            end = _find_parameter_separator(line, i)
            value = line[i:end].strip()
            i = end

        pairs.append((key, value))

        if i < n and line[i] == ",":
            i += 1
            if i < n and line[i] == " ":
                i += 1

    return pairs


def _decode_json_quoted_string(segment: str) -> tuple[str, int | None]:
    try:
        value, end = _JSON_DECODER.raw_decode(segment)
    except json.JSONDecodeError:
        return segment, None
    if isinstance(value, str):
        return value, end
    return segment, None


def _find_parameter_separator(line: str, start: int) -> int:
    cursor = start
    while True:
        comma = line.find(",", cursor)
        if comma < 0:
            return len(line)

        probe = comma + 1
        while probe < len(line) and line[probe] == " ":
            probe += 1
        if _looks_like_pair_start(line, probe):
            return comma

        cursor = comma + 1


def _looks_like_pair_start(line: str, start: int) -> bool:
    if start >= len(line):
        return False
    colon = line.find(":", start)
    if colon <= start:
        return False
    key = line[start:colon].strip()
    if not key:
        return False
    return "," not in key and "\n" not in key


def _assign_known_parameter(output: dict[str, Any], key: str, value: str) -> bool:
    if key == _PARAM_STEPS:
        parsed = _coerce_int(value)
        if parsed is None:
            return False
        output[IMAGEINFO_STEPS] = parsed
        return True

    if key == _PARAM_SAMPLER:
        text = _coerce_string(value)
        if not text:
            return False
        output[IMAGEINFO_SAMPLER] = text
        return True

    if key == _PARAM_SCHEDULE_TYPE:
        text = _coerce_string(value)
        if not text:
            return False
        output[IMAGEINFO_SCHEDULER] = text
        return True

    if key == _PARAM_CFG_SCALE:
        parsed = _coerce_float(value)
        if parsed is None:
            return False
        output[IMAGEINFO_CFG] = parsed
        return True

    if key == _PARAM_SEED:
        parsed = _coerce_int(value)
        if parsed is None:
            return False
        output[IMAGEINFO_SEED] = parsed
        return True

    if key == _PARAM_SIZE:
        size = _parse_size(value)
        if size is None:
            return False
        output[IMAGEINFO_WIDTH], output[IMAGEINFO_HEIGHT] = size
        return True

    if key == _PARAM_MODEL:
        text = _coerce_string(value)
        if not text:
            return False
        model = _ensure_model_dict(output, IMAGEINFO_MODEL)
        model["name"] = text
        if not _coerce_string(model.get("folder_paths")):
            model["folder_paths"] = _MODEL_FOLDER_CHECKPOINTS
        return True

    if key == _PARAM_REFINER:
        text = _coerce_string(value)
        if not text:
            return False
        refiner = _ensure_model_dict_with_default_folder(output, IMAGEINFO_REFINER_MODEL)
        refiner["name"] = text
        return True

    if key == _PARAM_DETAILER:
        text = _coerce_string(value)
        if not text:
            return False
        detailer = _ensure_model_dict_with_default_folder(output, IMAGEINFO_DETAILER_MODEL)
        detailer["name"] = text
        return True

    if key == _PARAM_MODEL_MERGE:
        return _assign_model_merge_payload(output, IMAGEINFO_MODEL, value)

    if key == _PARAM_REFINER_MERGE:
        return _assign_model_merge_payload(output, IMAGEINFO_REFINER_MODEL, value)

    if key == _PARAM_DETAILER_MERGE:
        return _assign_model_merge_payload(output, IMAGEINFO_DETAILER_MODEL, value)

    if key == _PARAM_MODEL_FOLDER_PATHS:
        text = _coerce_string(value)
        if not text:
            return False
        model = output.get(IMAGEINFO_MODEL)
        if isinstance(model, dict):
            if not is_model_merge_value(model):
                model["folder_paths"] = text
        else:
            model = _ensure_model_dict(output, IMAGEINFO_MODEL)
            model["folder_paths"] = text

        refiner = output.get(IMAGEINFO_REFINER_MODEL)
        if isinstance(refiner, dict) and not is_model_merge_value(refiner):
            refiner["folder_paths"] = text
        detailer = output.get(IMAGEINFO_DETAILER_MODEL)
        if isinstance(detailer, dict) and not is_model_merge_value(detailer):
            detailer["folder_paths"] = text
        return True

    clip_index = _parse_clip_parameter_index(key)
    if clip_index is not None:
        text = _coerce_string(value)
        if not text:
            return False
        clip = _ensure_clip_dict(output)
        clip_names = clip.setdefault("clip_names", [])
        if not isinstance(clip_names, list):
            clip_names = []
            clip["clip_names"] = clip_names
        while len(clip_names) < clip_index:
            clip_names.append(None)
        clip_names[clip_index - 1] = text
        return True

    if key == _PARAM_CLIP_TYPE:
        text = _coerce_string(value)
        if not text:
            return False
        clip = _ensure_clip_dict(output)
        clip["type"] = text
        return True

    if key == _PARAM_CLIP_DEVICE:
        text = _coerce_string(value)
        if not text:
            return False
        clip = _ensure_clip_dict(output)
        clip["device"] = text
        return True

    if key == _PARAM_VAE:
        text = _coerce_string(value)
        if not text:
            return False
        output[IMAGEINFO_VAE] = text
        return True

    if key == _PARAM_MODEL_WEIGHT_DTYPE:
        return _assign_model_metadata_string(output, IMAGEINFO_MODEL, _MODEL_WEIGHT_DTYPE_KEY, value)

    if key == _PARAM_MODEL_AURAFLOW_SHIFT:
        return _assign_model_metadata_float(output, IMAGEINFO_MODEL, _MODEL_AURAFLOW_SHIFT_KEY, value)

    if key == _PARAM_REFINER_WEIGHT_DTYPE:
        return _assign_model_metadata_string(output, IMAGEINFO_REFINER_MODEL, _MODEL_WEIGHT_DTYPE_KEY, value)

    if key == _PARAM_REFINER_AURAFLOW_SHIFT:
        return _assign_model_metadata_float(output, IMAGEINFO_REFINER_MODEL, _MODEL_AURAFLOW_SHIFT_KEY, value)

    if key == _PARAM_DETAILER_WEIGHT_DTYPE:
        return _assign_model_metadata_string(output, IMAGEINFO_DETAILER_MODEL, _MODEL_WEIGHT_DTYPE_KEY, value)

    if key == _PARAM_DETAILER_AURAFLOW_SHIFT:
        return _assign_model_metadata_float(output, IMAGEINFO_DETAILER_MODEL, _MODEL_AURAFLOW_SHIFT_KEY, value)

    return False


def _assign_model_merge_payload(output: dict[str, Any], model_key: str, value: str) -> bool:
    model_value = model_value_from_merge_json(value)
    if model_value is None:
        return False
    output[model_key] = model_value
    return True


def _ensure_model_dict(output: dict[str, Any], model_key: str) -> dict[str, Any]:
    current = output.get(model_key)
    if isinstance(current, dict):
        return current
    model: dict[str, Any] = {}
    output[model_key] = model
    return model


def _ensure_clip_dict(output: dict[str, Any]) -> dict[str, Any]:
    current = output.get(IMAGEINFO_CLIP)
    if isinstance(current, dict):
        if not _coerce_string(current.get("folder_paths")):
            current["folder_paths"] = _MODEL_FOLDER_TEXT_ENCODERS
        return current
    clip: dict[str, Any] = {"folder_paths": _MODEL_FOLDER_TEXT_ENCODERS}
    output[IMAGEINFO_CLIP] = clip
    return clip


def _parse_clip_parameter_index(key: str) -> int | None:
    key_text = str(key or "").strip()
    if not key_text.lower().startswith("clip "):
        return None
    suffix = key_text[5:].strip()
    if not suffix.isdigit():
        return None
    index = int(suffix)
    if index <= 0:
        return None
    return index


def _finalize_clip_value(output: dict[str, Any]) -> None:
    clip = output.get(IMAGEINFO_CLIP)
    if not isinstance(clip, dict):
        output.pop(IMAGEINFO_CLIP, None)
        return

    clip_names = clip.get("clip_names")
    if isinstance(clip_names, list):
        normalized_names = [text for text in (_coerce_string(item) for item in clip_names) if text]
        if normalized_names:
            clip["clip_names"] = normalized_names
        else:
            clip.pop("clip_names", None)

    if not clip.get("clip_names"):
        output.pop(IMAGEINFO_CLIP, None)


def _ensure_model_dict_with_default_folder(output: dict[str, Any], model_key: str) -> dict[str, Any]:
    model = _ensure_model_dict(output, model_key)
    if not _coerce_string(model.get("folder_paths")):
        shared_folder = _shared_model_folder(output)
        if shared_folder:
            model["folder_paths"] = shared_folder
    return model


def _shared_model_folder(output: dict[str, Any]) -> str:
    model = output.get(IMAGEINFO_MODEL)
    model_folder = model.get("folder_paths") if isinstance(model, dict) else None
    return _coerce_string(model_folder) or _MODEL_FOLDER_CHECKPOINTS


def _assign_model_metadata_string(
    output: dict[str, Any],
    model_key: str,
    metadata_key: str,
    value: str,
) -> bool:
    text = _coerce_string(value)
    if not text:
        return False
    current = output.get(model_key)
    if isinstance(current, dict) and is_model_merge_value(current):
        return False
    model = _ensure_model_dict_with_default_folder(output, model_key)
    model[metadata_key] = text
    return True


def _assign_model_metadata_float(
    output: dict[str, Any],
    model_key: str,
    metadata_key: str,
    value: str,
) -> bool:
    parsed = _coerce_float(value)
    if parsed is None:
        return False
    current = output.get(model_key)
    if isinstance(current, dict) and is_model_merge_value(current):
        return False
    model = _ensure_model_dict_with_default_folder(output, model_key)
    model[metadata_key] = parsed
    return True


def _parse_size(value: Any) -> tuple[int, int] | None:
    text = _coerce_string(value)
    if not text:
        return None
    match = _SIZE_RE.match(text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return format(value, "g")
    return str(value)


def _coerce_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return int(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(text, 10)
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    try:
        text = str(value).strip()
        if not text:
            return None
        numeric = float(text)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    except Exception:
        return None
