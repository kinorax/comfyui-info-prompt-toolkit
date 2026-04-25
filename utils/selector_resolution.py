# Copyright 2026 kinorax
from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

from .file_hash_cache import normalize_relative_path
from .model_lora_metadata_pipeline import get_shared_metadata_pipeline


def normalize_sha256_or_none(value: object) -> str | None:
    normalized = normalize_selector_value_or_none(value)
    if normalized is None or len(normalized) != 64:
        return None
    lowered = normalized.lower()
    if any(char not in "0123456789abcdef" for char in lowered):
        return None
    return lowered


def normalize_selector_value_or_none(value: object) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip()
    return normalized or None


def selector_basename(value: object) -> str:
    normalized = normalize_selector_value_or_none(value)
    if normalized is None:
        return ""

    parts = [part for part in normalized.replace("\\", "/").split("/") if part]
    if not parts:
        return normalized
    return parts[-1]


def _normalized_options(options: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(option) for option in options)


def _match_options_by_relative_paths(options: Sequence[str], relative_paths: Iterable[str]) -> list[str]:
    option_map: dict[str, str] = {}
    for option in options:
        normalized = normalize_relative_path(option)
        if normalized and normalized not in option_map:
            option_map[normalized] = option

    matches: list[str] = []
    for relative_path in relative_paths:
        normalized = normalize_relative_path(relative_path)
        if not normalized:
            continue
        option = option_map.get(normalized)
        if option and option not in matches:
            matches.append(option)
    return matches


def _render_matches(matches: Sequence[str], *, limit: int = 5) -> str:
    rendered_matches = ", ".join(matches[:limit])
    if len(matches) > limit:
        rendered_matches = f"{rendered_matches}, ..."
    return rendered_matches


def resolve_selector_value(
    selected_value: object,
    options: Sequence[str],
    *,
    value_label: str,
    folder_name: str,
    sha256: object | None = None,
) -> str | None:
    normalized = normalize_selector_value_or_none(selected_value)
    if normalized is None:
        return None

    normalized_options = _normalized_options(options)
    if normalized in normalized_options:
        return normalized

    normalized_sha256 = normalize_sha256_or_none(sha256)
    if normalized_sha256 is not None:
        pipeline = get_shared_metadata_pipeline(start=False)
        hash_matches = pipeline.find_relative_paths_by_hash(
            folder_name=folder_name,
            hash_prefix=normalized_sha256,
            preferred_algos=("sha256",),
        )
        option_matches = _match_options_by_relative_paths(normalized_options, hash_matches)
        if len(option_matches) == 1:
            return option_matches[0]
        if len(option_matches) > 1:
            raise RuntimeError(
                f"{value_label} '{normalized}' is not available and sha256 matched multiple files: "
                f"{_render_matches(option_matches)}"
            )

    basename = selector_basename(normalized)
    basename_key = basename.casefold()
    matches = [option for option in normalized_options if selector_basename(option).casefold() == basename_key]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise RuntimeError(
            f"{value_label} '{normalized}' is not available and no file named '{basename}' was found"
        )

    raise RuntimeError(
        f"{value_label} '{normalized}' is not available and file name '{basename}' matched multiple files: "
        f"{_render_matches(matches)}"
    )
