# Copyright 2026 kinorax
from __future__ import annotations

from collections.abc import Sequence

from .. import const as Const
from ..utils.selector_resolution import resolve_selector_value


def resolve_clip_selector_values(
    clip_names: Sequence[str],
    *,
    sha256_values: Sequence[str | None] | None = None,
    value_label_prefix: str = "clip_name",
) -> list[str | None]:
    clip_name_options = Const.get_CLIP_NAME_OPTIONS()
    normalized_sha256_values = list(sha256_values or ())

    resolved: list[str | None] = []
    for index, clip_name in enumerate(clip_names, start=1):
        sha256 = normalized_sha256_values[index - 1] if index - 1 < len(normalized_sha256_values) else None
        value = resolve_selector_value(
            clip_name,
            clip_name_options,
            value_label=f"{value_label_prefix}{index}",
            folder_name=Const.MODEL_FOLDER_PATH_TEXT_ENCODERS,
            sha256=sha256,
        )
        resolved.append(value)
    return resolved
