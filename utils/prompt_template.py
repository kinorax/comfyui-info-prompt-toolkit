from __future__ import annotations

import random
import re
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .prompt_wildcards import resolve_prompt_wildcards
from .prompt_text import normalize_prompt_prefix, remove_prompt_comments

_DYNAMIC_KEY_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_WEIGHT_PREFIX_RE = re.compile(r"^([+-]?(?:\d+(?:\.\d+)?|\.\d+))::(.*)$", re.DOTALL)
_UNESCAPE_NON_DOLLAR_RE = re.compile(r"\\([{}|#\\\\])")

_ESCAPED_DOLLAR_TOKEN = "\x00ESCAPED_DOLLAR\x00"

_SAMPLER_RANDOM = "random"
_SAMPLER_CYCLE = "cycle"
_DEFAULT_MULTI_SEPARATOR = ", "
_DEFAULT_CYCLE_START_MIN = 0
_DEFAULT_CYCLE_START_MAX = 1000


@dataclass(frozen=True)
class _DynamicOption:
    text: str
    weight: float


@dataclass(frozen=True)
class _MultiSelector:
    minimum: int
    maximum: int | None
    separator: str
    options_body: str


def render_prompt_template(
    template: object,
    suffix: object = None,
    extras: Mapping[str, object] | None = None,
    rng: object | None = None,
    cycle_index: int | None = None,
    wildcard_root: object | None = None,
) -> str:
    without_comments = remove_prompt_comments(_string_or_empty(template))
    extras_map = _normalize_extras(extras)
    wildcard_rendered = resolve_prompt_wildcards(
        without_comments,
        rng=rng,
        root_dir=wildcard_root,
    )
    dynamic_rendered = _render_dynamic_prompts(
        wildcard_rendered,
        rng=rng,
        cycle_index=cycle_index,
    )
    combined = _append_template_suffix(dynamic_rendered, suffix)
    unescaped = _unescape_non_dollar(combined)
    return _replace_dynamic_keys(unescaped, extras_map)


def _string_or_empty(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_extras(extras: Mapping[str, object] | None) -> dict[str, str]:
    if not isinstance(extras, Mapping):
        return {}

    output: dict[str, str] = {}
    for raw_key, raw_value in extras.items():
        key = str(raw_key).strip()
        if not key:
            continue
        output[key] = "" if raw_value is None else str(raw_value)
    return output


def _append_template_suffix(template_text: str, suffix: object) -> str:
    if suffix is None:
        return template_text
    return f"{normalize_prompt_prefix(template_text)}{_string_or_empty(suffix)}"


def _render_dynamic_prompts(
    text: str,
    rng: object | None = None,
    cycle_index: int | None = None,
) -> str:
    chunks: list[str] = []
    index = 0
    text_len = len(text)

    while index < text_len:
        if text[index] == "\\":
            if index + 1 < text_len:
                chunks.append(text[index : index + 2])
                index += 2
                continue
            chunks.append(text[index])
            index += 1
            continue

        if text[index] != "{":
            chunks.append(text[index])
            index += 1
            continue

        close_index = _find_matching_brace(text, index)
        if close_index is None:
            chunks.append(text[index])
            index += 1
            continue

        body = text[index + 1 : close_index]
        chunks.append(
            _render_dynamic_block(
                body,
                rng=rng,
                cycle_index=cycle_index,
            )
        )
        index = close_index + 1

    return "".join(chunks)


def _find_matching_brace(text: str, open_index: int) -> int | None:
    depth = 0
    index = open_index

    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2 if index + 1 < len(text) else 1
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1

    return None


def _render_dynamic_block(
    body: str,
    rng: object | None = None,
    cycle_index: int | None = None,
) -> str:
    sampler, body_without_sampler = _parse_sampler_prefix(body)
    multi_selector = _parse_multi_selector(body_without_sampler)

    if multi_selector is not None:
        options = _parse_dynamic_options(multi_selector.options_body)
        selected = _choose_multiple_options(
            options,
            selector=multi_selector,
            sampler=sampler,
            rng=rng,
            cycle_index=cycle_index,
        )
        nested_rendered = [
            _render_dynamic_prompts(
                option_text,
                rng=rng,
                cycle_index=cycle_index,
            )
            for option_text in selected
        ]
        return multi_selector.separator.join(nested_rendered)

    options = _parse_dynamic_options(body_without_sampler)
    selected = _choose_single_option(
        options,
        sampler=sampler,
        rng=rng,
        cycle_index=cycle_index,
    )
    return _render_dynamic_prompts(selected, rng=rng, cycle_index=cycle_index)


def _parse_sampler_prefix(body: str) -> tuple[Literal["random", "cycle"], str]:
    stripped = body.lstrip(" \t\r\n")
    if not stripped:
        return _SAMPLER_RANDOM, body

    marker = stripped[0]
    if marker == "~":
        return _SAMPLER_RANDOM, stripped[1:]
    if marker == "@":
        return _SAMPLER_CYCLE, stripped[1:]
    return _SAMPLER_RANDOM, body


def _parse_multi_selector(body: str) -> _MultiSelector | None:
    parts = _split_top_level(body, token="$$", maxsplit=2)
    if len(parts) < 2:
        return None

    bounds = _parse_multi_bounds(parts[0].strip())
    if bounds is None:
        return None

    if len(parts) >= 3:
        separator = parts[1]
        options_body = parts[2]
    else:
        separator = _DEFAULT_MULTI_SEPARATOR
        options_body = parts[1]

    return _MultiSelector(
        minimum=bounds[0],
        maximum=bounds[1],
        separator=separator,
        options_body=options_body,
    )


def _parse_multi_bounds(text: str) -> tuple[int, int | None] | None:
    if not text:
        return None

    if text.isdigit():
        count = int(text)
        return count, count

    if "-" not in text:
        return None

    left, right = text.split("-", 1)
    left_text = left.strip()
    right_text = right.strip()

    if left_text:
        if not left_text.isdigit():
            return None
        minimum = int(left_text)
    else:
        minimum = 1

    if right_text:
        if not right_text.isdigit():
            return None
        maximum = int(right_text)
    else:
        maximum = None

    minimum = max(0, minimum)
    if maximum is not None:
        maximum = max(0, maximum)
        if maximum < minimum:
            maximum = minimum

    return minimum, maximum


def _parse_dynamic_options(body: str) -> list[_DynamicOption]:
    return [_parse_dynamic_option(part) for part in _split_dynamic_options(body)]


def _parse_dynamic_option(option_text: str) -> _DynamicOption:
    cleaned = _trim_dynamic_option(option_text)
    m = _WEIGHT_PREFIX_RE.match(cleaned)
    if not m:
        return _DynamicOption(text=cleaned, weight=1.0)

    weight_text = m.group(1)
    content = _trim_dynamic_option(m.group(2))
    try:
        weight = float(weight_text)
    except Exception:
        return _DynamicOption(text=cleaned, weight=1.0)

    return _DynamicOption(text=content, weight=weight)


def _split_dynamic_options(body: str) -> list[str]:
    return _split_top_level(body, token="|")


def _split_top_level(text: str, token: str, maxsplit: int | None = None) -> list[str]:
    if not token:
        return [text]

    parts: list[str] = []
    depth = 0
    start = 0
    splits = 0
    index = 0
    text_len = len(text)
    token_len = len(token)

    while index < text_len:
        char = text[index]

        if char == "\\":
            index += 2 if index + 1 < text_len else 1
            continue
        if char == "{":
            depth += 1
            index += 1
            continue
        if char == "}":
            if depth > 0:
                depth -= 1
            index += 1
            continue

        can_split = maxsplit is None or splits < maxsplit
        if depth == 0 and can_split and text.startswith(token, index):
            parts.append(text[start:index])
            index += token_len
            start = index
            splits += 1
            continue

        index += 1

    parts.append(text[start:])
    return parts


def _trim_dynamic_option(option: str) -> str:
    return textwrap.dedent(option).strip(" \t\r\n")


def _choose_single_option(
    options: Sequence[_DynamicOption],
    sampler: Literal["random", "cycle"],
    rng: object | None = None,
    cycle_index: int | None = None,
) -> str:
    candidates = [option for option in options if option.weight > 0]
    if not candidates:
        return ""

    if sampler == _SAMPLER_CYCLE:
        cycle_value = _resolve_cycle_index(cycle_index, rng)
        return candidates[cycle_value % len(candidates)].text

    if all(option.weight == 1.0 for option in candidates):
        picked = _choose_one(candidates, rng)
        return picked.text if picked is not None else ""

    index = _choose_weighted_index(candidates, rng)
    return candidates[index].text


def _choose_multiple_options(
    options: Sequence[_DynamicOption],
    selector: _MultiSelector,
    sampler: Literal["random", "cycle"],
    rng: object | None = None,
    cycle_index: int | None = None,
) -> list[str]:
    candidates = [option for option in options if option.weight > 0]
    if not candidates:
        return []

    count = _resolve_selection_count(
        selector=selector,
        option_count=len(candidates),
        sampler=sampler,
        rng=rng,
        cycle_index=cycle_index,
    )
    if count <= 0:
        return []

    if sampler == _SAMPLER_CYCLE:
        cycle_value = _resolve_cycle_index(cycle_index, rng)
        start = cycle_value % len(candidates)
        return [candidates[(start + offset) % len(candidates)].text for offset in range(count)]

    pool = list(candidates)
    selected: list[str] = []
    while pool and len(selected) < count:
        if all(option.weight == 1.0 for option in pool):
            picked = _choose_one(pool, rng)
            if picked is None:
                break
            selected.append(picked.text)
            pool.remove(picked)
            continue

        picked_index = _choose_weighted_index(pool, rng)
        selected.append(pool.pop(picked_index).text)

    return selected


def _resolve_selection_count(
    selector: _MultiSelector,
    option_count: int,
    sampler: Literal["random", "cycle"],
    rng: object | None = None,
    cycle_index: int | None = None,
) -> int:
    maximum = option_count if selector.maximum is None else min(selector.maximum, option_count)
    minimum = min(max(selector.minimum, 0), maximum)

    if minimum >= maximum:
        return minimum

    if sampler == _SAMPLER_CYCLE:
        cycle_value = _resolve_cycle_index(cycle_index, rng)
        span = maximum - minimum + 1
        return minimum + (cycle_value % span)

    return _randint(minimum, maximum, rng)


def _resolve_cycle_index(cycle_index: int | None, rng: object | None = None) -> int:
    if cycle_index is None:
        return _randint(_DEFAULT_CYCLE_START_MIN, _DEFAULT_CYCLE_START_MAX, rng)

    try:
        value = int(cycle_index)
    except Exception:
        return 0
    return max(0, value)


def _choose_weighted_index(options: Sequence[_DynamicOption], rng: object | None = None) -> int:
    if not options:
        return 0

    total_weight = sum(option.weight for option in options)
    if total_weight <= 0:
        return 0

    random_value = _random_float(rng)
    if random_value is None:
        return 0

    target = random_value * total_weight
    cumulative = 0.0
    for index, option in enumerate(options):
        cumulative += option.weight
        if target < cumulative:
            return index
    return len(options) - 1


def _replace_dynamic_keys(text: str, extras: Mapping[str, str]) -> str:
    protected = text.replace(r"\$", _ESCAPED_DOLLAR_TOKEN)

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        resolved = extras.get(key)
        if resolved is not None:
            return resolved

        if "_" in key:
            space_key = key.replace("_", " ")
            resolved = extras.get(space_key)
            if resolved is not None:
                return resolved

        return f"${key}"

    replaced = _DYNAMIC_KEY_RE.sub(_replace, protected)
    return replaced.replace(_ESCAPED_DOLLAR_TOKEN, "$")


def _unescape_non_dollar(text: str) -> str:
    return _UNESCAPE_NON_DOLLAR_RE.sub(r"\1", text)


def _choose_one(options: Sequence[_DynamicOption], rng: object | None = None) -> _DynamicOption | None:
    if not options:
        return None

    chooser = rng if rng is not None else random
    if hasattr(chooser, "choice"):
        try:
            picked = chooser.choice(options)
            if isinstance(picked, _DynamicOption):
                return picked
        except Exception:
            pass

    index = _randint(0, len(options) - 1, chooser)
    return options[index]


def _random_float(rng: object | None = None) -> float | None:
    chooser = rng if rng is not None else random
    if not hasattr(chooser, "random"):
        return None

    try:
        raw = float(chooser.random())
    except Exception:
        return None

    if raw < 0.0:
        return 0.0
    if raw >= 1.0:
        return 0.999999999999
    return raw


def _randint(minimum: int, maximum: int, rng: object | None = None) -> int:
    if maximum <= minimum:
        return minimum

    chooser = rng if rng is not None else random
    if hasattr(chooser, "randrange"):
        try:
            return int(chooser.randrange(minimum, maximum + 1))
        except Exception:
            pass
    if hasattr(chooser, "randint"):
        try:
            return int(chooser.randint(minimum, maximum))
        except Exception:
            pass

    random_value = _random_float(chooser)
    if random_value is None:
        return minimum

    span = maximum - minimum + 1
    offset = int(random_value * span)
    if offset >= span:
        offset = span - 1
    return minimum + offset
