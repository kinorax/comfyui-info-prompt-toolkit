from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

try:
    import folder_paths  # type: ignore
except Exception:
    folder_paths = None  # type: ignore[assignment]


_WILDCARD_TOKEN_RE = re.compile(r"__([^\r\n]+?)__(?:#(\d+))?")
_WILDCARD_WEIGHT_PREFIX_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d+)?|\.\d+))::(.*)$",
    re.DOTALL,
)
_WILDCARD_SUBDIR = ("info_prompt_toolkit", "wildcards")


@dataclass(frozen=True)
class PromptWildcardItem:
    index: int
    value: str
    display_text: str
    description: str | None = None
    weight: float = 1.0


def get_wildcards_directory(root_dir: str | Path | None = None) -> Path | None:
    if root_dir is not None:
        return Path(root_dir).resolve()

    if folder_paths is None:
        return None

    getter = getattr(folder_paths, "get_user_directory", None)
    if not callable(getter):
        return None

    try:
        user_dir = getter()
    except Exception:
        return None

    if not isinstance(user_dir, str) or not user_dir:
        return None

    return (Path(user_dir) / Path(*_WILDCARD_SUBDIR)).resolve()


def resolve_prompt_wildcards(
    text: object,
    rng: object | None = None,
    root_dir: str | Path | None = None,
) -> str:
    source = "" if text is None else str(text)
    if not source:
        return ""

    root = get_wildcards_directory(root_dir)
    if root is None or not root.is_dir():
        return source

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        selector_digits = match.group(2)
        items = read_prompt_wildcard_items_by_token(token, root_dir=root)
        if not items:
            return match.group(0)

        selector_index = _parse_selector_index(selector_digits)
        if selector_index is not None:
            selected = get_prompt_wildcard_item_by_index(items, selector_index)
            if selected is not None:
                return selected.value

            random_item = _choose_prompt_wildcard_item(items, rng=rng)
            if random_item is None:
                return match.group(0)

            random_value = random_item.value
            return f"{random_value}#{selector_digits}"

        random_item = _choose_prompt_wildcard_item(items, rng=rng)
        if random_item is None:
            return match.group(0)
        return random_item.value

    return _WILDCARD_TOKEN_RE.sub(_replace, source)


def resolve_prompt_wildcard_token(
    token: object,
    rng: object | None = None,
    root_dir: str | Path | None = None,
    selector_index: int | None = None,
) -> str | None:
    items = read_prompt_wildcard_items_by_token(token, root_dir=root_dir)
    if not items:
        return None

    if selector_index is not None:
        selected = get_prompt_wildcard_item_by_index(items, selector_index)
        return selected.value if selected is not None else None

    random_item = _choose_prompt_wildcard_item(items, rng=rng)
    return random_item.value if random_item is not None else None


def list_prompt_wildcards(
    query: object = "",
    root_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[str]:
    root = get_wildcards_directory(root_dir)
    if root is None or not root.is_dir():
        return []

    normalized_query = _normalize_wildcard_query(query)
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".txt"
        ),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )

    output: list[str] = []
    for path in files:
        token = path.relative_to(root).with_suffix("").as_posix()
        if normalized_query and not token.lower().startswith(normalized_query):
            continue
        if not read_prompt_wildcard_items(path):
            continue

        output.append(token)
        if limit is not None and limit > 0 and len(output) >= limit:
            break

    return output


def list_prompt_wildcard_items(
    token: object,
    query: object = "",
    root_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[PromptWildcardItem]:
    items = read_prompt_wildcard_items_by_token(token, root_dir=root_dir)
    if not items:
        return []

    selector_query = str(query or "").strip()
    if selector_query:
        items = [
            item for item in items if str(item.index).startswith(selector_query)
        ]

    if limit is not None and limit > 0:
        return items[:limit]
    return items


def read_prompt_wildcard_candidates(path: str | Path) -> list[str]:
    return [item.value for item in read_prompt_wildcard_items(path)]


def read_prompt_wildcard_items_by_token(
    token: object,
    root_dir: str | Path | None = None,
) -> list[PromptWildcardItem]:
    wildcard_path = resolve_prompt_wildcard_path(token, root_dir=root_dir)
    if wildcard_path is None or not wildcard_path.is_file():
        return []
    return read_prompt_wildcard_items(wildcard_path)


def read_prompt_wildcard_items(path: str | Path) -> list[PromptWildcardItem]:
    try:
        raw_text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return []

    return _parse_prompt_wildcard_items(raw_text)


def resolve_prompt_wildcard_path(
    token: object,
    root_dir: str | Path | None = None,
) -> Path | None:
    root = get_wildcards_directory(root_dir)
    if root is None:
        return None

    normalized_token = _normalize_wildcard_token(token)
    if normalized_token is None:
        return None

    try:
        candidate = (root.joinpath(*normalized_token).with_suffix(".txt")).resolve()
    except Exception:
        return None

    try:
        candidate.relative_to(root)
    except Exception:
        return None

    return candidate


def get_prompt_wildcard_item_by_index(
    items: list[PromptWildcardItem],
    selector_index: int,
) -> PromptWildcardItem | None:
    if selector_index <= 0:
        return None
    for item in items:
        if item.index == selector_index:
            return item
    return None


def _parse_prompt_wildcard_items(raw_text: str) -> list[PromptWildcardItem]:
    items: list[PromptWildcardItem] = []
    pending_description_lines: list[str] = []
    in_block_comment = False

    for raw_line in raw_text.splitlines():
        line_result = _analyze_raw_wildcard_line(
            raw_line,
            in_block_comment=in_block_comment,
        )
        in_block_comment = line_result["in_block_comment"]

        cleaned_line = str(line_result["cleaned_line"] or "").strip(" \t\r\n")
        if cleaned_line:
            parsed_item = _parse_prompt_wildcard_item_line(
                cleaned_line=cleaned_line,
                display_text=raw_line.strip(" \t\r\n"),
            )
            if parsed_item is None:
                pending_description_lines = []
                continue

            description = "\n".join(pending_description_lines).strip()
            items.append(
                PromptWildcardItem(
                    index=len(items) + 1,
                    value=parsed_item["value"],
                    display_text=parsed_item["display_text"],
                    description=description or None,
                    weight=float(parsed_item["weight"]),
                )
            )
            pending_description_lines = []
            continue

        description_lines = [
            line for line in line_result["description_lines"] if line
        ]
        if description_lines:
            pending_description_lines.extend(description_lines)
            continue

        if not in_block_comment and not raw_line.strip():
            pending_description_lines = []

    return items


def _analyze_raw_wildcard_line(
    raw_line: str,
    *,
    in_block_comment: bool,
) -> dict[str, object]:
    cleaned_parts: list[str] = []
    description_lines: list[str] = []
    index = 0
    line_length = len(raw_line)
    saw_nonspace_outside_comment = False

    while index < line_length:
        if in_block_comment:
            close_index = raw_line.find("*/", index)
            if close_index < 0:
                description_lines.extend(
                    _extract_block_comment_description(raw_line[index:])
                )
                index = line_length
                break

            description_lines.extend(
                _extract_block_comment_description(raw_line[index:close_index])
            )
            index = close_index + 2
            in_block_comment = False
            continue

        if raw_line.startswith("//", index):
            if not saw_nonspace_outside_comment:
                description_text = raw_line[index + 2 :].strip()
                if description_text:
                    description_lines.append(description_text)
            break

        if raw_line.startswith("/*", index):
            close_index = raw_line.find("*/", index + 2)
            if close_index < 0:
                if not saw_nonspace_outside_comment and not "".join(cleaned_parts).strip():
                    description_lines.extend(
                        _extract_block_comment_description(raw_line[index + 2 :])
                    )
                in_block_comment = True
                break

            if not saw_nonspace_outside_comment and not "".join(cleaned_parts).strip():
                suffix_after_comment = raw_line[close_index + 2 :]
                if not suffix_after_comment.strip():
                    description_lines.extend(
                        _extract_block_comment_description(
                            raw_line[index + 2 : close_index]
                        )
                    )

            index = close_index + 2
            continue

        char = raw_line[index]
        cleaned_parts.append(char)
        if char not in (" ", "\t"):
            saw_nonspace_outside_comment = True
        index += 1

    cleaned_line = "".join(cleaned_parts)
    if cleaned_line.strip():
        description_lines = []

    return {
        "cleaned_line": cleaned_line,
        "description_lines": description_lines,
        "in_block_comment": in_block_comment,
    }


def _extract_block_comment_description(text: str) -> list[str]:
    output: list[str] = []
    for raw_line in text.splitlines():
        normalized = raw_line.strip()
        if not normalized:
            continue
        if normalized.startswith("*"):
            normalized = normalized[1:].lstrip()
        if normalized:
            output.append(normalized)
    return output


def _parse_prompt_wildcard_item_line(
    *,
    cleaned_line: str,
    display_text: str,
) -> dict[str, object] | None:
    stripped_value = cleaned_line.strip(" \t\r\n")
    stripped_display_text = display_text.strip(" \t\r\n")
    if not stripped_value:
        return None

    match = _WILDCARD_WEIGHT_PREFIX_RE.match(stripped_value)
    if not match:
        return {
            "value": stripped_value,
            "display_text": stripped_display_text,
            "weight": 1.0,
        }

    try:
        weight = float(match.group(1))
    except Exception:
        return {
            "value": stripped_value,
            "display_text": stripped_display_text,
            "weight": 1.0,
        }

    weighted_value = match.group(2).strip(" \t\r\n")
    if weight < 0 or not weighted_value:
        return None

    display_match = _WILDCARD_WEIGHT_PREFIX_RE.match(stripped_display_text)
    weighted_display_text = stripped_display_text
    if display_match:
        weighted_display_text = display_match.group(2).strip(" \t\r\n")

    if not weighted_display_text:
        weighted_display_text = weighted_value

    return {
        "value": weighted_value,
        "display_text": weighted_display_text,
        "weight": weight,
    }


def _normalize_wildcard_query(query: object) -> str:
    text = "" if query is None else str(query)
    return text.strip().replace("\\", "/").lower()


def _normalize_wildcard_token(token: object) -> tuple[str, ...] | None:
    text = "" if token is None else str(token)
    normalized = text.strip().replace("\\", "/")
    if not normalized:
        return None

    parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    return tuple(parts)


def _parse_selector_index(selector_digits: str | None) -> int | None:
    if selector_digits is None or not selector_digits:
        return None
    try:
        return int(selector_digits)
    except Exception:
        return None


def _choose_prompt_wildcard_item(
    items: list[PromptWildcardItem],
    rng: object | None = None,
) -> PromptWildcardItem | None:
    random_items = [item for item in items if item.weight > 0]
    if not random_items:
        return None

    if len(random_items) == 1:
        return random_items[0]

    if all(item.weight == 1.0 for item in random_items):
        chooser = rng if rng is not None else random
        if hasattr(chooser, "choice"):
            try:
                picked = chooser.choice(random_items)
                if isinstance(picked, PromptWildcardItem):
                    return picked
            except Exception:
                pass

        return random_items[_randint(0, len(random_items) - 1, rng)]

    picked_index = _choose_weighted_index(random_items, rng=rng)
    return random_items[picked_index]


def _choose_candidate(candidates: list[str], rng: object | None = None) -> str:
    chooser = rng if rng is not None else random
    if hasattr(chooser, "choice"):
        try:
            picked = chooser.choice(candidates)
            if isinstance(picked, str):
                return picked
        except Exception:
            pass

    if len(candidates) == 1:
        return candidates[0]

    if hasattr(chooser, "randrange"):
        try:
            index = int(chooser.randrange(0, len(candidates)))
            if 0 <= index < len(candidates):
                return candidates[index]
        except Exception:
            pass

    if hasattr(chooser, "randint"):
        try:
            index = int(chooser.randint(0, len(candidates) - 1))
            if 0 <= index < len(candidates):
                return candidates[index]
        except Exception:
            pass

    return candidates[0]


def _choose_weighted_index(
    items: list[PromptWildcardItem],
    rng: object | None = None,
) -> int:
    total_weight = sum(item.weight for item in items)
    if total_weight <= 0:
        return 0

    random_value = _random_float(rng)
    if random_value is None:
        return 0

    target = random_value * total_weight
    cumulative = 0.0
    for index, item in enumerate(items):
        cumulative += item.weight
        if target < cumulative:
            return index
    return len(items) - 1


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
