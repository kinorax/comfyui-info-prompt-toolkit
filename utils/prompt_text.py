from __future__ import annotations

import re

_COMMENT_ONLY_LINE_RE = re.compile(r"^[ \t]*//[^\r\n]*(?:\r?\n|$)", re.MULTILINE)
_INLINE_LINE_COMMENT_RE = re.compile(r"//[^\r\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_NEWLINE_RE = re.compile(r"\r\n|\r|\n")
_COMMA_WITHOUT_SPACE_RE = re.compile(r",(?=\S)")
_DOT_WITHOUT_SPACE_RE = re.compile(r"\.(?=[^\s\d])")
_CONSECUTIVE_SPACES_RE = re.compile(r" {2,}")
_SPACE_COMMA_AFTER_PUNCT_RE = re.compile(r"([,.])(?:\s+,)+")
_TRAILING_COMMAS_RE = re.compile(r"(?:,\s*)+$")
_CAPTION_TOKEN_DELIMITER_RE = re.compile(r", |\. ")
_NUMERIC_TEXT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)\Z")

CaptionToken = tuple[str, bool]


def remove_prompt_comments(text: str) -> str:
    without_block = _BLOCK_COMMENT_RE.sub("", text)
    without_comment_only_lines = _COMMENT_ONLY_LINE_RE.sub("", without_block)
    without_line_comments = _INLINE_LINE_COMMENT_RE.sub("", without_comment_only_lines)
    return without_line_comments


def normalize_prompt_prefix(prefix: object) -> str:
    if prefix is None:
        return ""

    trimmed = str(prefix).rstrip(" \r\n")
    if not trimmed:
        return ""

    if trimmed.endswith(",") or trimmed.endswith("."):
        return f"{trimmed} "
    return f"{trimmed}, "


def combine_prompt_text(start: str | None, end: str | None) -> str | None:
    if start is None and end is None:
        return None
    if start is None:
        return end
    if end is None or end == "":
        return start

    normalized_start = normalize_prompt_prefix(start)
    return f"{normalized_start}{end}"


def normalize_prompt_tokens(text: str) -> str:
    normalized = text.strip(" \r\n")
    normalized = _NEWLINE_RE.sub(",", normalized)
    normalized = _COMMA_WITHOUT_SPACE_RE.sub(", ", normalized)
    normalized = _DOT_WITHOUT_SPACE_RE.sub(". ", normalized)
    normalized = _CONSECUTIVE_SPACES_RE.sub(" ", normalized)
    normalized = _SPACE_COMMA_AFTER_PUNCT_RE.sub(r"\1", normalized)
    normalized = _COMMA_WITHOUT_SPACE_RE.sub(", ", normalized)
    normalized = _DOT_WITHOUT_SPACE_RE.sub(". ", normalized)
    normalized = _TRAILING_COMMAS_RE.sub("", normalized)
    return normalized


def strip_prompt_weights(text: str) -> str:
    stripped_chars: list[str] = []
    depth = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch == "\\" and i + 1 < n:
            next_char = text[i + 1]
            if next_char == "\\":
                stripped_chars.append("\\")
                i += 2
                continue
            if next_char in "()":
                stripped_chars.append(next_char)
                i += 2
                continue

        if ch == "(":
            depth += 1
            i += 1
            continue

        if ch == ")":
            if depth > 0:
                depth -= 1
            i += 1
            continue

        if ch == ":":
            weight_end = _prompt_weight_end(text, i)
            if weight_end is not None:
                i = weight_end
                continue

        stripped_chars.append(ch)
        i += 1

    return "".join(stripped_chars)


def _prompt_weight_end(text: str, colon_index: int) -> int | None:
    number_end = _prompt_weight_number_end(text, colon_index + 1)
    if number_end is None:
        return None

    boundary_index = number_end
    while boundary_index < len(text) and text[boundary_index] in " \t":
        boundary_index += 1

    if not _is_prompt_weight_boundary(text, boundary_index):
        return None

    if _looks_like_plain_numeric_segment(_segment_before_colon(text, colon_index)):
        return None

    return boundary_index


def _prompt_weight_number_end(text: str, start_index: int) -> int | None:
    i = start_index
    while i < len(text) and text[i] in " \t":
        i += 1

    if i < len(text) and text[i] in "+-":
        i += 1

    number_start = i
    while i < len(text) and text[i].isdigit():
        i += 1
    has_int_digits = i > number_start

    has_fraction_digits = False
    if i < len(text) and text[i] == ".":
        i += 1
        fraction_start = i
        while i < len(text) and text[i].isdigit():
            i += 1
        has_fraction_digits = i > fraction_start

    if not (has_int_digits or has_fraction_digits):
        return None
    return i


def _is_prompt_weight_boundary(text: str, index: int) -> bool:
    if index >= len(text):
        return True

    ch = text[index]
    if ch in ",)\r\n":
        return True
    if ch != ".":
        return False

    next_index = index + 1
    return next_index >= len(text) or text[next_index] in " \t\r\n"


def _segment_before_colon(text: str, colon_index: int) -> str:
    end_index = colon_index
    while end_index > 0 and text[end_index - 1] in " \t":
        end_index -= 1

    start_index = end_index
    while start_index > 0 and text[start_index - 1] not in " \t,()\r\n":
        start_index -= 1
    return text[start_index:end_index]


def _looks_like_plain_numeric_segment(text: str) -> bool:
    return bool(text) and _NUMERIC_TEXT_RE.fullmatch(text) is not None


def _split_caption_tokens(text: str) -> list[CaptionToken]:
    stripped = text.strip(" \r\n")
    if not stripped:
        return []

    items: list[CaptionToken] = []
    start = 0
    for match in _CAPTION_TOKEN_DELIMITER_RE.finditer(stripped):
        token = stripped[start : match.start()].strip(" \r\n")
        if token:
            items.append((token, match.group(0) == ". "))
        start = match.end()

    trailing = stripped[start:].strip(" \r\n")
    if not trailing:
        return items

    ends_with_dot = trailing.endswith(".")
    if ends_with_dot:
        trailing = trailing[:-1].rstrip(" \r\n")
    if trailing:
        items.append((trailing, ends_with_dot))
    return items


def _split_caption_remove_tokens(text: str) -> list[CaptionToken]:
    stripped = text.strip(" \r\n")
    if not stripped:
        return []

    items: list[CaptionToken] = []
    for raw_item in stripped.split(","):
        token = raw_item.strip(" \r\n")
        if not token:
            continue

        ends_with_dot = token.endswith(".")
        if ends_with_dot:
            token = token[:-1].rstrip(" \r\n")
        if token:
            items.append((token, ends_with_dot))
    return items


def _join_caption_tokens(tokens: list[CaptionToken]) -> str:
    if not tokens:
        return ""

    parts: list[str] = []
    last_index = len(tokens) - 1
    for index, (token, ends_with_dot) in enumerate(tokens):
        parts.append(token)
        if index >= last_index:
            if ends_with_dot:
                parts.append(".")
            continue
        parts.append(". " if ends_with_dot else ", ")
    return "".join(parts)


def merge_caption_tokens(start: str | None, end: str | None) -> str | None:
    if start is None and end is None:
        return None

    merged: list[CaptionToken] = []
    seen: set[CaptionToken] = set()

    for value in (start, end):
        if value is None:
            continue
        for token in _split_caption_tokens(value):
            if token in seen:
                continue
            seen.add(token)
            merged.append(token)

    if not merged:
        return ""

    return _join_caption_tokens(merged)


def remove_caption_tokens(string: str | None, remove: str | None) -> str | None:
    if string is None:
        return None

    source_tokens = _split_caption_tokens(string)
    if not source_tokens:
        return ""

    if remove is None:
        return _join_caption_tokens(source_tokens)

    remove_tokens = set(_split_caption_remove_tokens(remove))
    if not remove_tokens:
        return _join_caption_tokens(source_tokens)

    filtered_tokens = [token for token in source_tokens if token not in remove_tokens]
    return _join_caption_tokens(filtered_tokens)
