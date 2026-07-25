# Copyright 2026 kinorax
from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Mapping

WEBM_IMAGE_INFOTEXT_TAG = "_IPT_IMAGE_INFOTEXT"
WEBM_VIDEO_INFOTEXT_TAG = "_IPT_VIDEO_INFOTEXT"
WEBM_VIDEO_SAVER_TAG = "_IPT_VIDEO_SAVER"
WEBM_PRIVATE_METADATA_TAG = "_IPT_NPTC"

_LOGICAL_TO_WEBM_TAG = {
    "image_infotext": WEBM_IMAGE_INFOTEXT_TAG,
    "video_infotext": WEBM_VIDEO_INFOTEXT_TAG,
    "video_saver": WEBM_VIDEO_SAVER_TAG,
}


def build_matroska_metadata(
    metadata: Mapping[str, Any],
    encrypted_payload: bytes | None = None,
) -> dict[str, str]:
    output: dict[str, str] = {}
    for logical_key, tag_name in _LOGICAL_TO_WEBM_TAG.items():
        if logical_key not in metadata:
            continue
        value = metadata[logical_key]
        if isinstance(value, str):
            serialized = value
        else:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
        if serialized:
            output[tag_name] = serialized

    if encrypted_payload:
        output[WEBM_PRIVATE_METADATA_TAG] = encode_webm_private_metadata(encrypted_payload)
    return output


def build_webm_metadata(
    metadata: Mapping[str, Any],
    encrypted_payload: bytes | None = None,
) -> dict[str, str]:
    return build_matroska_metadata(metadata, encrypted_payload)


def extract_matroska_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    by_upper_name = {
        str(key).upper(): str(value)
        for key, value in metadata.items()
        if value is not None
    }
    return {
        logical_key: by_upper_name[tag_name]
        for logical_key, tag_name in _LOGICAL_TO_WEBM_TAG.items()
        if tag_name in by_upper_name
    }


def extract_webm_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    return extract_matroska_metadata(metadata)


def extract_matroska_private_metadata(metadata: Mapping[str, Any]) -> bytes | None:
    encoded = _find_case_insensitive_value(metadata, WEBM_PRIVATE_METADATA_TAG)
    if encoded is None:
        return None
    return decode_webm_private_metadata(encoded)


def extract_webm_private_metadata(metadata: Mapping[str, Any]) -> bytes | None:
    return extract_matroska_private_metadata(metadata)


def encode_webm_private_metadata(payload: bytes) -> str:
    return base64.urlsafe_b64encode(bytes(payload)).decode("ascii").rstrip("=")


def decode_webm_private_metadata(value: Any) -> bytes:
    try:
        encoded = str(value).encode("ascii")
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError("Matroska private metadata is not valid Base64URL") from exc
    if not encoded or len(encoded) % 4 == 1:
        raise ValueError("Matroska private metadata is not valid Base64URL")

    padded = encoded + (b"=" * ((-len(encoded)) % 4))
    try:
        return base64.b64decode(padded, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Matroska private metadata is not valid Base64URL") from exc


def _find_case_insensitive_value(metadata: Mapping[str, Any], target_key: str) -> Any | None:
    target_upper = target_key.upper()
    for key, value in metadata.items():
        if str(key).upper() == target_upper:
            return value
    return None
