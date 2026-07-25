# Copyright 2026 kinorax
from __future__ import annotations

import base64
import copy
import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .. import const as Const
from .a1111_infotext import a1111_infotext_to_image_info, image_info_to_a1111_infotext
from .image_info_encryption_targets import (
    TARGET_TO_IMAGE_INFO_KEYS,
    associated_hash_extra_keys,
    encryption_targets_from_image_info,
    normalize_encryption_targets,
    normalize_extra_keys,
    normalize_main_fields,
)
from .image_info_hash_extras import add_civitai_hash_extras, clear_representative_hash_extras
from .image_info_normalizer import normalize_image_info_with_comfy_options
from .settings import SettingsStorageError, get_metadata_encryption_settings

_PAYLOAD_VERSION = 1
_PACKED_VERSION = 1
_VIDEO_PACKED_VERSION = 2
_NONCE_BYTES = 12
_KEY_BYTES = 32
_AAD = b"IPT/metadata-encryption/v1"


@dataclass(frozen=True)
class PreparedMetadata:
    infotext: str
    encrypted_payload: bytes | None


def prepare_image_info_metadata(image_info: Any) -> PreparedMetadata:
    if not isinstance(image_info, Mapping):
        return PreparedMetadata("", None)

    targets = encryption_targets_from_image_info(image_info)
    image_info_without_hashes = clear_representative_hash_extras(image_info)
    image_info_with_hashes = add_civitai_hash_extras(image_info_without_hashes)

    if targets is None:
        public_info = dict(image_info_with_hashes)
        public_info.pop(Const.IMAGEINFO_ENCRYPTION_TARGETS, None)
        return PreparedMetadata(image_info_to_a1111_infotext(public_info), None)

    image_info_with_hashes = _restore_selected_original_extras_for_encryption(
        original_info=image_info,
        image_info=image_info_with_hashes,
        targets=targets,
    )
    public_info, encrypted_info = split_image_info_for_encryption(image_info_with_hashes, targets)
    encrypted_payload = encrypt_metadata_payload(
        {
            "version": _PAYLOAD_VERSION,
            "targets": targets,
            "image_info": encrypted_info,
        }
    )
    return PreparedMetadata(image_info_to_a1111_infotext(public_info), encrypted_payload)


def image_info_to_public_infotext(image_info: Any) -> str:
    return prepare_image_info_metadata(image_info).infotext


def pack_video_encrypted_payloads(
    *,
    image_payload: bytes | None = None,
    video_payload: bytes | None = None,
) -> bytes | None:
    entries: dict[str, str] = {}
    if image_payload:
        entries["i"] = _encode_base64url(bytes(image_payload))
    if video_payload:
        entries["v"] = _encode_base64url(bytes(video_payload))
    if not entries:
        return None
    return bytes([_VIDEO_PACKED_VERSION]) + json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def unpack_video_encrypted_payloads(
    payload: bytes | None,
    *,
    strict: bool = False,
) -> dict[str, bytes | None]:
    if not payload:
        return {"image": None, "video": None}
    try:
        return _unpack_video_encrypted_payloads(payload)
    except MetadataEncryptionFormatError:
        if strict:
            raise
        return {"image": None, "video": None}


def _unpack_video_encrypted_payloads(payload: bytes) -> dict[str, bytes | None]:
    if not isinstance(payload, (bytes, bytearray)):
        raise MetadataEncryptionFormatError("The video metadata wrapper must be bytes.")
    packed = bytes(payload)
    if not packed or packed[0] != _VIDEO_PACKED_VERSION:
        raise MetadataEncryptionFormatError("The video metadata wrapper version is unsupported.")
    try:
        entries = json.loads(packed[1:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataEncryptionFormatError("The video metadata wrapper is invalid JSON.") from exc
    if not isinstance(entries, Mapping):
        raise MetadataEncryptionFormatError("The video metadata wrapper must contain an object.")

    output: dict[str, bytes | None] = {"image": None, "video": None}
    for entry_key, output_key in (("i", "image"), ("v", "video")):
        if entry_key not in entries:
            continue
        decoded = _decode_optional_base64url(entries.get(entry_key))
        if decoded is None:
            raise MetadataEncryptionFormatError(
                f"The video metadata wrapper entry `{entry_key}` is invalid."
            )
        output[output_key] = decoded
    if not any(output.values()):
        raise MetadataEncryptionFormatError("The video metadata wrapper contains no payload.")
    return output


def split_image_info_for_encryption(
    image_info: Mapping[str, Any],
    encryption_targets: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    targets = normalize_encryption_targets(encryption_targets)
    public_info = dict(image_info)
    public_info.pop(Const.IMAGEINFO_ENCRYPTION_TARGETS, None)
    encrypted_info: dict[str, Any] = {}

    if targets is None:
        return public_info, encrypted_info

    main_fields = normalize_main_fields(targets.get("main_fields"))
    extra_keys = list(normalize_extra_keys(targets.get("extra_keys")))
    extra_keys.extend(associated_hash_extra_keys(main_fields))

    for main_field in main_fields:
        for image_info_key in TARGET_TO_IMAGE_INFO_KEYS.get(main_field, tuple()):
            if image_info_key not in public_info:
                continue
            encrypted_info[image_info_key] = copy.deepcopy(public_info.pop(image_info_key))

    public_extras_raw = public_info.get(Const.IMAGEINFO_EXTRAS)
    if isinstance(public_extras_raw, Mapping):
        public_extras = dict(public_extras_raw)
        encrypted_extras: dict[str, Any] = {}
        seen_extra_keys: set[str] = set()
        for extra_key in extra_keys:
            if extra_key in seen_extra_keys:
                continue
            seen_extra_keys.add(extra_key)
            if extra_key not in public_extras:
                continue
            encrypted_extras[extra_key] = copy.deepcopy(public_extras.pop(extra_key))

        if public_extras:
            public_info[Const.IMAGEINFO_EXTRAS] = public_extras
        else:
            public_info.pop(Const.IMAGEINFO_EXTRAS, None)
        if encrypted_extras:
            encrypted_info[Const.IMAGEINFO_EXTRAS] = encrypted_extras

    return public_info, encrypted_info


def _restore_selected_original_extras_for_encryption(
    *,
    original_info: Mapping[str, Any],
    image_info: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    original_extras = original_info.get(Const.IMAGEINFO_EXTRAS)
    if not isinstance(original_extras, Mapping):
        return dict(image_info)

    selected_extra_keys: list[str] = list(normalize_extra_keys(targets.get("extra_keys")))
    selected_extra_keys.extend(associated_hash_extra_keys(targets.get("main_fields")))
    if not selected_extra_keys:
        return dict(image_info)

    output = dict(image_info)
    extras_raw = output.get(Const.IMAGEINFO_EXTRAS)
    extras: dict[str, Any] = dict(extras_raw) if isinstance(extras_raw, Mapping) else {}

    changed = False
    seen: set[str] = set()
    for extra_key in selected_extra_keys:
        if extra_key in seen:
            continue
        seen.add(extra_key)
        if extra_key in extras or extra_key not in original_extras:
            continue
        extras[extra_key] = copy.deepcopy(original_extras[extra_key])
        changed = True

    if changed:
        output[Const.IMAGEINFO_EXTRAS] = extras
    return output


def image_info_from_metadata(
    infotext: str | None,
    encrypted_payload: bytes | None,
    *,
    file_name: str | None = None,
) -> dict[str, Any]:
    image_info = a1111_infotext_to_image_info(infotext)
    image_info = merge_encrypted_metadata(image_info, encrypted_payload, file_name=file_name)
    return normalize_image_info_with_comfy_options(image_info)


def merge_encrypted_metadata(
    image_info: Mapping[str, Any] | None,
    encrypted_payload: bytes | None,
    *,
    file_name: str | None = None,
) -> dict[str, Any]:
    output = dict(image_info) if isinstance(image_info, Mapping) else {}
    extras_raw = output.get(Const.IMAGEINFO_EXTRAS)
    if isinstance(extras_raw, Mapping):
        output[Const.IMAGEINFO_EXTRAS] = dict(extras_raw)

    if not encrypted_payload:
        return output

    try:
        payload = decrypt_metadata_payload(encrypted_payload)
        targets = normalize_encryption_targets(payload.get("targets"))
        encrypted_info = payload.get("image_info")
        if targets is None or not isinstance(encrypted_info, Mapping):
            raise MetadataEncryptionFormatError("The encrypted metadata payload is invalid.")
    except Exception:
        _warn_decryption_failed(file_name)
        return output

    encrypted_extras = encrypted_info.get(Const.IMAGEINFO_EXTRAS)
    for key, value in encrypted_info.items():
        if key == Const.IMAGEINFO_EXTRAS:
            continue
        output[str(key)] = copy.deepcopy(value)

    if isinstance(encrypted_extras, Mapping):
        merged_extras = dict(output.get(Const.IMAGEINFO_EXTRAS)) if isinstance(output.get(Const.IMAGEINFO_EXTRAS), Mapping) else {}
        for key, value in encrypted_extras.items():
            if not isinstance(key, str):
                continue
            merged_extras[key] = copy.deepcopy(value)
        if merged_extras:
            output[Const.IMAGEINFO_EXTRAS] = merged_extras
        else:
            output.pop(Const.IMAGEINFO_EXTRAS, None)

    output[Const.IMAGEINFO_ENCRYPTION_TARGETS] = targets
    return output


def encrypt_metadata_payload(payload: Mapping[str, Any]) -> bytes:
    key = _primary_key_bytes()
    plaintext = _canonical_json_bytes(payload)
    nonce = secrets.token_bytes(_NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, _AAD)
    return bytes([_PACKED_VERSION]) + nonce + ciphertext


def decrypt_metadata_payload(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, (bytes, bytearray)):
        raise MetadataEncryptionFormatError("The encrypted metadata payload must be bytes.")
    packed = bytes(payload)
    if len(packed) <= 1 + _NONCE_BYTES + 16:
        raise MetadataEncryptionFormatError("The encrypted metadata payload is too short.")
    if packed[0] != _PACKED_VERSION:
        raise MetadataEncryptionFormatError("The encrypted metadata payload version is unsupported.")

    nonce = packed[1 : 1 + _NONCE_BYTES]
    ciphertext = packed[1 + _NONCE_BYTES :]
    last_error: Exception | None = None
    for key in _decryption_key_candidates():
        try:
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, _AAD)
            decoded = json.loads(plaintext.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise MetadataEncryptionFormatError("The encrypted metadata plaintext must be an object.")
            if decoded.get("version") != _PAYLOAD_VERSION:
                raise MetadataEncryptionFormatError("The encrypted metadata plaintext version is unsupported.")
            return decoded
        except InvalidTag as error:
            last_error = error
            continue
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MetadataEncryptionFormatError("The encrypted metadata plaintext is invalid JSON.") from error

    raise MetadataEncryptionAuthenticationError("The encrypted metadata could not be authenticated.") from last_error


class MetadataEncryptionError(ValueError):
    """Base error for IPT metadata encryption payloads."""


class MetadataEncryptionFormatError(MetadataEncryptionError):
    """The encrypted metadata payload has an unsupported format."""


class MetadataEncryptionAuthenticationError(MetadataEncryptionError):
    """No configured key could authenticate the encrypted metadata."""


def _primary_key_bytes() -> bytes:
    settings = _metadata_encryption_settings()
    return _decode_key_text(settings.get("primary_key"))


def _decryption_key_candidates() -> tuple[bytes, ...]:
    settings = _metadata_encryption_settings()
    keys: list[bytes] = []
    primary_key = _decode_key_text(settings.get("primary_key"))
    keys.append(primary_key)
    fallback_key = settings.get("fallback_key")
    if isinstance(fallback_key, str) and fallback_key.strip():
        fallback = _decode_key_text(fallback_key)
        if fallback != primary_key:
            keys.append(fallback)
    return tuple(keys)


def _metadata_encryption_settings() -> Mapping[str, Any]:
    try:
        return get_metadata_encryption_settings()
    except SettingsStorageError as error:
        raise MetadataEncryptionFormatError("The metadata encryption setting is unavailable.") from error


def _decode_key_text(value: Any) -> bytes:
    if not isinstance(value, str):
        raise MetadataEncryptionFormatError("The metadata encryption key is invalid.")
    text = value.strip()
    padding = "=" * ((4 - len(text) % 4) % 4)
    try:
        decoded = base64.b64decode(text + padding, altchars=b"-_", validate=True)
    except (ValueError, TypeError) as error:
        raise MetadataEncryptionFormatError("The metadata encryption key is invalid.") from error
    if len(decoded) != _KEY_BYTES:
        raise MetadataEncryptionFormatError("The metadata encryption key length is invalid.")
    return decoded


def _encode_base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode_optional_base64url(value: Any) -> bytes | None:
    if not isinstance(value, str) or not value:
        return None
    padding = "=" * ((4 - len(value) % 4) % 4)
    try:
        return base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (ValueError, TypeError):
        return None


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MetadataEncryptionFormatError("The encrypted metadata plaintext is not valid JSON.") from error


def _warn_decryption_failed(file_name: str | None) -> None:
    name = str(file_name or "").strip()
    if name:
        message = f'[IPT] Encrypted metadata could not be decrypted for "{name}"; using public metadata only.'
    else:
        message = "[IPT] Encrypted metadata could not be decrypted; using public metadata only."
    try:
        print(message)
    except Exception:
        pass
