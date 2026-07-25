# Copyright 2026 kinorax
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
from typing import Any
import uuid

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


METADATA_ENCRYPTION_SETTING_ID = "InfoPromptToolkit.MetadataEncryption.99.keys"

ENVELOPE_FORMAT = "ipt.protected-setting"
ENVELOPE_VERSION = 1
ENVELOPE_CIPHER = "AES-256-GCM"
ENVELOPE_KDF = "HKDF-SHA256"

KEY_SOURCE_DEVICE = "device"
KEY_SOURCE_LOCAL_FILE = "local-file"
KEY_SOURCES = (KEY_SOURCE_DEVICE, KEY_SOURCE_LOCAL_FILE)

LOCAL_ROOT_FORMAT = "ipt.protected-settings-root"
LOCAL_ROOT_VERSION = 1
LOCAL_ROOT_FILE_NAME = "protected_settings.key"

KEY_BYTES = 32
SALT_BYTES = 32
NONCE_BYTES = 12
TAG_BYTES = 16
MAX_REVISION = (1 << 63) - 1

_BASE64URL_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")
_IDENTIFIER_PLACEHOLDERS = {
    "0",
    "00",
    "00000000-0000-0000-0000-000000000000",
    "ffffffff-ffff-ffff-ffff-ffffffffffff",
    "default string",
    "none",
    "not applicable",
    "not specified",
    "system serial number",
    "to be filled by o.e.m.",
    "to be filled by oem",
    "unknown",
}

_MACHINE_IDENTIFIER_LOCK = threading.Lock()
_LOCAL_ROOT_LOCK = threading.Lock()
_MACHINE_IDENTIFIER_UNSET = object()
_machine_identifier_cache: object | tuple[str, str] | None = _MACHINE_IDENTIFIER_UNSET


class ProtectedSettingError(ValueError):
    """Base error for protected-setting validation and crypto failures."""


class ProtectedSettingFormatError(ProtectedSettingError):
    """The envelope or protected value has an unsupported format."""


class ProtectedSettingUnavailableError(ProtectedSettingError):
    """The root material required to protect or unprotect a value is unavailable."""


class ProtectedSettingAuthenticationError(ProtectedSettingError):
    """The protected value failed AES-GCM authentication."""


@dataclass(frozen=True)
class ProtectedSettingRegistration:
    setting_id: str
    value_version: int
    normalizer: Callable[[Any], Any]
    default_factory: Callable[[], Any]
    frontend_readable: bool = False


_REGISTRY: dict[str, ProtectedSettingRegistration] = {}


def register_protected_setting(registration: ProtectedSettingRegistration) -> None:
    if registration.setting_id in _REGISTRY:
        raise RuntimeError(f"Protected setting already registered: {registration.setting_id}")
    _REGISTRY[registration.setting_id] = registration


def get_protected_setting_registration(setting_id: str) -> ProtectedSettingRegistration:
    registration = _REGISTRY.get(str(setting_id))
    if registration is None:
        raise ProtectedSettingFormatError("The protected setting is not registered.")
    return registration


def list_protected_setting_ids() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def normalize_metadata_key_text(value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProtectedSettingFormatError("The encryption key must be a Base64URL string.")

    text = value.strip()
    if not text and allow_empty:
        return ""
    if text.endswith("=") and text.count("=") == 1:
        text = text[:-1]
    if not _BASE64URL_KEY_RE.fullmatch(text):
        raise ProtectedSettingFormatError("The encryption key must be 43 Base64URL characters.")

    decoded = _decode_base64url(text, expected_length=KEY_BYTES, field_name="key")
    normalized = _encode_base64url(decoded)
    if normalized != text:
        raise ProtectedSettingFormatError("The encryption key is not canonical Base64URL.")
    return normalized


def normalize_metadata_encryption_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtectedSettingFormatError("The metadata encryption setting must be an object.")

    version = value.get("version")
    if isinstance(version, bool) or version != 1:
        raise ProtectedSettingFormatError("The metadata encryption setting version is unsupported.")

    primary_key = normalize_metadata_key_text(value.get("primary_key"))
    fallback_key = normalize_metadata_key_text(value.get("fallback_key", ""), allow_empty=True)
    if fallback_key and fallback_key == primary_key:
        raise ProtectedSettingFormatError("The fallback key must differ from the primary key.")

    return {
        "version": 1,
        "primary_key": primary_key,
        "fallback_key": fallback_key,
    }


def generate_random_metadata_key() -> str:
    return _encode_base64url(secrets.token_bytes(KEY_BYTES))


def generate_device_default_metadata_key() -> tuple[str, bool]:
    identifier = get_machine_identifier()
    if identifier is None:
        return generate_random_metadata_key(), False

    identifier_type, normalized_identifier = identifier
    material = (
        b"IPT/device-key/v1"
        + b"\x00"
        + identifier_type.encode("utf-8")
        + b"\x00"
        + normalized_identifier.encode("utf-8")
    )
    return _encode_base64url(hashlib.sha256(material).digest()), True


def create_default_value(setting_id: str) -> Any:
    registration = get_protected_setting_registration(setting_id)
    return registration.normalizer(registration.default_factory())


def protect_setting(
    setting_id: str,
    value: Any,
    revision: int,
    *,
    key_source: str | None = None,
) -> dict[str, Any]:
    registration = get_protected_setting_registration(setting_id)
    normalized = registration.normalizer(value)
    normalized_revision = _normalize_revision(revision)

    if key_source is None:
        root, normalized_key_source = _preferred_root_material()
    else:
        normalized_key_source = _normalize_key_source(key_source)
        root = _root_material_for_source(normalized_key_source, create_local=False)

    salt = secrets.token_bytes(SALT_BYTES)
    nonce = secrets.token_bytes(NONCE_BYTES)
    envelope_header = {
        "format": ENVELOPE_FORMAT,
        "version": ENVELOPE_VERSION,
        "cipher": ENVELOPE_CIPHER,
        "kdf": ENVELOPE_KDF,
        "key_source": normalized_key_source,
        "revision": normalized_revision,
    }
    setting_key = _derive_setting_key(root, salt, setting_id)
    plaintext = _canonical_json_bytes(normalized)
    aad = _build_aad(setting_id, envelope_header)
    ciphertext = AESGCM(setting_key).encrypt(nonce, plaintext, aad)

    return {
        **envelope_header,
        "salt": _encode_base64url(salt),
        "nonce": _encode_base64url(nonce),
        "ciphertext": _encode_base64url(ciphertext),
    }


def unprotect_setting(setting_id: str, envelope: Any) -> Any:
    registration = get_protected_setting_registration(setting_id)
    normalized_envelope = normalize_envelope(envelope)
    root = _root_material_for_source(normalized_envelope["key_source"], create_local=False)
    salt = _decode_base64url(
        normalized_envelope["salt"],
        expected_length=SALT_BYTES,
        field_name="salt",
    )
    nonce = _decode_base64url(
        normalized_envelope["nonce"],
        expected_length=NONCE_BYTES,
        field_name="nonce",
    )
    ciphertext = _decode_base64url(
        normalized_envelope["ciphertext"],
        minimum_length=TAG_BYTES,
        field_name="ciphertext",
    )
    setting_key = _derive_setting_key(root, salt, setting_id)
    aad = _build_aad(setting_id, normalized_envelope)

    try:
        plaintext = AESGCM(setting_key).decrypt(nonce, ciphertext, aad)
    except InvalidTag as error:
        raise ProtectedSettingAuthenticationError(
            "The protected setting could not be authenticated."
        ) from error

    try:
        value = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtectedSettingFormatError("The protected setting plaintext is invalid JSON.") from error
    return registration.normalizer(value)


def normalize_envelope(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtectedSettingFormatError("The protected setting envelope must be an object.")

    expected = {
        "format": ENVELOPE_FORMAT,
        "version": ENVELOPE_VERSION,
        "cipher": ENVELOPE_CIPHER,
        "kdf": ENVELOPE_KDF,
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            raise ProtectedSettingFormatError(f"The protected setting {field_name} is unsupported.")

    revision = _normalize_revision(value.get("revision"))
    key_source = _normalize_key_source(value.get("key_source"))
    salt = _normalize_base64url_field(value.get("salt"), SALT_BYTES, "salt")
    nonce = _normalize_base64url_field(value.get("nonce"), NONCE_BYTES, "nonce")
    ciphertext = _normalize_base64url_field(
        value.get("ciphertext"),
        None,
        "ciphertext",
        minimum_length=TAG_BYTES,
    )

    return {
        **expected,
        "key_source": key_source,
        "revision": revision,
        "salt": salt,
        "nonce": nonce,
        "ciphertext": ciphertext,
    }


def get_envelope_revision(value: Any) -> int:
    return normalize_envelope(value)["revision"]


def get_machine_identifier(*, refresh: bool = False) -> tuple[str, str] | None:
    global _machine_identifier_cache

    with _MACHINE_IDENTIFIER_LOCK:
        if refresh:
            _machine_identifier_cache = _MACHINE_IDENTIFIER_UNSET
        if _machine_identifier_cache is _MACHINE_IDENTIFIER_UNSET:
            _machine_identifier_cache = _detect_machine_identifier()
        cached = _machine_identifier_cache

    if cached is None:
        return None
    return cached  # type: ignore[return-value]


def _detect_machine_identifier() -> tuple[str, str] | None:
    candidates: list[tuple[str, str | None]] = []
    if sys.platform.startswith("win"):
        candidates.extend(_windows_identifier_candidates())
    elif sys.platform == "darwin":
        candidates.extend(_macos_identifier_candidates())
    else:
        candidates.extend(_linux_identifier_candidates())

    for identifier_type, raw_value in candidates:
        normalized = _normalize_identifier(identifier_type, raw_value)
        if normalized is not None:
            return identifier_type, normalized

    mac = _get_standard_library_mac()
    if mac is not None:
        return "mac", mac
    return None


def _windows_identifier_candidates() -> list[tuple[str, str | None]]:
    return [
        (
            "system_uuid",
            _run_powershell("(Get-CimInstance -ClassName Win32_ComputerSystemProduct).UUID"),
        ),
        (
            "board_serial",
            _run_powershell("(Get-CimInstance -ClassName Win32_BaseBoard).SerialNumber"),
        ),
    ]


def _linux_identifier_candidates() -> list[tuple[str, str | None]]:
    return [
        ("system_uuid", _read_first_line(Path("/sys/class/dmi/id/product_uuid"))),
        ("board_serial", _read_first_line(Path("/sys/class/dmi/id/board_serial"))),
    ]


def _macos_identifier_candidates() -> list[tuple[str, str | None]]:
    output = _run_command(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"])
    if not output:
        return []

    def value_for(name: str) -> str | None:
        match = re.search(rf'"{re.escape(name)}"\s*=\s*"([^"]+)"', output)
        return match.group(1) if match else None

    return [
        ("system_uuid", value_for("IOPlatformUUID")),
        ("hardware_serial", value_for("IOPlatformSerialNumber")),
    ]


def _run_powershell(command: str) -> str | None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        return None
    return _run_command(
        [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command]
    )


def _run_command(command: list[str]) -> str | None:
    kwargs: dict[str, Any] = {
        "capture_output": True,
        "check": False,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 5,
    }
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(command, **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    text = str(completed.stdout or "").strip()
    return text.splitlines()[0].strip() if text else None


def _read_first_line(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return text.splitlines()[0].strip() if text else None


def _normalize_identifier(identifier_type: str, value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split()).lower()
    if not text or text in _IDENTIFIER_PLACEHOLDERS:
        return None

    compact = re.sub(r"[^0-9a-z]", "", text)
    if not compact or set(compact) <= {"0"} or set(compact) <= {"f"}:
        return None

    if identifier_type == "system_uuid":
        try:
            parsed = uuid.UUID(text)
        except (ValueError, AttributeError):
            return None
        if parsed.int == 0 or parsed.int == (1 << 128) - 1:
            return None
        return str(parsed)
    return text


def _get_standard_library_mac() -> str | None:
    try:
        value = uuid.getnode()
    except Exception:
        return None
    first_octet = (value >> 40) & 0xFF
    if first_octet & 0x01:
        return None
    text = f"{value:012x}"
    if set(text) <= {"0"} or set(text) <= {"f"}:
        return None
    return text


def _preferred_root_material() -> tuple[bytes, str]:
    identifier = get_machine_identifier()
    if identifier is not None:
        return _device_root_material(identifier), KEY_SOURCE_DEVICE
    return _load_local_root_key(create=True), KEY_SOURCE_LOCAL_FILE


def _root_material_for_source(key_source: str, *, create_local: bool) -> bytes:
    if key_source == KEY_SOURCE_DEVICE:
        identifier = get_machine_identifier()
        if identifier is None:
            raise ProtectedSettingUnavailableError("The device identifier is unavailable.")
        return _device_root_material(identifier)
    if key_source == KEY_SOURCE_LOCAL_FILE:
        return _load_local_root_key(create=create_local)
    raise ProtectedSettingFormatError("The protected setting key source is unsupported.")


def _device_root_material(identifier: tuple[str, str]) -> bytes:
    identifier_type, normalized_identifier = identifier
    material = (
        b"IPT/protected-settings-root/v1"
        + b"\x00"
        + identifier_type.encode("utf-8")
        + b"\x00"
        + normalized_identifier.encode("utf-8")
    )
    return hashlib.sha256(material).digest()


def _load_local_root_key(*, create: bool) -> bytes:
    with _LOCAL_ROOT_LOCK:
        return _load_local_root_key_unlocked(create=create)


def _load_local_root_key_unlocked(*, create: bool) -> bytes:
    path = _resolve_local_root_key_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if not create:
            raise ProtectedSettingUnavailableError("The local root key file is missing.")
        key = secrets.token_bytes(KEY_BYTES)
        _write_local_root_key(path, key)
        return key
    except (OSError, json.JSONDecodeError) as error:
        raise ProtectedSettingUnavailableError("The local root key file cannot be read.") from error

    if not isinstance(raw, Mapping):
        raise ProtectedSettingFormatError("The local root key file must contain an object.")
    if raw.get("format") != LOCAL_ROOT_FORMAT or raw.get("version") != LOCAL_ROOT_VERSION:
        raise ProtectedSettingFormatError("The local root key file format is unsupported.")
    return _decode_base64url(raw.get("key"), expected_length=KEY_BYTES, field_name="key")


def _resolve_local_root_key_path() -> Path:
    from .settings import resolve_settings_path

    return resolve_settings_path().with_name(LOCAL_ROOT_FILE_NAME)


def _write_local_root_key(path: Path, key: bytes) -> None:
    payload = {
        "format": LOCAL_ROOT_FORMAT,
        "version": LOCAL_ROOT_VERSION,
        "key": _encode_base64url(key),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError as error:
        raise ProtectedSettingUnavailableError("The local root key file cannot be written.") from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _derive_setting_key(root: bytes, salt: bytes, setting_id: str) -> bytes:
    info = b"IPT/protected-setting/v1" + b"\x00" + setting_id.encode("utf-8")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_BYTES,
        salt=salt,
        info=info,
    ).derive(root)


def _build_aad(setting_id: str, envelope: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes(
        {
            "cipher": envelope["cipher"],
            "format": envelope["format"],
            "kdf": envelope["kdf"],
            "key_source": envelope["key_source"],
            "revision": envelope["revision"],
            "setting_id": setting_id,
            "version": envelope["version"],
        }
    )


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
        raise ProtectedSettingFormatError("The protected setting is not valid JSON.") from error


def _normalize_revision(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtectedSettingFormatError("The protected setting revision must be an integer.")
    if value < 1 or value > MAX_REVISION:
        raise ProtectedSettingFormatError("The protected setting revision is out of range.")
    return value


def _normalize_key_source(value: Any) -> str:
    if value not in KEY_SOURCES:
        raise ProtectedSettingFormatError("The protected setting key source is unsupported.")
    return str(value)


def _normalize_base64url_field(
    value: Any,
    expected_length: int | None,
    field_name: str,
    *,
    minimum_length: int | None = None,
) -> str:
    decoded = _decode_base64url(
        value,
        expected_length=expected_length,
        minimum_length=minimum_length,
        field_name=field_name,
    )
    normalized = _encode_base64url(decoded)
    if normalized != value:
        raise ProtectedSettingFormatError(f"The protected setting {field_name} is not canonical Base64URL.")
    return normalized


def _encode_base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode_base64url(
    value: Any,
    *,
    expected_length: int | None = None,
    minimum_length: int | None = None,
    field_name: str,
) -> bytes:
    if not isinstance(value, str) or not value or "=" in value:
        raise ProtectedSettingFormatError(f"The protected setting {field_name} is invalid Base64URL.")
    padding = "=" * ((4 - len(value) % 4) % 4)
    try:
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (ValueError, TypeError) as error:
        raise ProtectedSettingFormatError(
            f"The protected setting {field_name} is invalid Base64URL."
        ) from error
    if expected_length is not None and len(decoded) != expected_length:
        raise ProtectedSettingFormatError(
            f"The protected setting {field_name} has an invalid length."
        )
    if minimum_length is not None and len(decoded) < minimum_length:
        raise ProtectedSettingFormatError(
            f"The protected setting {field_name} has an invalid length."
        )
    return decoded


def _metadata_encryption_default_factory() -> dict[str, Any]:
    primary_key, _reproducible = generate_device_default_metadata_key()
    return {
        "version": 1,
        "primary_key": primary_key,
        "fallback_key": "",
    }


register_protected_setting(
    ProtectedSettingRegistration(
        setting_id=METADATA_ENCRYPTION_SETTING_ID,
        value_version=1,
        normalizer=normalize_metadata_encryption_settings,
        default_factory=_metadata_encryption_default_factory,
        frontend_readable=True,
    )
)
