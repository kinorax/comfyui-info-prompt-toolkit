# Copyright 2026 kinorax
from __future__ import annotations

import binascii
import os
from pathlib import Path

from PIL import Image

from . import exif as Exif

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SOI = b"\xff\xd8"
_RIFF = b"RIFF"
_WEBP = b"WEBP"
_PNG_PARAMETERS_KEY = b"parameters"
_EXIF_PREFIX = b"Exif\x00\x00"
_WEBP_EXIF_FLAG = 0x08
_WEBP_XMP_FLAG = 0x04
_WEBP_ANIMATION_FLAG = 0x02
_WEBP_ALPHA_FLAG = 0x10
_WEBP_ICC_FLAG = 0x20


def write_a1111_text_metadata_only(path: str | Path, text: str) -> None:
    target = Path(path)
    data = target.read_bytes()
    infotext = text if isinstance(text, str) else ""

    if data.startswith(_PNG_SIGNATURE):
        updated = _rewrite_png_parameters(data, infotext)
    elif data.startswith(_JPEG_SOI):
        updated = _rewrite_jpeg_exif(data, _build_user_comment_exif(target, infotext))
    elif data.startswith(_RIFF) and data[8:12] == _WEBP:
        updated = _rewrite_webp_exif(data, _build_user_comment_exif(target, infotext))
    else:
        raise ValueError(f"unsupported image format for metadata-only rewrite: {target.suffix or target.name}")

    _replace_file_bytes(target, updated)


def _replace_file_bytes(path: Path, data: bytes) -> None:
    temp_path = path.with_name(f".{path.name}.ipt-metadata-{os.getpid()}.tmp")
    try:
        temp_path.write_bytes(data)
        os.replace(temp_path, path)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


def _encode_user_comment(text: str) -> bytes:
    return b"UNICODE\x00" + text.encode("utf-16-be")


def _build_user_comment_exif(path: Path, text: str) -> bytes:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
    except Exception:
        exif = Image.Exif()

    if exif is None:
        exif = Image.Exif()

    if text:
        exif[Exif.EXIF_USERCOMMENT_TAG] = _encode_user_comment(text)
    else:
        try:
            del exif[Exif.EXIF_USERCOMMENT_TAG]
        except Exception:
            pass

    try:
        if len(exif) == 0:
            return b""
    except Exception:
        pass
    return bytes(exif.tobytes())


def _make_png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(chunk_type)
    crc = binascii.crc32(payload, crc) & 0xFFFFFFFF
    return len(payload).to_bytes(4, "big") + chunk_type + payload + crc.to_bytes(4, "big")


def _read_png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError("not a PNG file")

    chunks: list[tuple[bytes, bytes]] = []
    offset = len(_PNG_SIGNATURE)
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("truncated PNG chunk header")
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        crc_end = payload_end + 4
        if crc_end > len(data):
            raise ValueError("truncated PNG chunk payload")
        chunks.append((chunk_type, data[payload_start:payload_end]))
        offset = crc_end
        if chunk_type == b"IEND":
            break
    return chunks


def _png_text_keyword(chunk_type: bytes, payload: bytes) -> bytes | None:
    if chunk_type not in (b"tEXt", b"zTXt", b"iTXt"):
        return None
    keyword, separator, _ = payload.partition(b"\x00")
    if not separator:
        return None
    return keyword


def _make_png_parameters_chunk(text: str) -> tuple[bytes, bytes]:
    encoded = text.encode("utf-8")
    payload = _PNG_PARAMETERS_KEY + b"\x00\x00\x00\x00\x00" + encoded
    return b"iTXt", payload


def _rewrite_png_parameters(data: bytes, text: str) -> bytes:
    chunks = _read_png_chunks(data)
    output = bytearray(_PNG_SIGNATURE)
    inserted = False

    for chunk_type, payload in chunks:
        if _png_text_keyword(chunk_type, payload) == _PNG_PARAMETERS_KEY:
            continue

        if chunk_type == b"IDAT" and text and not inserted:
            output.extend(_make_png_chunk(*_make_png_parameters_chunk(text)))
            inserted = True

        output.extend(_make_png_chunk(chunk_type, payload))

    if text and not inserted:
        raise ValueError("PNG file did not contain an IDAT chunk")
    return bytes(output)


def _normalize_exif_app1_payload(exif_payload: bytes) -> bytes:
    if not exif_payload:
        return b""
    if exif_payload.startswith(_EXIF_PREFIX):
        return exif_payload
    return _EXIF_PREFIX + exif_payload


def _make_jpeg_segment(marker: int, payload: bytes) -> bytes:
    segment_length = len(payload) + 2
    if segment_length > 0xFFFF:
        raise ValueError("EXIF metadata is too large for a JPEG APP1 segment")
    return b"\xff" + bytes([marker]) + segment_length.to_bytes(2, "big") + payload


def _rewrite_jpeg_exif(data: bytes, exif_payload: bytes) -> bytes:
    if not data.startswith(_JPEG_SOI):
        raise ValueError("not a JPEG file")

    exif_app1_payload = _normalize_exif_app1_payload(exif_payload)
    output = bytearray(_JPEG_SOI)
    offset = 2
    inserted = False

    def insert_exif() -> None:
        nonlocal inserted
        if inserted:
            return
        if exif_app1_payload:
            output.extend(_make_jpeg_segment(0xE1, exif_app1_payload))
        inserted = True

    while offset < len(data):
        marker_start = offset
        if data[offset] != 0xFF:
            insert_exif()
            output.extend(data[offset:])
            return bytes(output)

        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            insert_exif()
            output.extend(data[marker_start:])
            return bytes(output)

        marker = data[offset]
        offset += 1

        if marker == 0x00 or marker == 0x01 or 0xD0 <= marker <= 0xD9:
            output.extend(data[marker_start:offset])
            continue

        if offset + 2 > len(data):
            raise ValueError("truncated JPEG segment length")
        segment_length = int.from_bytes(data[offset : offset + 2], "big")
        if segment_length < 2:
            raise ValueError("invalid JPEG segment length")

        segment_end = offset + segment_length
        if segment_end > len(data):
            raise ValueError("truncated JPEG segment payload")

        payload = data[offset + 2 : segment_end]
        segment = data[marker_start:segment_end]
        offset = segment_end

        if marker == 0xDA:
            insert_exif()
            output.extend(segment)
            output.extend(data[offset:])
            return bytes(output)

        if marker == 0xE1 and payload.startswith(_EXIF_PREFIX):
            continue

        if marker != 0xE0:
            insert_exif()
        output.extend(segment)

    insert_exif()
    return bytes(output)


def _read_webp_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    if not data.startswith(_RIFF) or data[8:12] != _WEBP:
        raise ValueError("not a WebP file")

    chunks: list[tuple[bytes, bytes]] = []
    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("truncated WebP chunk header")
        chunk_type = data[offset : offset + 4]
        length = int.from_bytes(data[offset + 4 : offset + 8], "little")
        payload_start = offset + 8
        payload_end = payload_start + length
        padded_end = payload_end + (length & 1)
        if padded_end > len(data):
            raise ValueError("truncated WebP chunk payload")
        chunks.append((chunk_type, data[payload_start:payload_end]))
        offset = padded_end
    return chunks


def _make_webp_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    padding = b"\x00" if len(payload) & 1 else b""
    return chunk_type + len(payload).to_bytes(4, "little") + payload + padding


def _parse_webp_canvas_size(chunks: list[tuple[bytes, bytes]]) -> tuple[int, int]:
    for chunk_type, payload in chunks:
        if chunk_type == b"VP8X" and len(payload) >= 10:
            width = int.from_bytes(payload[4:7], "little") + 1
            height = int.from_bytes(payload[7:10], "little") + 1
            return width, height
        if chunk_type == b"VP8 " and len(payload) >= 10 and payload[3:6] == b"\x9d\x01\x2a":
            width = int.from_bytes(payload[6:8], "little") & 0x3FFF
            height = int.from_bytes(payload[8:10], "little") & 0x3FFF
            return width, height
        if chunk_type == b"VP8L" and len(payload) >= 5 and payload[0] == 0x2F:
            bits = int.from_bytes(payload[1:5], "little")
            width = (bits & 0x3FFF) + 1
            height = ((bits >> 14) & 0x3FFF) + 1
            return width, height
    raise ValueError("failed to determine WebP canvas size")


def _infer_webp_feature_flags(chunks: list[tuple[bytes, bytes]], has_exif: bool) -> int:
    flags = _WEBP_EXIF_FLAG if has_exif else 0
    for chunk_type, payload in chunks:
        if chunk_type == b"ICCP":
            flags |= _WEBP_ICC_FLAG
        elif chunk_type == b"XMP ":
            flags |= _WEBP_XMP_FLAG
        elif chunk_type in (b"ANIM", b"ANMF"):
            flags |= _WEBP_ANIMATION_FLAG
        elif chunk_type == b"ALPH":
            flags |= _WEBP_ALPHA_FLAG
        elif chunk_type == b"VP8L" and len(payload) >= 5 and payload[0] == 0x2F:
            bits = int.from_bytes(payload[1:5], "little")
            if ((bits >> 28) & 1) != 0:
                flags |= _WEBP_ALPHA_FLAG
    return flags


def _make_vp8x_chunk(chunks: list[tuple[bytes, bytes]], has_exif: bool) -> tuple[bytes, bytes]:
    width, height = _parse_webp_canvas_size(chunks)
    if width <= 0 or height <= 0 or width > 0x1000000 or height > 0x1000000:
        raise ValueError("WebP canvas size is out of VP8X range")
    flags = _infer_webp_feature_flags(chunks, has_exif)
    payload = bytes([flags]) + b"\x00\x00\x00"
    payload += (width - 1).to_bytes(3, "little")
    payload += (height - 1).to_bytes(3, "little")
    return b"VP8X", payload


def _set_vp8x_exif_flag(payload: bytes, has_exif: bool) -> bytes:
    if len(payload) < 10:
        raise ValueError("invalid VP8X chunk")
    flags = payload[0] | _WEBP_EXIF_FLAG if has_exif else payload[0] & ~_WEBP_EXIF_FLAG
    return bytes([flags]) + payload[1:]


def _rewrite_webp_exif(data: bytes, exif_payload: bytes) -> bytes:
    chunks = _read_webp_chunks(data)
    has_vp8x = any(chunk_type == b"VP8X" for chunk_type, _ in chunks)
    has_exif = bool(exif_payload)

    rewritten_chunks: list[tuple[bytes, bytes]] = []
    for chunk_type, payload in chunks:
        if chunk_type == b"EXIF":
            continue
        if chunk_type == b"VP8X":
            rewritten_chunks.append((chunk_type, _set_vp8x_exif_flag(payload, has_exif)))
            continue
        rewritten_chunks.append((chunk_type, payload))

    if has_exif and not has_vp8x:
        rewritten_chunks.insert(0, _make_vp8x_chunk(chunks, has_exif=True))
    if has_exif:
        rewritten_chunks.append((b"EXIF", exif_payload))

    body = b"".join(_make_webp_chunk(chunk_type, payload) for chunk_type, payload in rewritten_chunks)
    riff_size = len(body) + 4
    return _RIFF + riff_size.to_bytes(4, "little") + _WEBP + body
