# Copyright 2026 kinorax
from __future__ import annotations

import os
from pathlib import Path

_BOX_UUID = b"uuid"
_IPT_PRIVATE_UUID = bytes.fromhex("8f4d3a2b5c714a099ec217c0d3f3a921")


def append_ipt_private_metadata(path: str | Path, payload: bytes | None) -> None:
    target = Path(path)
    data = target.read_bytes()
    payload_bytes = bytes(payload) if payload else b""

    updated = _remove_top_level_private_uuid_boxes(data)
    if payload_bytes:
        updated += _make_uuid_box(payload_bytes)
    _replace_file_bytes(target, updated)


def read_ipt_private_metadata(path: str | Path) -> bytes | None:
    data = Path(path).read_bytes()
    for box_type, payload_start, payload_end in _iter_top_level_box_payloads(data):
        if box_type != _BOX_UUID:
            continue
        payload = data[payload_start:payload_end]
        if len(payload) < 16 or payload[:16] != _IPT_PRIVATE_UUID:
            continue
        return payload[16:]
    return None


def _make_uuid_box(payload: bytes) -> bytes:
    box_size = 8 + len(_IPT_PRIVATE_UUID) + len(payload)
    if box_size <= 0xFFFFFFFF:
        return box_size.to_bytes(4, "big") + _BOX_UUID + _IPT_PRIVATE_UUID + payload
    largesize = box_size + 8
    return (
        (1).to_bytes(4, "big")
        + _BOX_UUID
        + largesize.to_bytes(8, "big")
        + _IPT_PRIVATE_UUID
        + payload
    )


def _iter_top_level_boxes(data: bytes) -> list[tuple[int, int, bytes, int, int]]:
    boxes: list[tuple[int, int, bytes, int, int]] = []
    offset = 0
    data_len = len(data)
    while offset + 8 <= data_len:
        box_start = offset
        size = int.from_bytes(data[offset : offset + 4], "big")
        box_type = data[offset + 4 : offset + 8]
        offset += 8
        header_size = 8

        if size == 1:
            if offset + 8 > data_len:
                break
            largesize = int.from_bytes(data[offset : offset + 8], "big")
            offset += 8
            header_size = 16
            if largesize < header_size:
                break
            box_end = box_start + largesize
        elif size == 0:
            box_end = data_len
        else:
            if size < header_size:
                break
            box_end = box_start + size

        if box_end > data_len:
            break

        boxes.append((box_start, box_end, box_type, box_start + header_size, box_end))
        offset = box_end
        if size == 0:
            break
    return boxes


def _iter_top_level_box_payloads(data: bytes) -> list[tuple[bytes, int, int]]:
    return [
        (box_type, payload_start, payload_end)
        for _box_start, _box_end, box_type, payload_start, payload_end in _iter_top_level_boxes(data)
    ]


def _remove_top_level_private_uuid_boxes(data: bytes) -> bytes:
    output = bytearray()
    copied_until = 0
    removed_any = False

    for box_start, box_end, box_type, payload_start, payload_end in _iter_top_level_boxes(data):
        if box_type == _BOX_UUID:
            payload = data[payload_start:payload_end]
            if len(payload) >= 16 and payload[:16] == _IPT_PRIVATE_UUID:
                output.extend(data[copied_until:box_start])
                copied_until = box_end
                removed_any = True

    if not removed_any:
        return data
    output.extend(data[copied_until:])
    return bytes(output)


def _replace_file_bytes(path: Path, data: bytes) -> None:
    temp_path = path.with_name(f".{path.name}.ipt-mp4-metadata-{os.getpid()}.tmp")
    try:
        temp_path.write_bytes(data)
        os.replace(temp_path, path)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
