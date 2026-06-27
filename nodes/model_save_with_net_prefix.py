# Copyright 2026 kinorax
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import folder_paths
from comfy_api.latest import io as c_io

from .. import const as Const
from ..utils import cast as Cast

MODEL_RUNTIME_TYPE = c_io.Custom("MODEL")

_SOURCE_PREFIX = "model.diffusion_model."
_TARGET_PREFIX = "net."
_OUTPUT_EXT = ".safetensors"
_DEFAULT_FILENAME_PREFIX = "diffusion_models/ComfyUI_net"


def _normalize_filename_prefix(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("filename_prefix must not be empty")
    return text


def _relative_to_output_root(path: Path) -> str:
    output_root = Path(folder_paths.get_output_directory()).resolve()
    try:
        return path.resolve().relative_to(output_root).as_posix()
    except Exception as exc:
        raise ValueError("saved file path is outside ComfyUI output directory") from exc


def _build_output_path(filename_prefix: str) -> Path:
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, _subfolder, _filename_prefix = folder_paths.get_save_image_path(
        filename_prefix,
        output_dir,
    )
    return Path(full_output_folder) / f"{filename}_{counter:05}_{_OUTPUT_EXT}"


def _validate_loaded_model(model: object) -> None:
    if model is None:
        raise ValueError("model is required")

    state_dict_for_saving = getattr(model, "state_dict_for_saving", None)
    if not callable(state_dict_for_saving):
        raise ValueError("model must be a loaded ComfyUI MODEL")


def _rewrite_safetensors_header_with_net_prefix(path: Path) -> None:
    with path.open("r+b") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))

        rewritten: dict[str, Any] = {}
        normalized_model_keys: set[str] = set()
        model_key_count = 0

        for key, value in header.items():
            if key == "__metadata__":
                continue
            if not isinstance(key, str):
                raise ValueError(f"safetensors key must be a string: {key!r}")

            if key.startswith(_SOURCE_PREFIX):
                normalized_key = key[len(_SOURCE_PREFIX) :]
                out_key = _TARGET_PREFIX + normalized_key
            elif key.startswith(_TARGET_PREFIX):
                normalized_key = key[len(_TARGET_PREFIX) :]
                out_key = key
            else:
                rewritten[key] = value
                continue

            if normalized_key in normalized_model_keys:
                raise ValueError(f"duplicate model key after prefix rewrite: {normalized_key}")
            normalized_model_keys.add(normalized_key)
            rewritten[out_key] = value
            model_key_count += 1

        if model_key_count == 0:
            raise ValueError("saved model contains no model.diffusion_model.* or net.* keys")

        rewritten_bytes = json.dumps(rewritten, separators=(",", ":")).encode("utf-8")
        if len(rewritten_bytes) > header_len:
            raise ValueError("rewritten safetensors header does not fit in the original header area")

        f.seek(8)
        f.write(rewritten_bytes)
        f.write(b" " * (header_len - len(rewritten_bytes)))


class ModelSaveWithNetPrefix(c_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-ModelSaveWithNetPrefix",
            display_name="ModelSave with Net Prefix",
            category=Const.CATEGORY_IMAGEINFO,
            is_output_node=True,
            not_idempotent=True,
            description=(
                "Save a loaded MODEL using ComfyUI's standard checkpoint path, then rewrite "
                "model.diffusion_model.* keys to net.* without changing tensor bytes."
            ),
            search_aliases=[
                "model save with net prefix",
                "save model net prefix",
                "save diffusion model as net",
                "net prefix model save",
            ],
            inputs=[
                MODEL_RUNTIME_TYPE.Input(
                    "model",
                    tooltip="Loaded runtime MODEL to save",
                ),
                c_io.String.Input(
                    "filename_prefix",
                    default=_DEFAULT_FILENAME_PREFIX,
                    tooltip="Output prefix under the ComfyUI output directory",
                ),
            ],
            outputs=[
                c_io.String.Output(
                    Cast.out_id("file_path"),
                    display_name="file_path",
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        model: object | None = None,
        filename_prefix: object | None = _DEFAULT_FILENAME_PREFIX,
    ) -> bool | str:
        try:
            _normalize_filename_prefix(filename_prefix)
        except Exception as exc:
            return str(exc)
        return True

    @classmethod
    def execute(
        cls,
        model: object,
        filename_prefix: object = _DEFAULT_FILENAME_PREFIX,
    ) -> c_io.NodeOutput:
        import comfy.sd

        _validate_loaded_model(model)
        normalized_prefix = _normalize_filename_prefix(filename_prefix)
        output_path = _build_output_path(normalized_prefix)

        try:
            comfy.sd.save_checkpoint(str(output_path), model, metadata={})
            _rewrite_safetensors_header_with_net_prefix(output_path)
        except Exception as exc:
            try:
                output_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(f"failed to save MODEL with net prefix: {output_path}") from exc

        rel_path = _relative_to_output_root(output_path)
        try:
            print(f"[IPT][ModelSaveWithNetPrefix] saved {rel_path}")
        except Exception:
            pass
        return c_io.NodeOutput(rel_path)
