# Copyright 2026 kinorax
import hashlib
import os
import re
from typing import Optional

import folder_paths
import node_helpers
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

from comfy_api.latest import io as c_io
from .. import const as Const
from ..utils import cast as Cast
from ..utils import clipspace_bridge as ClipspaceBridge
from ..utils import exif as Exif
from ..utils.a1111_infotext import a1111_infotext_to_image_info
from ..utils.image_reader_metadata import read_a1111_text_from_image_selection
from ..utils.image_info_normalizer import normalize_image_info_with_comfy_options


def _list_input_images() -> list[str]:
    # 公式 LoadImage と同様に input ディレクトリ直下の画像ファイルだけを列挙する。
    input_dir = folder_paths.get_input_directory()
    files = [
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    ]
    files = folder_paths.filter_files_content_types(files, ["image"])
    return sorted(files)


class ImageReader(c_io.ComfyNode):

    @staticmethod
    def _is_clipspace_masked_image(path: str) -> bool:
        norm = path.replace("\\", "/").lower()
        return "/clipspace/" in norm and bool(re.search(r"clipspace-painted-masked-\d+\.png$", norm))

    @classmethod
    def _read_metadata_from_annotated_source(cls, source_annotated: Optional[str]) -> Optional[str]:
        if not source_annotated:
            return None
        if not folder_paths.exists_annotated_filepath(source_annotated):
            return None
        try:
            source_path = folder_paths.get_annotated_filepath(source_annotated)
            source_img = node_helpers.pillow(Image.open, source_path)
            text = Exif.extract_a1111_text(source_img)
            return text
        except Exception:
            return None

    @classmethod
    def _recover_metadata_for_clipspace(
        cls,
        image: str,
        image_path: str,
        current_text: Optional[str],
    ) -> Optional[str]:
        if current_text:
            return current_text
        if not cls._is_clipspace_masked_image(image_path):
            return current_text

        inherited: Optional[str] = None

        # Frontend hook route can provide source image path even in a fresh session.
        mapped_source = (
            ClipspaceBridge.get_source_annotated(image)
            or ClipspaceBridge.get_source_annotated(image_path)
        )
        inherited = cls._read_metadata_from_annotated_source(mapped_source)

        if not inherited:
            return None

        # Persist recovered metadata into the clipspace file for future reads.
        Exif.inject_a1111_text_png(image_path, inherited)
        return inherited

    @classmethod
    def define_schema(cls) -> c_io.Schema:
        return c_io.Schema(
            node_id="IPT-ImageReader",
            display_name="Image Reader",
            category="image",
            search_aliases=["load image", "image reader", "open image", "import image"],
            inputs=[
                # V1 LoadImage の (sorted(files), {"image_upload": True}) に相当
                c_io.Combo.Input(
                    "image",
                    options=_list_input_images(),
                    upload=c_io.UploadType.image,
                    image_folder=c_io.FolderType.input,
                )
            ],
            outputs=[
                c_io.Image.Output(Cast.out_id("image"), display_name="image"),
                c_io.Mask.Output(Cast.out_id("mask"), display_name="mask"),
                Const.IMAGEINFO_TYPE.Output(Cast.out_id(Const.IMAGEINFO), display_name=Const.IMAGEINFO),
                c_io.String.Output("INFO", display_name="infotext(raw)"),
            ],
        )

    @classmethod
    def execute(cls, image: str) -> c_io.NodeOutput:
        image_path, a1111_text = read_a1111_text_from_image_selection(image)
        img = node_helpers.pillow(Image.open, image_path)
        image_info = a1111_infotext_to_image_info(a1111_text)
        image_info = normalize_image_info_with_comfy_options(image_info)

        output_images = []
        output_masks = []
        w, h = None, None

        for frame in ImageSequence.Iterator(img):
            frame = node_helpers.pillow(ImageOps.exif_transpose, frame)

            if frame.mode == "I":
                frame = frame.point(lambda i: i * (1 / 255))

            rgb = frame.convert("RGB")

            if len(output_images) == 0:
                w, h = rgb.size
            # フレーム間でサイズが違うものはスキップ（公式同様）
            if rgb.size[0] != w or rgb.size[1] != h:
                continue

            image_np = np.array(rgb).astype(np.float32) / 255.0
            image_t = torch.from_numpy(image_np)[None,]  # [1,H,W,3]

            # 公式 LoadImage は alpha を mask に変換（mask = 1 - alpha）
            if "A" in frame.getbands():
                alpha = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(alpha)
            # パレットPNG + transparency対策（あると地味に便利）
            elif frame.mode == "P" and "transparency" in frame.info:
                alpha = np.array(frame.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(alpha)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            output_images.append(image_t)
            output_masks.append(mask.unsqueeze(0))  # [1,H,W]

            # MPOは最初のフレームのみ（公式同様）
            if getattr(img, "format", None) == "MPO":
                break

        if len(output_images) > 1:
            out_image = torch.cat(output_images, dim=0)  # [B,H,W,3]
            out_mask = torch.cat(output_masks, dim=0)    # [B,H,W]
        else:
            out_image = output_images[0]
            out_mask = output_masks[0]

        return c_io.NodeOutput(out_image, out_mask, image_info, a1111_text)

    @classmethod
    def fingerprint_inputs(cls, image: str):
        # V1の IS_CHANGED 相当（V3では fingerprint_inputs）。ファイル内容ハッシュで変更検知。
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def validate_inputs(cls, image: str):
        # V1の VALIDATE_INPUTS 相当
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True
