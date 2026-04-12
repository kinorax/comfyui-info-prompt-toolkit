
from PIL import Image, PngImagePlugin
from typing import Optional, Any

EXIF_USERCOMMENT_TAG = 37510  # UserComment

def _decode_usercomment(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, (bytes, bytearray)):
        return str(raw)

    b = bytes(raw)
    if b.startswith(b"UNICODE\x00"):
        payload = b[8:].rstrip(b"\x00")
        try:
            return payload.decode("utf-16-be", errors="replace").rstrip("\x00")
        except Exception:
            return payload.decode("utf-16-le", errors="replace").rstrip("\x00")
    if b.startswith(b"ASCII\x00\x00\x00"):
        return b[8:].decode("ascii", errors="replace").rstrip("\x00")

    for enc in ("utf-8", "utf-16-be", "utf-16-le", "latin1"):
        try:
            return b.decode(enc)
        except Exception:
            pass
    return None


def extract_a1111_text(pil_image) -> Optional[str]:
    params = pil_image.info.get("parameters")
    if isinstance(params, str) and params.strip():
        return params

    try:
        exif = pil_image.getexif()
        if exif:
            uc = exif.get(EXIF_USERCOMMENT_TAG)
            dec = _decode_usercomment(uc)
            if dec and dec.strip():
                return dec
    except Exception:
        pass

    exif_bytes = pil_image.info.get("exif")
    if exif_bytes:
        try:
            import piexif
            exif_dict = piexif.load(exif_bytes)
            uc = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
            dec = _decode_usercomment(uc)
            if dec and dec.strip():
                return dec
        except Exception:
            pass

    return None


def inject_a1111_text_png(path: str, text: str) -> bool:
    """
    PNG の tEXt("parameters") に A1111 文字列を付与する。
    既存のテキストチャンクは維持しつつ上書きする。
    """
    if not text or not isinstance(text, str):
        return False

    try:
        with Image.open(path) as im:
            if (im.format or "").upper() != "PNG":
                return False

            pnginfo = PngImagePlugin.PngInfo()
            for k, v in (im.info or {}).items():
                if isinstance(v, str) and k != "parameters":
                    pnginfo.add_text(k, v)
            pnginfo.add_text("parameters", text)

            save_kwargs = {
                "pnginfo": pnginfo,
                "compress_level": 4,
            }
            if "icc_profile" in im.info:
                save_kwargs["icc_profile"] = im.info["icc_profile"]
            if "dpi" in im.info:
                save_kwargs["dpi"] = im.info["dpi"]

            im.save(path, **save_kwargs)
            return True
    except Exception:
        return False
