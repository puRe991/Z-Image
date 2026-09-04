"""Convert arbitrary image files to PNG for the thin client.

The 32-bit client can display PNG and GIF through Tk alone.  Anything else
(JPEG, WebP, TIFF, ...) is uploaded raw and converted here, where Pillow is
available.  Without Pillow the function still handles PNG - including
downscaling - through the pure-Python codec.
"""

from __future__ import annotations

import io
from typing import Optional, Tuple

from ..imaging import (
    MODE_RGB,
    ImageError,
    png_decode,
    png_encode,
    resize_nearest,
    sniff_format,
    sniff_size,
    to_rgb,
)

try:  # pragma: no cover - depends on the host
    from PIL import Image  # type: ignore

    HAVE_PILLOW = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    HAVE_PILLOW = False


def _target_size(width: int, height: int, max_size: int) -> Optional[Tuple[int, int]]:
    if max_size <= 0 or max(width, height) <= max_size:
        return None
    scale = max_size / float(max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def to_png(data: bytes, max_size: int = 0) -> Tuple[bytes, int, int]:
    """Return ``(png_bytes, width, height)`` for any supported input."""
    if HAVE_PILLOW:
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            resized = _target_size(image.width, image.height, max_size)
            if resized:
                image = image.resize(resized, Image.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=False)
            return buffer.getvalue(), image.width, image.height

    if sniff_format(data) != "png":
        raise ImageError(
            "cannot convert %s without Pillow - install Pillow on the server" % sniff_format(data)
        )
    width, height = sniff_size(data)
    resized = _target_size(width, height, max_size)
    if not resized:
        return data, width, height
    _, _, mode, pixels = png_decode(data)
    mode, rgb = to_rgb(mode, bytes(pixels))
    new_width, new_height = resized
    scaled = resize_nearest(width, height, bytes(rgb), MODE_RGB, new_width, new_height)
    return png_encode(new_width, new_height, bytes(scaled), MODE_RGB), new_width, new_height
