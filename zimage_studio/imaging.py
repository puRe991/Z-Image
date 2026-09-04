"""Dependency free image helpers (PNG codec, format sniffing, mask rasterising).

Pillow is *not* importable on every 32-bit Windows Python, and shipping a
compiled dependency would defeat the purpose of the thin client.  Everything the
GUI needs - measuring an image, painting a mask, writing a PNG - is implemented
here on top of :mod:`zlib` from the standard library.
"""

from __future__ import annotations

import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import zlib

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

MODE_L = "L"
MODE_RGB = "RGB"
MODE_RGBA = "RGBA"
_CHANNELS = {MODE_L: 1, MODE_RGB: 3, MODE_RGBA: 4}
_COLOR_TYPE = {MODE_L: 0, MODE_RGB: 2, MODE_RGBA: 6}


class ImageError(ValueError):
    """Raised for malformed or unsupported image data."""


# --------------------------------------------------------------------------
# Format sniffing
# --------------------------------------------------------------------------


def sniff_format(data: bytes) -> str:
    """Return ``png``/``jpeg``/``gif``/``bmp``/``webp`` or ``unknown``."""
    if data[:8] == PNG_MAGIC:
        return "png"
    if data[:2] == b"\xff\xd8":
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:2] == b"BM":
        return "bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "unknown"


def sniff_size(data: bytes) -> Tuple[int, int]:
    """Return ``(width, height)`` of an image without decoding the pixels."""
    fmt = sniff_format(data)
    if fmt == "png":
        if len(data) < 24 or data[12:16] != b"IHDR":
            raise ImageError("truncated PNG header")
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    if fmt == "gif":
        width, height = struct.unpack("<HH", data[6:10])
        return int(width), int(height)
    if fmt == "bmp":
        width, height = struct.unpack("<ii", data[18:26])
        return abs(int(width)), abs(int(height))
    if fmt == "jpeg":
        return _sniff_jpeg_size(data)
    if fmt == "webp":
        return _sniff_webp_size(data)
    raise ImageError("unsupported image format")


def _sniff_jpeg_size(data: bytes) -> Tuple[int, int]:
    index = 2
    length = len(data)
    while index + 9 < length:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        segment_len = struct.unpack(">H", data[index + 2 : index + 4])[0]
        # SOF0..SOF15 except the DHT/JPG/DAC markers carry the frame size.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return int(width), int(height)
        index += 2 + segment_len
    raise ImageError("no JPEG frame header found")


def _sniff_webp_size(data: bytes) -> Tuple[int, int]:
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height
    if chunk == b"VP8 ":
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    raise ImageError("unsupported WebP variant")


# --------------------------------------------------------------------------
# PNG encoding / decoding
# --------------------------------------------------------------------------


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def png_encode(width: int, height: int, pixels: bytes, mode: str = MODE_RGB, *, level: int = 6) -> bytes:
    """Encode raw, tightly packed 8-bit pixels as a PNG file."""
    if mode not in _CHANNELS:
        raise ImageError("unsupported mode %r" % mode)
    channels = _CHANNELS[mode]
    expected = width * height * channels
    if len(pixels) != expected:
        raise ImageError("expected %d bytes for %dx%d %s, got %d" % (expected, width, height, mode, len(pixels)))

    stride = width * channels
    raw = bytearray()
    view = memoryview(pixels)
    for y in range(height):
        raw.append(0)  # filter type 0 (None) - fast and good enough after zlib
        raw += view[y * stride : (y + 1) * stride]

    header = struct.pack(">IIBBBBB", width, height, 8, _COLOR_TYPE[mode], 0, 0, 0)
    return b"".join(
        [
            PNG_MAGIC,
            _chunk(b"IHDR", header),
            _chunk(b"IDAT", zlib.compress(bytes(raw), level)),
            _chunk(b"IEND", b""),
        ]
    )


def png_decode(data: bytes) -> Tuple[int, int, str, bytearray]:
    """Decode a non-interlaced 8/16-bit PNG into ``(w, h, mode, pixels)``.

    Palette and grayscale-alpha images are expanded to RGB / RGBA so that the
    caller only ever has to deal with the three modes in :data:`_CHANNELS`.
    """
    if data[:8] != PNG_MAGIC:
        raise ImageError("not a PNG file")

    index = 8
    width = height = bit_depth = color_type = interlace = 0
    palette = b""
    trns = b""
    idat = bytearray()
    seen_header = False

    while index + 8 <= len(data):
        (length,) = struct.unpack(">I", data[index : index + 4])
        tag = data[index + 4 : index + 8]
        payload = data[index + 8 : index + 8 + length]
        index += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filt, interlace = struct.unpack(">IIBBBBB", payload)
            seen_header = True
        elif tag == b"PLTE":
            palette = payload
        elif tag == b"tRNS":
            trns = payload
        elif tag == b"IDAT":
            idat += payload
        elif tag == b"IEND":
            break

    if not seen_header:
        raise ImageError("PNG without IHDR")
    if interlace:
        raise ImageError("interlaced PNG is not supported")
    if bit_depth not in (8, 16):
        raise ImageError("unsupported PNG bit depth %d" % bit_depth)
    if color_type not in (0, 2, 3, 4, 6):
        raise ImageError("unsupported PNG color type %d" % color_type)

    src_channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    sample_bytes = bit_depth // 8
    bytes_per_pixel = src_channels * sample_bytes
    stride = width * bytes_per_pixel

    raw = zlib.decompress(bytes(idat))
    if len(raw) < (stride + 1) * height:
        raise ImageError("truncated PNG image data")

    out = bytearray(stride * height)
    prev = bytearray(stride)
    pos = 0
    for y in range(height):
        filter_type = raw[pos]
        pos += 1
        line = bytearray(raw[pos : pos + stride])
        pos += stride
        _unfilter_line(filter_type, line, prev, bytes_per_pixel)
        out[y * stride : (y + 1) * stride] = line
        prev = line

    if sample_bytes == 2:  # drop the low byte of 16-bit samples
        out = bytearray(out[i] for i in range(0, len(out), 2))

    if color_type == 3:
        return width, height, *_expand_palette(width, height, out, palette, trns)
    if color_type == 4:  # gray + alpha -> RGBA
        rgba = bytearray(width * height * 4)
        for i in range(width * height):
            gray = out[i * 2]
            rgba[i * 4 : i * 4 + 4] = bytes((gray, gray, gray, out[i * 2 + 1]))
        return width, height, MODE_RGBA, rgba

    mode = {0: MODE_L, 2: MODE_RGB, 6: MODE_RGBA}[color_type]
    return width, height, mode, out


def _unfilter_line(filter_type: int, line: bytearray, prev: bytearray, bpp: int) -> None:
    if filter_type == 0:
        return
    if filter_type == 1:  # Sub
        for i in range(bpp, len(line)):
            line[i] = (line[i] + line[i - bpp]) & 0xFF
    elif filter_type == 2:  # Up
        for i in range(len(line)):
            line[i] = (line[i] + prev[i]) & 0xFF
    elif filter_type == 3:  # Average
        for i in range(len(line)):
            left = line[i - bpp] if i >= bpp else 0
            line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
    elif filter_type == 4:  # Paeth
        for i in range(len(line)):
            left = line[i - bpp] if i >= bpp else 0
            up = prev[i]
            up_left = prev[i - bpp] if i >= bpp else 0
            p = left + up - up_left
            pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
            if pa <= pb and pa <= pc:
                pred = left
            elif pb <= pc:
                pred = up
            else:
                pred = up_left
            line[i] = (line[i] + pred) & 0xFF
    else:
        raise ImageError("unknown PNG filter type %d" % filter_type)


def _expand_palette(
    width: int, height: int, indices: bytearray, palette: bytes, trns: bytes
) -> Tuple[str, bytearray]:
    if not palette:
        raise ImageError("palette PNG without PLTE chunk")
    entries = len(palette) // 3
    if trns:
        out = bytearray(width * height * 4)
        for i, value in enumerate(indices):
            base = min(value, entries - 1) * 3
            alpha = trns[value] if value < len(trns) else 255
            out[i * 4 : i * 4 + 4] = palette[base : base + 3] + bytes((alpha,))
        return MODE_RGBA, out
    out = bytearray(width * height * 3)
    for i, value in enumerate(indices):
        base = min(value, entries - 1) * 3
        out[i * 3 : i * 3 + 3] = palette[base : base + 3]
    return MODE_RGB, out


# --------------------------------------------------------------------------
# Mask rasterising
# --------------------------------------------------------------------------

#: One brush stroke as recorded by the canvas widget.  Coordinates are in image
#: space (pixels of the *source* image), which keeps the mask independent of the
#: current zoom level.
Stroke = Dict[str, object]


def make_stroke(points: Sequence[Tuple[float, float]], radius: float, erase: bool = False) -> Stroke:
    return {"points": [(float(x), float(y)) for x, y in points], "radius": float(radius), "erase": bool(erase)}


def _disc_spans(radius: int) -> List[Tuple[int, int]]:
    """Half-open x offsets per row of a filled circle, indexed by ``dy + radius``."""
    spans: List[Tuple[int, int]] = []
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        dx = int((max(0, r2 - dy * dy)) ** 0.5)
        spans.append((-dx, dx + 1))
    return spans


_SPAN_CACHE: Dict[int, List[Tuple[int, int]]] = {}


def new_mask(width: int, height: int) -> bytearray:
    """An all-zero (nothing selected) mask buffer."""
    if width <= 0 or height <= 0:
        raise ImageError("mask size must be positive")
    return bytearray(width * height)


def stamp_stroke(mask: bytearray, width: int, height: int, stroke: Stroke) -> None:
    """Paint (or erase) one stroke into an existing mask buffer, in place.

    Stamping incrementally is what keeps mask painting interactive: the cost is
    proportional to the brushed area, not to the size of the image.
    """
    radius = max(1, int(round(float(stroke.get("radius", 16)))))
    value = 0 if stroke.get("erase") else 255
    spans = _SPAN_CACHE.get(radius)
    if spans is None:
        spans = _SPAN_CACHE[radius] = _disc_spans(radius)
    for x, y in _densify(stroke.get("points") or [], radius):
        _stamp(mask, width, height, int(round(x)), int(round(y)), radius, spans, value)


def rasterize_strokes(width: int, height: int, strokes: Iterable[Stroke]) -> bytes:
    """Render brush strokes into an 8-bit mask (255 = edit, 0 = keep)."""
    mask = new_mask(width, height)
    for stroke in strokes:
        stamp_stroke(mask, width, height, stroke)
    return bytes(mask)


def _densify(points: Sequence[Tuple[float, float]], radius: int) -> Iterable[Tuple[float, float]]:
    """Yield points along the polyline so that consecutive discs overlap."""
    if not points:
        return
    step = max(1.0, radius * 0.5)
    previous = points[0]
    yield previous
    for point in points[1:]:
        dx = point[0] - previous[0]
        dy = point[1] - previous[1]
        distance = (dx * dx + dy * dy) ** 0.5
        if distance > step:
            count = int(distance / step)
            for i in range(1, count + 1):
                t = i / float(count + 1)
                yield (previous[0] + dx * t, previous[1] + dy * t)
        yield point
        previous = point


def _stamp(
    mask: bytearray,
    width: int,
    height: int,
    cx: int,
    cy: int,
    radius: int,
    spans: List[Tuple[int, int]],
    value: int,
) -> None:
    fill = bytes((value,))
    for dy in range(-radius, radius + 1):
        y = cy + dy
        if y < 0 or y >= height:
            continue
        x0, x1 = spans[dy + radius]
        start = max(0, cx + x0)
        end = min(width, cx + x1)
        if start >= end:
            continue
        row = y * width
        mask[row + start : row + end] = fill * (end - start)


def strokes_to_png(width: int, height: int, strokes: Iterable[Stroke]) -> bytes:
    """Convenience wrapper: rasterise strokes and encode them as a grayscale PNG."""
    return png_encode(width, height, rasterize_strokes(width, height, strokes), MODE_L)


def mask_is_empty(width: int, height: int, strokes: Iterable[Stroke]) -> bool:
    """True when the strokes would not paint a single pixel."""
    return 255 not in rasterize_strokes(width, height, strokes)


def solid_png(width: int, height: int, color: Tuple[int, int, int]) -> bytes:
    """A single colour PNG - used for placeholders and by the test-suite."""
    return png_encode(width, height, bytes(color) * (width * height), MODE_RGB)


def resize_nearest(
    width: int, height: int, pixels: bytes, mode: str, new_width: int, new_height: int
) -> bytearray:
    """Nearest-neighbour resample - enough for thumbnails and mask alignment."""
    if mode not in _CHANNELS:
        raise ImageError("unsupported mode %r" % mode)
    if new_width <= 0 or new_height <= 0:
        raise ImageError("target size must be positive")
    channels = _CHANNELS[mode]
    out = bytearray(new_width * new_height * channels)
    src = memoryview(pixels)
    x_map = [min(width - 1, (x * width) // new_width) * channels for x in range(new_width)]
    for y in range(new_height):
        src_row = min(height - 1, (y * height) // new_height) * width * channels
        dst_row = y * new_width * channels
        for x in range(new_width):
            src_off = src_row + x_map[x]
            dst_off = dst_row + x * channels
            out[dst_off : dst_off + channels] = src[src_off : src_off + channels]
    return out


def to_rgb(mode: str, pixels: bytes) -> Tuple[str, bytearray]:
    """Drop an alpha channel / expand grayscale so callers can assume RGB."""
    if mode == MODE_RGB:
        return MODE_RGB, bytearray(pixels)
    if mode == MODE_L:
        out = bytearray(len(pixels) * 3)
        for i, value in enumerate(pixels):
            out[i * 3 : i * 3 + 3] = bytes((value, value, value))
        return MODE_RGB, out
    if mode == MODE_RGBA:
        out = bytearray(len(pixels) // 4 * 3)
        for i in range(len(pixels) // 4):
            out[i * 3 : i * 3 + 3] = pixels[i * 4 : i * 4 + 3]
        return MODE_RGB, out
    raise ImageError("unsupported mode %r" % mode)


#: Palette used for the translucent mask overlay in the GUI: index 0 is fully
#: transparent, index 1 is a semi-transparent accent colour.
OVERLAY_PALETTE = (0, 0, 0, 255, 59, 48)
OVERLAY_ALPHA = (0, 140)

_MASK_TO_INDEX = bytes(1 if value else 0 for value in range(256))


def png_encode_indexed(
    width: int,
    height: int,
    indices: bytes,
    palette: Sequence[int] = OVERLAY_PALETTE,
    alpha: Sequence[int] = OVERLAY_ALPHA,
    *,
    level: int = 1,
) -> bytes:
    """Encode an 8-bit indexed PNG with per-entry alpha.

    The mask overlay is a two colour image, so shipping it as a palette PNG keeps
    the buffer at one byte per pixel - fast enough to re-encode on every brush
    stroke, even on a slow 32-bit machine.
    """
    if len(indices) != width * height:
        raise ImageError("expected %d index bytes, got %d" % (width * height, len(indices)))
    header = struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0)
    raw = bytearray()
    view = memoryview(indices)
    for y in range(height):
        raw.append(0)
        raw += view[y * width : (y + 1) * width]
    chunks = [
        PNG_MAGIC,
        _chunk(b"IHDR", header),
        _chunk(b"PLTE", bytes(palette)),
        _chunk(b"tRNS", bytes(alpha)),
        _chunk(b"IDAT", zlib.compress(bytes(raw), level)),
        _chunk(b"IEND", b""),
    ]
    return b"".join(chunks)


def mask_to_overlay_png(width: int, height: int, mask: bytes) -> bytes:
    """Turn an 8-bit mask into the translucent overlay the canvas displays."""
    return png_encode_indexed(width, height, bytes(mask).translate(_MASK_TO_INDEX))
