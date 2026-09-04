"""Pure-python PNG codec, format sniffing and mask rasterising."""

import struct
import zlib

import pytest

from zimage_studio import imaging as im


def test_encode_decode_roundtrip_rgb():
    pixels = bytes(range(48))
    data = im.png_encode(4, 4, pixels, im.MODE_RGB)
    width, height, mode, decoded = im.png_decode(data)
    assert (width, height, mode) == (4, 4, im.MODE_RGB)
    assert bytes(decoded) == pixels


def test_encode_decode_roundtrip_gray_and_rgba():
    gray = bytes(range(16))
    width, height, mode, decoded = im.png_decode(im.png_encode(4, 4, gray, im.MODE_L))
    assert mode == im.MODE_L and bytes(decoded) == gray

    rgba = bytes(range(64))
    _, _, mode, decoded = im.png_decode(im.png_encode(4, 4, rgba, im.MODE_RGBA))
    assert mode == im.MODE_RGBA and bytes(decoded) == rgba


def test_encode_rejects_wrong_buffer_size():
    with pytest.raises(im.ImageError):
        im.png_encode(4, 4, b"\x00" * 10, im.MODE_RGB)


def test_decode_handles_all_png_filters():
    """Pillow writes filtered scanlines; our decoder has to undo every type."""
    pil = pytest.importorskip("PIL.Image")
    import io

    image = pil.new("RGB", (17, 9))
    image.putdata([((x * 13) % 256, (y * 29) % 256, (x + y) % 256) for y in range(9) for x in range(17)])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    width, height, mode, decoded = im.png_decode(buffer.getvalue())
    assert (width, height, mode) == (17, 9, im.MODE_RGB)
    assert bytes(decoded) == image.tobytes()


def test_decode_palette_png():
    pil = pytest.importorskip("PIL.Image")
    import io

    image = pil.new("P", (6, 3))
    image.putpalette([255, 0, 0] + [0, 255, 0] + [0] * (256 * 3 - 6))
    image.putdata([1, 0, 1, 0, 1, 0] * 3)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    _, _, mode, decoded = im.png_decode(buffer.getvalue())
    assert mode == im.MODE_RGB
    assert bytes(decoded[:6]) == bytes((0, 255, 0, 255, 0, 0))


def test_decode_rejects_interlaced():
    data = bytearray(im.png_encode(2, 2, b"\x00" * 12, im.MODE_RGB))
    data[8 + 8 + 12] = 1  # interlace flag inside IHDR
    payload = bytes(data[16:29])
    data[29:33] = struct.pack(">I", zlib.crc32(b"IHDR" + payload) & 0xFFFFFFFF)
    with pytest.raises(im.ImageError):
        im.png_decode(bytes(data))


def test_decode_rejects_non_png():
    with pytest.raises(im.ImageError):
        im.png_decode(b"GIF89a not really")


@pytest.mark.parametrize("size", [(1, 1), (13, 7), (64, 64)])
def test_sniff_size_png(size):
    assert im.sniff_size(im.solid_png(size[0], size[1], (0, 0, 0))) == size


def test_sniff_size_jpeg_and_webp():
    pil = pytest.importorskip("PIL.Image")
    import io

    image = pil.new("RGB", (23, 41), (10, 20, 30))
    for fmt, expected in (("JPEG", "jpeg"), ("WEBP", "webp")):
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        data = buffer.getvalue()
        assert im.sniff_format(data) == expected
        assert im.sniff_size(data) == (23, 41)


def test_sniff_unknown_format():
    assert im.sniff_format(b"\x00\x01\x02\x03") == "unknown"
    with pytest.raises(im.ImageError):
        im.sniff_size(b"\x00\x01\x02\x03")


def test_rasterize_stroke_paints_a_disc():
    mask = im.rasterize_strokes(32, 32, [im.make_stroke([(16, 16)], 4)])
    assert mask[16 * 32 + 16] == 255
    assert mask[0] == 0
    painted = sum(1 for value in mask if value)
    assert 30 < painted < 90  # roughly pi*r^2


def test_rasterize_line_is_continuous():
    mask = im.rasterize_strokes(64, 64, [im.make_stroke([(5, 5), (58, 5)], 2)])
    row = mask[5 * 64 : 6 * 64]
    assert all(row[x] == 255 for x in range(6, 57))


def test_eraser_removes_paint():
    strokes = [im.make_stroke([(16, 16)], 8), im.make_stroke([(16, 16)], 8, erase=True)]
    assert im.mask_is_empty(32, 32, strokes)


def test_strokes_clip_at_the_border():
    mask = im.rasterize_strokes(16, 16, [im.make_stroke([(-20, -20), (40, 40)], 3)])
    assert len(mask) == 256
    assert mask[0] == 255


def test_strokes_to_png_is_grayscale():
    data = im.strokes_to_png(32, 32, [im.make_stroke([(8, 8)], 3)])
    _, _, mode, _ = im.png_decode(data)
    assert mode == im.MODE_L


def test_resize_nearest_changes_size_only():
    source = im.solid_png(4, 4, (7, 8, 9))
    _, _, mode, pixels = im.png_decode(source)
    scaled = im.resize_nearest(4, 4, bytes(pixels), mode, 8, 2)
    assert len(scaled) == 8 * 2 * 3
    assert bytes(scaled[:3]) == bytes((7, 8, 9))


def test_to_rgb_from_every_mode():
    assert bytes(im.to_rgb(im.MODE_L, b"\x05")[1]) == b"\x05\x05\x05"
    assert bytes(im.to_rgb(im.MODE_RGBA, b"\x01\x02\x03\x04")[1]) == b"\x01\x02\x03"
    assert bytes(im.to_rgb(im.MODE_RGB, b"\x01\x02\x03")[1]) == b"\x01\x02\x03"
