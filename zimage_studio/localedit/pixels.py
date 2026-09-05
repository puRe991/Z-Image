"""Fast pixel plumbing in pure Python.

A per-pixel Python loop costs roughly a second per megapixel on a modern
machine and several seconds on the kind of laptop this code targets, so the
primitives here avoid them:

* splitting an RGB buffer into three planes and merging them back are extended
  slice assignments, which run at C speed,
* per-pixel value mappings (brightness, contrast, tinting, ...) are
  :meth:`bytes.translate` with a 256 byte lookup table,
* neighbourhood averages are computed by widening each byte into a 16-bit slot,
  interpreting the whole plane as one big integer and adding the shifted copies
  in a single arithmetic operation.

The big-integer trick is exact as long as every slot stays below 65536, which
holds for the four-neighbour sums used here (at most 4 x 255).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

PlaneList = List[bytes]


# ----------------------------------------------------------------------
# planes
# ----------------------------------------------------------------------


def split_planes(rgb: bytes) -> PlaneList:
    """Split interleaved RGB bytes into three separate channel planes."""
    return [bytes(rgb[channel::3]) for channel in range(3)]


def merge_planes(planes: Sequence[bytes]) -> bytearray:
    """Interleave three channel planes back into RGB bytes."""
    size = len(planes[0])
    out = bytearray(size * 3)
    for channel in range(3):
        out[channel::3] = planes[channel]
    return out


def crop_plane(plane: bytes, width: int, box: Tuple[int, int, int, int]) -> bytes:
    """Cut a rectangle ``(x0, y0, x1, y1)`` out of a plane."""
    x0, y0, x1, y1 = box
    return b"".join(plane[y * width + x0 : y * width + x1] for y in range(y0, y1))


def paste_plane(
    plane: bytearray, width: int, box: Tuple[int, int, int, int], patch: bytes
) -> None:
    """Write a rectangle back into a plane, in place."""
    x0, y0, x1, y1 = box
    patch_width = x1 - x0
    for index, y in enumerate(range(y0, y1)):
        plane[y * width + x0 : y * width + x1] = patch[index * patch_width : (index + 1) * patch_width]


# ----------------------------------------------------------------------
# lookup tables
# ----------------------------------------------------------------------


def clamp_byte(value: float) -> int:
    if value <= 0:
        return 0
    if value >= 255:
        return 255
    return int(value)


def make_table(mapping) -> bytes:
    """Build a 256 entry translation table from ``f(value) -> value``."""
    return bytes(clamp_byte(mapping(value)) for value in range(256))


def apply_table(plane: bytes, table: bytes) -> bytes:
    return plane.translate(table)


def blend_tables(table: bytes, amount: float) -> bytes:
    """Scale a table's effect: ``amount`` 0 is identity, 1 is the full table."""
    amount = max(0.0, min(1.0, float(amount)))
    if amount >= 1.0:
        return table
    return bytes(clamp_byte(value + (table[value] - value) * amount) for value in range(256))


# ----------------------------------------------------------------------
# neighbourhood maths
# ----------------------------------------------------------------------


def _widen(plane: bytes) -> int:
    """Interpret a plane as one big integer with one 16-bit slot per pixel."""
    wide = bytearray(len(plane) * 2)
    wide[0::2] = plane
    return int.from_bytes(bytes(wide), "little")


def _narrow(value: int, count: int) -> bytes:
    """Inverse of :func:`_widen`, keeping the low byte of every slot."""
    return bytes(value.to_bytes(count * 2 + 2, "little")[0 : count * 2 : 2])


def _shift_rows(plane: bytes, width: int, rows: int) -> bytes:
    """Shift a plane vertically, repeating the edge row."""
    if rows > 0:  # take from above
        return plane[: width * rows] + plane[: -width * rows]
    if rows < 0:
        rows = -rows
        return plane[width * rows :] + plane[-width * rows :]
    return plane


def _shift_columns(plane: bytes, columns: int) -> bytes:
    """Shift a plane horizontally.  Wraps at row ends; callers fix the edges."""
    if columns > 0:
        return plane[:columns] + plane[:-columns]
    if columns < 0:
        columns = -columns
        return plane[columns:] + plane[-columns:]
    return plane


def neighbour_average(plane: bytes, width: int, height: int) -> bytes:
    """Average of the four direct neighbours of every pixel.

    One pass over a megapixel costs a few tens of milliseconds because the whole
    plane is summed as a single big integer.
    """
    count = len(plane)
    total = (
        _widen(_shift_rows(plane, width, 1))
        + _widen(_shift_rows(plane, width, -1))
        + _widen(_shift_columns(plane, 1))
        + _widen(_shift_columns(plane, -1))
    )
    averaged = bytearray(_narrow(total >> 2, count))
    # The horizontal shifts wrapped around the row ends, so restore both columns
    # from a vertical-only average.
    vertical = _narrow(
        (_widen(_shift_rows(plane, width, 1)) + _widen(_shift_rows(plane, width, -1))) >> 1, count
    )
    averaged[0::width] = vertical[0::width]
    averaged[width - 1 :: width] = vertical[width - 1 :: width]
    return bytes(averaged)


def add_planes(planes: Sequence[bytes]) -> bytes:
    """Byte-wise sum of planes whose total is known to stay below 256.

    Used for the luma plane, where the three weighted channels add up to at most
    255 by construction.  One big-integer addition replaces a per-pixel loop.
    """
    count = len(planes[0])
    total = 0
    for plane in planes:
        total += _widen(plane)
    return _narrow(total, count)


def weighted_mix(first: bytes, second: bytes, amount: float, *, steps: int = 64) -> bytes:
    """Blend two planes: ``amount`` 0 keeps ``first``, 1 gives ``second``.

    The weights are quantised to ``steps`` so the whole blend is one big-integer
    multiply-add per plane; at 64 steps the rounding error is under one value
    step and invisible.
    """
    amount = max(0.0, min(1.0, float(amount)))
    if amount <= 1.0 / (2 * steps):
        return bytes(first)
    if amount >= 1.0 - 1.0 / (2 * steps):
        return bytes(second)
    count = len(first)
    k = int(round(amount * steps))
    shift = steps.bit_length() - 1
    total = _widen(first) * (steps - k) + _widen(second) * k
    return _narrow(total >> shift, count)


_HARD_HIGH = bytes(255 if value >= 250 else 0 for value in range(256))
_HARD_LOW = bytes(255 if value <= 5 else 0 for value in range(256))


def composite_by_mask(
    base: Sequence[bytes], other: Sequence[bytes], mask: bytes
) -> List[bytes]:
    """Blend two images through a soft mask: 255 keeps ``base``, 0 takes ``other``.

    A segmentation mask is almost entirely 0 or 255 with a thin soft edge, so the
    two flat regions are selected with one big-integer bitwise operation per
    plane and only the edge pixels go through a Python loop.
    """
    keep = mask.translate(_HARD_HIGH)
    take = mask.translate(_HARD_LOW)
    keep_int = int.from_bytes(keep, "little")
    take_int = int.from_bytes(take, "little")
    count = len(mask)

    soft = [index for index, value in enumerate(mask) if 5 < value < 250]
    result: List[bytes] = []
    for base_plane, other_plane in zip(base, other):
        merged = bytearray(
            (
                (int.from_bytes(base_plane, "little") & keep_int)
                | (int.from_bytes(other_plane, "little") & take_int)
            ).to_bytes(count, "little")
        )
        for index in soft:
            weight = mask[index]
            merged[index] = (
                other_plane[index] + (base_plane[index] - other_plane[index]) * weight // 255
            ) & 0xFF
        result.append(bytes(merged))
    return result


def box_blur(plane: bytes, width: int, height: int, passes: int = 1) -> bytes:
    """Repeated neighbour averaging - a cheap approximation of a Gaussian."""
    result = plane
    for _ in range(max(0, int(passes))):
        result = neighbour_average(result, width, height)
    return result


# ----------------------------------------------------------------------
# scaling
# ----------------------------------------------------------------------


def downsample_plane(plane: bytes, width: int, height: int, factor: int = 2):
    """Nearest-neighbour shrink by an integer factor, using row slices only."""
    factor = max(1, int(factor))
    new_width = max(1, width // factor)
    new_height = max(1, height // factor)
    out = bytearray(new_width * new_height)
    for y in range(new_height):
        row = plane[(y * factor) * width : (y * factor) * width + width]
        out[y * new_width : (y + 1) * new_width] = row[::factor][:new_width]
    return bytes(out), new_width, new_height


def upsample_plane(plane: bytes, width: int, height: int, factor: int, target=None):
    """Nearest-neighbour enlarge by an integer factor.

    Each output row is built with ``factor`` strided slice assignments instead of
    a per-pixel loop.
    """
    factor = max(1, int(factor))
    new_width, new_height = (width * factor, height * factor)
    if target is not None:
        new_width, new_height = target
    out = bytearray(new_width * new_height)
    row_buffer = bytearray(new_width)
    # Every phase of the strided assignment needs exactly as many bytes as it
    # has slots; a target that is not an exact multiple of the factor leaves the
    # last slots to be filled from the edge pixel.
    slots = [len(range(phase, new_width, factor)) for phase in range(factor)]
    for y in range(new_height):
        source_row = plane[min(height - 1, y // factor) * width :][:width]
        edge = source_row[-1:] or b"\x00"
        for phase in range(factor):
            needed = slots[phase]
            chunk = source_row[:needed]
            if len(chunk) < needed:
                chunk = chunk + edge * (needed - len(chunk))
            row_buffer[phase::factor] = chunk
        out[y * new_width : (y + 1) * new_width] = row_buffer
    return bytes(out), new_width, new_height
