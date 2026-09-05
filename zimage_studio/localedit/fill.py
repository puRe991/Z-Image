"""Content-aware fill: replace the brushed area from its surroundings.

This is the local counterpart of inpainting.  Two classical techniques are
combined:

* **Diffusion** - the hole is smoothed inwards from its edge over an image
  pyramid, which reproduces gradients (sky, walls, blurred backgrounds) but
  loses texture.
* **Patch transplant** - the neighbourhood is searched for a translated
  rectangle whose visible border matches the border around the hole, and that
  rectangle is copied in.  This keeps texture (grass, gravel, fabric) intact.

The transplant is used when a good match exists and is feathered into the
diffusion result at the seam, otherwise the diffusion result stands alone.

None of this invents new content: it removes objects by continuing what is
already around them.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .pixels import (
    crop_plane,
    downsample_plane,
    neighbour_average,
    paste_plane,
    upsample_plane,
)

Box = Tuple[int, int, int, int]

#: Iterations of neighbour averaging per pyramid level.
LEVEL_ITERATIONS = 12
#: The pyramid stops once the hole fits in a box this small.
COARSE_TARGET = 24
#: Candidate transplant offsets are searched at these multiples of the hole size.
OFFSET_STEPS = (1.0, 1.5, 2.0)
#: Mean absolute border difference (0-255) below which a transplant is trusted.
TRANSPLANT_THRESHOLD = 26.0


def mask_bounding_box(mask: Sequence[int], width: int, height: int, margin: int = 0) -> Optional[Box]:
    """Smallest box containing every set pixel, grown by ``margin``."""
    min_x, min_y, max_x, max_y = width, height, -1, -1
    for y in range(height):
        row = bytes(mask[y * width : (y + 1) * width])
        if 255 not in row and not any(row):
            continue
        first = next((x for x, value in enumerate(row) if value), None)
        if first is None:
            continue
        last = len(row) - 1 - next(x for x, value in enumerate(reversed(row)) if value)
        min_x = min(min_x, first)
        max_x = max(max_x, last)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
    if max_x < 0:
        return None
    return (
        max(0, min_x - margin),
        max(0, min_y - margin),
        min(width, max_x + 1 + margin),
        min(height, max_y + 1 + margin),
    )


def _select(filled: bytes, original: bytes, hole_int: int, keep_int: int, count: int) -> bytes:
    """Keep ``original`` outside the hole and ``filled`` inside it.

    Done as two big-integer bitwise operations rather than a per-pixel loop.
    """
    filled_int = int.from_bytes(filled, "little")
    original_int = int.from_bytes(original, "little")
    merged = (filled_int & hole_int) | (original_int & keep_int)
    return merged.to_bytes(count, "little")


def _diffuse(plane: bytes, mask: bytes, width: int, height: int, iterations: int) -> bytes:
    count = len(plane)
    hole_int = int.from_bytes(mask, "little")
    keep_int = int.from_bytes(bytes(255 - value for value in mask), "little")
    current = plane
    for _ in range(iterations):
        averaged = neighbour_average(current, width, height)
        current = _select(averaged, plane, hole_int, keep_int, count)
    return current


def _seed_hole(plane: bytes, mask: bytes) -> bytes:
    """Start the coarsest level from the average colour around the hole."""
    known = [value for value, hole in zip(plane, mask) if not hole]
    average = sum(known) // len(known) if known else 128
    seeded = bytearray(plane)
    for index, hole in enumerate(mask):
        if hole:
            seeded[index] = average
    return bytes(seeded)


def _pyramid_fill(plane: bytes, mask: bytes, width: int, height: int) -> bytes:
    """Fill a hole by diffusing over an image pyramid (coarse to fine)."""
    levels: List[Tuple[bytes, bytes, int, int]] = [(plane, mask, width, height)]
    current_plane, current_mask, current_w, current_h = plane, mask, width, height
    while max(current_w, current_h) > COARSE_TARGET and min(current_w, current_h) >= 4:
        current_plane, current_w, current_h = downsample_plane(current_plane, current_w, current_h, 2)
        current_mask, _, _ = downsample_plane(current_mask, levels[-1][2], levels[-1][3], 2)
        levels.append((current_plane, current_mask, current_w, current_h))

    coarse_plane, coarse_mask, coarse_w, coarse_h = levels[-1]
    result = _diffuse(_seed_hole(coarse_plane, coarse_mask), coarse_mask, coarse_w, coarse_h, LEVEL_ITERATIONS)

    for level in range(len(levels) - 2, -1, -1):
        target_plane, target_mask, target_w, target_h = levels[level]
        upscaled, _, _ = upsample_plane(
            result, coarse_w, coarse_h, 2, target=(target_w, target_h)
        )
        count = len(target_plane)
        hole_int = int.from_bytes(target_mask, "little")
        keep_int = int.from_bytes(bytes(255 - value for value in target_mask), "little")
        seeded = _select(upscaled, target_plane, hole_int, keep_int, count)
        result = _diffuse_from(seeded, target_plane, target_mask, target_w, target_h, LEVEL_ITERATIONS // 2)
        coarse_w, coarse_h = target_w, target_h
    return result


def _diffuse_from(
    seeded: bytes, original: bytes, mask: bytes, width: int, height: int, iterations: int
) -> bytes:
    count = len(original)
    hole_int = int.from_bytes(mask, "little")
    keep_int = int.from_bytes(bytes(255 - value for value in mask), "little")
    current = seeded
    for _ in range(iterations):
        averaged = neighbour_average(current, width, height)
        current = _select(averaged, original, hole_int, keep_int, count)
    return current


def _border_indices(mask: bytes, width: int, height: int) -> List[int]:
    """Known pixels that touch the hole - the evidence a transplant must match."""
    border = []
    for index, value in enumerate(mask):
        if value:
            continue
        x, y = index % width, index // width
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and mask[ny * width + nx]:
                border.append(index)
                break
    return border


def _best_offset(
    planes: Sequence[bytes], mask: bytes, width: int, height: int
) -> Tuple[Optional[Tuple[int, int]], float]:
    """Find the translation whose pixels best continue the hole's border."""
    border = _border_indices(mask, width, height)
    if not border:
        return None, 255.0
    if len(border) > 600:  # subsample: the estimate does not need every pixel
        border = border[:: len(border) // 600 + 1]

    hole_width = max(1, sum(1 for value in mask[: width] if value) or width // 4)
    span_x = max(4, min(width - 1, int(hole_width)))
    span_y = max(4, min(height - 1, int(hole_width)))

    candidates = []
    for factor in OFFSET_STEPS:
        dx = int(span_x * factor)
        dy = int(span_y * factor)
        candidates.extend([(dx, 0), (-dx, 0), (0, dy), (0, -dy), (dx, dy), (-dx, -dy), (dx, -dy), (-dx, dy)])

    best_offset = None
    best_score = 255.0
    for offset_x, offset_y in candidates:
        total = 0
        counted = 0
        for index in border:
            x = index % width + offset_x
            y = index // width + offset_y
            if not (0 <= x < width and 0 <= y < height):
                counted = 0
                break
            source = y * width + x
            if mask[source]:  # the candidate would copy from inside the hole
                counted = 0
                break
            for plane in planes:
                total += abs(plane[index] - plane[source])
            counted += 3
        if counted and total / float(counted) < best_score:
            best_score = total / float(counted)
            best_offset = (offset_x, offset_y)
    return best_offset, best_score


def content_aware_fill(
    planes: Sequence[bytes],
    width: int,
    height: int,
    mask: bytes,
    *,
    texture: bool = True,
    progress=None,
) -> List[bytes]:
    """Fill the masked area of an RGB image, returning new planes.

    ``mask`` is one byte per pixel, 255 where the pixel must be replaced.
    """
    box = mask_bounding_box(mask, width, height, margin=max(8, min(width, height) // 16))
    if box is None:
        return [bytes(plane) for plane in planes]

    x0, y0, x1, y1 = box
    crop_w, crop_h = x1 - x0, y1 - y0
    crop_mask = crop_plane(mask, width, box)
    crop_mask = bytes(255 if value >= 128 else 0 for value in crop_mask)
    if 255 not in crop_mask:
        return [bytes(plane) for plane in planes]

    crops = [crop_plane(plane, width, box) for plane in planes]

    offset = None
    if texture:
        offset, score = _best_offset(crops, crop_mask, crop_w, crop_h)
        if score > TRANSPLANT_THRESHOLD:
            offset = None

    hole_indices = [index for index, value in enumerate(crop_mask) if value]
    # A feather weight per hole pixel: 0 at the seam, 1 deep inside, so a
    # transplant fades into the diffused result instead of showing an edge.
    feather = _feather_weights(crop_mask, crop_w, crop_h) if offset else None

    results = []
    for channel, crop in enumerate(crops):
        if progress is not None:
            progress(channel + 1, 3)
        filled = _pyramid_fill(crop, crop_mask, crop_w, crop_h)
        if offset is not None:
            filled = _apply_transplant(filled, crop, crop_mask, crop_w, crop_h, offset, hole_indices, feather)
        merged = bytearray(planes[channel])
        paste_plane(merged, width, box, filled)
        results.append(bytes(merged))
    return results


def _feather_weights(mask: bytes, width: int, height: int) -> List[float]:
    """How far each hole pixel is from the hole's edge, normalised to 0..1."""
    softened = mask
    for _ in range(3):
        softened = neighbour_average(softened, width, height)
    return [softened[index] / 255.0 for index, value in enumerate(mask) if value]


def _apply_transplant(
    filled: bytes,
    original: bytes,
    mask: bytes,
    width: int,
    height: int,
    offset: Tuple[int, int],
    hole_indices: Sequence[int],
    feather: Sequence[float],
) -> bytes:
    offset_x, offset_y = offset
    out = bytearray(filled)
    for position, index in enumerate(hole_indices):
        x = index % width + offset_x
        y = index // width + offset_y
        if not (0 <= x < width and 0 <= y < height):
            continue
        source = y * width + x
        if mask[source]:
            continue
        weight = feather[position]
        base = out[index]
        out[index] = int(base + (original[source] - base) * weight) & 0xFF
    return bytes(out)
