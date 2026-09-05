"""Colour and detail adjustments, implemented without any dependency.

Every operation takes and returns three 8-bit channel planes.  Wherever the
result of a pixel depends only on its own value, the work is a 256-entry lookup
table applied with :meth:`bytes.translate`, which costs about a millisecond per
megapixel.  The few operations that mix channels or neighbours fall back to the
big-integer arithmetic in :mod:`~zimage_studio.localedit.pixels`.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

from .pixels import PlaneList, add_planes, apply_table, box_blur, make_table, weighted_mix

#: Named colours the "fill with ..." command understands.
COLORS: Dict[str, Tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "grey": (128, 128, 128),
    "red": (200, 40, 40),
    "green": (60, 160, 70),
    "blue": (50, 90, 200),
    "yellow": (230, 200, 60),
    "orange": (235, 140, 50),
    "brown": (120, 85, 55),
    "pink": (235, 150, 180),
    "purple": (130, 70, 180),
}


def _lut(planes: Sequence[bytes], tables: Sequence[bytes]) -> PlaneList:
    return [apply_table(plane, table) for plane, table in zip(planes, tables)]


def brighter(planes, width, height, amount=0.5):
    shift = 90.0 * amount
    table = make_table(lambda value: value + shift)
    return _lut(planes, [table] * 3)


def darker(planes, width, height, amount=0.5):
    factor = max(0.05, 1.0 - 0.8 * amount)
    table = make_table(lambda value: value * factor)
    return _lut(planes, [table] * 3)


def more_contrast(planes, width, height, amount=0.5):
    factor = 1.0 + 1.4 * amount
    table = make_table(lambda value: 128 + (value - 128) * factor)
    return _lut(planes, [table] * 3)


def less_contrast(planes, width, height, amount=0.5):
    factor = max(0.1, 1.0 - 0.7 * amount)
    table = make_table(lambda value: 128 + (value - 128) * factor)
    return _lut(planes, [table] * 3)


def grayscale(planes, width, height, amount=1.0):
    """Rec. 601 luma, mixed back in by ``amount``."""
    gray = _luma(planes)
    if amount >= 0.999:
        return [gray, gray, gray]
    return [_mix(plane, gray, amount) for plane in planes]


def sepia(planes, width, height, amount=1.0):
    gray = _luma(planes)
    tinted = [
        apply_table(gray, make_table(lambda value: value * 1.07 + 18)),
        apply_table(gray, make_table(lambda value: value * 0.94 + 8)),
        apply_table(gray, make_table(lambda value: value * 0.72)),
    ]
    if amount >= 0.999:
        return tinted
    return [_mix(plane, tint, amount) for plane, tint in zip(planes, tinted)]


def warmer(planes, width, height, amount=0.5):
    return _lut(
        planes,
        [
            make_table(lambda value: value + 40 * amount),
            make_table(lambda value: value + 8 * amount),
            make_table(lambda value: value - 30 * amount),
        ],
    )


def cooler(planes, width, height, amount=0.5):
    return _lut(
        planes,
        [
            make_table(lambda value: value - 30 * amount),
            make_table(lambda value: value - 4 * amount),
            make_table(lambda value: value + 40 * amount),
        ],
    )


def saturate(planes, width, height, amount=0.5):
    gray = _luma(planes)
    factor = 1.0 + 1.5 * amount
    return [_scale_around(plane, gray, factor) for plane in planes]


def desaturate(planes, width, height, amount=0.5):
    return grayscale(planes, width, height, min(1.0, amount))


def blur(planes, width, height, amount=0.5):
    passes = max(1, int(round(1 + 7 * amount)))
    return [box_blur(plane, width, height, passes) for plane in planes]


def sharpen(planes, width, height, amount=0.5):
    """Unsharp mask: the image plus its difference from a blurred copy."""
    strength = 0.4 + 1.2 * amount
    result = []
    for plane in planes:
        soft = box_blur(plane, width, height, 1)
        result.append(
            bytes(
                min(255, max(0, int(value + (value - blurred) * strength)))
                for value, blurred in zip(plane, soft)
            )
        )
    return result


def denoise(planes, width, height, amount=0.5):
    return [box_blur(plane, width, height, 1 if amount < 0.6 else 2) for plane in planes]


def invert(planes, width, height, amount=1.0):
    table = make_table(lambda value: 255 - value)
    return _lut(planes, [table] * 3)


def fill_color(planes, width, height, amount=1.0, color=(128, 128, 128)):
    size = len(planes[0])
    solid = [bytes((channel,)) * size for channel in color]
    if amount >= 0.999:
        return solid
    return [_mix(plane, fill, amount) for plane, fill in zip(planes, solid)]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _luma(planes: Sequence[bytes]) -> bytes:
    """Weighted grey plane (0.299 R + 0.587 G + 0.114 B), one pass in C."""
    red = apply_table(planes[0], make_table(lambda value: value * 0.299))
    green = apply_table(planes[1], make_table(lambda value: value * 0.587))
    blue = apply_table(planes[2], make_table(lambda value: value * 0.114))
    # The three parts sum to at most 255, so they can simply be added.
    return add_planes([red, green, blue])


def _mix(plane: bytes, other: bytes, amount: float) -> bytes:
    return weighted_mix(plane, other, amount)


def _scale_around(plane: bytes, center: bytes, factor: float) -> bytes:
    return bytes(
        min(255, max(0, int(c + (value - c) * factor))) for value, c in zip(plane, center)
    )


#: Every operation the prompt parser can select.
OPERATIONS: Dict[str, Callable[..., PlaneList]] = {
    "brighter": brighter,
    "darker": darker,
    "more_contrast": more_contrast,
    "less_contrast": less_contrast,
    "grayscale": grayscale,
    "sepia": sepia,
    "warmer": warmer,
    "cooler": cooler,
    "saturate": saturate,
    "desaturate": desaturate,
    "blur": blur,
    "sharpen": sharpen,
    "denoise": denoise,
    "invert": invert,
    "fill_color": fill_color,
}


def apply_operation(name: str, planes, width: int, height: int, amount: float = 0.5, **kwargs):
    """Run one named operation, returning new planes."""
    try:
        operation = OPERATIONS[name]
    except KeyError:
        raise KeyError("unknown operation %r" % name) from None
    return operation(planes, width, height, amount, **kwargs)
