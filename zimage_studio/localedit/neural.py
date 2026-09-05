"""Neural inpainting on the local machine.

Wraps the LaMa network (a 51 million parameter convolutional model) behind a
plain function.  The network has a fixed 512x512 input, so an image of any size
is handled by cutting a square region around the brushed area - with enough
context around it for the network to continue structures - scaling that region
to 512x512, and pasting the result back through a feathered mask.  Everything
beyond that mask (plus the few pixels of the seam, or none at all with
``feather=0``) is left bit-identical to the original.

NumPy is required here (and only here); it is the one compiled dependency that
still ships 32-bit Windows wheels.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Callable, List, Optional, Sequence, Tuple

MODEL_FILENAME = "lama_fp32.onnx"
MODEL_SIZE = 512

#: How much context around the brushed area the network is given, as a multiple
#: of the longer side of that area.
CONTEXT_FACTOR = 2.2
#: The mask is grown by this fraction of the crop before inpainting, so that the
#: object's own soft edge is covered.
DILATE_FRACTION = 0.012

ProgressCallback = Callable[[int, int, str], None]


class NeuralUnavailable(RuntimeError):
    """Raised when the model or NumPy is missing."""


#: The first import of a large C extension must not happen in several threads at
#: once - CPython's import lock plus a concurrent garbage collection can abort
#: the process.  The result is therefore resolved once, under a lock, and cached.
_NUMPY_LOCK = threading.Lock()
_NUMPY_STATE: Optional[bool] = None


def numpy_available() -> bool:
    """Whether NumPy can be imported here (cached, thread-safe)."""
    global _NUMPY_STATE
    if _NUMPY_STATE is None:
        with _NUMPY_LOCK:
            if _NUMPY_STATE is None:
                try:
                    import numpy  # noqa: F401

                    _NUMPY_STATE = True
                except ImportError:
                    _NUMPY_STATE = False
    return _NUMPY_STATE


def search_paths() -> List[Path]:
    """Where a model file is looked for, in order."""
    candidates: List[Path] = []
    override = os.environ.get("ZIMAGE_LAMA_MODEL")
    if override:
        candidates.append(Path(override))
    here = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            Path.cwd() / "models" / MODEL_FILENAME,
            here / "models" / MODEL_FILENAME,
            Path.cwd() / MODEL_FILENAME,
        ]
    )
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "ZImageStudio" / "models" / MODEL_FILENAME)
    return candidates


def find_model(explicit: Optional[str] = None) -> Optional[Path]:
    """First existing model file, or ``None``."""
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    for candidate in search_paths():
        if candidate.is_file():
            return candidate
    return None


class NeuralInpainter:
    """Loads the network once and inpaints images with it."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = find_model(model_path)
        self._interpreter = None

    @property
    def available(self) -> bool:
        return self.model_path is not None and numpy_available()

    def describe(self) -> str:
        if not numpy_available():
            return "NumPy is missing - install numpy to enable neural inpainting."
        if self.model_path is None:
            return "Model file %s not found (looked in %s)." % (
                MODEL_FILENAME,
                ", ".join(str(item.parent) for item in search_paths()),
            )
        return "LaMa inpainting, %s" % self.model_path

    def load(self):
        """Load the graph; safe to call repeatedly."""
        if self._interpreter is not None:
            return self._interpreter
        if not self.available:
            raise NeuralUnavailable(self.describe())
        from ..nn.onnx_model import load
        from ..nn.runtime import Interpreter

        self._interpreter = Interpreter(load(self.model_path))
        return self._interpreter

    # -- the actual work ------------------------------------------------

    def inpaint(
        self,
        planes: Sequence[bytes],
        width: int,
        height: int,
        mask: bytes,
        *,
        progress: Optional[ProgressCallback] = None,
        feather: int = 3,
    ) -> List[bytes]:
        """Replace the masked pixels; returns new planes."""
        import numpy as np

        interpreter = self.load()

        image = np.frombuffer(b"".join(bytes(plane) for plane in planes), dtype=np.uint8)
        image = image.reshape(3, height, width).astype(np.float32)
        mask_array = np.frombuffer(bytes(mask), dtype=np.uint8).reshape(height, width)

        box = _crop_box(mask_array, width, height)
        if box is None:
            return [bytes(plane) for plane in planes]
        x0, y0, x1, y1 = box

        crop = image[:, y0:y1, x0:x1]
        crop_mask = mask_array[y0:y1, x0:x1].astype(np.float32) / 255.0

        small = _resize(crop, MODEL_SIZE, MODEL_SIZE) / 255.0
        small_mask = _resize(crop_mask[None], MODEL_SIZE, MODEL_SIZE)[0]
        small_mask = _dilate(small_mask, max(1, int(MODEL_SIZE * DILATE_FRACTION)))
        small_mask = (small_mask > 0.05).astype(np.float32)

        result = interpreter.run(
            {"image": small[None].astype(np.float32), "mask": small_mask[None, None]},
            progress=progress,
        )["output"][0]

        restored = _resize(np.clip(result, 0, 255), y1 - y0, x1 - x0)
        blend = _feather(crop_mask, feather)
        merged = crop * (1.0 - blend) + restored * blend

        out = image.copy()
        out[:, y0:y1, x0:x1] = merged
        out = np.clip(out, 0, 255).astype(np.uint8)
        return [out[channel].tobytes() for channel in range(3)]


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------


def _crop_box(mask_array, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """A square region around the mask, with context, clamped to the image."""
    import numpy as np

    rows = np.nonzero(mask_array.any(axis=1))[0]
    cols = np.nonzero(mask_array.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return None

    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    span = max(bottom - top, right - left)
    side = int(min(max(width, height), max(64, span * CONTEXT_FACTOR)))
    side = min(side, min(width, height)) if min(width, height) >= 64 else min(width, height)

    center_y = (top + bottom) // 2
    center_x = (left + right) // 2
    y0 = max(0, min(height - side, center_y - side // 2))
    x0 = max(0, min(width - side, center_x - side // 2))
    return x0, y0, x0 + side, y0 + side


def _resize(array, out_height: int, out_width: int):
    """Bilinear resize of the two trailing axes."""
    import numpy as np

    in_height, in_width = array.shape[-2], array.shape[-1]
    if (in_height, in_width) == (out_height, out_width):
        return np.asarray(array, dtype=np.float32)

    y = (np.arange(out_height, dtype=np.float32) + 0.5) * in_height / out_height - 0.5
    x = (np.arange(out_width, dtype=np.float32) + 0.5) * in_width / out_width - 0.5
    y0 = np.clip(np.floor(y).astype(np.int64), 0, in_height - 1)
    x0 = np.clip(np.floor(x).astype(np.int64), 0, in_width - 1)
    y1 = np.clip(y0 + 1, 0, in_height - 1)
    x1 = np.clip(x0 + 1, 0, in_width - 1)
    wy = np.clip(y - y0, 0, 1).reshape(-1, 1).astype(np.float32)
    wx = np.clip(x - x0, 0, 1).reshape(1, -1).astype(np.float32)

    source = np.asarray(array, dtype=np.float32)
    top = source[..., y0, :][..., :, x0] * (1 - wx) + source[..., y0, :][..., :, x1] * wx
    bottom = source[..., y1, :][..., :, x0] * (1 - wx) + source[..., y1, :][..., :, x1] * wx
    return top * (1 - wy) + bottom * wy


def _dilate(mask, radius: int):
    """Grow a mask by ``radius`` pixels (a separable maximum filter)."""
    import numpy as np

    grown = mask
    for axis in (0, 1):
        stack = [grown]
        for shift in range(1, radius + 1):
            stack.append(np.roll(grown, shift, axis=axis))
            stack.append(np.roll(grown, -shift, axis=axis))
        grown = np.maximum.reduce(stack)
    return grown


def _feather(mask, radius: int):
    """Soften the mask edge so the pasted region has no visible seam."""
    import numpy as np

    result = np.asarray(mask, dtype=np.float32)
    for _ in range(max(0, radius)):
        padded = np.pad(result, 1, mode="edge")
        result = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        ) / 4.0
    return np.clip(result, 0.0, 1.0)
