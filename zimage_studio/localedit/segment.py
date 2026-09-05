"""Find the main subject of a picture.

Uses U²-Net (the small "u2netp" variant: 1.1 million parameters, 5 MB) through
the same NumPy runtime as the inpainting network.  It returns a soft mask of
whatever stands out in the image, which is what "freistellen" and "Hintergrund
entfernen" need - and, unlike the brush, it needs no manual work.

The network is class-agnostic: it finds *the salient object*, not "the person"
or "the car".  Asking it to remove one specific thing among several will not
work; that is what the brush is for.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from .neural import numpy_available

MODEL_FILENAME = "u2netp.onnx"
MODEL_SIZE = 320

#: ImageNet statistics the network was trained with.
CHANNEL_MEAN = (0.485, 0.456, 0.406)
CHANNEL_STD = (0.229, 0.224, 0.225)

ProgressCallback = Callable[[int, int, str], None]


def search_paths() -> List[Path]:
    candidates: List[Path] = []
    override = os.environ.get("ZIMAGE_U2NET_MODEL")
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            Path.cwd() / "models" / MODEL_FILENAME,
            root / "models" / MODEL_FILENAME,
            Path.cwd() / MODEL_FILENAME,
        ]
    )
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "ZImageStudio" / "models" / MODEL_FILENAME)
    return candidates


def find_model(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    for candidate in search_paths():
        if candidate.is_file():
            return candidate
    return None


class SubjectSegmenter:
    """Produces a soft foreground mask for an image."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = find_model(model_path)
        self._interpreter = None

    @property
    def available(self) -> bool:
        return self.model_path is not None and numpy_available()

    def describe(self) -> str:
        if not numpy_available():
            return "NumPy is missing - install numpy to enable subject detection."
        if self.model_path is None:
            return "Model file %s not found." % MODEL_FILENAME
        return "U^2-Net subject detection, %s" % self.model_path

    def load(self):
        if self._interpreter is None:
            if not self.available:
                raise RuntimeError(self.describe())
            from ..nn.onnx_model import load
            from ..nn.runtime import Interpreter

            self._interpreter = Interpreter(load(self.model_path))
        return self._interpreter

    def mask(
        self,
        planes: Sequence[bytes],
        width: int,
        height: int,
        *,
        progress: Optional[ProgressCallback] = None,
    ) -> bytes:
        """Return an 8-bit mask: 255 where the subject is, 0 for background."""
        import numpy as np

        from .neural import _resize

        interpreter = self.load()

        image = np.frombuffer(b"".join(bytes(plane) for plane in planes), dtype=np.uint8)
        image = image.reshape(3, height, width).astype(np.float32)

        small = _resize(image, MODEL_SIZE, MODEL_SIZE)
        peak = float(small.max()) or 255.0
        small = small / peak
        for channel in range(3):
            small[channel] = (small[channel] - CHANNEL_MEAN[channel]) / CHANNEL_STD[channel]

        name = interpreter.graph.input_names[0]
        outputs = interpreter.run({name: small[None].astype(np.float32)}, progress=progress)
        # The first output is the fused prediction; the others are side outputs
        # from the intermediate stages.
        prediction = outputs[list(outputs)[0]][0, 0]

        low, high = float(prediction.min()), float(prediction.max())
        if high - low < 1e-6:
            normalised = np.zeros_like(prediction)
        else:
            normalised = (prediction - low) / (high - low)

        full = _resize(normalised[None], height, width)[0]
        return np.clip(full * 255.0, 0, 255).astype(np.uint8).tobytes()
