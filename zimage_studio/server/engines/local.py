"""The engine that runs on the user's own machine.

It combines two things that need no server, no GPU and no internet:

* **Neural inpainting** - the LaMa network removes whatever is brushed and
  continues the surroundings.  This is what "remove this" in the prompt does.
* **Classical adjustments** - brightness, contrast, colour, detail; driven by
  the same text box through a small instruction vocabulary.

If the model file or NumPy is missing the engine still works: removal then falls
back to the dependency-free content-aware fill, at lower quality.  The
capability is reported in :meth:`info` so the interface can say which of the two
is active.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

from ...imaging import (
    MODE_L,
    MODE_RGB,
    ImageError,
    png_decode,
    png_encode,
    resize_nearest,
    to_rgb,
)
from ...localedit.adjust import apply_operation
from ...localedit.commands import REMOVE, Command, describe_vocabulary, parse_prompt
from ...localedit.fill import content_aware_fill, mask_bounding_box
from ...localedit.neural import NeuralInpainter, numpy_available
from ...localedit.pixels import merge_planes, split_planes
from ...protocol import MODE_IMG2IMG, MODE_INPAINT, GenerateRequest, ServerInfo, decode_image
from ...version import __version__
from .base import Engine, EngineError, GenerationContext

#: The default word strength, used to normalise the interface's strength slider.
NEUTRAL_AMOUNT = 0.6


class LocalEngine(Engine):
    name = "local"

    def __init__(self, model_path: Optional[str] = None, language: str = "de") -> None:
        self.language = language
        self.inpainter = NeuralInpainter(model_path)
        self._loaded = False

    # -- capabilities --------------------------------------------------

    @property
    def neural(self) -> bool:
        return self.inpainter.available

    def info(self) -> ServerInfo:
        if self.neural:
            model = "LaMa (51M) + local adjustments"
            detail = "Neural inpainting active: %s" % self.inpainter.model_path
        else:
            model = "local adjustments (no neural model)"
            detail = self.inpainter.describe()
        return ServerInfo(
            name="Z-Image Studio (local)",
            version=__version__,
            engine=self.name,
            model=model,
            device="cpu",
            dtype="uint8",
            ready=True,
            can_convert=False,
            modes=[MODE_IMG2IMG, MODE_INPAINT],
            detail=detail,
        )

    def prepare(self) -> None:
        """Load the network up front so the first edit is not the slow one."""
        if self.neural and not self._loaded:
            self.inpainter.load()
            self._loaded = True

    # -- generation ----------------------------------------------------

    def generate(self, request: GenerateRequest, ctx: GenerationContext) -> Tuple[List[bytes], int]:
        if not request.image:
            raise EngineError(
                "Der lokale Modus bearbeitet vorhandene Bilder - bitte zuerst ein Bild öffnen."
            )

        width, height, mode, pixels = png_decode(decode_image(request.image))
        _, rgb = to_rgb(mode, bytes(pixels))
        planes = split_planes(bytes(rgb))

        mask = self._mask(request, width, height)
        commands = parse_prompt(request.prompt, has_mask=mask is not None)
        if not commands:
            raise EngineError(
                "Diese Anweisung kenne ich nicht.\n\n%s" % describe_vocabulary(self.language)
            )

        strength = float(request.strength)
        total = len(commands)
        for index, command in enumerate(commands):
            ctx.progress(index, total, command.name)
            if command.is_removal:
                if mask is None:
                    raise EngineError(
                        "Zum Entfernen bitte den Bereich mit dem Pinsel markieren."
                    )
                planes = self._remove(planes, width, height, mask, ctx, index, total, request)
            else:
                planes = self._adjust(planes, width, height, mask, command, strength)
        ctx.progress(total, total, "done")

        merged = merge_planes(planes)
        png = png_encode(width, height, bytes(merged), MODE_RGB)
        seed = request.seed if request.seed >= 0 else int(time.time()) % (2**31 - 1)
        return [png], seed

    # -- steps ---------------------------------------------------------

    def _mask(self, request: GenerateRequest, width: int, height: int) -> Optional[bytes]:
        if not request.mask or request.mode != MODE_INPAINT:
            return None
        mask_w, mask_h, mask_mode, mask_pixels = png_decode(decode_image(request.mask))
        if mask_mode != MODE_L:
            _, rgb = to_rgb(mask_mode, bytes(mask_pixels))
            mask_pixels = bytearray(rgb[0::3])
        mask = bytes(mask_pixels)
        if (mask_w, mask_h) != (width, height):
            mask = bytes(resize_nearest(mask_w, mask_h, mask, MODE_L, width, height))
        if request.mask_invert:
            mask = bytes(255 - value for value in mask)
        if 255 not in mask and not any(mask):
            return None
        return mask

    def _remove(
        self,
        planes: Sequence[bytes],
        width: int,
        height: int,
        mask: bytes,
        ctx: GenerationContext,
        step: int,
        total: int,
        request: GenerateRequest,
    ) -> List[bytes]:
        """Inpaint the masked area, preferring the neural network."""
        if not self.neural:
            ctx.progress(step, total, "content-aware fill")
            return content_aware_fill(planes, width, height, mask)

        def on_node(done: int, nodes: int, op: str) -> None:
            # Map the network's progress onto this command's slice of the bar.
            ctx.progress(
                int((step + done / float(max(1, nodes))) * 1000 / max(1, total)),
                1000,
                "neural inpainting (%d%%)" % int(100 * done / max(1, nodes)),
            )

        feather = 0 if request.keep_unmasked else max(1, int(request.mask_blur) // 4)
        try:
            return self.inpainter.inpaint(
                planes, width, height, mask, progress=on_node, feather=feather
            )
        except MemoryError as exc:
            raise EngineError(
                "Zu wenig Arbeitsspeicher für das Modell. Bitte andere Programme schließen "
                "oder einen kleineren Bereich markieren."
            ) from exc

    def _adjust(
        self,
        planes: Sequence[bytes],
        width: int,
        height: int,
        mask: Optional[bytes],
        command: Command,
        strength: float,
    ) -> List[bytes]:
        """Apply one adjustment, restricted to the mask when there is one."""
        amount = max(0.0, min(1.0, strength * (command.amount / NEUTRAL_AMOUNT)))
        options = dict(command.options)
        adjusted = apply_operation(command.name, planes, width, height, amount, **options)
        if mask is None:
            return list(adjusted)
        return _blend_masked(planes, adjusted, mask, width, height)


def _blend_masked(
    base: Sequence[bytes], other: Sequence[bytes], mask: bytes, width: int, height: int
) -> List[bytes]:
    """Mix ``other`` into ``base`` where the mask is set.

    Only the mask's bounding box is walked, so brushing a small area stays cheap
    on a large image.
    """
    box = mask_bounding_box(mask, width, height, margin=1)
    if box is None:
        return [bytes(plane) for plane in base]
    x0, y0, x1, y1 = box

    result = [bytearray(plane) for plane in base]
    for y in range(y0, y1):
        row = y * width
        for x in range(x0, x1):
            index = row + x
            weight = mask[index]
            if not weight:
                continue
            if weight == 255:
                for channel in range(3):
                    result[channel][index] = other[channel][index]
                continue
            for channel in range(3):
                start = result[channel][index]
                result[channel][index] = (
                    start + (other[channel][index] - start) * weight // 255
                ) & 0xFF
    return [bytes(plane) for plane in result]
