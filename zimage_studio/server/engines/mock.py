"""A GPU-free engine.

It powers the automated test-suite and the GUI's *demo* mode: everything around
the model - queueing, progress, cancellation, masks, history - can be exercised
on a laptop without downloading 12 GB of weights.  The images it produces are
procedural, deterministic for a given seed, and clearly not model output.
"""

from __future__ import annotations

import hashlib
import time
from typing import List, Tuple

from ...imaging import MODE_L, MODE_RGB, png_decode, png_encode, resize_nearest, to_rgb
from ...protocol import MODE_IMG2IMG, MODE_INPAINT, GenerateRequest, ServerInfo, decode_image
from ...version import __version__
from ..convert import to_png
from .base import Engine, GenerationContext


class MockEngine(Engine):
    name = "mock"

    def __init__(self, step_delay: float = 0.02) -> None:
        self.step_delay = max(0.0, float(step_delay))

    def info(self) -> ServerInfo:
        return ServerInfo(
            engine=self.name,
            version=__version__,
            model="procedural-demo",
            device="cpu",
            dtype="uint8",
            ready=True,
            can_convert=True,
            detail="Demo engine - generates procedural images, no Z-Image weights loaded.",
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _palette(seed: int, prompt: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        digest = hashlib.sha256(("%d|%s" % (seed, prompt)).encode("utf-8")).digest()
        return (digest[0], digest[1], digest[2]), (digest[3], digest[4], digest[5])

    def _gradient(self, width: int, height: int, seed: int, prompt: str) -> bytearray:
        (r0, g0, b0), (r1, g1, b1) = self._palette(seed, prompt)
        out = bytearray(width * height * 3)
        for y in range(height):
            fy = y / float(max(1, height - 1))
            row = y * width * 3
            for x in range(width):
                fx = x / float(max(1, width - 1))
                t = (fx + fy) * 0.5
                out[row + x * 3] = int(r0 + (r1 - r0) * t) & 0xFF
                out[row + x * 3 + 1] = int(g0 + (g1 - g0) * fx) & 0xFF
                out[row + x * 3 + 2] = int(b0 + (b1 - b0) * fy) & 0xFF
        return out

    # -- Engine API --------------------------------------------------------

    def generate(self, request: GenerateRequest, ctx: GenerationContext) -> Tuple[List[bytes], int]:
        seed = request.seed if request.seed >= 0 else int(time.time() * 1000) % (2**31 - 1)
        width, height = request.width, request.height

        source = None
        if request.image:
            src_w, src_h, src_mode, src_px = png_decode(decode_image(request.image))
            _, rgb = to_rgb(src_mode, bytes(src_px))
            source = resize_nearest(src_w, src_h, bytes(rgb), MODE_RGB, width, height)

        mask = None
        if request.mask:
            m_w, m_h, m_mode, m_px = png_decode(decode_image(request.mask))
            gray = m_px if m_mode == MODE_L else to_rgb(m_mode, bytes(m_px))[1][0::3]
            mask = resize_nearest(m_w, m_h, bytes(gray), MODE_L, width, height)
            if request.mask_invert:
                mask = bytearray(255 - value for value in mask)

        results: List[bytes] = []
        for index in range(request.num_images):
            image_seed = seed + index
            target = self._gradient(width, height, image_seed, request.prompt)
            total = max(1, request.steps)
            for step in range(1, total + 1):
                ctx.progress(step, total, "demo step %d/%d" % (step, total))
                if self.step_delay:
                    time.sleep(self.step_delay)
            if source is not None:
                strength = request.strength if request.mode in (MODE_IMG2IMG, MODE_INPAINT) else 1.0
                target = _blend(source, target, strength, mask if request.mode == MODE_INPAINT else None)
            results.append(png_encode(width, height, bytes(target), MODE_RGB))
        return results, seed

    def convert(self, data: bytes, max_size: int = 0) -> Tuple[bytes, int, int]:
        return to_png(data, max_size)


def _blend(
    source: bytearray, target: bytearray, strength: float, mask: "bytearray | None"
) -> bytearray:
    """Mix ``target`` into ``source`` by ``strength``, restricted to ``mask``."""
    strength = max(0.0, min(1.0, float(strength)))
    out = bytearray(source)
    for i in range(0, len(out), 3):
        weight = strength
        if mask is not None:
            weight *= mask[i // 3] / 255.0
        if weight <= 0.0:
            continue
        for channel in range(3):
            a = out[i + channel]
            b = target[i + channel]
            out[i + channel] = int(a + (b - a) * weight) & 0xFF
    return out
