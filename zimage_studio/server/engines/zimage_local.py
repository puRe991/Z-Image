"""The real engine: Z-Image weights running locally through PyTorch.

Supports the three modes of the protocol:

``txt2img``
    Plain sampling from noise.
``img2img``
    SDEdit: the source image is encoded with the VAE and the sampler starts
    partway down the sigma schedule, controlled by ``strength``.
``inpaint``
    Same, plus a latent mask that re-anchors everything outside the brushed area
    on the source after every step.  The untouched pixels are additionally
    composited back at full resolution so they stay bit-exact.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from ...protocol import MODE_IMG2IMG, MODE_INPAINT, GenerateRequest, ServerInfo, decode_image
from ...version import __version__
from ..convert import HAVE_PILLOW, to_png
from .base import Engine, EngineError, GenerationContext

#: Repository root - ``src`` holds the reference implementation this engine drives.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "fp16": "float16", "float32": "float32", "fp32": "float32"}


def _ensure_src_on_path() -> None:
    src = _REPO_ROOT / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


class ZImageEngine(Engine):
    name = "zimage"

    def __init__(
        self,
        model_path: str = "ckpts/Z-Image-Turbo",
        device: str = "auto",
        dtype: str = "bfloat16",
        attention_backend: Optional[str] = None,
        allow_download: bool = True,
        preload: bool = False,
    ) -> None:
        self.model_path = model_path
        self.requested_device = device
        self.requested_dtype = _DTYPES.get(str(dtype).lower(), "bfloat16")
        self.attention_backend = attention_backend or os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
        self.allow_download = allow_download
        self.device = ""
        self.resolved_model_path = ""
        self.components: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._load_error = ""
        if preload:
            self.prepare()

    # -- loading -----------------------------------------------------------

    @property
    def ready(self) -> bool:
        return bool(self.components)

    def _resolve_device(self, torch) -> str:
        requested = (self.requested_device or "auto").lower()
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def prepare(self) -> None:
        with self._lock:
            if self.components:
                return
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - depends on the host
                self._load_error = (
                    "PyTorch is not installed in this Python. Install a 64-bit Python with torch "
                    "on the machine that runs the server."
                )
                raise EngineError(self._load_error) from exc
            if not HAVE_PILLOW:
                self._load_error = "Pillow is required by the Z-Image engine (pip install pillow)."
                raise EngineError(self._load_error)

            _ensure_src_on_path()
            try:
                from utils import ensure_model_weights, load_from_local_dir, set_attention_backend
            except ImportError as exc:  # pragma: no cover
                raise EngineError(
                    "Could not import the Z-Image reference implementation from %s/src" % _REPO_ROOT
                ) from exc

            device = self._resolve_device(torch)
            dtype = getattr(torch, self.requested_dtype)
            if device in ("cpu", "mps") and self.requested_dtype == "bfloat16":
                # bf16 is painfully slow (or unsupported) outside CUDA.
                dtype = torch.float32

            path = Path(self.model_path)
            if self.allow_download:
                resolved = ensure_model_weights(str(path), verify=False)
            else:
                if not path.is_dir():
                    raise EngineError(
                        "Model directory %s does not exist and --no-download was given." % path
                    )
                resolved = str(path)

            components = load_from_local_dir(resolved, device=device, dtype=dtype, compile=False)
            try:
                set_attention_backend(self.attention_backend)
            except Exception as exc:  # noqa: BLE001 - a bad backend must not be fatal
                import logging

                logging.getLogger(__name__).warning(
                    "Attention backend %r unavailable (%s), keeping the default.",
                    self.attention_backend,
                    exc,
                )

            self.device = device
            self.resolved_model_path = resolved
            self.requested_dtype = str(dtype).replace("torch.", "")
            self.components = components

    # -- info --------------------------------------------------------------

    def info(self) -> ServerInfo:
        return ServerInfo(
            engine=self.name,
            version=__version__,
            model=self.resolved_model_path or self.model_path,
            device=self.device or ("%s (not loaded yet)" % self.requested_device),
            dtype=self.requested_dtype,
            ready=self.ready,
            can_convert=HAVE_PILLOW,
            detail=self._load_error
            or ("Weights loaded." if self.ready else "Weights load on the first job."),
        )

    # -- generation --------------------------------------------------------

    def generate(self, request: GenerateRequest, ctx: GenerationContext) -> Tuple[List[bytes], int]:
        # prepare() puts the reference implementation on sys.path, so it has to
        # run before zimage can be imported.
        self.prepare()

        import torch
        from PIL import Image, ImageFilter

        from zimage import generate as zimage_generate

        components = self.components
        vae = components["vae"]
        device = self.device

        seed = request.seed if request.seed >= 0 else int(time.time() * 1_000_000) % (2**31 - 1)
        width, height = request.width, request.height

        source: Optional["Image.Image"] = None
        init_latents = None
        mask_latents = None
        feather_mask: Optional["Image.Image"] = None

        if request.image and request.mode in (MODE_IMG2IMG, MODE_INPAINT):
            source = _load_image(decode_image(request.image)).resize((width, height), Image.LANCZOS)
            ctx.progress(0, request.steps, "encoding source image")
            init_latents = self._encode(torch, vae, source, seed)

        if request.mask and request.mode == MODE_INPAINT:
            mask_image = _load_mask(decode_image(request.mask), width, height, request.mask_invert)
            if request.mask_blur:
                mask_image = mask_image.filter(ImageFilter.GaussianBlur(float(request.mask_blur)))
            feather_mask = mask_image
            mask_latents = self._mask_to_latents(torch, mask_image, init_latents)

        total_steps = max(1, request.steps)
        results: List[bytes] = []

        for index in range(request.num_images):
            ctx.check_cancelled()
            image_seed = seed + index
            generator = torch.Generator(device=_generator_device(device)).manual_seed(image_seed)

            def on_step(step: int, total: int, _offset: int = index) -> None:
                overall_total = total * request.num_images
                ctx.progress(_offset * total + step, overall_total, "step %d/%d" % (step, total))

            images = zimage_generate(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                **components,
                height=height,
                width=width,
                num_inference_steps=total_steps,
                guidance_scale=float(request.guidance),
                generator=generator,
                init_latents=init_latents,
                strength=float(request.strength) if init_latents is not None else 1.0,
                mask_latents=mask_latents,
                callback=on_step,
            )

            result = images[0]
            if source is not None and feather_mask is not None and request.keep_unmasked:
                result = Image.composite(result.convert("RGB"), source.convert("RGB"), feather_mask)
            results.append(_encode_png(result))

        return results, seed

    def convert(self, data: bytes, max_size: int = 0) -> Tuple[bytes, int, int]:
        return to_png(data, max_size)

    # -- helpers -----------------------------------------------------------

    def _encode(self, torch, vae, image, seed: int):
        """VAE-encode a PIL image into scaled latents."""
        import numpy as np

        array = np.asarray(image.convert("RGB"), dtype="float32") / 127.5 - 1.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=vae.dtype)
        with torch.no_grad():
            posterior = vae.encode(tensor).latent_dist
            generator = torch.Generator(device=_generator_device(self.device)).manual_seed(seed)
            latents = posterior.sample(generator)
        shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
        return ((latents - shift) * vae.config.scaling_factor).to(torch.float32)

    def _mask_to_latents(self, torch, mask_image, init_latents):
        """Downsample the pixel mask onto the latent grid (1 = repaint)."""
        import numpy as np
        from PIL import Image

        if init_latents is None:
            raise EngineError("inpainting needs a source image")
        latent_height, latent_width = int(init_latents.shape[-2]), int(init_latents.shape[-1])
        small = mask_image.resize((latent_width, latent_height), Image.BILINEAR)
        array = np.asarray(small, dtype="float32") / 255.0
        return torch.from_numpy(array)[None, None].to(device=self.device, dtype=torch.float32)


def _generator_device(device: str) -> str:
    # torch.Generator does not support the mps backend.
    return "cpu" if str(device).startswith("mps") else device


def _load_image(data: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")


def _load_mask(data: bytes, width: int, height: int, invert: bool):
    from PIL import Image, ImageOps

    mask = Image.open(io.BytesIO(data)).convert("L").resize((width, height), Image.BILINEAR)
    if invert:
        mask = ImageOps.invert(mask)
    return mask


def _encode_png(image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()
