"""Wire protocol shared by the Z-Image Studio client and server.

Standard library only: the client half of the application must import this
module on a 32-bit Windows Python where no compiled wheels are available.

All payloads are JSON.  Binary image data travels as base64 encoded PNG in the
``image``/``mask``/``images`` fields.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional, Tuple

from .version import API_VERSION, __version__

# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

ROUTE_INFO = "/api/v1/info"
ROUTE_GENERATE = "/api/v1/generate"
ROUTE_JOBS = "/api/v1/jobs"  # + /<job_id>, + /<job_id>/cancel
ROUTE_CONVERT = "/api/v1/convert"

# --------------------------------------------------------------------------
# Limits and defaults
# --------------------------------------------------------------------------

MODE_TXT2IMG = "txt2img"
MODE_IMG2IMG = "img2img"
MODE_INPAINT = "inpaint"
MODES = (MODE_TXT2IMG, MODE_IMG2IMG, MODE_INPAINT)

#: Latent alignment.  The transformer patchifies 2x2 latents on top of an /8 VAE,
#: so both edges must be a multiple of 16; we align to 32 to stay on the safe
#: side of the model's sequence packing.
SIZE_ALIGN = 32

MIN_SIZE = 256
MAX_SIZE = 2048
MAX_STEPS = 100
MAX_IMAGES = 4
#: Hard cap for a single uploaded image (base64 expanded), 24 MiB.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024

DEFAULTS: Dict[str, Any] = {
    "mode": MODE_IMG2IMG,
    "prompt": "",
    "negative_prompt": "",
    "steps": 8,
    "guidance": 1.0,
    "strength": 0.6,
    "seed": -1,
    "width": 1024,
    "height": 1024,
    "num_images": 1,
    "mask_blur": 8,
    "mask_invert": False,
    "keep_unmasked": True,
}

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"
TERMINAL_STATUSES = (STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED)


class ProtocolError(ValueError):
    """Raised when a payload cannot be understood."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def encode_image(data: bytes) -> str:
    """base64-encode raw image bytes for transport."""
    return base64.b64encode(data).decode("ascii")


def decode_image(payload: str, *, field_name: str = "image") -> bytes:
    """Decode a base64 image field, raising :class:`ProtocolError` on garbage."""
    if not isinstance(payload, str):
        raise ProtocolError("%s must be a base64 string" % field_name)
    # Tolerate data URLs, some GUI toolkits paste them.
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError("%s is not valid base64: %s" % (field_name, exc)) from exc
    if not raw:
        raise ProtocolError("%s is empty" % field_name)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ProtocolError("%s exceeds %d bytes" % (field_name, MAX_UPLOAD_BYTES))
    return raw


def align_size(value: int, *, align: int = SIZE_ALIGN) -> int:
    """Clamp ``value`` into the supported range and snap it onto the model grid."""
    value = int(round(float(value)))
    value = max(MIN_SIZE, min(MAX_SIZE, value))
    return max(align, int(round(value / float(align))) * align)


def fit_size(width: int, height: int, *, target_pixels: int = 1024 * 1024) -> Tuple[int, int]:
    """Scale ``width``x``height`` to roughly ``target_pixels`` keeping the aspect.

    Both returned edges are aligned to the model grid, which is what the server
    expects and what the GUI shows as the effective output resolution.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    scale = (float(target_pixels) / float(width * height)) ** 0.5
    return align_size(width * scale), align_size(height * scale)


def _as_float(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if result != result or result in (float("inf"), float("-inf")):  # NaN / inf
        return fallback
    return result


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# --------------------------------------------------------------------------
# Request / response objects
# --------------------------------------------------------------------------


@dataclass
class GenerateRequest:
    """A single generation request, already validated and clamped."""

    mode: str = DEFAULTS["mode"]
    prompt: str = DEFAULTS["prompt"]
    negative_prompt: str = DEFAULTS["negative_prompt"]
    image: Optional[str] = None  # base64 PNG of the source image
    mask: Optional[str] = None  # base64 8-bit PNG, white = edit here
    steps: int = DEFAULTS["steps"]
    guidance: float = DEFAULTS["guidance"]
    strength: float = DEFAULTS["strength"]
    seed: int = DEFAULTS["seed"]
    width: int = DEFAULTS["width"]
    height: int = DEFAULTS["height"]
    num_images: int = DEFAULTS["num_images"]
    mask_blur: int = DEFAULTS["mask_blur"]
    mask_invert: bool = DEFAULTS["mask_invert"]
    keep_unmasked: bool = DEFAULTS["keep_unmasked"]
    api_version: str = API_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def summary(self) -> Dict[str, Any]:
        """Everything except the bulky base64 payloads - used for logs and history."""
        data = asdict(self)
        for key in ("image", "mask"):
            data[key] = bool(data.get(key))
        return data


def parse_generate_request(payload: Dict[str, Any]) -> GenerateRequest:
    """Validate an incoming payload and return a clamped :class:`GenerateRequest`.

    Unknown fields are ignored, out-of-range numbers are clamped rather than
    rejected so that an older client can still talk to a newer server.
    """
    if not isinstance(payload, dict):
        raise ProtocolError("payload must be a JSON object")

    mode = str(payload.get("mode", DEFAULTS["mode"])).strip().lower()
    if mode not in MODES:
        raise ProtocolError("unknown mode %r (expected one of %s)" % (mode, ", ".join(MODES)))

    prompt = str(payload.get("prompt") or "")
    if mode == MODE_TXT2IMG and not prompt.strip():
        raise ProtocolError("prompt must not be empty for txt2img")

    image = payload.get("image")
    mask = payload.get("mask")
    if mode in (MODE_IMG2IMG, MODE_INPAINT) and not image:
        raise ProtocolError("mode %s requires a source image" % mode)
    if mode == MODE_INPAINT and not mask:
        raise ProtocolError("mode inpaint requires a mask")
    if image is not None:
        decode_image(image, field_name="image")
    if mask is not None:
        decode_image(mask, field_name="mask")

    steps = int(_clamp(_as_int(payload.get("steps"), DEFAULTS["steps"]), 1, MAX_STEPS))
    guidance = _clamp(_as_float(payload.get("guidance"), DEFAULTS["guidance"]), 0.0, 20.0)
    strength = _clamp(_as_float(payload.get("strength"), DEFAULTS["strength"]), 0.0, 1.0)
    num_images = int(_clamp(_as_int(payload.get("num_images"), DEFAULTS["num_images"]), 1, MAX_IMAGES))
    mask_blur = int(_clamp(_as_int(payload.get("mask_blur"), DEFAULTS["mask_blur"]), 0, 64))

    seed = _as_int(payload.get("seed"), DEFAULTS["seed"])
    if seed < 0:
        seed = -1
    else:
        seed = seed % (2**31 - 1)

    return GenerateRequest(
        mode=mode,
        prompt=prompt,
        negative_prompt=str(payload.get("negative_prompt") or ""),
        image=image,
        mask=mask,
        steps=steps,
        guidance=guidance,
        strength=strength,
        seed=seed,
        width=align_size(_as_int(payload.get("width"), DEFAULTS["width"])),
        height=align_size(_as_int(payload.get("height"), DEFAULTS["height"])),
        num_images=num_images,
        mask_blur=mask_blur,
        mask_invert=bool(payload.get("mask_invert", DEFAULTS["mask_invert"])),
        keep_unmasked=bool(payload.get("keep_unmasked", DEFAULTS["keep_unmasked"])),
        api_version=str(payload.get("api_version") or API_VERSION),
    )


@dataclass
class JobStatus:
    """Progress and result of one generation job."""

    job_id: str
    status: str = STATUS_QUEUED
    progress: float = 0.0
    step: int = 0
    total_steps: int = 0
    message: str = ""
    error: str = ""
    seed: int = -1
    images: List[str] = field(default_factory=list)
    elapsed: float = 0.0
    queue_position: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL_STATUSES


def parse_job_status(payload: Dict[str, Any]) -> JobStatus:
    """Build a :class:`JobStatus` from a server response, tolerating omissions."""
    if not isinstance(payload, dict):
        raise ProtocolError("job status must be a JSON object")
    job_id = payload.get("job_id")
    if not job_id:
        raise ProtocolError("job status is missing job_id")
    images = payload.get("images") or []
    if not isinstance(images, list):
        raise ProtocolError("images must be a list")
    return JobStatus(
        job_id=str(job_id),
        status=str(payload.get("status") or STATUS_QUEUED),
        progress=_clamp(_as_float(payload.get("progress"), 0.0), 0.0, 1.0),
        step=_as_int(payload.get("step"), 0),
        total_steps=_as_int(payload.get("total_steps"), 0),
        message=str(payload.get("message") or ""),
        error=str(payload.get("error") or ""),
        seed=_as_int(payload.get("seed"), -1),
        images=[str(item) for item in images],
        elapsed=_as_float(payload.get("elapsed"), 0.0),
        queue_position=_as_int(payload.get("queue_position"), 0),
    )


@dataclass
class ServerInfo:
    """Capability announcement returned by ``GET /api/v1/info``."""

    name: str = "Z-Image Studio Server"
    version: str = __version__
    api_version: str = API_VERSION
    engine: str = "unknown"
    model: str = ""
    device: str = ""
    dtype: str = ""
    ready: bool = False
    modes: List[str] = field(default_factory=lambda: list(MODES))
    max_size: int = MAX_SIZE
    can_convert: bool = False
    queue_length: int = 0
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_server_info(payload: Dict[str, Any]) -> ServerInfo:
    if not isinstance(payload, dict):
        raise ProtocolError("info must be a JSON object")
    info = ServerInfo()
    for key in info.to_dict():
        if key in payload and payload[key] is not None:
            setattr(info, key, payload[key])
    info.modes = [str(m) for m in (info.modes or []) if str(m) in MODES] or list(MODES)
    info.ready = bool(info.ready)
    info.can_convert = bool(info.can_convert)
    return info
