"""Engine interface used by the job queue."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from ...protocol import GenerateRequest, ServerInfo


class Cancelled(Exception):
    """Raised inside an engine when the client cancelled the job."""


class EngineError(RuntimeError):
    """Raised for engine failures that should be reported to the user verbatim."""


class GenerationContext:
    """Progress reporting and cooperative cancellation handed to an engine."""

    def __init__(
        self,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._on_progress = on_progress
        self._is_cancelled = is_cancelled

    def progress(self, step: int, total: int, message: str = "") -> None:
        """Report ``step`` of ``total``; also a cancellation checkpoint."""
        self.check_cancelled()
        if self._on_progress is not None:
            self._on_progress(int(step), int(total), str(message))

    def cancelled(self) -> bool:
        return bool(self._is_cancelled and self._is_cancelled())

    def check_cancelled(self) -> None:
        if self.cancelled():
            raise Cancelled()


class Engine:
    """Base class for every backend that can turn a request into images."""

    name = "base"

    def prepare(self) -> None:
        """Load weights.  Called once from the worker thread before the first job."""

    def info(self) -> ServerInfo:
        raise NotImplementedError

    def generate(self, request: GenerateRequest, ctx: GenerationContext) -> Tuple[List[bytes], int]:
        """Run one request and return ``(png_bytes_list, effective_seed)``."""
        raise NotImplementedError

    def convert(self, data: bytes, max_size: int = 0) -> Tuple[bytes, int, int]:
        """Convert arbitrary image bytes to PNG, optionally shrinking the long edge.

        Engines that cannot decode foreign formats raise :class:`EngineError`;
        the client then falls back to its own limited PNG/GIF support.
        """
        raise EngineError("this engine cannot convert images")
