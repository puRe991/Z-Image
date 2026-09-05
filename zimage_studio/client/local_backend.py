"""Run the editing engine inside the client process.

On a machine that hosts no server - the case this application was built for -
there is no reason to send images through a socket.  :class:`LocalClient`
exposes exactly the interface :class:`~zimage_studio.client.api.StudioClient`
does, so the user interface and the job runner cannot tell the difference, but
every call goes straight to the local engine.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

from ..protocol import GenerateRequest, JobStatus, ServerInfo
from ..server.engines import create_engine
from ..server.engines.base import EngineError
from ..server.jobs import JobQueue
from .api import ClientError


class LocalClient:
    """In-process stand-in for the HTTP client."""

    def __init__(self, model_path: Optional[str] = None, language: str = "de") -> None:
        self.base_url = "local://"
        self.token = ""
        self.model_path = model_path
        self.language = language
        self._engine = None
        self._queue: Optional[JobQueue] = None
        # Reentrant on purpose: the queue property resolves the engine
        # property while already holding the lock.
        self._lock = threading.RLock()

    # -- lifecycle -----------------------------------------------------

    @property
    def engine(self):
        with self._lock:
            if self._engine is None:
                self._engine = create_engine(
                    "local", model_path=self.model_path, language=self.language
                )
            return self._engine

    @property
    def queue(self) -> JobQueue:
        with self._lock:
            if self._queue is None:
                self._queue = JobQueue(self.engine)
                self._queue.start()
            return self._queue

    def close(self) -> None:
        with self._lock:
            if self._queue is not None:
                self._queue.stop()
                self._queue = None

    def reload(self, model_path: Optional[str]) -> None:
        """Point the engine at a different model file."""
        self.close()
        with self._lock:
            self.model_path = model_path
            self._engine = None

    # -- client interface ----------------------------------------------

    def info(self, timeout: float = 0.0) -> ServerInfo:
        try:
            info = self.engine.info()
        except EngineError as exc:
            raise ClientError(str(exc)) from exc
        info.queue_length = self._queue.queue_length() if self._queue else 0
        return info

    def generate(self, request: GenerateRequest) -> JobStatus:
        return self.queue.submit(request)

    def job_status(self, job_id: str) -> JobStatus:
        status = self.queue.status(job_id)
        if status is None:
            raise ClientError("unknown job %s" % job_id)
        return status

    def cancel(self, job_id: str) -> bool:
        return self.queue.cancel(job_id)

    def convert(self, data: bytes, max_size: int = 0) -> Tuple[bytes, int, int]:
        """Only PNG can be converted without Pillow; the caller handles the rest."""
        from ..server.convert import to_png

        try:
            return to_png(data, max_size)
        except Exception as exc:  # noqa: BLE001 - surfaced as a dialog
            raise ClientError(str(exc)) from exc
