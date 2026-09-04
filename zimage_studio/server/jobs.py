"""Serial job queue.

Only one generation may touch the GPU at a time, so jobs are executed by a
single worker thread while the HTTP handler threads only ever read status.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
import traceback
from typing import Dict, List, Optional

from ..protocol import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_QUEUED,
    STATUS_RUNNING,
    GenerateRequest,
    JobStatus,
    encode_image,
)
from .engines.base import Cancelled, Engine, GenerationContext

logger = logging.getLogger(__name__)

#: Finished jobs are dropped after this many seconds so long-running servers do
#: not accumulate base64 blobs forever.
RETENTION_SECONDS = 30 * 60
MAX_FINISHED_JOBS = 24


class _Job:
    def __init__(self, job_id: str, request: GenerateRequest) -> None:
        self.request = request
        self.status = JobStatus(job_id=job_id, status=STATUS_QUEUED, total_steps=request.steps)
        self.cancel_event = threading.Event()
        self.created = time.time()
        self.started = 0.0
        self.finished_at = 0.0


class JobQueue:
    """Accepts requests, runs them one by one and keeps their status around."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._jobs: Dict[str, _Job] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._ids = itertools.count(1)
        self._stop = threading.Event()
        self._prepared = False
        self._worker = threading.Thread(target=self._run, name="zimage-worker", daemon=True)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if not self._worker.is_alive():
            self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._queue.put("")  # wake the worker
        if self._worker.is_alive():
            self._worker.join(timeout)

    # -- public API --------------------------------------------------------

    def submit(self, request: GenerateRequest) -> JobStatus:
        job_id = "job-%d-%d" % (int(time.time()), next(self._ids))
        job = _Job(job_id, request)
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            job.status.queue_position = self._pending_count()
            self._evict_locked()
        self._queue.put(job_id)
        logger.info("Queued %s (%s)", job_id, request.mode)
        return self.status(job_id)

    def status(self, job_id: str) -> Optional[JobStatus]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            # Return a copy so the caller can serialise it without holding the lock.
            return JobStatus(**job.status.to_dict())

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status.finished:
                return False
            job.cancel_event.set()
            if job.status.status == STATUS_QUEUED:
                job.status.status = STATUS_CANCELLED
                job.status.message = "cancelled before start"
                job.finished_at = time.time()
            return True

    def queue_length(self) -> int:
        with self._lock:
            return self._pending_count()

    # -- internals ---------------------------------------------------------

    def _pending_count(self) -> int:
        return sum(1 for job in self._jobs.values() if job.status.status in (STATUS_QUEUED, STATUS_RUNNING))

    def _evict_locked(self) -> None:
        now = time.time()
        finished = [
            jid
            for jid in self._order
            if self._jobs[jid].status.finished and self._jobs[jid].finished_at
        ]
        stale = [jid for jid in finished if now - self._jobs[jid].finished_at > RETENTION_SECONDS]
        overflow = finished[: max(0, len(finished) - MAX_FINISHED_JOBS)]
        for jid in set(stale) | set(overflow):
            self._jobs.pop(jid, None)
            if jid in self._order:
                self._order.remove(jid)

    def _run(self) -> None:
        while not self._stop.is_set():
            job_id = self._queue.get()
            if self._stop.is_set() or not job_id:
                break
            with self._lock:
                job = self._jobs.get(job_id)
            if job is None or job.cancel_event.is_set():
                continue
            self._execute(job)

    def _execute(self, job: _Job) -> None:
        job.started = time.time()
        with self._lock:
            job.status.status = STATUS_RUNNING
            job.status.message = "starting"
            job.status.queue_position = 0

        def on_progress(step: int, total: int, message: str) -> None:
            with self._lock:
                job.status.step = step
                job.status.total_steps = total or job.status.total_steps
                job.status.progress = min(1.0, step / float(total)) if total else 0.0
                if message:
                    job.status.message = message
                job.status.elapsed = time.time() - job.started

        ctx = GenerationContext(on_progress=on_progress, is_cancelled=job.cancel_event.is_set)
        try:
            if not self._prepared:
                with self._lock:
                    job.status.message = "loading model"
                self.engine.prepare()
                self._prepared = True
            images, seed = self.engine.generate(job.request, ctx)
            with self._lock:
                job.status.images = [encode_image(png) for png in images]
                job.status.seed = seed
                job.status.status = STATUS_DONE
                job.status.progress = 1.0
                job.status.message = "done"
        except Cancelled:
            with self._lock:
                job.status.status = STATUS_CANCELLED
                job.status.message = "cancelled"
            logger.info("Cancelled %s", job.status.job_id)
        except Exception as exc:  # noqa: BLE001 - reported back to the client
            logger.error("Job %s failed: %s", job.status.job_id, exc)
            logger.debug(traceback.format_exc())
            with self._lock:
                job.status.status = STATUS_ERROR
                job.status.error = "%s: %s" % (type(exc).__name__, exc)
                job.status.message = "failed"
        finally:
            with self._lock:
                job.status.elapsed = time.time() - job.started
                job.finished_at = time.time()
                self._evict_locked()
