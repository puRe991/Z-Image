"""HTTP client for the Z-Image Studio backend.

Uses :mod:`urllib` only - the GUI has to run on a 32-bit Python where
``requests`` and friends may not be installed.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, Optional
import urllib.error
import urllib.request

from ..protocol import (
    ROUTE_CONVERT,
    ROUTE_GENERATE,
    ROUTE_INFO,
    ROUTE_JOBS,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_ERROR,
    GenerateRequest,
    JobStatus,
    ProtocolError,
    ServerInfo,
    decode_image,
    encode_image,
    parse_job_status,
    parse_server_info,
)
from ..version import __version__

DEFAULT_BASE_URL = "http://127.0.0.1:8787"
DEFAULT_TIMEOUT = 30.0
#: Generation can take minutes on a slow GPU; polling stays cheap.
POLL_INTERVAL = 0.5


class ClientError(RuntimeError):
    """Any problem talking to the backend, with a message fit for a dialog."""


class StudioClient:
    """Thin, synchronous wrapper around the REST API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        token: str = "",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.token = token or ""
        self.timeout = timeout

    # -- low level ---------------------------------------------------------

    def _request(
        self, method: str, route: str, payload: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        url = self.base_url + route
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Accept", "application/json")
        request.add_header("User-Agent", "ZImageStudio/%s" % __version__)
        if data is not None:
            request.add_header("Content-Type", "application/json; charset=utf-8")
        if self.token:
            request.add_header("X-Auth-Token", self.token)
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise ClientError(_http_error_message(exc, url)) from exc
        except urllib.error.URLError as exc:
            raise ClientError("Cannot reach %s (%s)." % (url, getattr(exc, "reason", exc))) from exc
        except OSError as exc:
            raise ClientError("Network error talking to %s: %s" % (url, exc)) from exc
        if not body:
            return {}
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ClientError("Server sent a malformed response: %s" % exc) from exc
        if not isinstance(parsed, dict):
            raise ClientError("Server sent an unexpected response type.")
        return parsed

    # -- API ---------------------------------------------------------------

    def info(self, timeout: float = 8.0) -> ServerInfo:
        try:
            return parse_server_info(self._request("GET", ROUTE_INFO, timeout=timeout))
        except ProtocolError as exc:
            raise ClientError(str(exc)) from exc

    def generate(self, request: GenerateRequest) -> JobStatus:
        payload = json.loads(request.to_json())
        try:
            return parse_job_status(self._request("POST", ROUTE_GENERATE, payload, timeout=120.0))
        except ProtocolError as exc:
            raise ClientError(str(exc)) from exc

    def job_status(self, job_id: str) -> JobStatus:
        try:
            return parse_job_status(self._request("GET", "%s/%s" % (ROUTE_JOBS, job_id)))
        except ProtocolError as exc:
            raise ClientError(str(exc)) from exc

    def cancel(self, job_id: str) -> bool:
        try:
            self._request("POST", "%s/%s/cancel" % (ROUTE_JOBS, job_id), {})
            return True
        except ClientError:
            return False

    def convert(self, data: bytes, max_size: int = 0) -> "tuple[bytes, int, int]":
        """Ask the server to turn any image format into a PNG the GUI can show."""
        response = self._request(
            "POST", ROUTE_CONVERT, {"image": encode_image(data), "max_size": int(max_size)}, timeout=60.0
        )
        try:
            png = decode_image(response.get("image", ""), field_name="image")
        except ProtocolError as exc:
            raise ClientError(str(exc)) from exc
        return png, int(response.get("width") or 0), int(response.get("height") or 0)


def _http_error_message(exc: urllib.error.HTTPError, url: str) -> str:
    detail = ""
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        detail = str(payload.get("error") or "")
    except Exception:  # noqa: BLE001 - error bodies are best effort
        detail = ""
    if exc.code == 401:
        return "Authentication failed - check the access token in the server settings."
    if exc.code == 404:
        return detail or "%s not found on the server." % url
    return detail or "Server returned HTTP %s for %s." % (exc.code, url)


class JobRunner:
    """Runs one job on a worker thread and reports back through callbacks.

    The GUI passes callbacks that marshal into the Tk thread; nothing in here
    touches Tk itself, which keeps the class unit-testable.
    """

    def __init__(self, client: StudioClient) -> None:
        self.client = client
        self._thread: Optional[threading.Thread] = None
        self._job_id = ""
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    @property
    def job_id(self) -> str:
        return self._job_id

    def start(
        self,
        request: GenerateRequest,
        on_progress: Callable[[JobStatus], None],
        on_done: Callable[[JobStatus], None],
        on_error: Callable[[str], None],
    ) -> bool:
        """Begin a job.  Returns ``False`` when one is already running."""
        with self._lock:
            if self.busy:
                return False
            self._cancel.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(request, on_progress, on_done, on_error),
                name="zimage-job",
                daemon=True,
            )
            self._thread.start()
            return True

    def cancel(self) -> None:
        self._cancel.set()
        job_id = self._job_id
        if job_id:
            threading.Thread(target=self.client.cancel, args=(job_id,), daemon=True).start()

    def _run(
        self,
        request: GenerateRequest,
        on_progress: Callable[[JobStatus], None],
        on_done: Callable[[JobStatus], None],
        on_error: Callable[[str], None],
    ) -> None:
        try:
            status = self.client.generate(request)
            self._job_id = status.job_id
            on_progress(status)
            while not status.finished:
                if self._cancel.is_set():
                    self.client.cancel(status.job_id)
                time.sleep(POLL_INTERVAL)
                status = self.client.job_status(status.job_id)
                on_progress(status)
            if status.status == STATUS_DONE:
                on_done(status)
            elif status.status == STATUS_CANCELLED:
                on_error("Cancelled.")
            elif status.status == STATUS_ERROR:
                on_error(status.error or "The server reported an unknown error.")
        except ClientError as exc:
            on_error(str(exc))
        except Exception as exc:  # noqa: BLE001 - never kill the worker silently
            on_error("Unexpected client error: %s" % exc)
        finally:
            self._job_id = ""
