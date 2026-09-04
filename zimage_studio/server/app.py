"""HTTP front-end for the generation queue.

Deliberately built on :mod:`http.server` from the standard library: the server
half already needs PyTorch, adding a web framework on top buys nothing and makes
the deployment story on a fresh GPU box harder.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import socket
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

from ..protocol import (
    MAX_UPLOAD_BYTES,
    ROUTE_CONVERT,
    ROUTE_GENERATE,
    ROUTE_INFO,
    ROUTE_JOBS,
    ProtocolError,
    decode_image,
    encode_image,
    parse_generate_request,
)
from ..version import APP_NAME, __version__
from .convert import to_png
from .engines.base import Engine, EngineError
from .jobs import JobQueue

logger = logging.getLogger(__name__)

#: Largest JSON body we accept (source image + mask, base64 expanded).
MAX_BODY_BYTES = 3 * MAX_UPLOAD_BYTES


class StudioServer(ThreadingHTTPServer):
    """Threading HTTP server that carries the queue and the auth token."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: Tuple[str, int], engine: Engine, token: str = "") -> None:
        self.queue = JobQueue(engine)
        self.engine = engine
        self.token = token or ""
        super().__init__(address, StudioHandler)

    def server_activate(self) -> None:  # noqa: D102
        super().server_activate()
        self.queue.start()

    def server_close(self) -> None:  # noqa: D102
        self.queue.stop()
        super().server_close()


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "ZImageStudio/%s" % __version__
    protocol_version = "HTTP/1.1"

    # -- plumbing ----------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug("%s - %s", self.address_string(), format % args)

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):  # client gave up mid-poll
            logger.debug("client disconnected before the response was written")

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message, "status": status})

    def _read_json(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(400, "invalid Content-Length")
            return None
        if length <= 0:
            self._error(400, "empty request body")
            return None
        if length > MAX_BODY_BYTES:
            self._error(413, "request body larger than %d bytes" % MAX_BODY_BYTES)
            return None
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._error(400, "invalid JSON: %s" % exc)
            return None
        if not isinstance(payload, dict):
            self._error(400, "request body must be a JSON object")
            return None
        return payload

    def _authorised(self) -> bool:
        token = getattr(self.server, "token", "")
        if not token:
            return True
        if self.headers.get("X-Auth-Token") == token:
            return True
        self._error(401, "missing or invalid X-Auth-Token")
        return False

    # -- routes ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/":
            self._send_json(200, {"name": APP_NAME, "version": __version__, "info": ROUTE_INFO})
            return
        if not self._authorised():
            return
        if path == ROUTE_INFO:
            self._handle_info()
            return
        if path.startswith(ROUTE_JOBS + "/"):
            self._handle_job_status(path[len(ROUTE_JOBS) + 1 :])
            return
        self._error(404, "unknown endpoint %s" % path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if not self._authorised():
            return
        if path == ROUTE_GENERATE:
            self._handle_generate()
            return
        if path == ROUTE_CONVERT:
            self._handle_convert()
            return
        if path.startswith(ROUTE_JOBS + "/") and path.endswith("/cancel"):
            self._handle_cancel(path[len(ROUTE_JOBS) + 1 : -len("/cancel")])
            return
        self._error(404, "unknown endpoint %s" % path)

    # -- handlers ----------------------------------------------------------

    def _handle_info(self) -> None:
        info = self.server.engine.info()
        info.queue_length = self.server.queue.queue_length()
        self._send_json(200, info.to_dict())

    def _handle_generate(self) -> None:
        payload = self._read_json()
        if payload is None:
            return
        try:
            request = parse_generate_request(payload)
        except ProtocolError as exc:
            self._error(400, str(exc))
            return
        status = self.server.queue.submit(request)
        self._send_json(202, status.to_dict())

    def _handle_job_status(self, job_id: str) -> None:
        status = self.server.queue.status(job_id)
        if status is None:
            self._error(404, "unknown job %s" % job_id)
            return
        self._send_json(200, status.to_dict())

    def _handle_cancel(self, job_id: str) -> None:
        if not self.server.queue.cancel(job_id):
            status = self.server.queue.status(job_id)
            if status is None:
                self._error(404, "unknown job %s" % job_id)
            else:
                self._send_json(409, {"error": "job already finished", "status_name": status.status})
            return
        self._send_json(200, {"job_id": job_id, "cancelled": True})

    def _handle_convert(self) -> None:
        payload = self._read_json()
        if payload is None:
            return
        try:
            raw = decode_image(payload.get("image"), field_name="image")
            max_size = int(payload.get("max_size") or 0)
            png, width, height = to_png(raw, max_size)
        except (ProtocolError, EngineError, ValueError) as exc:
            self._error(400, str(exc))
            return
        self._send_json(200, {"image": encode_image(png), "width": width, "height": height})


def serve(
    host: str = "127.0.0.1",
    port: int = 8787,
    engine_name: str = "zimage",
    token: str = "",
    engine_kwargs: Optional[Dict[str, Any]] = None,
    ready_callback: Optional[Callable[[StudioServer], None]] = None,
) -> None:
    """Run the server until interrupted."""
    from .engines import create_engine

    engine = create_engine(engine_name, **(engine_kwargs or {}))
    server = StudioServer((host, port), engine, token=token)
    bound_host, bound_port = server.server_address[:2]
    logger.info("%s %s listening on http://%s:%s", APP_NAME, __version__, bound_host, bound_port)
    if not token:
        logger.warning("No --token set: anybody who can reach this port can use the GPU.")
    if ready_callback is not None:
        ready_callback(server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        server.server_close()


def local_ip() -> str:
    """Best effort LAN address, shown in the startup banner."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
