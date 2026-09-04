"""Shared fixtures: a live server with the GPU-free demo engine."""

from __future__ import annotations

import pathlib
import sys
import threading

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from zimage_studio.client.api import StudioClient  # noqa: E402
from zimage_studio.server.app import StudioServer  # noqa: E402
from zimage_studio.server.engines import create_engine  # noqa: E402


@pytest.fixture()
def server():
    """A running server on an ephemeral port, torn down after the test."""
    instance = StudioServer(("127.0.0.1", 0), create_engine("mock", step_delay=0.0))
    thread = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=5)


@pytest.fixture()
def client(server) -> StudioClient:
    host, port = server.server_address[:2]
    return StudioClient("http://%s:%d" % (host, port), timeout=15.0)
