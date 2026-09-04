"""End-to-end tests over real HTTP against the demo engine."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from zimage_studio import imaging as im, protocol as p
from zimage_studio.client.api import ClientError, JobRunner, StudioClient
from zimage_studio.server.app import StudioServer
from zimage_studio.server.engines import create_engine


def _wait(client: StudioClient, job_id: str, timeout: float = 20.0) -> p.JobStatus:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.job_status(job_id)
        if status.finished:
            return status
        time.sleep(0.02)
    raise AssertionError("job %s did not finish within %.0fs" % (job_id, timeout))


def _source(width: int = 256, height: int = 256) -> str:
    return p.encode_image(im.solid_png(width, height, (200, 40, 40)))


def test_info_reports_capabilities(client):
    info = client.info()
    assert info.engine == "mock" and info.ready
    assert p.MODE_IMG2IMG in info.modes


def test_txt2img_roundtrip(client):
    request = p.parse_generate_request(
        {"mode": "txt2img", "prompt": "a cat", "width": 256, "height": 256, "steps": 3, "seed": 7}
    )
    status = _wait(client, client.generate(request).job_id)
    assert status.status == p.STATUS_DONE
    assert status.seed == 7
    png = p.decode_image(status.images[0])
    assert im.sniff_size(png) == (256, 256)


def test_img2img_keeps_size_and_uses_the_source(client):
    request = p.parse_generate_request(
        {
            "mode": "img2img",
            "prompt": "repaint",
            "image": _source(),
            "width": 256,
            "height": 256,
            "strength": 0.0,
            "steps": 2,
            "seed": 1,
        }
    )
    status = _wait(client, client.generate(request).job_id)
    _, _, mode, pixels = im.png_decode(p.decode_image(status.images[0]))
    # strength 0 means "keep the source", so the flat red must survive.
    assert bytes(pixels[:3]) == bytes((200, 40, 40))
    assert mode == im.MODE_RGB


def test_inpaint_only_touches_the_masked_area(client):
    mask = p.encode_image(im.strokes_to_png(256, 256, [im.make_stroke([(192, 192)], 24)]))
    request = p.parse_generate_request(
        {
            "mode": "inpaint",
            "prompt": "patch",
            "image": _source(),
            "mask": mask,
            "width": 256,
            "height": 256,
            "strength": 1.0,
            "steps": 2,
            "seed": 3,
        }
    )
    status = _wait(client, client.generate(request).job_id)
    _, _, _, pixels = im.png_decode(p.decode_image(status.images[0]))
    top_left = bytes(pixels[:3])
    offset = (192 * 256 + 192) * 3
    centre_of_mask = bytes(pixels[offset : offset + 3])
    assert top_left == bytes((200, 40, 40))  # outside the mask: untouched
    assert centre_of_mask != top_left  # inside the mask: repainted


def test_multiple_images_are_returned(client):
    request = p.parse_generate_request(
        {"mode": "txt2img", "prompt": "x", "num_images": 3, "steps": 1, "width": 64, "height": 64}
    )
    status = _wait(client, client.generate(request).job_id)
    assert len(status.images) == 3
    assert len(set(status.images)) == 3  # different seeds -> different images


def test_random_seed_is_reported_back(client):
    request = p.parse_generate_request({"mode": "txt2img", "prompt": "x", "seed": -1, "steps": 1})
    status = _wait(client, client.generate(request).job_id)
    assert status.seed >= 0


def test_progress_is_reported_while_running():
    server = StudioServer(("127.0.0.1", 0), create_engine("mock", step_delay=0.05))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        client = StudioClient("http://%s:%d" % (host, port))
        job = client.generate(
            p.parse_generate_request(
                {"mode": "txt2img", "prompt": "x", "steps": 12, "width": 64, "height": 64}
            )
        )
        seen = []
        deadline = time.time() + 20
        while time.time() < deadline:
            status = client.job_status(job.job_id)
            seen.append(status.progress)
            if status.finished:
                break
            time.sleep(0.02)
        assert any(0.0 < value < 1.0 for value in seen), seen
        assert seen[-1] == 1.0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cancel_stops_a_running_job():
    server = StudioServer(("127.0.0.1", 0), create_engine("mock", step_delay=0.05))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        client = StudioClient("http://%s:%d" % (host, port))
        job = client.generate(
            p.parse_generate_request(
                {"mode": "txt2img", "prompt": "x", "steps": 60, "width": 64, "height": 64}
            )
        )
        time.sleep(0.15)
        assert client.cancel(job.job_id)
        status = _wait(client, job.job_id, timeout=10)
        assert status.status == p.STATUS_CANCELLED
        assert not status.images
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cancelling_a_finished_job_is_reported(client):
    job = client.generate(p.parse_generate_request({"mode": "txt2img", "prompt": "x", "steps": 1}))
    _wait(client, job.job_id)
    assert client.cancel(job.job_id) is False


def test_unknown_job_returns_404(client):
    with pytest.raises(ClientError):
        client.job_status("does-not-exist")


def test_invalid_request_is_rejected_with_a_message(server):
    host, port = server.server_address[:2]
    url = "http://%s:%d%s" % (host, port, p.ROUTE_GENERATE)
    body = json.dumps({"mode": "img2img", "prompt": "x"}).encode()
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=10)
    assert excinfo.value.code == 400
    assert "source image" in json.loads(excinfo.value.read())["error"]


def test_malformed_json_is_rejected(server):
    host, port = server.server_address[:2]
    url = "http://%s:%d%s" % (host, port, p.ROUTE_GENERATE)
    request = urllib.request.Request(url, data=b"{not json", method="POST")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=10)
    assert excinfo.value.code == 400


def test_unknown_endpoint_returns_404(server):
    host, port = server.server_address[:2]
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen("http://%s:%d/api/v1/nope" % (host, port), timeout=10)
    assert excinfo.value.code == 404


def test_token_is_enforced():
    engine = create_engine("mock", step_delay=0.0)
    server = StudioServer(("127.0.0.1", 0), engine, token="s3cret")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        base = "http://%s:%d" % (host, port)
        with pytest.raises(ClientError) as excinfo:
            StudioClient(base).info()
        assert "Authentication failed" in str(excinfo.value)
        assert StudioClient(base, token="s3cret").info().ready
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_convert_endpoint_returns_png(client):
    pil = pytest.importorskip("PIL.Image")
    import io

    buffer = io.BytesIO()
    pil.new("RGB", (120, 60), (5, 6, 7)).save(buffer, format="JPEG")
    png, width, height = client.convert(buffer.getvalue(), max_size=60)
    assert im.sniff_format(png) == "png"
    assert (width, height) == (60, 30)


def test_unreachable_server_gives_a_readable_error():
    with pytest.raises(ClientError) as excinfo:
        StudioClient("http://127.0.0.1:9", timeout=2.0).info()
    assert "Cannot reach" in str(excinfo.value)


def test_job_runner_reports_progress_and_result(client):
    runner = JobRunner(client)
    done: list = []
    errors: list = []
    progress: list = []
    request = p.parse_generate_request({"mode": "txt2img", "prompt": "x", "steps": 4, "width": 64, "height": 64})
    assert runner.start(request, progress.append, done.append, errors.append)
    deadline = time.time() + 20
    while runner.busy and time.time() < deadline:
        time.sleep(0.02)
    assert not errors, errors
    assert len(done) == 1 and done[0].images
    assert progress


def test_job_runner_refuses_a_second_job(client):
    runner = JobRunner(client)
    request = p.parse_generate_request({"mode": "txt2img", "prompt": "x", "steps": 1})
    assert runner.start(request, lambda s: None, lambda s: None, lambda e: None)
    second = runner.start(request, lambda s: None, lambda s: None, lambda e: None)
    deadline = time.time() + 20
    while runner.busy and time.time() < deadline:
        time.sleep(0.02)
    assert second is False


def test_job_runner_surfaces_server_errors():
    runner = JobRunner(StudioClient("http://127.0.0.1:9", timeout=2.0))
    errors: list = []
    request = p.parse_generate_request({"mode": "txt2img", "prompt": "x"})
    runner.start(request, lambda s: None, lambda s: None, errors.append)
    deadline = time.time() + 20
    while runner.busy and time.time() < deadline:
        time.sleep(0.02)
    assert errors and "Cannot reach" in errors[0]
