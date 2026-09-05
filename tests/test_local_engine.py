"""The engine that edits images on the user's own machine."""

from __future__ import annotations

import pathlib
import time

import pytest

from zimage_studio import imaging as im, protocol as p
from zimage_studio.client.api import JobRunner
from zimage_studio.client.local_backend import LocalClient
from zimage_studio.server.engines import create_engine
from zimage_studio.server.engines.base import EngineError, GenerationContext

MODEL = pathlib.Path(__file__).resolve().parents[1] / "models" / "lama_fp32.onnx"
needs_model = pytest.mark.skipif(not MODEL.is_file(), reason="model file not present")


def _image(width=96, height=96):
    """A vertical gradient with a bright square to remove."""
    pixels = bytearray(width * height * 3)
    for y in range(height):
        value = (y * 200) // height
        for x in range(width):
            index = (y * width + x) * 3
            inside = 40 <= x < 56 and 40 <= y < 56
            pixels[index : index + 3] = bytes((250, 40, 40) if inside else (value, value, 200 - value))
    return im.png_encode(width, height, bytes(pixels), im.MODE_RGB)


def _mask(width=96, height=96):
    return im.strokes_to_png(width, height, [im.make_stroke([(47, 47)], 12)])


def _request(**overrides):
    payload = {
        "mode": p.MODE_INPAINT,
        "prompt": "",
        "image": p.encode_image(_image()),
        "mask": p.encode_image(_mask()),
        "width": 256,
        "height": 256,
        "strength": 0.6,
    }
    payload.update(overrides)
    return p.parse_generate_request(payload)


def _decode(png):
    width, height, mode, pixels = im.png_decode(png)
    return width, height, mode, bytes(pixels)


# ----------------------------------------------------------------------
# capabilities
# ----------------------------------------------------------------------


def test_info_without_any_model_is_honest_about_it():
    engine = create_engine(
        "local", model_path="/nonexistent/model.onnx", segment_model_path="/nonexistent/seg.onnx"
    )
    info = engine.info()
    assert info.ready  # the classical operations still work
    assert "no neural model" in info.model
    assert "not found" in info.detail
    assert p.MODE_TXT2IMG not in info.modes


@needs_model
def test_info_with_a_model_reports_the_network():
    engine = create_engine("local", model_path=str(MODEL))
    info = engine.info()
    assert "LaMa" in info.model
    assert info.device == "cpu"


# ----------------------------------------------------------------------
# editing
# ----------------------------------------------------------------------


def test_removal_falls_back_to_content_aware_fill_without_a_model():
    engine = create_engine(
        "local", model_path="/nonexistent/model.onnx", segment_model_path="/nonexistent/seg.onnx"
    )
    images, seed = engine.generate(_request(prompt="entferne das"), GenerationContext())
    assert len(images) == 1 and seed >= 0
    width, height, _mode, pixels = _decode(images[0])
    assert (width, height) == (96, 96)  # the source size is kept
    centre = pixels[(47 * width + 47) * 3 : (47 * width + 47) * 3 + 3]
    assert centre[0] < 200  # the red square is gone


def test_adjustments_only_touch_the_masked_area():
    engine = create_engine("local", model_path=None)
    images, _seed = engine.generate(_request(prompt="viel heller"), GenerationContext())
    width, height, _mode, pixels = _decode(images[0])
    _w, _h, _m, original = _decode(_image())
    corner = 0
    assert pixels[corner : corner + 3] == original[corner : corner + 3]
    inside = (47 * width + 47) * 3
    assert pixels[inside] >= original[inside]


def test_image_to_image_applies_to_the_whole_picture():
    engine = create_engine("local", model_path=None)
    request = _request(mode=p.MODE_IMG2IMG, prompt="schwarzweiß", mask=None, strength=1.0)
    images, _seed = engine.generate(request, GenerationContext())
    _w, _h, _mode, pixels = _decode(images[0])
    assert pixels[0] == pixels[1] == pixels[2]  # grey everywhere, including the corner


def test_an_unknown_instruction_explains_the_vocabulary():
    engine = create_engine("local", model_path=None)
    with pytest.raises(EngineError) as excinfo:
        engine.generate(_request(mode=p.MODE_IMG2IMG, mask=None, prompt="ein Leuchtturm bei Nacht"), GenerationContext())
    assert "entfernen" in str(excinfo.value)


def test_removal_without_a_mask_is_refused():
    engine = create_engine("local", model_path=None)
    with pytest.raises(EngineError, match="Pinsel"):
        engine.generate(
            _request(mode=p.MODE_IMG2IMG, mask=None, prompt="entferne das"), GenerationContext()
        )


def test_a_request_without_an_image_is_refused():
    engine = create_engine("local", model_path=None)
    request = p.parse_generate_request({"mode": p.MODE_TXT2IMG, "prompt": "entferne das"})
    with pytest.raises(EngineError, match="Bild"):
        engine.generate(request, GenerationContext())


def test_progress_is_reported():
    engine = create_engine("local", model_path=None)
    seen = []
    engine.generate(
        _request(prompt="heller und schärfer"),
        GenerationContext(on_progress=lambda step, total, message: seen.append((step, total))),
    )
    assert seen and seen[-1][0] == seen[-1][1]


@needs_model
def test_neural_removal_changes_the_masked_area_only():
    engine = create_engine("local", model_path=str(MODEL))
    images, _seed = engine.generate(_request(prompt="entferne das"), GenerationContext())
    width, height, _mode, pixels = _decode(images[0])
    _w, _h, _m, original = _decode(_image())
    inside = (47 * width + 47) * 3
    assert pixels[inside : inside + 3] != original[inside : inside + 3]
    assert pixels[0:3] == original[0:3]


# ----------------------------------------------------------------------
# in-process client
# ----------------------------------------------------------------------


def test_local_client_speaks_the_same_protocol():
    client = LocalClient(model_path=None)
    try:
        info = client.info()
        assert info.engine == "local"
        status = client.generate(_request(prompt="entferne das"))
        deadline = time.time() + 120
        while not status.finished and time.time() < deadline:
            time.sleep(0.05)
            status = client.job_status(status.job_id)
        assert status.status == p.STATUS_DONE, status.error
        assert status.images
    finally:
        client.close()


def test_local_client_reports_unknown_jobs():
    from zimage_studio.client.api import ClientError

    client = LocalClient(model_path=None)
    try:
        client.generate(_request(prompt="entferne das"))
        with pytest.raises(ClientError):
            client.job_status("nope")
    finally:
        client.close()


def test_job_runner_drives_the_local_client():
    client = LocalClient(model_path=None)
    runner = JobRunner(client)
    done, errors = [], []
    try:
        assert runner.start(_request(prompt="entferne das"), lambda s: None, done.append, errors.append)
        deadline = time.time() + 120
        while runner.busy and time.time() < deadline:
            time.sleep(0.05)
        assert not errors, errors
        assert done and done[0].images
    finally:
        client.close()


def test_engine_errors_reach_the_client_as_a_failed_job():
    client = LocalClient(model_path=None)
    try:
        status = client.generate(
            _request(mode=p.MODE_IMG2IMG, mask=None, prompt="ein Leuchtturm bei Nacht")
        )
        deadline = time.time() + 60
        while not status.finished and time.time() < deadline:
            time.sleep(0.05)
            status = client.job_status(status.job_id)
        assert status.status == p.STATUS_ERROR
        assert "EngineError" in status.error
    finally:
        client.close()


# ----------------------------------------------------------------------
# subject detection
# ----------------------------------------------------------------------

SEGMENT_MODEL = pathlib.Path(__file__).resolve().parents[1] / "models" / "u2netp.onnx"
needs_segment_model = pytest.mark.skipif(
    not SEGMENT_MODEL.is_file(), reason="subject-detection model not present"
)


def _subject_image(width=128, height=128):
    """A plain background with one obvious object in the middle."""
    pixels = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            index = (y * width + x) * 3
            inside = (x - width // 2) ** 2 + (y - height // 2) ** 2 < (width // 4) ** 2
            pixels[index : index + 3] = bytes((230, 40, 40) if inside else (150, 190, 220))
    return im.png_encode(width, height, bytes(pixels), im.MODE_RGB)


def test_subject_operations_need_the_model():
    engine = create_engine("local", model_path=None, segment_model_path="/nonexistent.onnx")
    request = p.parse_generate_request(
        {"mode": p.MODE_IMG2IMG, "prompt": "freistellen", "image": p.encode_image(_subject_image())}
    )
    with pytest.raises(EngineError, match="u2netp"):
        engine.generate(request, GenerationContext())


@needs_segment_model
def test_cutout_returns_an_image_with_transparency():
    engine = create_engine("local", model_path=None, segment_model_path=str(SEGMENT_MODEL))
    request = p.parse_generate_request(
        {"mode": p.MODE_IMG2IMG, "prompt": "freistellen", "image": p.encode_image(_subject_image())}
    )
    images, _seed = engine.generate(request, GenerationContext())
    width, height, mode, pixels = im.png_decode(images[0])
    assert mode == im.MODE_RGBA
    centre = (height // 2 * width + width // 2) * 4
    assert pixels[centre + 3] > 200  # the subject is opaque
    assert pixels[3] < 60  # the corner is transparent


@needs_segment_model
def test_background_colour_replaces_only_the_background():
    engine = create_engine("local", model_path=None, segment_model_path=str(SEGMENT_MODEL))
    request = p.parse_generate_request(
        {
            "mode": p.MODE_IMG2IMG,
            "prompt": "hintergrund schwarz",
            "image": p.encode_image(_subject_image()),
        }
    )
    images, _seed = engine.generate(request, GenerationContext())
    width, height, mode, pixels = im.png_decode(images[0])
    assert mode == im.MODE_RGB
    corner = pixels[0:3]
    centre_index = (height // 2 * width + width // 2) * 3
    assert list(corner) == [0, 0, 0]
    assert pixels[centre_index] > 150  # the subject kept its colour


@needs_segment_model
def test_info_lists_both_networks():
    engine = create_engine(
        "local", model_path=str(MODEL) if MODEL.is_file() else None, segment_model_path=str(SEGMENT_MODEL)
    )
    info = engine.info()
    assert "U^2-Net" in info.model
