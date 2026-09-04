"""Validation rules of the wire protocol."""

import pytest

from zimage_studio import imaging, protocol as p


def _png() -> str:
    return p.encode_image(imaging.solid_png(8, 8, (1, 2, 3)))


def test_defaults_are_clamped_and_aligned():
    request = p.parse_generate_request({"mode": "txt2img", "prompt": "x", "width": 1000, "height": 30})
    assert request.width % p.SIZE_ALIGN == 0
    assert request.height == p.MIN_SIZE


def test_out_of_range_numbers_are_clamped_not_rejected():
    request = p.parse_generate_request(
        {"mode": "txt2img", "prompt": "x", "steps": 10**6, "guidance": -4, "strength": 9, "num_images": 99}
    )
    assert request.steps == p.MAX_STEPS
    assert request.guidance == 0.0
    assert request.strength == 1.0
    assert request.num_images == p.MAX_IMAGES


def test_garbage_numbers_fall_back_to_defaults():
    request = p.parse_generate_request({"mode": "txt2img", "prompt": "x", "steps": "many", "seed": None})
    assert request.steps == p.DEFAULTS["steps"]
    assert request.seed == -1


def test_txt2img_requires_a_prompt():
    with pytest.raises(p.ProtocolError):
        p.parse_generate_request({"mode": "txt2img", "prompt": "   "})


def test_img2img_requires_an_image():
    with pytest.raises(p.ProtocolError):
        p.parse_generate_request({"mode": "img2img", "prompt": "x"})


def test_inpaint_requires_a_mask():
    with pytest.raises(p.ProtocolError):
        p.parse_generate_request({"mode": "inpaint", "prompt": "x", "image": _png()})


def test_unknown_mode_is_rejected():
    with pytest.raises(p.ProtocolError):
        p.parse_generate_request({"mode": "upscale", "prompt": "x"})


def test_broken_base64_is_rejected():
    with pytest.raises(p.ProtocolError):
        p.parse_generate_request({"mode": "img2img", "prompt": "x", "image": "!!!not base64!!!"})


def test_data_urls_are_accepted():
    assert p.decode_image("data:image/png;base64," + _png())


def test_seed_is_normalised():
    assert p.parse_generate_request({"mode": "txt2img", "prompt": "x", "seed": -99}).seed == -1
    assert p.parse_generate_request({"mode": "txt2img", "prompt": "x", "seed": 2**40}).seed >= 0


def test_fit_size_keeps_aspect_and_grid():
    width, height = p.fit_size(1920, 1080)
    assert width % p.SIZE_ALIGN == 0 and height % p.SIZE_ALIGN == 0
    assert width > height
    assert 0.6 * 1024 * 1024 < width * height < 1.6 * 1024 * 1024


def test_summary_hides_payloads():
    summary = p.parse_generate_request({"mode": "img2img", "prompt": "x", "image": _png()}).summary()
    assert summary["image"] is True and isinstance(summary["image"], bool)


def test_job_status_roundtrip():
    status = p.parse_job_status({"job_id": "a", "status": "done", "progress": 5, "images": ["x"]})
    assert status.progress == 1.0 and status.finished


def test_job_status_needs_an_id():
    with pytest.raises(p.ProtocolError):
        p.parse_job_status({"status": "done"})


def test_server_info_defaults_survive_partial_payloads():
    info = p.parse_server_info({"engine": "mock", "modes": ["img2img", "bogus"]})
    assert info.engine == "mock" and info.modes == ["img2img"]
