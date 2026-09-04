"""Integration tests for the desktop client.

They drive the real Tk widgets (under Xvfb in CI) against the in-process demo
backend, so the whole path is covered: open an image, brush a mask, submit a
job, receive the result and put it into the history.
"""

from __future__ import annotations

import pathlib
import tempfile
import time

import pytest

tk = pytest.importorskip("tkinter")

from zimage_studio import imaging as im, protocol as p  # noqa: E402
from zimage_studio.client.settings import Settings  # noqa: E402


@pytest.fixture()
def app(tmp_path):
    """A fully built main window backed by the demo engine."""
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # no display
        pytest.skip("no X display available: %s" % exc)
    root.withdraw()

    from zimage_studio.client.app import StudioApp

    settings = Settings(tmp_path / "settings.json")
    settings.set("auto_connect", False)
    instance = StudioApp(root=root, settings=settings)
    instance.start_demo_backend()
    _pump(instance, 0.4)
    try:
        yield instance
    finally:
        instance.quit()
        try:
            root.destroy()
        except tk.TclError:
            pass


def _pump(app, seconds: float) -> None:
    """Run the Tk event loop for a while so background callbacks land."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        app.root.update()
        time.sleep(0.01)


def _wait_for(app, predicate, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.root.update()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _write_png(tmp_path, width=320, height=200, color=(180, 90, 40)) -> str:
    path = tmp_path / "source.png"
    path.write_bytes(im.solid_png(width, height, color))
    return str(path)


# ----------------------------------------------------------------------
# construction and connection
# ----------------------------------------------------------------------


def test_window_builds_with_every_panel(app):
    assert app.btn_generate.winfo_exists()
    assert app.canvas.canvas.winfo_exists()
    assert app.history.winfo_exists()


def test_demo_backend_connects(app):
    assert _wait_for(app, lambda: app.server_info is not None)
    assert app.server_info.engine == "mock"


def test_language_can_be_switched_at_runtime(app):
    app.tr.switch("de")
    app._retranslate()
    assert app.btn_generate.cget("text") == "Generieren"
    app.toggle_language()
    assert app.btn_generate.cget("text") == "Generate"


# ----------------------------------------------------------------------
# opening images
# ----------------------------------------------------------------------


def test_open_png_sets_source_and_mode(app, tmp_path):
    assert app.open_image(_write_png(tmp_path))
    assert app.canvas.source_size == (320, 200)
    assert app.mode_var.get() == p.MODE_IMG2IMG


def test_open_rejects_garbage_without_a_dialog_crash(app, tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"definitely not an image")
    import tkinter.messagebox as messagebox

    calls = []
    original = messagebox.showerror
    messagebox.showerror = lambda *args, **kwargs: calls.append(args)
    try:
        assert app.open_image(str(broken)) is False
    finally:
        messagebox.showerror = original
    assert calls


def test_output_size_follows_the_source_aspect(app, tmp_path):
    app.open_image(_write_png(app_tmp := tmp_path, width=1600, height=800))
    width, height = app.output_size()
    assert width > height
    assert width % p.SIZE_ALIGN == 0 and height % p.SIZE_ALIGN == 0
    assert abs((width / float(height)) - 2.0) < 0.15


def test_output_size_uses_the_aspect_box_for_txt2img(app):
    app.mode_var.set(p.MODE_TXT2IMG)
    app.aspect_var.set("16:9")
    app._update_size_label()
    width, height = app.output_size()
    assert width > height


# ----------------------------------------------------------------------
# masking
# ----------------------------------------------------------------------


def _paint(app, points, tool="brush"):
    """Simulate a brush drag on the canvas widget."""
    app._set_tool(tool)
    canvas = app.canvas

    class _Event:
        def __init__(self, x, y):
            self.x, self.y = x, y
            self.state = 0

    scale = canvas.scale
    canvas._on_press(_Event(int(points[0][0] * scale), int(points[0][1] * scale)))
    for x, y in points[1:]:
        canvas._on_drag(_Event(int(x * scale), int(y * scale)))
    canvas._on_release(_Event(int(points[-1][0] * scale), int(points[-1][1] * scale)))
    app.root.update()


def test_painting_creates_a_mask_and_switches_to_inpaint(app, tmp_path):
    app.open_image(_write_png(tmp_path))
    assert not app.canvas.has_mask
    _paint(app, [(60, 60), (120, 90)])
    assert app.canvas.has_mask
    assert app.mode_var.get() == p.MODE_INPAINT


def test_mask_png_matches_the_source_size(app, tmp_path):
    app.open_image(_write_png(tmp_path, 320, 200))
    _paint(app, [(100, 100)])
    mask = app.canvas.mask_png()
    assert im.sniff_size(mask) == (320, 200)
    _, _, mode, pixels = im.png_decode(mask)
    assert mode == "L" and 255 in pixels


def test_undo_redo_and_clear_mask(app, tmp_path):
    app.open_image(_write_png(tmp_path))
    _paint(app, [(80, 80), (140, 120)])
    assert app.canvas.can_undo
    app.undo()
    assert not app.canvas.has_mask
    app.redo()
    assert app.canvas.has_mask
    app.clear_mask()
    assert not app.canvas.has_mask
    assert app.mode_var.get() == p.MODE_IMG2IMG


def test_eraser_removes_painted_area(app, tmp_path):
    app.open_image(_write_png(tmp_path))
    _paint(app, [(100, 100)])
    app.brush_var.set(200)
    app._on_brush_size()
    _paint(app, [(100, 100)], tool="eraser")
    assert not app.canvas.has_mask


def test_zoom_changes_the_scale(app, tmp_path):
    app.open_image(_write_png(tmp_path))
    app.zoom_actual()
    assert app.canvas.scale == 1.0
    app.zoom_in()
    assert app.canvas.scale > 1.0
    app.zoom_out()
    app.zoom_out()
    assert app.canvas.scale < 1.0


# ----------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------


def test_request_validation_blocks_empty_prompt(app):
    app.mode_var.set(p.MODE_TXT2IMG)
    app.prompt_box.set("")
    with pytest.raises(p.ProtocolError):
        app.build_request()


def test_request_for_inpaint_carries_image_and_mask(app, tmp_path):
    app.open_image(_write_png(tmp_path))
    app.prompt_box.set("a red car")
    _paint(app, [(100, 100), (150, 120)])
    request = app.build_request()
    assert request.mode == p.MODE_INPAINT
    assert request.image and request.mask
    assert im.sniff_size(p.decode_image(request.mask)) == (320, 200)


def test_generate_end_to_end_updates_history_and_preview(app, tmp_path):
    assert _wait_for(app, lambda: app.server_info is not None)
    app.open_image(_write_png(tmp_path))
    app.prompt_box.set("a lighthouse at dusk")
    app.steps_scale.set(2)
    app.strength_scale.set(0.5)

    assert app.generate()
    assert _wait_for(app, lambda: len(app.history) == 1, timeout=60)
    assert app.canvas.preview_png is not None
    assert str(app.btn_generate.cget("state")) == "normal"
    assert app.last_seed >= 0


def test_generate_twice_is_refused_while_running(app, tmp_path):
    assert _wait_for(app, lambda: app.server_info is not None)
    app.mode_var.set(p.MODE_TXT2IMG)
    app.prompt_box.set("clouds")
    app.steps_scale.set(20)
    assert app.generate()
    app.root.update()
    assert app.runner.busy
    app.cancel()
    assert _wait_for(app, lambda: not app.runner.busy, timeout=60)


def test_cancel_leaves_the_ui_usable(app):
    assert _wait_for(app, lambda: app.server_info is not None)
    app.mode_var.set(p.MODE_TXT2IMG)
    app.prompt_box.set("mountains")
    app.steps_scale.set(30)
    app.generate()
    _pump(app, 0.3)
    app.cancel()
    assert _wait_for(app, lambda: str(app.btn_generate.cget("state")) == "normal", timeout=60)


def test_result_can_be_used_as_the_next_source(app, tmp_path):
    assert _wait_for(app, lambda: app.server_info is not None)
    app.mode_var.set(p.MODE_TXT2IMG)
    app.prompt_box.set("a forest")
    app.steps_scale.set(2)
    app.megapixel_var.set("0.5")
    assert app.generate()
    assert _wait_for(app, lambda: len(app.history) == 1, timeout=60)

    app.use_result_as_source(0)
    assert app.canvas.has_image
    assert app.mode_var.get() == p.MODE_IMG2IMG


def test_saving_the_result_writes_a_png(app, tmp_path, monkeypatch):
    app.open_image(_write_png(tmp_path))
    target = tmp_path / "out.png"
    monkeypatch.setattr(
        "zimage_studio.client.app.filedialog.asksaveasfilename", lambda **kwargs: str(target)
    )
    assert app.save_result() == str(target)
    assert im.sniff_format(target.read_bytes()) == "png"


def test_saving_the_mask_writes_a_grayscale_png(app, tmp_path, monkeypatch):
    app.open_image(_write_png(tmp_path))
    _paint(app, [(100, 100)])
    target = tmp_path / "mask.png"
    monkeypatch.setattr(
        "zimage_studio.client.app.filedialog.asksaveasfilename", lambda **kwargs: str(target)
    )
    assert app.save_mask() == str(target)
    _, _, mode, _ = im.png_decode(target.read_bytes())
    assert mode == "L"


def test_settings_survive_a_restart(app, tmp_path):
    app.prompt_box.set("persisted prompt")
    app.steps_scale.set(11)
    app._store_settings()
    reloaded = Settings(app.settings.path)
    assert reloaded.get("prompt") == "persisted prompt"
    assert reloaded.get("steps") == 11
