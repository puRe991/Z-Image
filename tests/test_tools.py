"""The helper scripts a user actually runs: the system check and the bundle builder."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location("tool_" + name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------
# check_system.py
# ----------------------------------------------------------------------


def test_system_check_runs_and_reports():
    check = _load("check_system")
    assert check.main([]) in (0, 1)  # 1 only means something is missing here


def test_system_check_detects_memory():
    check = _load("check_system")
    assert check.total_memory_mb() > 0


def test_system_check_benchmark_and_estimate():
    pytest.importorskip("numpy")
    check = _load("check_system")
    speed = check.benchmark()
    assert speed > 0.01
    assert check.estimate() is True


def test_system_check_reports_a_missing_model(monkeypatch, capsys):
    check = _load("check_system")
    from zimage_studio.localedit import neural

    monkeypatch.setattr(neural, "find_model", lambda explicit=None: None)
    assert check.check_model() is True  # a warning, not a failure
    assert "not found" in capsys.readouterr().out


def test_line_marks_failures():
    check = _load("check_system")
    assert check.line(check.OK, "fine") is True
    assert check.line(check.WARN, "hmm") is True
    assert check.line(check.BAD, "broken", "do this") is False


# ----------------------------------------------------------------------
# build_offline_bundle.py
# ----------------------------------------------------------------------


def test_bundle_copies_the_application(tmp_path):
    bundle = _load("build_offline_bundle")
    target = tmp_path / "app"
    bundle.copy_application(target)
    assert (target / "zimage_studio" / "client" / "app.py").is_file()
    assert (target / "docs" / "OFFLINE-INSTALL.md").is_file()
    # the 200 MB model must never be copied in by accident
    assert not list(target.rglob("*.onnx"))
    assert not list(target.rglob("__pycache__"))


def test_bundle_readme_names_every_step(tmp_path):
    bundle = _load("build_offline_bundle")
    bundle.write_readme(tmp_path, wheels=4, model_ok=True)
    text = (tmp_path / "LIESMICH.txt").read_text(encoding="utf-8")
    assert "python.org" in text
    assert "pip install --no-index" in text
    assert bundle.MODEL_NAME in text
    assert "Apache-2.0" in text


def test_bundle_readme_warns_when_the_model_is_missing(tmp_path):
    bundle = _load("build_offline_bundle")
    bundle.write_readme(tmp_path, wheels=0, model_ok=False)
    assert "FEHLT" in (tmp_path / "LIESMICH.txt").read_text(encoding="utf-8")


def test_bundle_checksum_matches_hashlib(tmp_path):
    import hashlib

    bundle = _load("build_offline_bundle")
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"z-image" * 1000)
    assert bundle.sha256(sample) == hashlib.sha256(sample.read_bytes()).hexdigest()


def test_bundle_download_rejects_a_wrong_checksum(tmp_path, monkeypatch):
    bundle = _load("build_offline_bundle")
    target = tmp_path / "model.onnx"
    target.write_bytes(b"not the model")
    # An existing file with the wrong checksum must be re-downloaded, and a
    # failing download must be reported rather than silently accepted.
    monkeypatch.setattr(
        bundle.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
    )
    assert bundle.download("https://example.invalid/model", target, "0" * 64) is False


def test_bundle_download_accepts_a_matching_file(tmp_path):
    import hashlib

    bundle = _load("build_offline_bundle")
    target = tmp_path / "model.onnx"
    target.write_bytes(b"content")
    digest = hashlib.sha256(b"content").hexdigest()
    assert bundle.download("https://example.invalid/model", target, digest) is True
