"""Guards for the promise that makes the 32-bit build possible.

The client half must stay importable on an old, 32-bit Python: no compiled
dependencies, and no syntax newer than 3.8.  Both are easy to break by accident
and impossible to notice on a modern developer machine, so they are asserted
here.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLIENT = ROOT / "zimage_studio" / "client"
SHARED = [ROOT / "zimage_studio" / "protocol.py", ROOT / "zimage_studio" / "imaging.py"]

#: Packages that have no wheel for 32-bit Windows, or that pull one in.
FORBIDDEN = {
    "torch",
    "numpy",
    "PIL",
    "Pillow",
    "transformers",
    "safetensors",
    "accelerate",
    "requests",
    "loguru",
    "tqdm",
    "huggingface_hub",
}


def _modules():
    return sorted(CLIENT.rglob("*.py")) + SHARED


def _imported_names(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module.split(".")[0]


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_third_party_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = sorted(set(_imported_names(tree)) & FORBIDDEN)
    assert not offenders, "%s imports %s, which cannot be installed on 32-bit Windows" % (
        path.name,
        ", ".join(offenders),
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_parses_as_python_38(path):
    """Python 3.8 is the oldest release we claim to support."""
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))


def test_pillow_is_only_used_behind_a_guard():
    """The optional Pillow fast path must never be a hard import."""
    source = (ROOT / "zimage_studio" / "server" / "convert.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    pillow_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "PIL"
    ]
    assert pillow_imports
    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for child in ast.walk(node)
        if child in pillow_imports
    ]
    assert len(guarded) == len(pillow_imports)


def test_protocol_and_imaging_are_shared_but_dependency_free():
    """Both halves import these two modules, so they carry the strictest rules."""
    for path in SHARED:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not set(_imported_names(tree)) & FORBIDDEN
