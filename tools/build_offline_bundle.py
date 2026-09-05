"""Assemble everything the 32-bit laptop needs, on a machine that has internet.

The target machine may have no network at all, so this script collects the
application, the NumPy wheels for 32-bit Windows and the neural network file
into one folder that can be copied over on a USB stick.

    python tools/build_offline_bundle.py --out ZImageStudio-Offline

Then copy that folder to the laptop and follow docs/OFFLINE-INSTALL.md.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

REPO = Path(__file__).resolve().parent.parent

#: The last NumPy release with 32-bit Windows wheels, for every CPython version
#: that still has a 32-bit Windows installer.
NUMPY_VERSION = "1.24.4"
PYTHON_TAGS = ("38", "39", "310", "311")

#: Optional, but worth having: with Pillow the client can also open JPEG and
#: WebP.  Unlike NumPy, Pillow still publishes 32-bit Windows wheels, so pip
#: picks whatever is newest for each interpreter.
OPTIONAL_PACKAGES = ("pillow",)

#: The neural networks the local mode can use.  Both are permissively licensed
#: and small enough for a 32-bit process.
MODELS = (
    {
        "name": "lama_fp32.onnx",
        "url": "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        "sha256": "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
        "license": "Apache-2.0 (LaMa, Samsung AI Center; ONNX export by Carve)",
        "purpose": "Bereich entfernen / retuschieren",
        "size_mb": 204,
        "required": True,
    },
    {
        "name": "u2netp.onnx",
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        "sha256": "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
        "license": "Apache-2.0 (U^2-Net, Xuebin Qin et al.)",
        "purpose": "Motiv freistellen / Hintergrund bearbeiten",
        "size_mb": 5,
        "required": False,
    },
)

MODEL_NAME = MODELS[0]["name"]
MODEL_LICENSE = MODELS[0]["license"]

#: Application files to copy.  Tests and assets are left out to keep the stick
#: small; everything needed to run is included.
INCLUDE = ("zimage_studio", "tools", "docs", "README.md", "LICENSE", "pyproject.toml")
EXCLUDE_SUFFIXES = (".pyc",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, expected: str = "") -> bool:
    if target.is_file() and (not expected or sha256(target) == expected):
        print("  already present: %s" % target.name)
        return True
    print("  downloading %s …" % target.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
    except OSError as exc:
        print("  FAILED: %s" % exc)
        return False
    if expected:
        actual = sha256(target)
        if actual != expected:
            print("  FAILED: checksum mismatch\n    expected %s\n    got      %s" % (expected, actual))
            return False
        print("  checksum ok")
    return True


def _download_wheel(spec: str, tag: str, destination: Path) -> bool:
    command = [
        sys.executable, "-m", "pip", "download",
        "--only-binary=:all:", "--no-deps",
        "--platform", "win32",
        "--python-version", tag,
        "--dest", str(destination),
        spec,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        print("    %s for CPython %s: not available %s" % (spec, tag, detail[-1:] or ""))
        return False
    print("    %s for CPython %s: ok" % (spec, tag))
    return True


def fetch_wheels(destination: Path) -> int:
    """Download the 32-bit Windows wheels for every supported CPython."""
    destination.mkdir(parents=True, exist_ok=True)
    collected = 0
    for tag in PYTHON_TAGS:
        if _download_wheel("numpy==%s" % NUMPY_VERSION, tag, destination):
            collected += 1
        for package in OPTIONAL_PACKAGES:
            _download_wheel(package, tag, destination)
    return collected


def copy_application(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in INCLUDE:
        source = REPO / name
        if not source.exists():
            continue
        target = destination / name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.onnx"),
            )
        else:
            shutil.copy2(source, target)
    print("  application files copied")


def write_readme(bundle: Path, wheels: int, model_ok: bool) -> None:
    text = """Z-Image Studio - Offline-Paket
==============================

Inhalt
------
  app/                Das Programm
  wheels/             NumPy (+ Pillow) für 32-Bit-Windows
  models/             Die KI-Modelle {model_state}
                        lama_fp32.onnx  Bereich entfernen   (204 MB, Apache-2.0)
                        u2netp.onnx     Motiv freistellen   (5 MB, Apache-2.0)

Installation auf dem Laptop
---------------------------
 1. Python 32-Bit installieren (3.8 bis 3.11, von python.org,
    "Windows installer (32-bit)"), dabei "tcl/tk and IDLE" aktiviert lassen.
 2. Diesen Ordner auf den Laptop kopieren, z. B. nach C:\\ZImageStudio
 3. Eingabeaufforderung im Ordner öffnen und NumPy installieren:

        py -3-32 -m pip install --no-index --find-links wheels numpy pillow

 4. Programm starten:

        cd app
        py -3-32 -m zimage_studio.client

 5. Im Programm: Verarbeitung → KI-Modelldatei wählen… und
    ..\\models\\{model} auswählen (nur einmal nötig).
    Der Ordner models\\ wird auch automatisch durchsucht.

Prüfen, ob alles passt:  py -3-32 app\\tools\\check_system.py
Ausführliche Anleitung:  app\\docs\\OFFLINE-INSTALL.md
"""
    (bundle / "LIESMICH.txt").write_text(
        text.format(
            wheels=wheels,
            model=MODEL_NAME,
            model_state="" if model_ok else "(UNVOLLSTÄNDIG - bitte nachladen!)",
        ),
        encoding="utf-8",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="ZImageStudio-Offline", help="output folder")
    parser.add_argument("--skip-model", action="store_true", help="do not download the 204 MB model")
    parser.add_argument("--skip-wheels", action="store_true", help="do not download the NumPy wheels")
    args = parser.parse_args(argv)

    bundle = Path(args.out).resolve()
    print("Building the offline bundle in %s" % bundle)

    print("\n[1/3] Application")
    copy_application(bundle / "app")

    wheels = 0
    if not args.skip_wheels:
        print("\n[2/3] NumPy wheels for 32-bit Windows")
        wheels = fetch_wheels(bundle / "wheels")

    model_ok = True
    if not args.skip_model:
        total_mb = sum(model["size_mb"] for model in MODELS)
        print("\n[3/3] Neural networks (%d MB)" % total_mb)
        for model in MODELS:
            print("  %s - %s" % (model["name"], model["purpose"]))
            local = REPO / "models" / model["name"]
            target = bundle / "models" / model["name"]
            if local.is_file() and sha256(local) == model["sha256"]:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local, target)
                print("    copied from %s" % local)
                continue
            if not download(model["url"], target, model["sha256"]) and model["required"]:
                model_ok = False

    write_readme(bundle, wheels, model_ok)
    print("\nDone. Copy %s to the laptop and read LIESMICH.txt there." % bundle)
    if not model_ok and not args.skip_model:
        print("WARNING: the model is missing - object removal will fall back to the simple fill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
