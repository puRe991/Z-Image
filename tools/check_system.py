"""Check whether this machine can run Z-Image Studio, and how fast.

Run it on the laptop:

    py -3-32 tools\\check_system.py

It reports the Python build, the interface toolkit, NumPy, memory and the model
file, estimates how long one neural edit will take, and says what to do about
anything that is missing.  Nothing is installed or changed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Reference measurement of the development machine: at 395 GFLOP/s on a
#: 512x512 float32 matrix product, one 512x512 edit took 11 seconds.  A slower
#: machine is assumed to scale with that number - a rough but honest guide.
REFERENCE_GFLOPS = 395.0
REFERENCE_SECONDS = 11.0

OK = "[ ok ]"
WARN = "[warn]"
BAD = "[fail]"


def line(status: str, text: str, hint: str = "") -> bool:
    print("%s %s" % (status, text))
    if hint:
        print("       -> %s" % hint)
    return status != BAD


def check_python() -> bool:
    bits = struct.calcsize("P") * 8
    version = "%d.%d.%d" % sys.version_info[:3]
    print("Python %s, %d-bit, on %s" % (version, bits, platform.platform()))
    print("  %s" % sys.executable)
    good = (3, 8) <= sys.version_info[:2] <= (3, 13)
    if bits == 32 and sys.version_info[:2] > (3, 11):
        return line(
            WARN,
            "32-bit Python %s: NumPy has no wheel for this version" % version,
            "use a 32-bit Python 3.8 - 3.11 for the neural mode",
        )
    if not good:
        return line(WARN, "Python %s is outside the tested range (3.8 - 3.13)" % version)
    return line(OK, "Python version and build are supported")


def check_tkinter() -> bool:
    try:
        import tkinter
    except ImportError:
        return line(
            BAD,
            "Tkinter is missing - the interface cannot start",
            "re-run the Python installer, choose Modify, tick 'tcl/tk and IDLE'",
        )
    return line(OK, "Tkinter %s (interface)" % tkinter.TkVersion)


def check_numpy() -> bool:
    try:
        import numpy
    except ImportError:
        return line(
            WARN,
            "NumPy is missing - object removal falls back to the simple fill",
            "install it offline:  py -3-32 -m pip install --no-index --find-links wheels numpy",
        )
    return line(OK, "NumPy %s (neural network)" % numpy.__version__)


def total_memory_mb() -> float:
    """Physical memory, without any third-party module."""
    if sys.platform.startswith("win"):
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = Status()
        status.dwLength = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullTotalPhys / (1024.0 * 1024.0)
    try:
        with open("/proc/meminfo") as handle:
            for row in handle:
                if row.startswith("MemTotal:"):
                    return int(row.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def check_memory() -> bool:
    total = total_memory_mb()
    if not total:
        return line(WARN, "could not determine the amount of memory")
    if total < 1400:
        return line(
            WARN,
            "%.0f MB of memory - the network needs about 500 MB" % total,
            "close other programs before editing",
        )
    return line(OK, "%.0f MB of memory (the network needs about 500 MB)" % total)


def check_model() -> bool:
    from zimage_studio.localedit.neural import MODEL_FILENAME, find_model, search_paths

    path = find_model()
    if path is None:
        return line(
            WARN,
            "%s not found - object removal falls back to the simple fill" % MODEL_FILENAME,
            "copy the file into: %s" % search_paths()[0].parent,
        )
    size = path.stat().st_size / (1024.0 * 1024.0)
    return line(OK, "model found: %s (%.0f MB)" % (path, size))


def benchmark() -> float:
    """Single-threaded matrix product performance, in GFLOP/s."""
    import numpy as np

    size = 384
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)
    a @ b  # warm up
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - start)
    return 2.0 * size**3 / best / 1e9


def estimate() -> bool:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return True
    speed = benchmark()
    seconds = REFERENCE_SECONDS * REFERENCE_GFLOPS / max(speed, 0.01)
    if seconds < 90:
        text = "about %.0f seconds" % seconds
        status = OK
    elif seconds < 600:
        text = "about %.0f minutes" % (seconds / 60.0)
        status = OK
    else:
        text = "roughly %.0f minutes" % (seconds / 60.0)
        status = WARN
    return line(
        status,
        "%.1f GFLOP/s -> one neural edit takes %s" % (speed, text),
        "rough estimate; adjustments like brightness stay instant",
    )


def full_run() -> bool:
    """Actually run the network once and report the real time."""
    from zimage_studio.localedit.neural import NeuralInpainter

    inpainter = NeuralInpainter()
    if not inpainter.available:
        return line(WARN, "cannot run the network: %s" % inpainter.describe())
    print("       running one real edit, please wait …")
    width = height = 256
    planes = [bytes([120]) * (width * height) for _ in range(3)]
    mask = bytearray(width * height)
    for y in range(110, 150):
        mask[y * width + 110 : y * width + 150] = b"\xff" * 40
    start = time.time()
    inpainter.inpaint(planes, width, height, bytes(mask))
    return line(OK, "one real edit took %.1f seconds" % (time.time() - start))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check this machine for Z-Image Studio.")
    parser.add_argument("--full", action="store_true", help="run the network once (slow, but exact)")
    args = parser.parse_args(argv)

    print("=" * 66)
    print("Z-Image Studio - system check")
    print("=" * 66)
    results = [check_python(), check_tkinter(), check_numpy(), check_memory(), check_model()]
    print("-" * 66)
    results.append(full_run() if args.full else estimate())
    print("=" * 66)

    if all(results):
        print("Everything needed is in place.")
        print("Start with:  py -3-32 -m zimage_studio.client")
    else:
        print("Something essential is missing - see the lines marked [fail] above.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
