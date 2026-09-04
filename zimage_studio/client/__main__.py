"""Entry point of the desktop client: ``python -m zimage_studio.client``."""

from __future__ import annotations

import argparse
import sys

from ..version import APP_NAME, __version__


def _enable_dpi_awareness() -> None:
    """Ask Windows not to bitmap-stretch the window on high-DPI screens."""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # Windows 8.1+
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()  # Vista .. 8
    except Exception:  # noqa: BLE001 - never let cosmetics stop the app
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m zimage_studio.client",
        description="%s - image-to-image editing front-end for Z-Image." % APP_NAME,
    )
    parser.add_argument("image", nargs="?", help="image to open on start-up")
    parser.add_argument("--server", default="", help="backend URL, e.g. http://192.168.1.20:8787")
    parser.add_argument("--token", default="", help="access token, if the server requires one")
    parser.add_argument("--lang", default="", choices=["", "de", "en"], help="interface language")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="start the GPU-free demo backend in this process (no weights needed)",
    )
    parser.add_argument("--version", action="version", version="%s %s" % (APP_NAME, __version__))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _enable_dpi_awareness()

    try:
        import tkinter  # noqa: F401
    except ImportError:
        print(
            "Tkinter is missing from this Python installation.\n"
            "On Windows re-run the python.org installer and enable 'tcl/tk and IDLE'.",
            file=sys.stderr,
        )
        return 2

    from .app import StudioApp
    from .settings import Settings

    settings = Settings()
    if args.server:
        settings.set("server_url", args.server)
    if args.token:
        settings.set("token", args.token)
    if args.lang:
        settings.set("language", args.lang)

    app = StudioApp(settings=settings, demo=args.demo)
    if args.image:
        app.root.after(200, lambda: app.open_image(args.image))
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
