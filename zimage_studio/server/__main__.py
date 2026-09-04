"""Command line entry point: ``python -m zimage_studio.server``."""

from __future__ import annotations

import argparse
import logging

from ..version import APP_NAME, __version__
from .app import local_ip, serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m zimage_studio.server",
        description="%s backend - serves Z-Image txt2img / img2img / inpaint over HTTP." % APP_NAME,
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to expose on the LAN)")
    parser.add_argument("--port", type=int, default=8787, help="TCP port (default: 8787)")
    parser.add_argument(
        "--engine",
        default="zimage",
        choices=["zimage", "mock"],
        help="'zimage' runs the real model, 'mock' is a GPU-free demo engine",
    )
    parser.add_argument("--model-path", default="ckpts/Z-Image-Turbo", help="local weights directory")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--dtype", default="bfloat16", help="bfloat16 | float16 | float32")
    parser.add_argument("--attention", default="", help="override the Z-Image attention backend")
    parser.add_argument("--offload", action="store_true", help="move modules to CPU between stages")
    parser.add_argument("--no-download", action="store_true", help="fail instead of fetching missing weights")
    parser.add_argument("--preload", action="store_true", help="load the weights at start-up, not on first job")
    parser.add_argument("--token", default="", help="require this value in the X-Auth-Token header")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    parser.add_argument("--version", action="version", version="%s %s" % (APP_NAME, __version__))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.engine == "mock":
        engine_kwargs = {}
    else:
        engine_kwargs = {
            "model_path": args.model_path,
            "device": args.device,
            "dtype": args.dtype,
            "attention_backend": args.attention or None,
            "offload": args.offload,
            "allow_download": not args.no_download,
            "preload": args.preload,
        }

    print("%s %s - engine=%s" % (APP_NAME, __version__, args.engine))
    print("  local:   http://127.0.0.1:%d" % args.port)
    if args.host not in ("127.0.0.1", "localhost"):
        print("  network: http://%s:%d" % (local_ip(), args.port))
    print("  paste that URL into the client's server settings\n")

    serve(
        host=args.host,
        port=args.port,
        engine_name=args.engine,
        token=args.token,
        engine_kwargs=engine_kwargs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
