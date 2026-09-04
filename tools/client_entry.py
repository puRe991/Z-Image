"""Entry point used by the PyInstaller build (see zimage_studio_client.spec)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zimage_studio.client.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
