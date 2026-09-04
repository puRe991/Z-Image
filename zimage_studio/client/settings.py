"""Persisted user settings.

Stored as JSON next to the other per-user application data
(``%APPDATA%\\ZImageStudio`` on Windows, ``~/.config/zimage-studio`` elsewhere).
A broken or unreadable file is never fatal - the defaults are used instead.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from ..protocol import DEFAULTS

APP_DIR_NAME = "ZImageStudio"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "server_url": "http://127.0.0.1:8787",
    "token": "",
    "language": "de",
    "theme": "dark",
    "last_open_dir": "",
    "last_save_dir": "",
    "brush_size": 48,
    "mask_blur": DEFAULTS["mask_blur"],
    "prompt": "",
    "negative_prompt": "",
    "steps": DEFAULTS["steps"],
    "guidance": DEFAULTS["guidance"],
    "strength": DEFAULTS["strength"],
    "num_images": DEFAULTS["num_images"],
    "megapixels": 1.0,
    "keep_unmasked": True,
    "mask_invert": False,
    "window_geometry": "",
    "auto_connect": True,
}


def config_dir() -> Path:
    """Directory holding the settings file, created on demand."""
    appdata = os.environ.get("APPDATA")
    if appdata:  # Windows
        base = Path(appdata) / APP_DIR_NAME
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "zimage-studio"
    return base


def settings_path() -> Path:
    return config_dir() / "settings.json"


class Settings:
    """Dictionary-like settings holder with defaults and disk persistence."""

    def __init__(self, path: Path = None) -> None:
        self.path = Path(path) if path else settings_path()
        self.data: Dict[str, Any] = dict(DEFAULT_SETTINGS)
        self.load()

    def load(self) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                stored = json.load(handle)
        except (OSError, ValueError):
            return
        if isinstance(stored, dict):
            for key, value in stored.items():
                if key in DEFAULT_SETTINGS and value is not None:
                    self.data[key] = value

    def save(self) -> bool:
        """Write the settings out; returns ``False`` when the disk said no."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self.data, handle, indent=2, sort_keys=True)
            temporary.replace(self.path)
            return True
        except OSError:
            return False

    def get(self, key: str, fallback: Any = None) -> Any:
        return self.data.get(key, DEFAULT_SETTINGS.get(key, fallback))

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def update(self, values: Dict[str, Any]) -> None:
        self.data.update(values)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)
