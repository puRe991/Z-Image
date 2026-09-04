"""Colour palettes and ttk styling.

The GUI ships its own flat theme instead of the platform default: the stock
Windows ttk theme has no dark variant and looks dated on a modern desktop.
"""

from __future__ import annotations

from typing import Dict

PALETTES: Dict[str, Dict[str, str]] = {
    "dark": {
        "bg": "#1e1f24",
        "panel": "#26282f",
        "panel_alt": "#2f323a",
        "fg": "#e8eaed",
        "muted": "#9aa0a6",
        "accent": "#4c8dff",
        "accent_active": "#6ba0ff",
        "accent_fg": "#ffffff",
        "border": "#3a3d46",
        "canvas": "#141519",
        "danger": "#ff5f56",
        "ok": "#3ddc84",
        "entry": "#1b1c21",
    },
    "light": {
        "bg": "#f2f3f5",
        "panel": "#ffffff",
        "panel_alt": "#e9ebef",
        "fg": "#1c1e21",
        "muted": "#606770",
        "accent": "#1a73e8",
        "accent_active": "#3b8bfd",
        "accent_fg": "#ffffff",
        "border": "#cfd3d9",
        "canvas": "#d8dade",
        "danger": "#d93025",
        "ok": "#1e8e3e",
        "entry": "#ffffff",
    },
}

DEFAULT_FONT = ("Segoe UI", 9)
HEADING_FONT = ("Segoe UI", 10, "bold")
MONO_FONT = ("Consolas", 9)


def palette(name: str) -> Dict[str, str]:
    return PALETTES.get(name, PALETTES["dark"])


def apply_theme(root, name: str) -> Dict[str, str]:
    """Style every ttk widget class the application uses; returns the palette."""
    from tkinter import ttk

    colors = palette(name)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")  # the only stock theme that honours every option
    except Exception:  # noqa: BLE001 - fall back to whatever is available
        pass

    root.configure(background=colors["bg"])

    style.configure(".", background=colors["bg"], foreground=colors["fg"], font=DEFAULT_FONT)
    style.configure("TFrame", background=colors["bg"])
    style.configure("Panel.TFrame", background=colors["panel"])
    style.configure("Toolbar.TFrame", background=colors["panel"])
    style.configure("TLabel", background=colors["bg"], foreground=colors["fg"])
    style.configure("Panel.TLabel", background=colors["panel"], foreground=colors["fg"])
    style.configure("Heading.TLabel", background=colors["panel"], foreground=colors["fg"], font=HEADING_FONT)
    style.configure("Muted.TLabel", background=colors["panel"], foreground=colors["muted"])
    style.configure("Status.TLabel", background=colors["panel"], foreground=colors["muted"])
    style.configure("Value.TLabel", background=colors["panel"], foreground=colors["accent"], font=MONO_FONT)

    style.configure(
        "TButton",
        background=colors["panel_alt"],
        foreground=colors["fg"],
        bordercolor=colors["border"],
        focuscolor=colors["accent"],
        padding=(10, 5),
        relief="flat",
    )
    style.map(
        "TButton",
        background=[("active", colors["border"]), ("disabled", colors["panel"])],
        foreground=[("disabled", colors["muted"])],
    )
    style.configure(
        "Accent.TButton",
        background=colors["accent"],
        foreground=colors["accent_fg"],
        padding=(12, 8),
        font=HEADING_FONT,
    )
    style.map(
        "Accent.TButton",
        background=[("active", colors["accent_active"]), ("disabled", colors["panel_alt"])],
        foreground=[("disabled", colors["muted"])],
    )
    style.configure("Tool.TButton", padding=(8, 4))
    style.configure(
        "Toggle.TButton", background=colors["panel_alt"], foreground=colors["fg"], padding=(8, 4)
    )
    style.map("Toggle.TButton", background=[("pressed", colors["accent"]), ("active", colors["border"])])

    style.configure(
        "TCheckbutton", background=colors["panel"], foreground=colors["fg"], focuscolor=colors["accent"]
    )
    style.map("TCheckbutton", background=[("active", colors["panel"])])
    style.configure("TRadiobutton", background=colors["panel"], foreground=colors["fg"])
    style.map("TRadiobutton", background=[("active", colors["panel"])])

    style.configure(
        "TEntry",
        fieldbackground=colors["entry"],
        foreground=colors["fg"],
        bordercolor=colors["border"],
        insertcolor=colors["fg"],
        padding=4,
    )
    style.configure(
        "TSpinbox",
        fieldbackground=colors["entry"],
        foreground=colors["fg"],
        background=colors["panel_alt"],
        bordercolor=colors["border"],
        arrowcolor=colors["fg"],
        padding=3,
    )
    style.configure(
        "TCombobox",
        fieldbackground=colors["entry"],
        background=colors["panel_alt"],
        foreground=colors["fg"],
        arrowcolor=colors["fg"],
        bordercolor=colors["border"],
        padding=3,
    )

    style.configure(
        "TScale", background=colors["panel"], troughcolor=colors["panel_alt"], bordercolor=colors["border"]
    )
    style.configure(
        "Horizontal.TProgressbar",
        background=colors["accent"],
        troughcolor=colors["panel_alt"],
        bordercolor=colors["border"],
        lightcolor=colors["accent"],
        darkcolor=colors["accent"],
    )
    style.configure("TLabelframe", background=colors["panel"], bordercolor=colors["border"])
    style.configure("TLabelframe.Label", background=colors["panel"], foreground=colors["muted"])
    style.configure("TPanedwindow", background=colors["bg"])
    style.configure("TSeparator", background=colors["border"])
    style.configure(
        "Vertical.TScrollbar",
        background=colors["panel_alt"],
        troughcolor=colors["bg"],
        bordercolor=colors["bg"],
        arrowcolor=colors["muted"],
    )
    style.configure(
        "Horizontal.TScrollbar",
        background=colors["panel_alt"],
        troughcolor=colors["bg"],
        bordercolor=colors["bg"],
        arrowcolor=colors["muted"],
    )
    return colors
