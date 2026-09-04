"""Small reusable Tk widgets: labelled sliders, hint-aware text boxes, history strip."""

from __future__ import annotations

import base64
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional

from ..imaging import sniff_size


class LabeledScale(ttk.Frame):
    """A slider with a caption and a live value readout."""

    def __init__(
        self,
        master,
        colors: Dict[str, str],
        text: str,
        minimum: float,
        maximum: float,
        value: float,
        *,
        step: float = 0.0,
        fmt: str = "%.2f",
        tooltip: str = "",
        command: Optional[Callable[[float], None]] = None,
    ) -> None:
        super().__init__(master, style="Panel.TFrame")
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.step = float(step)
        self.fmt = fmt
        self.command = command
        self._var = tk.DoubleVar(value=float(value))

        header = ttk.Frame(self, style="Panel.TFrame")
        header.pack(fill="x")
        self.caption = ttk.Label(header, text=text, style="Panel.TLabel")
        self.caption.pack(side="left")
        self.readout = ttk.Label(header, text=self.fmt % float(value), style="Value.TLabel")
        self.readout.pack(side="right")

        self.scale = ttk.Scale(
            self, from_=self.minimum, to=self.maximum, variable=self._var, command=self._on_change
        )
        self.scale.pack(fill="x", pady=(2, 0))

        if tooltip:
            self.hint = ttk.Label(self, text=tooltip, style="Muted.TLabel", wraplength=240)
            self.hint.pack(fill="x")
        else:
            self.hint = None

    def _snap(self, value: float) -> float:
        if self.step > 0:
            value = round(value / self.step) * self.step
        return max(self.minimum, min(self.maximum, value))

    def _on_change(self, _value) -> None:
        value = self._snap(self._var.get())
        self.readout.configure(text=self.fmt % value)
        if self.command is not None:
            self.command(value)

    def get(self) -> float:
        return self._snap(self._var.get())

    def set(self, value: float) -> None:
        self._var.set(self._snap(float(value)))
        self.readout.configure(text=self.fmt % self.get())

    def set_text(self, text: str, tooltip: str = "") -> None:
        self.caption.configure(text=text)
        if self.hint is not None and tooltip:
            self.hint.configure(text=tooltip)


class PromptBox(ttk.Frame):
    """Multi-line text input with a placeholder and a character counter."""

    def __init__(
        self,
        master,
        colors: Dict[str, str],
        title: str,
        placeholder: str = "",
        height: int = 5,
    ) -> None:
        super().__init__(master, style="Panel.TFrame")
        self.colors = colors
        self.placeholder = placeholder
        self._showing_placeholder = False

        header = ttk.Frame(self, style="Panel.TFrame")
        header.pack(fill="x")
        self.title_label = ttk.Label(header, text=title, style="Heading.TLabel")
        self.title_label.pack(side="left")
        self.counter = ttk.Label(header, text="0", style="Muted.TLabel")
        self.counter.pack(side="right")

        self.text = tk.Text(
            self,
            height=height,
            wrap="word",
            background=colors["entry"],
            foreground=colors["fg"],
            insertbackground=colors["fg"],
            selectbackground=colors["accent"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=colors["border"],
            highlightcolor=colors["accent"],
            padx=6,
            pady=5,
            undo=True,
        )
        self.text.pack(fill="both", expand=True, pady=(3, 0))
        self.text.bind("<KeyRelease>", self._update_counter)
        self.text.bind("<FocusIn>", self._clear_placeholder)
        self.text.bind("<FocusOut>", self._restore_placeholder)
        self._restore_placeholder()

    def _clear_placeholder(self, _event=None) -> None:
        if self._showing_placeholder:
            self.text.delete("1.0", "end")
            self.text.configure(foreground=self.colors["fg"])
            self._showing_placeholder = False

    def _restore_placeholder(self, _event=None) -> None:
        if not self.text.get("1.0", "end").strip() and self.placeholder:
            self.text.delete("1.0", "end")
            self.text.insert("1.0", self.placeholder)
            self.text.configure(foreground=self.colors["muted"])
            self._showing_placeholder = True
        self._update_counter()

    def _update_counter(self, _event=None) -> None:
        self.counter.configure(text="0" if self._showing_placeholder else str(len(self.get())))

    def get(self) -> str:
        if self._showing_placeholder:
            return ""
        return self.text.get("1.0", "end").strip()

    def set(self, value: str) -> None:
        self._showing_placeholder = False
        self.text.configure(foreground=self.colors["fg"])
        self.text.delete("1.0", "end")
        self.text.insert("1.0", value or "")
        self._restore_placeholder()

    def set_title(self, title: str, placeholder: str = "") -> None:
        self.title_label.configure(text=title)
        if placeholder:
            was_placeholder = self._showing_placeholder
            self.placeholder = placeholder
            if was_placeholder:
                self._showing_placeholder = False
                self.text.delete("1.0", "end")
                self._restore_placeholder()


class HistoryStrip(ttk.Frame):
    """Horizontal thumbnail strip of past results."""

    THUMB = 88

    def __init__(
        self,
        master,
        colors: Dict[str, str],
        title: str,
        empty_text: str,
        on_select: Optional[Callable[[int], None]] = None,
        on_activate: Optional[Callable[[int], None]] = None,
    ) -> None:
        super().__init__(master, style="Panel.TFrame")
        self.colors = colors
        self.on_select = on_select
        self.on_activate = on_activate
        self._items: List[Dict[str, object]] = []
        self._photos: List[tk.PhotoImage] = []
        self.selected = -1

        self.title_label = ttk.Label(self, text=title, style="Heading.TLabel")
        self.title_label.pack(anchor="w", padx=8, pady=(6, 2))

        self.canvas = tk.Canvas(
            self,
            height=self.THUMB + 16,
            background=colors["panel_alt"],
            highlightthickness=0,
            borderwidth=0,
        )
        self.scroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.scroll.set)
        self.canvas.pack(fill="x", padx=8)
        self.scroll.pack(fill="x", padx=8, pady=(0, 6))

        self.empty_text = empty_text
        self._empty_item = self.canvas.create_text(
            12, (self.THUMB + 16) // 2, anchor="w", fill=colors["muted"], text=empty_text
        )
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)

    def set_texts(self, title: str, empty_text: str) -> None:
        self.title_label.configure(text=title)
        self.empty_text = empty_text
        self.canvas.itemconfigure(self._empty_item, text="" if self._items else empty_text)

    def add(self, png: bytes, meta: Dict[str, object]) -> int:
        width, height = sniff_size(png)
        photo = tk.PhotoImage(master=self, data=base64.b64encode(png).decode("ascii"))
        factor = max(1, int(max(width, height) / float(self.THUMB)) or 1)
        thumb = photo.subsample(factor, factor)
        self._photos.append(thumb)
        entry = dict(meta)
        entry["png"] = png
        self._items.append(entry)
        self._redraw()
        index = len(self._items) - 1
        self.select(index)
        return index

    def item(self, index: int) -> Optional[Dict[str, object]]:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def __len__(self) -> int:
        return len(self._items)

    def select(self, index: int) -> None:
        self.selected = index
        self._redraw()
        self.canvas.xview_moveto(max(0.0, index / float(max(1, len(self._items)))))

    def _redraw(self) -> None:
        self.canvas.delete("thumb")
        self.canvas.itemconfigure(self._empty_item, text="" if self._items else self.empty_text)
        for index, photo in enumerate(self._photos):
            x = 8 + index * (self.THUMB + 10)
            border = self.colors["accent"] if index == self.selected else self.colors["border"]
            self.canvas.create_rectangle(
                x - 3,
                5,
                x + self.THUMB + 3,
                self.THUMB + 11,
                outline=border,
                width=2,
                tags=("thumb", "item-%d" % index),
            )
            self.canvas.create_image(x, 8, anchor="nw", image=photo, tags=("thumb", "item-%d" % index))
        width = 16 + len(self._photos) * (self.THUMB + 10)
        self.canvas.configure(scrollregion=(0, 0, max(width, 1), self.THUMB + 16))

    def _index_at(self, event) -> int:
        x = self.canvas.canvasx(event.x)
        index = int((x - 5) // (self.THUMB + 10))
        return index if 0 <= index < len(self._items) else -1

    def _on_click(self, event) -> None:
        index = self._index_at(event)
        if index >= 0:
            self.select(index)
            if self.on_select is not None:
                self.on_select(index)

    def _on_double_click(self, event) -> None:
        index = self._index_at(event)
        if index >= 0 and self.on_activate is not None:
            self.on_activate(index)
