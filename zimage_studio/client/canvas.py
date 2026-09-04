"""Image canvas with zoom, panning and a mask brush.

Everything is drawn with plain Tk primitives:

* the image and the mask overlay are :class:`tkinter.PhotoImage` objects,
  scaled with Tk's integer ``zoom``/``subsample`` (no Pillow required),
* the mask itself lives in a ``bytearray`` at source resolution and is stamped
  incrementally while the mouse moves, so painting stays responsive even on a
  slow 32-bit machine,
* the overlay is re-encoded as a two-colour indexed PNG (a few milliseconds for
  a 1 megapixel image) and handed back to Tk.
"""

from __future__ import annotations

import base64
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..imaging import (
    ImageError,
    make_stroke,
    mask_to_overlay_png,
    new_mask,
    png_encode,
    sniff_size,
    stamp_stroke,
)

#: Zoom factors as exact integer ratios, because that is what Tk can do losslessly.
ZOOM_STEPS: Sequence[Tuple[int, int]] = (
    (1, 8), (1, 6), (1, 4), (1, 3), (1, 2), (2, 3), (1, 1), (3, 2), (2, 1), (3, 1), (4, 1),
)
IDENTITY_ZOOM = ZOOM_STEPS.index((1, 1))

#: Guard against blowing up the address space of a 32-bit process: a scaled
#: PhotoImage costs four bytes per pixel.
MAX_SCALED_PIXELS = 4_000_000

TOOL_BRUSH = "brush"
TOOL_ERASER = "eraser"

#: Milliseconds between overlay refreshes while the mouse is down.
OVERLAY_THROTTLE_MS = 50


class ImageCanvas(ttk.Frame):
    """Displays the source image (or the latest result) and edits the mask."""

    def __init__(
        self,
        master,
        colors: Dict[str, str],
        on_status: Optional[Callable[[str], None]] = None,
        on_mask_change: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(master, style="TFrame")
        self.colors = colors
        self.on_status = on_status
        self.on_mask_change = on_mask_change

        # -- image state
        self.source_png: Optional[bytes] = None
        self.source_size: Tuple[int, int] = (0, 0)
        self.preview_png: Optional[bytes] = None
        self.preview_size: Tuple[int, int] = (0, 0)
        self.showing_preview = False

        # -- mask state
        self.mask: Optional[bytearray] = None
        self.strokes: List[dict] = []
        self._redo: List[dict] = []
        self._active_stroke: Optional[dict] = None
        self.show_mask = True

        # -- tools
        self.tool = TOOL_BRUSH
        self.brush_size = 48  # diameter in source pixels
        self.zoom_index = IDENTITY_ZOOM

        # -- Tk objects kept alive on purpose (PhotoImage is garbage collected)
        self._base_photo: Optional[tk.PhotoImage] = None
        self._scaled_photo: Optional[tk.PhotoImage] = None
        self._overlay_base: Optional[tk.PhotoImage] = None
        self._overlay_photo: Optional[tk.PhotoImage] = None
        self._image_item = None
        self._overlay_item = None
        self._cursor_item = None
        self._overlay_dirty = False
        self._overlay_job = None
        self._pan_origin: Optional[Tuple[int, int]] = None
        self._space_down = False

        self._build()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        self.canvas = tk.Canvas(
            self,
            background=self.colors["canvas"],
            highlightthickness=0,
            borderwidth=0,
            cursor="crosshair",
        )
        self.h_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._placeholder = self.canvas.create_text(
            10,
            10,
            anchor="nw",
            fill=self.colors["muted"],
            text="",
        )

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_move)
        self.canvas.bind("<Leave>", lambda _event: self._hide_cursor())
        self.canvas.bind("<ButtonPress-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._do_pan)
        self.canvas.bind("<ButtonRelease-2>", self._end_pan)
        self.canvas.bind("<ButtonPress-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._do_pan)
        self.canvas.bind("<ButtonRelease-3>", self._end_pan)
        self.canvas.bind("<Configure>", self._on_configure)
        # Wheel: vertical scroll, Shift horizontal, Control zooms.
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def has_image(self) -> bool:
        return self.source_png is not None

    @property
    def has_mask(self) -> bool:
        return bool(self.mask) and 255 in self.mask

    @property
    def can_undo(self) -> bool:
        return bool(self.strokes)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def set_source(self, png: bytes) -> Tuple[int, int]:
        """Load a PNG (or GIF) as the image being edited and reset the mask."""
        width, height = sniff_size(png)
        photo = tk.PhotoImage(master=self, data=base64.b64encode(png).decode("ascii"))
        self.source_png = png
        self.source_size = (width, height)
        self._base_photo = photo
        self.mask = new_mask(width, height)
        self.strokes = []
        self._redo = []
        self.preview_png = None
        self.showing_preview = False
        self.canvas.itemconfigure(self._placeholder, text="")
        self.zoom_index = IDENTITY_ZOOM
        self._rebuild_overlay()
        self.zoom_fit()
        self._notify_mask()
        return width, height

    def set_preview(self, png: bytes) -> None:
        """Show a generated result on top of the source (mask overlay hidden)."""
        self.preview_size = sniff_size(png)
        self.preview_png = png
        self.showing_preview = True
        self._render()

    def clear_preview(self) -> None:
        self.preview_png = None
        self.showing_preview = False
        self._render()

    def toggle_preview(self) -> bool:
        """Flip between the source and the last result; returns what is shown."""
        if self.preview_png is None:
            return False
        self.showing_preview = not self.showing_preview
        self._render()
        return self.showing_preview

    def displayed_png(self) -> Optional[bytes]:
        return self.preview_png if self.showing_preview else self.source_png

    def mask_png(self) -> Optional[bytes]:
        """The current mask as an 8-bit PNG, or ``None`` when nothing is painted."""
        if not self.has_mask or self.mask is None:
            return None
        width, height = self.source_size
        return png_encode(width, height, bytes(self.mask), "L")

    def set_tool(self, tool: str) -> None:
        self.tool = TOOL_ERASER if tool == TOOL_ERASER else TOOL_BRUSH
        self._draw_cursor_at_last()

    def set_brush_size(self, size: int) -> None:
        self.brush_size = max(2, min(512, int(size)))
        self._draw_cursor_at_last()

    def set_show_mask(self, visible: bool) -> None:
        self.show_mask = bool(visible)
        self._render()

    def clear_mask(self) -> None:
        if not self.has_image:
            return
        self._redo = list(self.strokes)
        self.strokes = []
        self.mask = new_mask(*self.source_size)
        self._rebuild_overlay()
        self._render()
        self._notify_mask()

    def undo(self) -> None:
        if not self.strokes:
            return
        self._redo.append(self.strokes.pop())
        self._replay()

    def redo(self) -> None:
        if not self._redo:
            return
        self.strokes.append(self._redo.pop())
        self._replay()

    # -- zoom ----------------------------------------------------------

    @property
    def scale(self) -> float:
        num, den = ZOOM_STEPS[self.zoom_index]
        return num / float(den)

    def zoom_in(self) -> None:
        self._set_zoom(self.zoom_index + 1)

    def zoom_out(self) -> None:
        self._set_zoom(self.zoom_index - 1)

    def zoom_actual(self) -> None:
        self._set_zoom(IDENTITY_ZOOM)

    def zoom_fit(self) -> None:
        """Pick the largest zoom step that shows the whole image."""
        if not self.has_image:
            return
        self.update_idletasks()
        view_width = max(1, self.canvas.winfo_width() - 4)
        view_height = max(1, self.canvas.winfo_height() - 4)
        if view_width < 20 or view_height < 20:  # not mapped yet
            self.after(60, self.zoom_fit)
            return
        width, height = self._displayed_size()
        best = 0
        for index, (num, den) in enumerate(ZOOM_STEPS):
            if width * num / den <= view_width and height * num / den <= view_height:
                best = index
        self._set_zoom(best)

    def _set_zoom(self, index: int) -> None:
        index = max(0, min(len(ZOOM_STEPS) - 1, index))
        num, den = ZOOM_STEPS[index]
        width, height = self._displayed_size()
        if width and height and (width * num // den) * (height * num // den) > MAX_SCALED_PIXELS:
            return  # would need more memory than a 32-bit process can spare
        self.zoom_index = index
        self._render()
        self._status("%d %%" % round(self.scale * 100))

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def _displayed_size(self) -> Tuple[int, int]:
        if self.showing_preview and self.preview_png is not None:
            return self.preview_size
        return self.source_size

    def _scaled(self, photo: tk.PhotoImage) -> tk.PhotoImage:
        num, den = ZOOM_STEPS[self.zoom_index]
        result = photo
        if num != 1:
            result = result.zoom(num, num)
        if den != 1:
            result = result.subsample(den, den)
        return result

    def _render(self) -> None:
        if not self.has_image:
            return
        photo = self._base_photo
        if self.showing_preview and self.preview_png is not None:
            photo = tk.PhotoImage(master=self, data=base64.b64encode(self.preview_png).decode("ascii"))
            self._preview_photo = photo  # keep a reference alive
        if photo is None:
            return

        self._scaled_photo = self._scaled(photo)
        if self._image_item is None:
            self._image_item = self.canvas.create_image(0, 0, anchor="nw", image=self._scaled_photo)
        else:
            self.canvas.itemconfigure(self._image_item, image=self._scaled_photo)

        self._render_overlay()
        width = self._scaled_photo.width()
        height = self._scaled_photo.height()
        self.canvas.configure(scrollregion=(0, 0, width, height))
        if self._cursor_item is not None:
            self.canvas.tag_raise(self._cursor_item)

    def _render_overlay(self) -> None:
        visible = self.show_mask and not self.showing_preview and self._overlay_base is not None
        if not visible:
            if self._overlay_item is not None:
                self.canvas.itemconfigure(self._overlay_item, state="hidden")
            return
        self._overlay_photo = self._scaled(self._overlay_base)
        if self._overlay_item is None:
            self._overlay_item = self.canvas.create_image(0, 0, anchor="nw", image=self._overlay_photo)
        else:
            self.canvas.itemconfigure(self._overlay_item, image=self._overlay_photo, state="normal")
        self.canvas.tag_raise(self._overlay_item, self._image_item)

    def _rebuild_overlay(self) -> None:
        """Re-encode the mask overlay from the mask buffer."""
        if self.mask is None:
            self._overlay_base = None
            return
        width, height = self.source_size
        try:
            png = mask_to_overlay_png(width, height, bytes(self.mask))
        except ImageError:
            self._overlay_base = None
            return
        self._overlay_base = tk.PhotoImage(master=self, data=base64.b64encode(png).decode("ascii"))

    def _replay(self) -> None:
        """Rebuild the mask from the stroke list (used by undo/redo)."""
        if not self.has_image:
            return
        width, height = self.source_size
        self.mask = new_mask(width, height)
        for stroke in self.strokes:
            stamp_stroke(self.mask, width, height, stroke)
        self._rebuild_overlay()
        self._render()
        self._notify_mask()

    def _schedule_overlay(self) -> None:
        if self._overlay_job is not None:
            return
        self._overlay_job = self.after(OVERLAY_THROTTLE_MS, self._flush_overlay)

    def _flush_overlay(self) -> None:
        self._overlay_job = None
        if self._overlay_dirty:
            self._overlay_dirty = False
            self._rebuild_overlay()
            self._render_overlay()

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------

    def _to_image(self, event) -> Tuple[float, float]:
        scale = self.scale
        return self.canvas.canvasx(event.x) / scale, self.canvas.canvasy(event.y) / scale

    def _paintable(self) -> bool:
        return self.has_image and not self.showing_preview and not self._space_down

    def _on_press(self, event) -> None:
        if self._space_down or not self.has_image:
            self._start_pan(event)
            return
        if self.showing_preview:
            self.toggle_preview()
            return
        x, y = self._to_image(event)
        self._active_stroke = make_stroke([(x, y)], self.brush_size / 2.0, self.tool == TOOL_ERASER)
        stamp_stroke(self.mask, *self.source_size, self._active_stroke)
        self._overlay_dirty = True
        self._schedule_overlay()

    def _on_drag(self, event) -> None:
        if self._pan_origin is not None:
            self._do_pan(event)
            return
        if self._active_stroke is None:
            return
        x, y = self._to_image(event)
        points = self._active_stroke["points"]
        last = points[-1]
        if abs(last[0] - x) < 0.5 and abs(last[1] - y) < 0.5:
            return
        points.append((x, y))
        # Only stamp the new segment - stamping the whole stroke again would make
        # long strokes quadratic.
        segment = make_stroke([last, (x, y)], self._active_stroke["radius"], self._active_stroke["erase"])
        stamp_stroke(self.mask, *self.source_size, segment)
        self._overlay_dirty = True
        self._schedule_overlay()
        self._draw_cursor(event)

    def _on_release(self, event) -> None:
        if self._pan_origin is not None:
            self._end_pan(event)
            return
        if self._active_stroke is None:
            return
        self.strokes.append(self._active_stroke)
        self._redo = []
        self._active_stroke = None
        self._overlay_dirty = True
        self._flush_overlay()
        self._notify_mask()

    def _on_move(self, event) -> None:
        self._draw_cursor(event)

    def _on_configure(self, _event) -> None:
        if self._cursor_item is not None:
            self.canvas.tag_raise(self._cursor_item)

    def _on_wheel(self, event) -> None:
        delta = getattr(event, "delta", 0)
        if event.num == 4:  # X11 wheel up
            delta = 120
        elif event.num == 5:
            delta = -120
        control = bool(event.state & 0x0004)
        shift = bool(event.state & 0x0001)
        if control:
            self.zoom_in() if delta > 0 else self.zoom_out()
        elif shift:
            self.canvas.xview_scroll(-1 if delta > 0 else 1, "units")
        else:
            self.canvas.yview_scroll(-1 if delta > 0 else 1, "units")

    # -- panning -------------------------------------------------------

    def _start_pan(self, event) -> None:
        self._pan_origin = (event.x, event.y)
        self.canvas.configure(cursor="fleur")
        self.canvas.scan_mark(event.x, event.y)

    def _do_pan(self, event) -> None:
        if self._pan_origin is None:
            return
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _end_pan(self, _event) -> None:
        self._pan_origin = None
        self.canvas.configure(cursor="crosshair")

    def set_space_down(self, down: bool) -> None:
        """The main window forwards the space bar so dragging pans the view."""
        self._space_down = bool(down)
        self.canvas.configure(cursor="fleur" if down else "crosshair")

    # -- brush cursor --------------------------------------------------

    def _draw_cursor(self, event) -> None:
        self._last_event = event
        if not self._paintable():
            self._hide_cursor()
            return
        radius = max(2.0, self.brush_size / 2.0 * self.scale)
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        colour = self.colors["danger"] if self.tool == TOOL_ERASER else self.colors["accent"]
        box = (x - radius, y - radius, x + radius, y + radius)
        if self._cursor_item is None:
            self._cursor_item = self.canvas.create_oval(*box, outline=colour, width=1)
        else:
            self.canvas.coords(self._cursor_item, *box)
            self.canvas.itemconfigure(self._cursor_item, outline=colour, state="normal")
        self.canvas.tag_raise(self._cursor_item)

    def _draw_cursor_at_last(self) -> None:
        event = getattr(self, "_last_event", None)
        if event is not None:
            self._draw_cursor(event)

    def _hide_cursor(self) -> None:
        if self._cursor_item is not None:
            self.canvas.itemconfigure(self._cursor_item, state="hidden")

    # -- misc ----------------------------------------------------------

    def destroy(self) -> None:  # noqa: D102 - cancel pending work first
        if self._overlay_job is not None:
            try:
                self.after_cancel(self._overlay_job)
            except tk.TclError:
                pass
            self._overlay_job = None
        super().destroy()

    def set_placeholder(self, text: str) -> None:
        self.canvas.itemconfigure(self._placeholder, text=text)

    def _status(self, message: str) -> None:
        if self.on_status is not None:
            self.on_status(message)

    def _notify_mask(self) -> None:
        if self.on_mask_change is not None:
            self.on_mask_change()
