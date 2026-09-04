"""Main window of Z-Image Studio.

The whole GUI is Tkinter/ttk and the standard library, so it runs on a 32-bit
Windows Python where PyTorch (and often Pillow) cannot be installed.  Anything
that needs real image decoding or the model itself is delegated to the server.
"""

from __future__ import annotations

import os
import queue
import random
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

from ..imaging import ImageError, sniff_format, sniff_size
from ..protocol import (
    MAX_UPLOAD_BYTES,
    MODE_IMG2IMG,
    MODE_INPAINT,
    MODE_TXT2IMG,
    STATUS_QUEUED,
    STATUS_RUNNING,
    GenerateRequest,
    JobStatus,
    ProtocolError,
    ServerInfo,
    align_size,
    encode_image,
    decode_image,
    fit_size,
    parse_generate_request,
)
from ..version import APP_NAME, __version__
from .api import ClientError, JobRunner, StudioClient
from .canvas import TOOL_BRUSH, TOOL_ERASER, ImageCanvas
from .i18n import Translator
from .settings import Settings
from .theme import apply_theme
from .widgets import HistoryStrip, LabeledScale, PromptBox

#: Aspect presets offered for text-to-image, where there is no source image.
ASPECTS = (("1:1", 1.0), ("3:2", 1.5), ("2:3", 1 / 1.5), ("16:9", 16 / 9.0), ("9:16", 9 / 16.0))
MEGAPIXELS = ("0.5", "1.0", "1.5", "2.0")
#: Anything larger is downscaled on load - both to stay inside the upload limit
#: and to keep a 32-bit process from running out of address space.
MAX_SOURCE_EDGE = 2048


class StudioApp:
    """Wires the widgets, the settings and the API client together."""

    def __init__(self, root: Optional[tk.Tk] = None, settings: Optional[Settings] = None, demo: bool = False):
        self.owns_root = root is None
        self.root = root or tk.Tk()
        self.settings = settings or Settings()
        self.tr = Translator(self.settings.get("language", "de"))
        self.colors = apply_theme(self.root, self.settings.get("theme", "dark"))

        self.client = StudioClient(self.settings.get("server_url"), self.settings.get("token"))
        self.runner = JobRunner(self.client)
        self.server_info: Optional[ServerInfo] = None
        self.last_seed = -1
        self.demo_server = None
        self._job_started = 0.0
        # Tk may only be touched from the thread that created it, so worker
        # threads hand their callbacks over through this queue instead of
        # calling root.after() themselves.
        self._events: "queue.Queue" = queue.Queue()
        self._event_job = None

        self.root.title("%s %s" % (APP_NAME, __version__))
        self.root.minsize(1000, 640)
        geometry = self.settings.get("window_geometry")
        if geometry:
            try:
                self.root.geometry(geometry)
            except tk.TclError:
                pass
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self._build_menu()
        self._build_layout()
        self._bind_keys()
        self._retranslate()
        self._update_mode_state()

        self._pump_events()

        if demo:
            self.root.after(120, self.start_demo_backend)
        elif self.settings.get("auto_connect", True):
            self.root.after(120, self.connect)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        colors = self.colors
        menu_options = dict(
            background=colors["panel"],
            foreground=colors["fg"],
            activebackground=colors["accent"],
            activeforeground=colors["accent_fg"],
            borderwidth=0,
        )
        self.menubar = tk.Menu(self.root, **menu_options)
        self.menu_file = tk.Menu(self.menubar, tearoff=0, **menu_options)
        self.menu_edit = tk.Menu(self.menubar, tearoff=0, **menu_options)
        self.menu_view = tk.Menu(self.menubar, tearoff=0, **menu_options)
        self.menu_server = tk.Menu(self.menubar, tearoff=0, **menu_options)
        self.menu_help = tk.Menu(self.menubar, tearoff=0, **menu_options)

        self.menu_file.add_command(command=self.open_image, accelerator="Ctrl+O")
        self.menu_file.add_command(command=self.save_result, accelerator="Ctrl+S")
        self.menu_file.add_command(command=self.save_mask)
        self.menu_file.add_separator()
        self.menu_file.add_command(command=self.quit)

        self.menu_edit.add_command(command=self.undo, accelerator="Ctrl+Z")
        self.menu_edit.add_command(command=self.redo, accelerator="Ctrl+Y")
        self.menu_edit.add_command(command=self.clear_mask)
        self.menu_edit.add_separator()
        self.menu_edit.add_command(command=self.use_result_as_source)

        self.show_mask_var = tk.BooleanVar(value=True)
        self.menu_view.add_command(command=self.zoom_in, accelerator="+")
        self.menu_view.add_command(command=self.zoom_out, accelerator="-")
        self.menu_view.add_command(command=self.zoom_fit, accelerator="0")
        self.menu_view.add_command(command=self.zoom_actual)
        self.menu_view.add_separator()
        self.menu_view.add_checkbutton(variable=self.show_mask_var, command=self.toggle_mask, accelerator="M")
        self.menu_view.add_command(command=self.toggle_compare)
        self.menu_view.add_separator()
        self.menu_view.add_command(command=self.toggle_theme)
        self.menu_view.add_command(command=self.toggle_language)

        self.menu_server.add_command(command=self.open_server_dialog)
        self.menu_server.add_command(command=self.connect)
        self.menu_server.add_separator()
        self.menu_server.add_command(command=self.start_demo_backend)

        self.menu_help.add_command(command=self.show_shortcuts)
        self.menu_help.add_command(command=self.show_about)

        for menu in (self.menu_file, self.menu_edit, self.menu_view, self.menu_server, self.menu_help):
            self.menubar.add_cascade(menu=menu)
        self.root.configure(menu=self.menubar)

    def _build_layout(self) -> None:
        colors = self.colors
        self.toolbar = ttk.Frame(self.root, style="Toolbar.TFrame", padding=(8, 6))
        self.toolbar.pack(side="top", fill="x")

        self.btn_open = ttk.Button(self.toolbar, style="Tool.TButton", command=self.open_image)
        self.btn_open.pack(side="left")
        ttk.Separator(self.toolbar, orient="vertical").pack(side="left", fill="y", padx=8)

        self.tool_var = tk.StringVar(value=TOOL_BRUSH)
        self.btn_brush = ttk.Radiobutton(
            self.toolbar,
            variable=self.tool_var,
            value=TOOL_BRUSH,
            style="Toolbutton",
            command=self._on_tool_change,
        )
        self.btn_eraser = ttk.Radiobutton(
            self.toolbar,
            variable=self.tool_var,
            value=TOOL_ERASER,
            style="Toolbutton",
            command=self._on_tool_change,
        )
        self.btn_brush.pack(side="left", padx=2)
        self.btn_eraser.pack(side="left", padx=2)

        self.brush_label = ttk.Label(self.toolbar, style="Panel.TLabel")
        self.brush_label.pack(side="left", padx=(12, 4))
        self.brush_var = tk.IntVar(value=int(self.settings.get("brush_size", 48)))
        self.brush_scale = ttk.Scale(
            self.toolbar,
            from_=4,
            to=256,
            variable=self.brush_var,
            length=140,
            command=lambda _v: self._on_brush_size(),
        )
        self.brush_scale.pack(side="left")
        self.brush_value = ttk.Label(self.toolbar, style="Value.TLabel", width=4)
        self.brush_value.pack(side="left", padx=(6, 0))

        ttk.Separator(self.toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        self.btn_mask = ttk.Checkbutton(
            self.toolbar, variable=self.show_mask_var, style="Toolbutton", command=self.toggle_mask
        )
        self.btn_mask.pack(side="left", padx=2)
        self.btn_compare = ttk.Button(self.toolbar, style="Tool.TButton", command=self.toggle_compare)
        self.btn_compare.pack(side="left", padx=2)
        self.btn_fit = ttk.Button(self.toolbar, style="Tool.TButton", command=self.zoom_fit)
        self.btn_fit.pack(side="left", padx=2)

        self.server_dot = tk.Canvas(
            self.toolbar, width=10, height=10, highlightthickness=0, background=colors["panel"]
        )
        self.server_dot.pack(side="right", padx=(6, 2))
        self._dot = self.server_dot.create_oval(1, 1, 9, 9, fill=colors["danger"], outline="")
        self.server_label = ttk.Label(self.toolbar, style="Status.TLabel")
        self.server_label.pack(side="right")

        # -- status bar and history first: pack() hands out space in call order,
        #    and the paned window below expands into whatever is left.
        status_bar = ttk.Frame(self.root, style="Toolbar.TFrame", padding=(8, 4))
        status_bar.pack(side="bottom", fill="x")
        self.status_label = ttk.Label(status_bar, style="Status.TLabel")
        self.status_label.pack(side="left")
        self.progress = ttk.Progressbar(status_bar, mode="determinate", maximum=1000, length=220)
        self.progress.pack(side="right")

        self.history = HistoryStrip(
            self.root,
            colors,
            "",
            "",
            on_select=self._on_history_select,
            on_activate=lambda index: self.use_result_as_source(index),
        )
        self.history.pack(side="bottom", fill="x")

        # -- main split
        self.paned = ttk.PanedWindow(self.root, orient="horizontal")
        self.paned.pack(side="top", fill="both", expand=True)

        left = ttk.Frame(self.paned, style="TFrame")
        self.canvas = ImageCanvas(
            left, colors, on_status=self.set_status, on_mask_change=self._on_mask_change
        )
        self.canvas.pack(fill="both", expand=True)
        self.paned.add(left, weight=3)

        right = ttk.Frame(self.paned, style="Panel.TFrame", width=320)
        right.pack_propagate(False)
        self.paned.add(right, weight=0)
        self._build_panel(right)


    def _build_panel(self, parent) -> None:
        colors = self.colors
        # The action buttons must never scroll out of reach, so they are packed
        # against the bottom of the panel before the scrollable settings area.
        actions = ttk.Frame(parent, style="Panel.TFrame", padding=(10, 8))
        actions.pack(side="bottom", fill="x")
        self.btn_generate = ttk.Button(actions, style="Accent.TButton", command=self.generate)
        self.btn_generate.pack(fill="x")
        self.btn_cancel = ttk.Button(actions, style="Tool.TButton", command=self.cancel, state="disabled")
        self.btn_cancel.pack(fill="x", pady=(6, 0))
        ttk.Separator(parent, orient="horizontal").pack(side="bottom", fill="x")

        outer = tk.Canvas(parent, background=colors["panel"], highlightthickness=0)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=outer.yview)
        panel = ttk.Frame(outer, style="Panel.TFrame", padding=10)
        panel.bind("<Configure>", lambda _e: outer.configure(scrollregion=outer.bbox("all")))
        window = outer.create_window((0, 0), window=panel, anchor="nw")
        outer.bind("<Configure>", lambda event: outer.itemconfigure(window, width=event.width))
        outer.configure(yscrollcommand=scroll.set)
        outer.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # mode
        self.mode_frame = ttk.LabelFrame(panel, style="TLabelframe", padding=8)
        self.mode_frame.pack(fill="x")
        self.mode_var = tk.StringVar(value=MODE_TXT2IMG)
        self.mode_buttons = {}
        for mode in (MODE_TXT2IMG, MODE_IMG2IMG, MODE_INPAINT):
            button = ttk.Radiobutton(
                self.mode_frame, variable=self.mode_var, value=mode, command=self._update_mode_state
            )
            button.pack(anchor="w")
            self.mode_buttons[mode] = button

        self.prompt_box = PromptBox(panel, colors, "", "", height=5)
        self.prompt_box.pack(fill="x", pady=(10, 0))
        self.prompt_box.set(self.settings.get("prompt", ""))

        self.negative_box = PromptBox(panel, colors, "", "", height=3)
        self.negative_box.pack(fill="x", pady=(8, 0))
        self.negative_box.set(self.settings.get("negative_prompt", ""))

        self.strength_scale = LabeledScale(
            panel, colors, "", 0.0, 1.0, float(self.settings.get("strength")), step=0.01, fmt="%.2f"
        )
        self.strength_scale.pack(fill="x", pady=(10, 0))

        self.steps_scale = LabeledScale(
            panel, colors, "", 1, 50, int(self.settings.get("steps")), step=1, fmt="%.0f"
        )
        self.steps_scale.pack(fill="x", pady=(8, 0))

        self.guidance_scale = LabeledScale(
            panel, colors, "", 0.0, 10.0, float(self.settings.get("guidance")), step=0.1, fmt="%.1f"
        )
        self.guidance_scale.pack(fill="x", pady=(8, 0))

        # seed
        seed_frame = ttk.Frame(panel, style="Panel.TFrame")
        seed_frame.pack(fill="x", pady=(10, 0))
        self.seed_label = ttk.Label(seed_frame, style="Panel.TLabel")
        self.seed_label.pack(side="left")
        self.seed_var = tk.StringVar(value="-1")
        self.seed_entry = ttk.Entry(seed_frame, textvariable=self.seed_var, width=12)
        self.seed_entry.pack(side="right")
        seed_buttons = ttk.Frame(panel, style="Panel.TFrame")
        seed_buttons.pack(fill="x", pady=(4, 0))
        self.btn_seed_random = ttk.Button(seed_buttons, style="Tool.TButton", command=self.random_seed)
        self.btn_seed_random.pack(side="left")
        self.btn_seed_reuse = ttk.Button(seed_buttons, style="Tool.TButton", command=self.reuse_seed)
        self.btn_seed_reuse.pack(side="left", padx=6)

        # size
        self.size_frame = ttk.LabelFrame(panel, style="TLabelframe", padding=8)
        self.size_frame.pack(fill="x", pady=(12, 0))
        row = ttk.Frame(self.size_frame, style="Panel.TFrame")
        row.pack(fill="x")
        self.megapixel_var = tk.StringVar(value=str(self.settings.get("megapixels", 1.0)))
        self.megapixel_box = ttk.Combobox(
            row, values=MEGAPIXELS, textvariable=self.megapixel_var, width=6, state="readonly"
        )
        self.megapixel_box.pack(side="left")
        self.megapixel_box.bind("<<ComboboxSelected>>", lambda _e: self._update_size_label())
        ttk.Label(row, text="MP", style="Muted.TLabel").pack(side="left", padx=4)
        self.aspect_var = tk.StringVar(value=ASPECTS[0][0])
        self.aspect_box = ttk.Combobox(
            row, values=[name for name, _ in ASPECTS], textvariable=self.aspect_var, width=6, state="readonly"
        )
        self.aspect_box.pack(side="right")
        self.aspect_box.bind("<<ComboboxSelected>>", lambda _e: self._update_size_label())
        self.size_label = ttk.Label(self.size_frame, style="Value.TLabel")
        self.size_label.pack(anchor="w", pady=(6, 0))

        self.images_frame = ttk.Frame(panel, style="Panel.TFrame")
        self.images_frame.pack(fill="x", pady=(10, 0))
        self.images_label = ttk.Label(self.images_frame, style="Panel.TLabel")
        self.images_label.pack(side="left")
        self.images_var = tk.IntVar(value=int(self.settings.get("num_images", 1)))
        self.images_spin = ttk.Spinbox(self.images_frame, from_=1, to=4, textvariable=self.images_var, width=5)
        self.images_spin.pack(side="right")

        # inpainting
        self.inpaint_frame = ttk.LabelFrame(panel, style="TLabelframe", padding=8)
        self.inpaint_frame.pack(fill="x", pady=(12, 0))
        self.blur_scale = LabeledScale(
            self.inpaint_frame, colors, "", 0, 48, int(self.settings.get("mask_blur")), step=1, fmt="%.0f"
        )
        self.blur_scale.pack(fill="x")
        self.invert_var = tk.BooleanVar(value=bool(self.settings.get("mask_invert")))
        self.invert_check = ttk.Checkbutton(self.inpaint_frame, variable=self.invert_var)
        self.invert_check.pack(anchor="w", pady=(6, 0))
        self.keep_var = tk.BooleanVar(value=bool(self.settings.get("keep_unmasked", True)))
        self.keep_check = ttk.Checkbutton(self.inpaint_frame, variable=self.keep_var)
        self.keep_check.pack(anchor="w")


    def _bind_keys(self) -> None:
        root = self.root
        root.bind("<Control-o>", lambda _e: self.open_image())
        root.bind("<Control-s>", lambda _e: self.save_result())
        root.bind("<Control-z>", lambda _e: self.undo())
        root.bind("<Control-y>", lambda _e: self.redo())
        root.bind("<Control-Return>", lambda _e: self.generate())
        root.bind("<Escape>", lambda _e: self.cancel())
        root.bind("<KeyPress-b>", lambda _e: self._set_tool(TOOL_BRUSH))
        root.bind("<KeyPress-e>", lambda _e: self._set_tool(TOOL_ERASER))
        root.bind("<KeyPress-m>", lambda _e: self._toggle_mask_key())
        root.bind("<KeyPress-bracketleft>", lambda _e: self._nudge_brush(-8))
        root.bind("<KeyPress-bracketright>", lambda _e: self._nudge_brush(8))
        root.bind("<KeyPress-plus>", lambda _e: self.zoom_in())
        root.bind("<KeyPress-minus>", lambda _e: self.zoom_out())
        root.bind("<KeyPress-0>", lambda _e: self.zoom_fit())
        root.bind("<KeyPress-space>", lambda _e: self.canvas.set_space_down(True))
        root.bind("<KeyRelease-space>", lambda _e: self.canvas.set_space_down(False))

    # ------------------------------------------------------------------
    # translation
    # ------------------------------------------------------------------

    def _retranslate(self) -> None:
        tr = self.tr
        self.root.title("%s %s" % (tr("app.title"), __version__))

        labels = [
            (self.menubar, 0, tr("menu.file")),
            (self.menubar, 1, tr("menu.edit")),
            (self.menubar, 2, tr("menu.view")),
            (self.menubar, 3, tr("menu.server")),
            (self.menubar, 4, tr("menu.help")),
        ]
        for menu, index, text in labels:
            menu.entryconfigure(index + 1, label=text)

        for index, key in enumerate(
            ("menu.file.open", "menu.file.save", "menu.file.save_mask", None, "menu.file.exit")
        ):
            if key:
                self.menu_file.entryconfigure(index, label=tr(key))
        for index, key in enumerate(
            ("menu.edit.undo", "menu.edit.redo", "menu.edit.clear_mask", None, "menu.edit.use_result")
        ):
            if key:
                self.menu_edit.entryconfigure(index, label=tr(key))
        for index, key in enumerate(
            (
                "menu.view.zoom_in",
                "menu.view.zoom_out",
                "menu.view.zoom_fit",
                "menu.view.zoom_100",
                None,
                "menu.view.toggle_mask",
                "menu.view.compare",
                None,
                "menu.view.theme",
                "menu.view.language",
            )
        ):
            if key:
                self.menu_view.entryconfigure(index, label=tr(key))
        for index, key in enumerate(
            ("menu.server.settings", "menu.server.reconnect", None, "menu.server.demo")
        ):
            if key:
                self.menu_server.entryconfigure(index, label=tr(key))
        for index, key in enumerate(("menu.help.shortcuts", "menu.help.about")):
            self.menu_help.entryconfigure(index, label=tr(key))

        self.btn_open.configure(text=tr("tool.open"))
        self.btn_brush.configure(text=tr("tool.brush"))
        self.btn_eraser.configure(text=tr("tool.eraser"))
        self.brush_label.configure(text=tr("tool.size"))
        self.btn_mask.configure(text=tr("tool.mask"))
        self.btn_compare.configure(text=tr("tool.compare"))
        self.btn_fit.configure(text=tr("tool.fit"))

        self.mode_frame.configure(text=tr("panel.mode"))
        for mode, button in self.mode_buttons.items():
            button.configure(text=tr("mode.%s" % mode))

        self.prompt_box.set_title(tr("panel.prompt"), tr("panel.prompt.hint"))
        self.negative_box.set_title(tr("panel.negative"), tr("panel.negative.hint"))
        self.strength_scale.set_text(tr("panel.strength"), tr("panel.strength.hint"))
        self.steps_scale.set_text(tr("panel.steps"))
        self.guidance_scale.set_text(tr("panel.guidance"))
        self.seed_label.configure(text=tr("panel.seed"))
        self.btn_seed_random.configure(text=tr("panel.seed.random"))
        self.btn_seed_reuse.configure(text=tr("panel.seed.reuse"))
        self.size_frame.configure(text=tr("panel.size"))
        self.images_label.configure(text=tr("panel.images"))
        self.inpaint_frame.configure(text=tr("panel.inpaint"))
        self.blur_scale.set_text(tr("panel.mask_blur"))
        self.invert_check.configure(text=tr("panel.mask_invert"))
        self.keep_check.configure(text=tr("panel.keep_unmasked"))
        self.btn_generate.configure(text=tr("panel.generate"))
        self.btn_cancel.configure(text=tr("panel.cancel"))
        self.history.set_texts(tr("history.title"), tr("history.empty"))
        self.canvas.set_placeholder(tr("panel.prompt.hint") if not self.canvas.has_image else "")
        self._update_brush_label()
        self._update_size_label()
        self._update_server_label()
        self.set_status(tr("status.ready"))

    # ------------------------------------------------------------------
    # server connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self.client.base_url = (self.settings.get("server_url") or "").rstrip("/")
        self.client.token = self.settings.get("token") or ""
        self._set_server_state(None, self.tr("status.connecting"))

        def worker() -> None:
            try:
                info = self.client.info()
            except ClientError as exc:
                self._later(self._on_connect_failed, str(exc))
                return
            self._later(self._on_connected, info)

        threading.Thread(target=worker, name="zimage-connect", daemon=True).start()

    def _on_connected(self, info: ServerInfo) -> None:
        self.server_info = info
        self._set_server_state(
            True, self.tr("status.connected", engine=info.engine, device=info.device or "?")
        )

    def _on_connect_failed(self, detail: str) -> None:
        self.server_info = None
        self._set_server_state(False, self.tr("status.no_server"))
        self.set_status(detail)

    def _set_server_state(self, connected: Optional[bool], text: str) -> None:
        colour = {True: self.colors["ok"], False: self.colors["danger"], None: self.colors["muted"]}[connected]
        self.server_dot.itemconfigure(self._dot, fill=colour)
        self.server_label.configure(text=text)

    def _update_server_label(self) -> None:
        if self.server_info is not None:
            self._set_server_state(
                True,
                self.tr(
                    "status.connected",
                    engine=self.server_info.engine,
                    device=self.server_info.device or "?",
                ),
            )
        else:
            self._set_server_state(False, self.tr("status.no_server"))

    def start_demo_backend(self) -> str:
        """Run the GPU-free demo engine in this process and connect to it."""
        if self.demo_server is None:
            from ..server.app import StudioServer
            from ..server.engines import create_engine

            self.demo_server = StudioServer(("127.0.0.1", 0), create_engine("mock", step_delay=0.05))
            threading.Thread(
                target=self.demo_server.serve_forever,
                kwargs={"poll_interval": 0.1},
                name="zimage-demo-server",
                daemon=True,
            ).start()
        host, port = self.demo_server.server_address[:2]
        url = "http://%s:%d" % (host, port)
        self.settings.set("server_url", url)
        self.connect()
        return url

    def open_server_dialog(self) -> None:
        ServerDialog(self)

    # ------------------------------------------------------------------
    # file handling
    # ------------------------------------------------------------------

    def open_image(self, path: Optional[str] = None) -> bool:
        if path is None:
            path = filedialog.askopenfilename(
                parent=self.root,
                initialdir=self.settings.get("last_open_dir") or os.getcwd(),
                filetypes=[
                    ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff"),
                    ("PNG", "*.png"),
                    ("All files", "*.*"),
                ],
            )
        if not path:
            return False
        try:
            with open(path, "rb") as handle:
                data = handle.read()
            png = self._to_displayable_png(data)
            width, height = self.canvas.set_source(png)
        except (OSError, ImageError, ClientError, tk.TclError) as exc:
            messagebox.showerror(self.tr("error.title"), self.tr("error.open", detail=exc), parent=self.root)
            return False
        self.settings.set("last_open_dir", os.path.dirname(path))
        self.mode_var.set(MODE_IMG2IMG)
        self._update_mode_state()
        self.set_status(self.tr("status.loaded", width=width, height=height))
        return True

    def _to_displayable_png(self, data: bytes) -> bytes:
        """Return PNG bytes Tk can display, converting through the server if needed."""
        fmt = sniff_format(data)
        needs_resize = False
        if fmt in ("png", "gif"):
            try:
                width, height = sniff_size(data)
                needs_resize = max(width, height) > MAX_SOURCE_EDGE
            except ImageError:
                needs_resize = True
            if fmt == "png" and not needs_resize and len(data) <= MAX_UPLOAD_BYTES:
                return data

        # Pillow if the host happens to have it (64-bit desktops usually do)...
        try:
            from ..server.convert import HAVE_PILLOW, to_png

            if HAVE_PILLOW:
                png, _, _ = to_png(data, MAX_SOURCE_EDGE)
                return png
        except ImportError:
            pass

        # ...otherwise let the server do it.
        if not self.client.base_url:
            raise ImageError(self.tr("error.unsupported"))
        png, _, _ = self.client.convert(data, MAX_SOURCE_EDGE)
        return png

    def save_result(self) -> Optional[str]:
        png = self.canvas.preview_png or self.canvas.source_png
        if png is None:
            return None
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".png",
            initialdir=self.settings.get("last_save_dir") or os.getcwd(),
            initialfile="z-image-%d.png" % int(time.time()),
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return None
        try:
            with open(path, "wb") as handle:
                handle.write(png)
        except OSError as exc:
            messagebox.showerror(self.tr("error.title"), self.tr("error.save", detail=exc), parent=self.root)
            return None
        self.settings.set("last_save_dir", os.path.dirname(path))
        self.set_status(self.tr("status.saved", path=path))
        return path

    def save_mask(self) -> Optional[str]:
        mask = self.canvas.mask_png()
        if mask is None:
            return None
        path = filedialog.asksaveasfilename(
            parent=self.root, defaultextension=".png", initialfile="mask.png", filetypes=[("PNG", "*.png")]
        )
        if not path:
            return None
        try:
            with open(path, "wb") as handle:
                handle.write(mask)
        except OSError as exc:
            messagebox.showerror(self.tr("error.title"), self.tr("error.save", detail=exc), parent=self.root)
            return None
        return path

    # ------------------------------------------------------------------
    # editing helpers
    # ------------------------------------------------------------------

    def _set_tool(self, tool: str) -> None:
        self.tool_var.set(tool)
        self._on_tool_change()

    def _on_tool_change(self) -> None:
        self.canvas.set_tool(self.tool_var.get())

    def _nudge_brush(self, delta: int) -> None:
        self.brush_var.set(max(4, min(256, self.brush_var.get() + delta)))
        self._on_brush_size()

    def _on_brush_size(self) -> None:
        size = int(self.brush_var.get())
        self.canvas.set_brush_size(size)
        self.settings.set("brush_size", size)
        self._update_brush_label()

    def _update_brush_label(self) -> None:
        self.brush_value.configure(text=str(int(self.brush_var.get())))

    def _toggle_mask_key(self) -> None:
        self.show_mask_var.set(not self.show_mask_var.get())
        self.toggle_mask()

    def toggle_mask(self) -> None:
        self.canvas.set_show_mask(self.show_mask_var.get())

    def toggle_compare(self) -> None:
        self.canvas.toggle_preview()

    def undo(self) -> None:
        self.canvas.undo()

    def redo(self) -> None:
        self.canvas.redo()

    def clear_mask(self) -> None:
        self.canvas.clear_mask()

    def zoom_in(self) -> None:
        self.canvas.zoom_in()

    def zoom_out(self) -> None:
        self.canvas.zoom_out()

    def zoom_fit(self) -> None:
        self.canvas.zoom_fit()

    def zoom_actual(self) -> None:
        self.canvas.zoom_actual()

    def random_seed(self) -> None:
        self.seed_var.set(str(random.randint(0, 2**31 - 2)))

    def reuse_seed(self) -> None:
        if self.last_seed >= 0:
            self.seed_var.set(str(self.last_seed))

    def use_result_as_source(self, index: Optional[int] = None) -> None:
        """Continue editing on top of a generated image."""
        png = None
        if index is not None:
            entry = self.history.item(index)
            png = entry.get("png") if entry else None
        if png is None:
            png = self.canvas.preview_png
        if png is None:
            return
        width, height = self.canvas.set_source(png)
        self.mode_var.set(MODE_IMG2IMG)
        self._update_mode_state()
        self.set_status(self.tr("status.loaded", width=width, height=height))

    def _on_history_select(self, index: int) -> None:
        entry = self.history.item(index)
        if entry:
            self.canvas.set_preview(entry["png"])
            seed = entry.get("seed")
            if isinstance(seed, int):
                self.last_seed = seed

    def _on_mask_change(self) -> None:
        if self.canvas.has_mask and self.mode_var.get() == MODE_IMG2IMG:
            self.mode_var.set(MODE_INPAINT)
        elif not self.canvas.has_mask and self.mode_var.get() == MODE_INPAINT:
            self.mode_var.set(MODE_IMG2IMG)
        self._update_mode_state()

    def _update_mode_state(self) -> None:
        mode = self.mode_var.get()
        has_image = self.canvas.has_image
        self.mode_buttons[MODE_IMG2IMG].configure(state="normal" if has_image else "disabled")
        self.mode_buttons[MODE_INPAINT].configure(state="normal" if has_image else "disabled")
        if not has_image and mode != MODE_TXT2IMG:
            self.mode_var.set(MODE_TXT2IMG)
            mode = MODE_TXT2IMG

        image_mode = mode in (MODE_IMG2IMG, MODE_INPAINT)
        state = "normal" if image_mode else "disabled"
        self.strength_scale.scale.configure(state=state)
        for child in (self.blur_scale.scale, self.invert_check, self.keep_check):
            child.configure(state="normal" if mode == MODE_INPAINT else "disabled")
        self.aspect_box.configure(state="readonly" if mode == MODE_TXT2IMG else "disabled")
        self._update_size_label()

    # ------------------------------------------------------------------
    # size handling
    # ------------------------------------------------------------------

    def output_size(self) -> "tuple[int, int]":
        """Resolution the server will be asked for, aligned to the model grid."""
        try:
            megapixels = float(self.megapixel_var.get())
        except ValueError:
            megapixels = 1.0
        target = max(0.25, min(4.0, megapixels)) * 1024 * 1024
        if self.canvas.has_image and self.mode_var.get() != MODE_TXT2IMG:
            width, height = self.canvas.source_size
        else:
            ratio = dict(ASPECTS).get(self.aspect_var.get(), 1.0)
            width, height = (int(1024 * ratio), 1024) if ratio >= 1 else (1024, int(1024 / ratio))
        return fit_size(width, height, target_pixels=int(target))

    def _update_size_label(self) -> None:
        width, height = self.output_size()
        self.size_label.configure(text="%d × %d px" % (width, height))

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------

    def build_request(self) -> GenerateRequest:
        """Collect the panel state into a validated request (raises on problems)."""
        mode = self.mode_var.get()
        prompt = self.prompt_box.get()
        if mode == MODE_TXT2IMG and not prompt.strip():
            raise ProtocolError(self.tr("error.no_prompt"))
        if mode in (MODE_IMG2IMG, MODE_INPAINT) and not self.canvas.has_image:
            raise ProtocolError(self.tr("error.no_image"))
        if mode == MODE_INPAINT and not self.canvas.has_mask:
            raise ProtocolError(self.tr("error.no_mask"))

        try:
            seed = int(self.seed_var.get())
        except (TypeError, ValueError):
            seed = -1

        width, height = self.output_size()
        payload = {
            "mode": mode,
            "prompt": prompt,
            "negative_prompt": self.negative_box.get(),
            "steps": int(self.steps_scale.get()),
            "guidance": float(self.guidance_scale.get()),
            "strength": float(self.strength_scale.get()),
            "seed": seed,
            "width": width,
            "height": height,
            "num_images": int(self.images_var.get() or 1),
            "mask_blur": int(self.blur_scale.get()),
            "mask_invert": bool(self.invert_var.get()),
            "keep_unmasked": bool(self.keep_var.get()),
        }
        if mode in (MODE_IMG2IMG, MODE_INPAINT) and self.canvas.source_png:
            payload["image"] = encode_image(self.canvas.source_png)
        if mode == MODE_INPAINT:
            mask = self.canvas.mask_png()
            if mask:
                payload["mask"] = encode_image(mask)
        return parse_generate_request(payload)

    def generate(self) -> bool:
        if self.runner.busy:
            messagebox.showinfo(self.tr("error.title"), self.tr("error.busy"), parent=self.root)
            return False
        try:
            request = self.build_request()
        except ProtocolError as exc:
            messagebox.showwarning(self.tr("error.title"), str(exc), parent=self.root)
            return False

        self._job_started = time.time()
        self._set_busy(True)
        self.set_status(self.tr("status.queued"))
        self.progress.configure(value=0)
        started = self.runner.start(
            request,
            lambda status: self._later(self._on_progress, status),
            lambda status: self._later(self._on_done, status),
            lambda detail: self._later(self._on_failed, detail),
        )
        if not started:
            self._set_busy(False)
        return started

    def cancel(self) -> None:
        if self.runner.busy:
            self.runner.cancel()
            self.set_status(self.tr("status.cancelled"))

    def _on_progress(self, status: JobStatus) -> None:
        if status.status == STATUS_QUEUED:
            self.set_status(self.tr("status.queued"))
        elif status.status == STATUS_RUNNING:
            self.set_status(
                self.tr("status.running", step=max(1, status.step), total=max(1, status.total_steps))
            )
        self.progress.configure(value=int(status.progress * 1000))

    def _on_done(self, status: JobStatus) -> None:
        self._set_busy(False)
        self.progress.configure(value=1000)
        self.last_seed = status.seed
        for index, payload in enumerate(status.images):
            try:
                png = decode_image(payload)
            except ProtocolError:
                continue
            self.history.add(png, {"seed": status.seed + index, "elapsed": status.elapsed})
            if index == 0:
                self.canvas.set_preview(png)
        self.set_status(
            self.tr("status.done", seconds=status.elapsed or (time.time() - self._job_started), seed=status.seed)
        )

    def _on_failed(self, detail: str) -> None:
        self._set_busy(False)
        self.progress.configure(value=0)
        self.set_status(detail)

    def _set_busy(self, busy: bool) -> None:
        self.btn_generate.configure(state="disabled" if busy else "normal")
        self.btn_cancel.configure(state="normal" if busy else "disabled")

    # ------------------------------------------------------------------
    # dialogs, misc
    # ------------------------------------------------------------------

    def show_about(self) -> None:
        messagebox.showinfo(
            self.tr("menu.help.about"),
            self.tr("about.text", app=APP_NAME, version=__version__),
            parent=self.root,
        )

    def show_shortcuts(self) -> None:
        messagebox.showinfo(self.tr("menu.help.shortcuts"), self.tr("shortcuts.text"), parent=self.root)

    def toggle_theme(self) -> None:
        name = "light" if self.settings.get("theme") == "dark" else "dark"
        self.settings.set("theme", name)
        messagebox.showinfo(
            self.tr("menu.view.theme"),
            {"de": "Das neue Design wird beim nächsten Start verwendet.",
             "en": "The new theme is applied on the next start."}[self.tr.language],
            parent=self.root,
        )

    def toggle_language(self) -> None:
        self.tr.toggle()
        self.settings.set("language", self.tr.language)
        self._retranslate()

    def set_status(self, message: str) -> None:
        self.status_label.configure(text=message)

    def _later(self, callback, *args) -> None:
        """Queue a worker-thread callback for execution on the Tk thread."""
        self._events.put((callback, args))

    def _pump_events(self) -> None:
        """Drain the callback queue; rescheduled on the Tk thread every 40 ms."""
        while True:
            try:
                callback, args = self._events.get_nowait()
            except queue.Empty:
                break
            try:
                callback(*args)
            except tk.TclError:  # widgets already destroyed
                break
        try:
            self._event_job = self.root.after(40, self._pump_events)
        except tk.TclError:
            self._event_job = None

    def _store_settings(self) -> None:
        self.settings.update(
            {
                "prompt": self.prompt_box.get(),
                "negative_prompt": self.negative_box.get(),
                "steps": int(self.steps_scale.get()),
                "guidance": float(self.guidance_scale.get()),
                "strength": float(self.strength_scale.get()),
                "num_images": int(self.images_var.get() or 1),
                "mask_blur": int(self.blur_scale.get()),
                "mask_invert": bool(self.invert_var.get()),
                "keep_unmasked": bool(self.keep_var.get()),
                "brush_size": int(self.brush_var.get()),
                "language": self.tr.language,
            }
        )
        try:
            self.settings.set("megapixels", float(self.megapixel_var.get()))
        except ValueError:
            pass
        try:
            self.settings.set("window_geometry", self.root.winfo_geometry())
        except tk.TclError:
            pass
        self.settings.save()

    def quit(self) -> None:
        self._store_settings()
        if self._event_job is not None:
            try:
                self.root.after_cancel(self._event_job)
            except tk.TclError:
                pass
            self._event_job = None
        if self.demo_server is not None:
            self.demo_server.shutdown()
            self.demo_server.server_close()
            self.demo_server = None
        if self.owns_root:
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


class ServerDialog(tk.Toplevel):
    """Modal dialog for the backend address and token."""

    def __init__(self, app: StudioApp) -> None:
        super().__init__(app.root)
        self.app = app
        colors = app.colors
        tr = app.tr
        self.title(tr("dialog.server.title"))
        self.configure(background=colors["panel"])
        self.transient(app.root)
        self.resizable(False, False)

        body = ttk.Frame(self, style="Panel.TFrame", padding=14)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text=tr("dialog.server.hint"), style="Muted.TLabel", justify="left").pack(
            anchor="w", pady=(0, 10)
        )
        ttk.Label(body, text=tr("dialog.server.url"), style="Panel.TLabel").pack(anchor="w")
        self.url_var = tk.StringVar(value=app.settings.get("server_url"))
        ttk.Entry(body, textvariable=self.url_var, width=44).pack(fill="x", pady=(2, 8))

        ttk.Label(body, text=tr("dialog.server.token"), style="Panel.TLabel").pack(anchor="w")
        self.token_var = tk.StringVar(value=app.settings.get("token"))
        ttk.Entry(body, textvariable=self.token_var, width=44, show="•").pack(fill="x", pady=(2, 8))

        self.result_label = ttk.Label(body, text="", style="Muted.TLabel", wraplength=380, justify="left")
        self.result_label.pack(anchor="w", pady=(0, 8))

        buttons = ttk.Frame(body, style="Panel.TFrame")
        buttons.pack(fill="x")
        ttk.Button(buttons, text=tr("dialog.server.test"), command=self.test).pack(side="left")
        ttk.Button(buttons, text=tr("dialog.server.cancel"), command=self.destroy).pack(side="right")
        ttk.Button(buttons, text=tr("dialog.server.ok"), style="Accent.TButton", command=self.apply).pack(
            side="right", padx=6
        )

        self.grab_set()

    def test(self) -> None:
        client = StudioClient(self.url_var.get(), self.token_var.get(), timeout=8.0)

        def worker() -> None:
            try:
                info = client.info()
                message = "%s %s — %s / %s" % (info.name, info.version, info.engine, info.device or "?")
            except ClientError as exc:
                message = str(exc)
            self.app._later(self._show_result, message)

        self.result_label.configure(text=self.app.tr("status.connecting"))
        threading.Thread(target=worker, daemon=True).start()

    def _show_result(self, message: str) -> None:
        try:
            self.result_label.configure(text=message)
        except tk.TclError:  # dialog already closed
            pass

    def apply(self) -> None:
        self.app.settings.set("server_url", self.url_var.get().strip())
        self.app.settings.set("token", self.token_var.get().strip())
        self.app.settings.save()
        self.app.connect()
        self.destroy()
