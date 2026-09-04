"""Interface strings.

German is the default because that is the language this application was
commissioned in; English is a full alternative and can be switched at runtime.
"""

from __future__ import annotations

from typing import Dict

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "de": {
        "app.title": "Z-Image Studio",
        "menu.file": "Datei",
        "menu.file.open": "Bild öffnen…",
        "menu.file.save": "Ergebnis speichern…",
        "menu.file.save_mask": "Maske speichern…",
        "menu.file.exit": "Beenden",
        "menu.edit": "Bearbeiten",
        "menu.edit.undo": "Rückgängig",
        "menu.edit.redo": "Wiederholen",
        "menu.edit.clear_mask": "Maske leeren",
        "menu.edit.use_result": "Ergebnis als Quelle übernehmen",
        "menu.view": "Ansicht",
        "menu.view.zoom_in": "Vergrößern",
        "menu.view.zoom_out": "Verkleinern",
        "menu.view.zoom_fit": "Einpassen",
        "menu.view.zoom_100": "Originalgröße",
        "menu.view.toggle_mask": "Maske anzeigen",
        "menu.view.compare": "Vorher/Nachher",
        "menu.view.theme": "Design wechseln",
        "menu.view.language": "Sprache: English",
        "menu.server": "Server",
        "menu.server.settings": "Verbindung…",
        "menu.server.reconnect": "Neu verbinden",
        "menu.server.demo": "Demo-Backend starten",
        "menu.help": "Hilfe",
        "menu.help.shortcuts": "Tastenkürzel",
        "menu.help.about": "Über",
        "tool.open": "Öffnen",
        "tool.brush": "Pinsel",
        "tool.eraser": "Radierer",
        "tool.size": "Größe",
        "tool.mask": "Maske",
        "tool.compare": "Vergleich",
        "tool.fit": "Einpassen",
        "panel.mode": "Modus",
        "mode.txt2img": "Text → Bild",
        "mode.img2img": "Bild → Bild",
        "mode.inpaint": "Bereich ersetzen",
        "panel.prompt": "Prompt",
        "panel.prompt.hint": "Beschreibe, wie das Bild aussehen soll…",
        "panel.negative": "Negativer Prompt",
        "panel.negative.hint": "Was vermieden werden soll (nur bei Guidance > 1)",
        "panel.strength": "Stärke",
        "panel.strength.hint": "Wie stark das Ausgangsbild verändert wird",
        "panel.steps": "Schritte",
        "panel.guidance": "Guidance",
        "panel.seed": "Seed",
        "panel.seed.random": "Zufall",
        "panel.seed.reuse": "Letzten übernehmen",
        "panel.size": "Ausgabegröße",
        "panel.size.match": "An Quellbild anpassen",
        "panel.images": "Anzahl Bilder",
        "panel.inpaint": "Bereich ersetzen",
        "panel.mask_blur": "Kantenweichzeichnung",
        "panel.mask_invert": "Maske umkehren",
        "panel.keep_unmasked": "Unmaskierte Pixel exakt behalten",
        "panel.generate": "Generieren",
        "panel.cancel": "Abbrechen",
        "history.title": "Verlauf",
        "history.empty": "Noch keine Ergebnisse",
        "history.use": "Als Quelle verwenden",
        "history.save": "Speichern…",
        "status.ready": "Bereit",
        "status.no_server": "Nicht verbunden",
        "status.connected": "Verbunden: {engine} auf {device}",
        "status.connecting": "Verbinde…",
        "status.queued": "In der Warteschlange…",
        "status.running": "Schritt {step}/{total}",
        "status.done": "Fertig in {seconds:.1f} s (Seed {seed})",
        "status.cancelled": "Abgebrochen",
        "status.loaded": "{width}×{height} geladen",
        "status.saved": "Gespeichert: {path}",
        "dialog.server.title": "Serververbindung",
        "dialog.server.url": "Server-URL",
        "dialog.server.token": "Zugriffstoken (optional)",
        "dialog.server.test": "Verbindung testen",
        "dialog.server.ok": "Übernehmen",
        "dialog.server.cancel": "Abbrechen",
        "dialog.server.hint": (
            "Die Z-Image-Gewichte laufen auf einem 64-Bit-Rechner mit GPU.\n"
            "Trage hier dessen Adresse ein, z. B. http://192.168.1.20:8787"
        ),
        "error.title": "Fehler",
        "error.open": "Das Bild konnte nicht geladen werden:\n{detail}",
        "error.save": "Das Bild konnte nicht gespeichert werden:\n{detail}",
        "error.no_prompt": "Bitte gib zuerst einen Prompt ein.",
        "error.no_image": "Für diesen Modus wird ein Ausgangsbild benötigt.",
        "error.no_mask": "Bitte markiere zuerst den zu ersetzenden Bereich mit dem Pinsel.",
        "error.busy": "Es läuft bereits eine Generierung.",
        "error.unsupported": (
            "Dieses Format kann ohne Serververbindung nicht gelesen werden.\n"
            "Verbinde dich mit dem Server oder verwende PNG bzw. GIF."
        ),
        "about.text": (
            "{app} {version}\n\n"
            "Bild-zu-Bild-Bearbeitung mit Z-Image.\n"
            "Die Oberfläche läuft auch auf 32-Bit-Windows; die Modellgewichte\n"
            "rechnen auf einem verbundenen Server."
        ),
        "shortcuts.text": (
            "Strg+O   Bild öffnen\n"
            "Strg+S   Ergebnis speichern\n"
            "Strg+Z / Strg+Y   Rückgängig / Wiederholen\n"
            "Strg+Eingabe   Generieren\n"
            "Esc   Abbrechen\n"
            "B / E   Pinsel / Radierer\n"
            "[ / ]   Pinsel kleiner / größer\n"
            "+ / -   Zoom\n"
            "0   Einpassen\n"
            "M   Maske ein-/ausblenden\n"
            "Leertaste+Ziehen   Verschieben"
        ),
    },
    "en": {
        "app.title": "Z-Image Studio",
        "menu.file": "File",
        "menu.file.open": "Open image…",
        "menu.file.save": "Save result…",
        "menu.file.save_mask": "Save mask…",
        "menu.file.exit": "Quit",
        "menu.edit": "Edit",
        "menu.edit.undo": "Undo",
        "menu.edit.redo": "Redo",
        "menu.edit.clear_mask": "Clear mask",
        "menu.edit.use_result": "Use result as source",
        "menu.view": "View",
        "menu.view.zoom_in": "Zoom in",
        "menu.view.zoom_out": "Zoom out",
        "menu.view.zoom_fit": "Fit to window",
        "menu.view.zoom_100": "Actual size",
        "menu.view.toggle_mask": "Show mask",
        "menu.view.compare": "Before/after",
        "menu.view.theme": "Switch theme",
        "menu.view.language": "Sprache: Deutsch",
        "menu.server": "Server",
        "menu.server.settings": "Connection…",
        "menu.server.reconnect": "Reconnect",
        "menu.server.demo": "Start demo backend",
        "menu.help": "Help",
        "menu.help.shortcuts": "Keyboard shortcuts",
        "menu.help.about": "About",
        "tool.open": "Open",
        "tool.brush": "Brush",
        "tool.eraser": "Eraser",
        "tool.size": "Size",
        "tool.mask": "Mask",
        "tool.compare": "Compare",
        "tool.fit": "Fit",
        "panel.mode": "Mode",
        "mode.txt2img": "Text to image",
        "mode.img2img": "Image to image",
        "mode.inpaint": "Replace area",
        "panel.prompt": "Prompt",
        "panel.prompt.hint": "Describe what the image should look like…",
        "panel.negative": "Negative prompt",
        "panel.negative.hint": "What to avoid (only used with guidance > 1)",
        "panel.strength": "Strength",
        "panel.strength.hint": "How far the result may move from the source",
        "panel.steps": "Steps",
        "panel.guidance": "Guidance",
        "panel.seed": "Seed",
        "panel.seed.random": "Random",
        "panel.seed.reuse": "Reuse last",
        "panel.size": "Output size",
        "panel.size.match": "Match source image",
        "panel.images": "Images",
        "panel.inpaint": "Replace area",
        "panel.mask_blur": "Edge feathering",
        "panel.mask_invert": "Invert mask",
        "panel.keep_unmasked": "Keep unmasked pixels exact",
        "panel.generate": "Generate",
        "panel.cancel": "Cancel",
        "history.title": "History",
        "history.empty": "No results yet",
        "history.use": "Use as source",
        "history.save": "Save…",
        "status.ready": "Ready",
        "status.no_server": "Not connected",
        "status.connected": "Connected: {engine} on {device}",
        "status.connecting": "Connecting…",
        "status.queued": "Queued…",
        "status.running": "Step {step}/{total}",
        "status.done": "Finished in {seconds:.1f} s (seed {seed})",
        "status.cancelled": "Cancelled",
        "status.loaded": "Loaded {width}×{height}",
        "status.saved": "Saved: {path}",
        "dialog.server.title": "Server connection",
        "dialog.server.url": "Server URL",
        "dialog.server.token": "Access token (optional)",
        "dialog.server.test": "Test connection",
        "dialog.server.ok": "Apply",
        "dialog.server.cancel": "Cancel",
        "dialog.server.hint": (
            "The Z-Image weights run on a 64-bit machine with a GPU.\n"
            "Enter its address here, for example http://192.168.1.20:8787"
        ),
        "error.title": "Error",
        "error.open": "The image could not be loaded:\n{detail}",
        "error.save": "The image could not be saved:\n{detail}",
        "error.no_prompt": "Please enter a prompt first.",
        "error.no_image": "This mode needs a source image.",
        "error.no_mask": "Please brush the area you want to replace first.",
        "error.busy": "A generation is already running.",
        "error.unsupported": (
            "This format cannot be read without a server connection.\n"
            "Connect to the server or use PNG or GIF."
        ),
        "about.text": (
            "{app} {version}\n\n"
            "Image-to-image editing with Z-Image.\n"
            "The interface runs on 32-bit Windows as well; the model weights\n"
            "run on a connected server."
        ),
        "shortcuts.text": (
            "Ctrl+O   Open image\n"
            "Ctrl+S   Save result\n"
            "Ctrl+Z / Ctrl+Y   Undo / redo\n"
            "Ctrl+Enter   Generate\n"
            "Esc   Cancel\n"
            "B / E   Brush / eraser\n"
            "[ / ]   Smaller / larger brush\n"
            "+ / -   Zoom\n"
            "0   Fit to window\n"
            "M   Toggle mask\n"
            "Space+drag   Pan"
        ),
    },
}

LANGUAGES = tuple(TRANSLATIONS)


class Translator:
    """Callable string lookup with ``{}`` formatting and a safe fallback."""

    def __init__(self, language: str = "de") -> None:
        self.language = language if language in TRANSLATIONS else "de"

    def switch(self, language: str) -> None:
        if language in TRANSLATIONS:
            self.language = language

    def toggle(self) -> str:
        order = list(LANGUAGES)
        self.language = order[(order.index(self.language) + 1) % len(order)]
        return self.language

    def __call__(self, key: str, **kwargs: object) -> str:
        table = TRANSLATIONS.get(self.language, {})
        text = table.get(key) or TRANSLATIONS["en"].get(key) or key
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, IndexError, ValueError):
                return text
        return text
