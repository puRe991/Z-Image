"""Turn a written instruction into local editing operations.

The text box stays useful without a text-conditioned model: the prompt is read
as an instruction ("entferne das", "etwas heller", "make it black and white")
and mapped onto the operations this machine can actually perform - object
removal by the neural network, everything else by the classical adjustments.

German and English are recognised, and the wording is matched on stems so that
inflections ("heller", "aufhellen", "hell") all land on the same operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .adjust import COLORS

#: The operation that hands the work to the inpainting network.
REMOVE = "remove"

#: Stems that select an operation.  Longer stems win, so "weniger kontrast"
#: beats "kontrast".
VOCABULARY: Sequence[Tuple[str, str]] = (
    # removal / inpainting
    ("entfern", REMOVE),
    ("wegmach", REMOVE),
    ("wegretusch", REMOVE),
    ("retusch", REMOVE),
    ("ausbessern", REMOVE),
    ("weglösch", REMOVE),
    ("lösch", REMOVE),
    ("weg damit", REMOVE),
    ("verschwind", REMOVE),
    ("remove", REMOVE),
    ("delete", REMOVE),
    ("erase", REMOVE),
    ("get rid", REMOVE),
    ("clean up", REMOVE),
    ("inpaint", REMOVE),
    ("fill in", REMOVE),
    # brightness
    ("aufhell", "brighter"),
    ("heller", "brighter"),
    ("brighten", "brighter"),
    ("brighter", "brighter"),
    ("lighter", "brighter"),
    ("abdunk", "darker"),
    ("dunkler", "darker"),
    ("darken", "darker"),
    ("darker", "darker"),
    # contrast
    ("weniger kontrast", "less_contrast"),
    ("kontrast raus", "less_contrast"),
    ("flacher", "less_contrast"),
    ("less contrast", "less_contrast"),
    ("flat", "less_contrast"),
    ("mehr kontrast", "more_contrast"),
    ("kontrast", "more_contrast"),
    ("contrast", "more_contrast"),
    # colour
    ("schwarzweiss", "grayscale"),
    ("schwarzweiß", "grayscale"),
    ("schwarz-weiss", "grayscale"),
    ("schwarz-weiß", "grayscale"),
    ("graustufen", "grayscale"),
    ("monochrom", "grayscale"),
    ("black and white", "grayscale"),
    ("grayscale", "grayscale"),
    ("greyscale", "grayscale"),
    ("monochrome", "grayscale"),
    ("sepia", "sepia"),
    ("vintage", "sepia"),
    ("wärmer", "warmer"),
    ("waermer", "warmer"),
    ("warmer", "warmer"),
    ("kühler", "cooler"),
    ("kuehler", "cooler"),
    ("kälter", "cooler"),
    ("cooler", "cooler"),
    ("colder", "cooler"),
    ("entsättig", "desaturate"),
    ("blasser", "desaturate"),
    ("desaturate", "desaturate"),
    ("muted", "desaturate"),
    ("farbiger", "saturate"),
    ("kräftiger", "saturate"),
    ("sättigung", "saturate"),
    ("saturate", "saturate"),
    ("vivid", "saturate"),
    # detail
    ("weichzeichn", "blur"),
    ("unscharf", "blur"),
    ("verwisch", "blur"),
    ("blur", "blur"),
    ("soften", "blur"),
    ("schärf", "sharpen"),
    ("schaerf", "sharpen"),
    ("sharpen", "sharpen"),
    ("sharper", "sharpen"),
    ("entrausch", "denoise"),
    ("rauschen", "denoise"),
    ("denoise", "denoise"),
    ("invertier", "invert"),
    ("negativ", "invert"),
    ("invert", "invert"),
)

#: Words that scale how strongly an operation is applied.
INTENSIFIERS: Dict[str, float] = {
    "ganz leicht": 0.2,
    "leicht": 0.3,
    "etwas": 0.35,
    "ein bisschen": 0.3,
    "ein wenig": 0.3,
    "slightly": 0.3,
    "a bit": 0.3,
    "a little": 0.3,
    "deutlich": 0.8,
    "stark": 0.85,
    "stärker": 0.85,
    "sehr": 0.9,
    "viel": 0.8,
    "extrem": 1.0,
    "very": 0.9,
    "much": 0.8,
    "strongly": 0.85,
    "a lot": 0.85,
}

#: Colour words for "fill with ...".
COLOR_WORDS: Dict[str, str] = {
    "schwarz": "black",
    "weiß": "white",
    "weiss": "white",
    "grau": "grey",
    "rot": "red",
    "grün": "green",
    "gruen": "green",
    "blau": "blue",
    "gelb": "yellow",
    "orange": "orange",
    "braun": "brown",
    "rosa": "pink",
    "pink": "pink",
    "lila": "purple",
    "violett": "purple",
    "black": "black",
    "white": "white",
    "grey": "grey",
    "gray": "grey",
    "red": "red",
    "green": "green",
    "blue": "blue",
    "yellow": "yellow",
    "brown": "brown",
    "purple": "purple",
}

FILL_STEMS = ("füll", "fuell", "fill", "einfärb", "einfaerb", "male", "paint", "make it")

DEFAULT_AMOUNT = 0.6


@dataclass
class Command:
    """One editing step derived from the prompt."""

    name: str
    amount: float = DEFAULT_AMOUNT
    options: Dict[str, object] = field(default_factory=dict)

    @property
    def is_removal(self) -> bool:
        return self.name == REMOVE


def _intensity(text: str) -> float:
    for word in sorted(INTENSIFIERS, key=len, reverse=True):
        if word in text:
            return INTENSIFIERS[word]
    return DEFAULT_AMOUNT


def _find_color(text: str) -> Optional[str]:
    for word in sorted(COLOR_WORDS, key=len, reverse=True):
        if word in text:
            return COLOR_WORDS[word]
    return None


def parse_prompt(prompt: str, *, has_mask: bool = False) -> List[Command]:
    """Read an instruction and return the operations it asks for.

    An empty instruction on a brushed area means "remove this", which is the
    most common request and what the network is for.
    """
    text = " %s " % (prompt or "").strip().lower()
    amount = _intensity(text)
    commands: List[Command] = []
    seen = set()

    for stem, operation in VOCABULARY:
        if stem not in text:
            continue
        # Blank the match out so a shorter stem cannot fire on the same words:
        # "weniger kontrast" must not also trigger plain "kontrast".
        text = text.replace(stem, " ")
        if operation not in seen:
            seen.add(operation)
            commands.append(Command(operation, amount))

    if any(stem in text for stem in FILL_STEMS):
        color = _find_color(text)
        if color:
            commands = [command for command in commands if command.name != REMOVE]
            commands.append(Command("fill_color", amount, {"color": COLORS[color]}))
    elif not commands:
        color = _find_color(text)
        if color:
            commands.append(Command("fill_color", amount, {"color": COLORS[color]}))

    if not commands and has_mask:
        commands.append(Command(REMOVE, 1.0))
    return commands


def describe_vocabulary(language: str = "de") -> str:
    """A short cheat sheet for the interface."""
    if language.startswith("de"):
        return (
            "Was der lokale Modus versteht:\n"
            "  entfernen / retuschieren / weg    – markierten Bereich wegrechnen (KI)\n"
            "  heller · dunkler · kontrast      – Helligkeit und Kontrast\n"
            "  schwarzweiß · sepia · farbiger   – Farbwirkung\n"
            "  wärmer · kühler · blasser        – Farbstimmung\n"
            "  weichzeichnen · schärfen         – Detailgrad\n"
            "  entrauschen · invertieren        – Rauschen, Negativ\n"
            "  füllen mit rot/blau/schwarz …    – Fläche einfärben\n"
            "Stärke: „leicht“, „etwas“, „stark“, „sehr“.\n"
            "Leeres Feld bei markiertem Bereich = entfernen."
        )
    return (
        "What the local mode understands:\n"
        "  remove / erase / clean up        - inpaint the brushed area (neural network)\n"
        "  brighter · darker · contrast     - brightness and contrast\n"
        "  black and white · sepia · vivid  - colour treatment\n"
        "  warmer · cooler · muted          - colour mood\n"
        "  blur · sharpen                   - detail\n"
        "  denoise · invert                 - noise, negative\n"
        "  fill with red/blue/black …       - flat colour\n"
        "Strength: \"slightly\", \"a bit\", \"strongly\", \"very\".\n"
        "An empty box with a brushed area means remove."
    )
