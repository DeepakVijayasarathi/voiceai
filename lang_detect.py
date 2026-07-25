"""
Script-based language detection for the 11 supported Indian languages.

Each language's script maps to a contiguous Unicode block, which is a
reliable signal on its own -- except two pairs share a script (Hindi/
Marathi both use Devanagari, Bengali/Assamese both use Bengali script).
For those pairs we disambiguate using a short list of common function
words that differ between the two languages; if none match we fall back
to the more widely spoken language in the pair.
"""
import re

SCRIPT_RANGES = {
    "hin": (0x0900, 0x097F),  # Devanagari (Hindi, Marathi)
    "ben": (0x0980, 0x09FF),  # Bengali (Bengali, Assamese)
    "guj": (0x0A80, 0x0AFF),
    "pan": (0x0A00, 0x0A7F),  # Gurmukhi
    "ory": (0x0B00, 0x0B7F),
    "tam": (0x0B80, 0x0BFF),
    "tel": (0x0C00, 0x0C7F),
    "kan": (0x0C80, 0x0CFF),
    "mal": (0x0D00, 0x0D7F),
}

# Devanagari-script disambiguation: Marathi-only function words/markers.
MARATHI_MARKERS = {"आहे", "आहेत", "मी", "तू", "आम्ही", "तुम्ही", "काय", "नाही"}
HINDI_MARKERS = {"है", "हैं", "मैं", "तुम", "हम", "क्या", "नहीं", "था", "थी"}

# Bengali-script disambiguation: Assamese uses a distinct "ৰ"/"ৱ" and
# a handful of markers not used in Bengali.
ASSAMESE_MARKERS = {"আছে", "কৰি", "নাই", "মই", "তুমি", "কি"}
BENGALI_MARKERS = {"আছে", "করে", "নেই", "আমি", "তুমি", "কী"}
ASSAMESE_ONLY_CHARS = {"ৰ", "ৱ"}


def _dominant_script(text: str) -> str | None:
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lang, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
    if not counts:
        return None
    return max(counts, key=counts.get)


def detect_language(text: str) -> str:
    """Return the ISO 639-3 code (matching our MMS-TTS model keys) for the
    dominant script in `text`. Defaults to 'hin' if no Indic script is
    found (e.g. pure Latin-script/English input)."""
    script_lang = _dominant_script(text)
    if script_lang is None:
        return "hin"

    if script_lang == "hin":
        words = set(re.findall(r"[ऀ-ॿ]+", text))
        if words & MARATHI_MARKERS and not (words & HINDI_MARKERS):
            return "mar"
        return "hin"

    if script_lang == "ben":
        if any(ch in ASSAMESE_ONLY_CHARS for ch in text):
            return "asm"
        words = set(re.findall(r"[ঀ-৿]+", text))
        if words & ASSAMESE_MARKERS and not (words & BENGALI_MARKERS):
            return "asm"
        return "ben"

    return script_lang
