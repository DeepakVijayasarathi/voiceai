"""
Mixed-language sentence handling, e.g. "நாளைக்கு office meeting இருக்குது."

MMS-TTS's grapheme-level tokenizer for each language only recognizes that
language's native script -- embedded Latin-script (English) words are out
of vocabulary and get silently dropped or mispronounced. We detect Latin
runs and transliterate them into the target script via the existing
IndicXlit service (ai4bharat/IndicXlit, already deployed as xlit-api on
this box) before synthesis, so the model sees native-script text only.
"""
import os
import re

import httpx

XLIT_BASE_URL = os.environ.get("XLIT_BASE_URL", "http://127.0.0.1:8501")
XLIT_API_KEY = os.environ.get("XLIT_API_KEY")

# MMS-TTS ISO 639-3 code -> IndicXlit's own language code
XLIT_LANG_MAP = {
    "hin": "hi", "mar": "mr", "ben": "bn", "guj": "gu", "ory": "or",
    "pan": "pa", "tam": "ta", "tel": "te", "kan": "kn", "mal": "ml",
    "asm": "as",
}

_LATIN_RUN_RE = re.compile(r"[A-Za-z][A-Za-z\s]*[A-Za-z]|[A-Za-z]")

# xlit-api lazily loads each language's transliteration model on first use
# (~10-15s cold start), then caches it in-process for fast subsequent
# calls - the generous timeout here only bites on that first call per
# language per xlit-api process lifetime.
_client = httpx.Client(timeout=20.0)


def _transliterate_run(text: str, lang: str) -> str:
    xlit_lang = XLIT_LANG_MAP.get(lang)
    if not xlit_lang:
        return text
    try:
        resp = _client.post(
            f"{XLIT_BASE_URL}/transliterate",
            headers={"x-api-key": XLIT_API_KEY} if XLIT_API_KEY else {},
            json={"text": text, "preset": lang},
        )
        resp.raise_for_status()
        return resp.json()["text"]
    except (httpx.HTTPError, KeyError):
        # Transliteration service unreachable/erroring - better to pass the
        # original Latin text through (silently mispronounced) than fail
        # the whole request.
        return text


def has_mixed_script(text: str, lang: str) -> bool:
    return lang != "eng" and bool(_LATIN_RUN_RE.search(text))


def resolve_mixed_language(text: str, lang: str) -> str:
    """Replace embedded Latin-script runs with their phonetic transliteration
    into the target Indic script, leaving native-script portions untouched."""
    if not has_mixed_script(text, lang):
        return text

    def _replace(match: re.Match) -> str:
        run = match.group(0).strip()
        if not run:
            return match.group(0)
        return _transliterate_run(run, lang)

    return _LATIN_RUN_RE.sub(_replace, text)
