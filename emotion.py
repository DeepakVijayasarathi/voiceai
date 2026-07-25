"""
Auto emotion/genre detection from script text, via OpenAI's chat
completions API.

This is a classification call, not a creative one: low max_tokens, low
temperature, and a constrained output (must be one of our known presets)
so a malformed/unexpected response degrades to "neutral" rather than
breaking synthesis.

The category set doubles as both a speech-prosody hint (speed, see
app.py's EMOTION_PRESETS) and a background-music genre selector (see
bgm.py's GENRES) - basic emotions (happy/sad/etc.) plus story genres
(horror/romance/action/comedy/devotional/mystery) so narration content
gets a genre-appropriate feel, not just a mood.
"""
import json
import logging
import os

import httpx

from sfx import SFX_TYPES

logger = logging.getLogger("indic_tts.emotion")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_EMOTION_MODEL", "gpt-4o")
OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"

VALID_EMOTIONS = (
    "neutral", "happy", "sad", "excited", "calm",
    "horror", "romance", "action", "comedy", "devotional", "mystery",
)

_client = httpx.Client(timeout=15.0)

_SYSTEM_PROMPT = (
    "You will receive text in various Indian languages (Tamil, Hindi, "
    "Telugu, Malayalam, Kannada, Marathi, Gujarati, Bengali, Punjabi, "
    "Odia, Assamese) or English. Analyze its emotional tone AND narrative "
    "genre regardless of language, for text-to-speech narration with "
    "genre-matched background music. Pick whichever single category best "
    "fits the overall scene - prefer a specific genre (horror, romance, "
    "action, comedy, devotional, mystery) over a generic emotion if the "
    "content clearly fits one. Respond with exactly one word, no "
    "punctuation, no explanation: neutral, happy, sad, excited, calm, "
    "horror, romance, action, comedy, devotional, or mystery."
)


def detect_emotion(text: str) -> str:
    """Returns one of VALID_EMOTIONS. Falls back to 'neutral' on any
    failure (missing key, network error, unexpected response) rather than
    failing the TTS request over a classification hiccup."""
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set - defaulting emotion to neutral")
        return "neutral"

    try:
        resp = _client.post(
            OPENAI_BASE_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text[:2000]},
                ],
                "max_tokens": 5,
                "temperature": 0,
            },
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip().lower()
        raw = raw.strip(".,!?\"' ")
        if raw in VALID_EMOTIONS:
            return raw
        logger.warning("unexpected emotion classification output: %r", raw)
        return "neutral"
    except (httpx.HTTPError, KeyError, IndexError) as e:
        logger.warning("emotion detection failed, defaulting to neutral: %s", e)
        return "neutral"


def _batch_system_prompt(known_characters: list[str]) -> str:
    speaker_clause = ""
    if known_characters:
        names = ", ".join(known_characters)
        speaker_clause = (
            f' Known characters in this story: {names}. Also tag "speaker": '
            f'if a segment is clearly a specific one of these characters '
            f'speaking (a direct quote attributed to them, not narration '
            f'about them), use their exact name; otherwise use "narrator". '
            f"Don't guess - use \"narrator\" whenever it's ambiguous."
        )
    return (
        "You will receive a JSON array of story segments in reading order, in "
        "various Indian languages or English. This is one continuous story "
        "being narrated with background music and environmental sound effects "
        "- classify EACH segment's own mood/genre AND any environmental sound "
        "cue independently, so both can shift scene-by-scene as the story "
        "unfolds (e.g. a calm opening, then a tense reveal, then a resolution "
        "should get three different genre labels, not one for everything). "
        "Genre categories: neutral, happy, sad, excited, calm, horror, "
        "romance, action, comedy, devotional, mystery. Also give "
        "\"bgm_intensity\" (0.0-1.0) for how dramatically prominent the "
        "MUSIC itself should be in this segment - a quiet transitional "
        "moment or gentle scene should be low (e.g. 0.3), a climactic, "
        "tense, or high-drama beat should be high (e.g. 0.9), independent "
        "of which genre was picked (the same genre can still swell or "
        "recede across different segments). "
        f"SFX categories: {', '.join(SFX_TYPES)} - pick whichever fits best if "
        "the segment's text actually describes or clearly implies that sound "
        "happening in the scene (e.g. rain/storm -> rain, a car/vehicle -> car, "
        "a busy street -> traffic, a market/gathering -> crowd, a door opening "
        "-> door_creak, something shattering -> glass_break, a river/stream -> "
        "water, an animal -> dog_bark if it fits, forest/garden -> birds); "
        "otherwise use 'none'. Don't force a match - 'none' is correct far "
        "more often than not. Also give \"sfx_intensity\" (0.0-1.0) for how "
        "prominent that SOUND EFFECT should be in the scene (a light drizzle "
        "mentioned in passing = low, e.g. 0.3; a violent storm the scene "
        "centers on = high, e.g. 0.9) - irrelevant/ignored when sfx is "
        f"'none'.{speaker_clause} "
        "CRITICAL: the output array must have EXACTLY the same number of "
        "elements as the input array, in the same order, one classification "
        "per input element - even if a single input element's text clearly "
        "spans multiple scenes or sounds, pick the ONE genre and ONE sfx that "
        "best represents that whole element rather than splitting it into "
        "more output elements than were given as input. Never add or remove "
        "array elements. "
        'Respond with ONLY a JSON array of objects, same length and order as '
        'the input, no markdown fences, no other text: '
        '[{"genre": "...", "bgm_intensity": 0.0, "sfx": "...", "sfx_intensity": 0.0, "speaker": "narrator"}]'
    )


def _normalize_item(item, known_characters: list[str]) -> dict:
    default = {"genre": "neutral", "bgm_intensity": 0.6, "sfx": "none", "sfx_intensity": 0.6, "speaker": "narrator"}
    if not isinstance(item, dict):
        return default
    genre = item.get("genre", "").lower() if isinstance(item.get("genre"), str) else ""
    sfx = item.get("sfx", "").lower() if isinstance(item.get("sfx"), str) else ""

    def _clamped_float(value, fallback):
        return min(max(float(value), 0.0), 1.0) if isinstance(value, (int, float)) else fallback

    speaker = item.get("speaker") if isinstance(item.get("speaker"), str) else "narrator"
    return {
        "genre": genre if genre in VALID_EMOTIONS else "neutral",
        "bgm_intensity": _clamped_float(item.get("bgm_intensity"), 0.6),
        "sfx": sfx if sfx in SFX_TYPES else "none",
        "sfx_intensity": _clamped_float(item.get("sfx_intensity"), 0.6),
        "speaker": speaker if speaker in known_characters else "narrator",
    }


def _reconcile_length(items: list, n_expected: int, known_characters: list[str]) -> list[dict]:
    """Handles the case where the model ignores the 'same length as input'
    instruction (observed in practice: a single input element whose text
    clearly spans multiple scenes sometimes gets split into several output
    elements anyway). Rather than discard a real, usable classification
    over a count mismatch, proportionally maps the M returned items across
    the N expected chunks and picks the first valid genre / first non-none
    sfx within each chunk's share - still uses real model output instead
    of blanket-falling-back to neutral/none."""
    normalized = [_normalize_item(item, known_characters) for item in items]
    if not normalized:
        return [
            {"genre": "neutral", "bgm_intensity": 0.6, "sfx": "none", "sfx_intensity": 0.6, "speaker": "narrator"}
            for _ in range(n_expected)
        ]

    results = []
    for i in range(n_expected):
        lo = int(i * len(normalized) / n_expected)
        hi = max(lo + 1, int((i + 1) * len(normalized) / n_expected))
        bucket = normalized[lo:hi]
        genre_item = next((b for b in bucket if b["genre"] != "neutral"), bucket[0])
        sfx_item = next((b for b in bucket if b["sfx"] != "none"), bucket[0])
        speaker = next((b["speaker"] for b in bucket if b["speaker"] != "narrator"), "narrator")
        results.append({
            "genre": genre_item["genre"],
            "bgm_intensity": genre_item["bgm_intensity"],
            "sfx": sfx_item["sfx"],
            "sfx_intensity": sfx_item["sfx_intensity"],
            "speaker": speaker,
        })
    return results


def detect_scenes_batch(texts: list[str], known_characters: list[str] | None = None) -> list[dict]:
    """Scene-aware classification: for each of `texts` (typically the same
    sentence-grouped chunks used for chunked TTS synthesis), independently
    classifies mood/genre plus intensity (background music), an
    environmental sound cue plus intensity (layered SFX track), and - when
    `known_characters` is given - which of those characters (if any) is
    speaking that segment, so their audio can get a distinguishing voice
    treatment (see voice_profile.py). One batched API call covers every
    chunk, not one call per chunk. Falls back to neutral/none/narrator
    per-item only when the API call itself fails outright; a count
    mismatch in an otherwise-valid response is reconciled (see
    _reconcile_length) rather than discarded. Always returns a list the
    same length as `texts`."""
    known_characters = known_characters or []
    fallback = [
        {"genre": "neutral", "bgm_intensity": 0.6, "sfx": "none", "sfx_intensity": 0.6, "speaker": "narrator"}
        for _ in texts
    ]
    if not texts:
        return []
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set - defaulting all segments to neutral/none/narrator")
        return fallback

    try:
        resp = _client.post(
            OPENAI_BASE_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": _batch_system_prompt(known_characters)},
                    {"role": "user", "content": json.dumps([t[:500] for t in texts], ensure_ascii=False)},
                ],
                # Speaker names in known_characters may be non-Latin script,
                # which costs more tokens per character than English field
                # values - budget generously so responses don't truncate
                # mid-string (observed: "Unterminated string" JSON errors
                # under a tighter budget with Devanagari character names).
                "max_tokens": 80 * len(texts) + 80,
                "temperature": 0,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        items = json.loads(raw)
        if not isinstance(items, list) or not items:
            logger.warning("batch scene classification returned unusable output, using fallback")
            return fallback
        if len(items) != len(texts):
            logger.warning(
                "batch scene classification returned %d items for %d inputs, reconciling",
                len(items), len(texts),
            )
            return _reconcile_length(items, len(texts), known_characters)
        return [_normalize_item(item, known_characters) for item in items]
    except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
        logger.warning("batch scene detection failed, defaulting all segments to neutral/none/narrator: %s", e)
        return fallback
