"""
Text-level quality scoring for AI-generated story narration, plus a single
automatic rewrite attempt when a story scores low.

Scope: this scores the generated TEXT only, via a second, independent
OpenAI call - it does NOT attempt to judge the synthesized AUDIO's
pronunciation/prosody, since nothing on this CPU-only box can evaluate
audio quality (see engine.py's docstring on why this service exists in the
first place). "A second model" here means a second, separate judging call
to the same OpenAI model generation used, not a different provider - no
other model on this box is suited to this kind of open-ended judgment.

Non-fatal throughout, same posture as emotion.py: a scoring or rewrite
failure never blocks story generation, it just means the story goes out
unscored/un-rewritten.
"""
import json
import logging
import os

import httpx

from story_gen import LANGUAGE_NAMES, StoryGenError, _chat_completion

logger = logging.getLogger("indic_tts.quality")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_QUALITY_MODEL", os.environ.get("OPENAI_STORY_MODEL", "gpt-4o"))
OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"

QUALITY_AXES = ("native_sounding", "tone_appropriateness", "cultural_accuracy", "emotional_delivery")
QUALITY_THRESHOLD = int(os.environ.get("INDIC_TTS_QUALITY_THRESHOLD", "65"))

_client = httpx.Client(timeout=20.0)

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a strict quality judge for AI-generated {lang} audio-drama "
    "narration. Score the text on each of these axes, 0-100: "
    '"native_sounding" (does the phrasing read like a native speaker wrote '
    'it, not a stiff translation), "tone_appropriateness" (does the '
    "register/tone fit who's speaking and the scene), \"cultural_accuracy\" "
    "(are any named places/festivals/customs actually correct and not "
    'misused), "emotional_delivery" (does the writing actually land the '
    "emotion the scene calls for). Respond with ONLY a JSON object, no "
    'markdown fences, no other text: {{"native_sounding": 0, '
    '"tone_appropriateness": 0, "cultural_accuracy": 0, "emotional_delivery": 0}}'
)


def _fallback_scores() -> dict:
    scores = {axis: 100 for axis in QUALITY_AXES}
    scores["overall"] = 100
    scores["scored"] = False
    return scores


def score_story(text: str, language: str) -> dict:
    """Scores `text` 0-100 on each of QUALITY_AXES via an independent
    OpenAI judging call, plus "overall" (the minimum of the four axes - a
    story with one badly-failing axis isn't "good on average"). Falls back
    to an all-100 "unscored" sentinel (scored=False) on any failure -
    missing key, network error, unparseable response - so a judging
    hiccup never blocks or degrades a story that was never actually
    evaluated, same posture as emotion.detect_emotion's neutral fallback."""
    if not OPENAI_API_KEY:
        return _fallback_scores()
    lang_name = LANGUAGE_NAMES.get(language, language)
    try:
        resp = _client.post(
            OPENAI_BASE_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT_TEMPLATE.format(lang=lang_name)},
                    {"role": "user", "content": text[:2000]},
                ],
                "max_tokens": 100,
                "temperature": 0,
            },
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return _fallback_scores()
        result = {}
        for axis in QUALITY_AXES:
            value = parsed.get(axis)
            result[axis] = min(max(int(value), 0), 100) if isinstance(value, (int, float)) else 100
        result["overall"] = min(result[axis] for axis in QUALITY_AXES)
        result["scored"] = True
        return result
    except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("quality scoring failed, defaulting to unscored: %s", e)
        return _fallback_scores()


def rewrite_story(text: str, scores: dict, language: str) -> str:
    """One rewrite attempt addressing whichever axes scored below
    QUALITY_THRESHOLD, reusing story_gen's own chat-completion call. Raises
    StoryGenError on failure, same as any other generation call in this
    service - callers should catch this and keep the original text."""
    lang_name = LANGUAGE_NAMES.get(language, language)
    weak_axes = [axis for axis in QUALITY_AXES if scores.get(axis, 100) < QUALITY_THRESHOLD]
    weak_note = ", ".join(a.replace("_", " ") for a in weak_axes) or "overall quality"
    system_prompt = (
        f"Rewrite the following {lang_name} audio-drama narration to "
        f"specifically improve its {weak_note}, while keeping the same "
        f"plot, characters, and roughly the same length. Write only the "
        f"rewritten narrative itself - no title, no notes, no markdown, no "
        f"quotation marks around it."
    )
    return _chat_completion(system_prompt, text, max_tokens=500, temperature=0.8)[:950]


def score_and_maybe_rewrite(text: str, language: str) -> tuple[str, dict]:
    """Scores `text`; if it's actually scored and its overall is below
    QUALITY_THRESHOLD, attempts ONE rewrite addressing the weak axes,
    re-scores the rewrite, and keeps whichever version scored higher.
    Never raises - a failed rewrite attempt just keeps the original text
    and its (low) score. Returns (final_text, final_scores), where
    final_scores also carries a "rewrite_attempted" bool so callers can
    tell a genuinely-good first draft apart from a rewrite that still
    didn't clear the bar."""
    scores = score_story(text, language)
    scores["rewrite_attempted"] = False
    if not scores.get("scored") or scores["overall"] >= QUALITY_THRESHOLD:
        return text, scores
    scores["rewrite_attempted"] = True
    try:
        rewritten_text = rewrite_story(text, scores, language)
        rewritten_scores = score_story(rewritten_text, language)
        rewritten_scores["rewrite_attempted"] = True
        if rewritten_scores.get("scored") and rewritten_scores["overall"] > scores["overall"]:
            return rewritten_text, rewritten_scores
    except StoryGenError as e:
        logger.warning("quality rewrite failed, keeping original text: %s", e)
    return text, scores
