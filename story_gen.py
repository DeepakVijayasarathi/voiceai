"""
Dream/memory description -> short narrative, via OpenAI. Also: character
extraction, story branching/continuation, character revival into a new
setting, and personalized-villain generation - the real, scoped pieces of
the "AI native storytelling" wishlist that don't need infrastructure this
service doesn't have (see db.py's module docstring for what's explicitly
NOT attempted here and why).

Unlike emotion.py's classification call, these are genuinely creative
generation - higher temperature, more tokens. Output feeds directly into
this service's own TTS pipeline (normalize -> mixed-language resolution
-> synthesize -> optional auto-emotion/bgm), so it's capped well under
MAX_TEXT_LEN and asked to write only in the target script so nothing
downstream needs a second translation pass.
"""
import json
import logging
import os

import httpx

from voice_profile import PROFILE_NAMES

logger = logging.getLogger("indic_tts.story_gen")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_STORY_MODEL", "gpt-4o")
OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"

# Keep real headroom under the TTS engine's MAX_TEXT_LEN (1000 chars) -
# the model doesn't count characters precisely, so ask for less than the
# hard limit and still truncate defensively afterward.
TARGET_CHAR_BUDGET = 700

LANGUAGE_NAMES = {
    "tam": "Tamil", "hin": "Hindi", "tel": "Telugu", "mal": "Malayalam",
    "kan": "Kannada", "mar": "Marathi", "guj": "Gujarati", "ben": "Bengali",
    "pan": "Punjabi", "ory": "Odia", "asm": "Assamese",
}


class StoryGenError(Exception):
    pass


def _chat_completion(system_prompt: str, user_content: str, max_tokens: int = 500, temperature: float = 0.9) -> str:
    if not OPENAI_API_KEY:
        raise StoryGenError("OPENAI_API_KEY not configured on this service")
    try:
        resp = httpx.post(
            OPENAI_BASE_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content[:2000]},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, IndexError) as e:
        logger.error("OpenAI call failed: %s", e)
        raise StoryGenError(f"Generation failed: {e}")
    if not content:
        raise StoryGenError("Generation returned empty text")
    return content


def generate_story(description: str, language: str) -> str:
    """Returns a short narrative in `language`'s native script, generated
    from the user's dream/memory description. Raises StoryGenError on
    failure - unlike emotion detection, there's no safe silent fallback
    for a request whose entire point is the generated text."""
    lang_name = LANGUAGE_NAMES.get(language, language)
    system_prompt = (
        f"You are a creative audio-drama writer. Turn the user's description "
        f"of a dream or memory into a short, vivid, cinematic narrative "
        f"suitable for spoken audio narration, written entirely in {lang_name} "
        f"(native script, not transliterated). Keep it under {TARGET_CHAR_BUDGET} "
        f"characters. Write only the narrative itself - no title, no notes, "
        f"no markdown, no quotation marks around it."
    )
    story = _chat_completion(system_prompt, description, max_tokens=500, temperature=0.9)
    return story[:950]  # hard safety margin under MAX_TEXT_LEN


def extract_characters(story_text: str, language: str) -> list[dict]:
    """Identifies named characters in a story and returns a short
    personality/backstory summary for each, plus a suggested voice_profile
    (see voice_profile.py) picked to fit that personality - a menacing
    villain and a cheerful child shouldn't be assigned arbitrarily by hash
    alone. voice_profile.assign_profiles still does the final assignment
    (resolving collisions within a story's cast); this is a preference,
    not a guarantee. Suitable for saving to db.py so characters can be
    pulled into a later story. Best-effort: returns an empty list (never
    raises) if the story has no clearly named characters or the
    classification fails - character extraction isn't the point of the
    request that calls this, story generation is."""
    profile_options = ", ".join(PROFILE_NAMES)
    system_prompt = (
        "Identify the named characters in this story (skip unnamed "
        "background figures). For each, write a 1-2 sentence personality "
        "summary and a 1-2 sentence backstory/role summary, both in "
        "English regardless of the story's language (so they're reusable "
        "across languages later). Also suggest a \"voice_profile\" that "
        f"fits their personality, one of: {profile_options} - deepest/deep "
        "suit menacing, authoritative, or older/gruff characters; "
        "bright/high/highest suit youthful, cheerful, or high-energy "
        "characters; warm/neutral_shift suit gentle or unremarkable ones. "
        "This is a preference based on personality, not a hard rule. "
        "Respond with ONLY a JSON array, no markdown fences, no other text: "
        '[{"name": "...", "personality": "...", "backstory": "...", "voice_profile": "..."}]. '
        "If there are no clearly named characters, respond with []."
    )
    try:
        raw = _chat_completion(system_prompt, story_text, max_tokens=700, temperature=0.3)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        characters = json.loads(raw)
        if not isinstance(characters, list):
            return []
        results = []
        for c in characters:
            if not (isinstance(c, dict) and c.get("name") and c.get("personality") and c.get("backstory")):
                continue
            voice_profile = c.get("voice_profile")
            c["voice_profile"] = voice_profile if voice_profile in PROFILE_NAMES else None
            results.append(c)
        return results
    except (StoryGenError, json.JSONDecodeError, AttributeError) as e:
        logger.warning("character extraction failed (non-fatal): %s", e)
        return []


def generate_continuation(prior_text: str, changed_decision: str, language: str) -> str:
    """'Story Time Machine' subset: generates a new continuation from a
    user-specified changed decision, given the prior story text as
    context. This does NOT verify consistency against anything beyond
    `prior_text` itself - if this story was previously branched from
    elsewhere, only the direct lineage passed in here is considered, not
    a full-universe canon (see db.py's docstring)."""
    lang_name = LANGUAGE_NAMES.get(language, language)
    system_prompt = (
        f"You are continuing an existing story in {lang_name} (native script). "
        f"The user will give you the story so far, then a changed decision at "
        f"some point in it. Write ONLY the new continuation from that changed "
        f"decision onward - do not repeat the original text, do not include "
        f"the parts before the change. Keep it under {TARGET_CHAR_BUDGET} "
        f"characters. No title, no markdown, no notes."
    )
    user_content = f"STORY SO FAR:\n{prior_text}\n\nCHANGED DECISION:\n{changed_decision}"
    continuation = _chat_completion(system_prompt, user_content, max_tokens=500, temperature=0.9)
    return continuation[:950]


def revive_character(name: str, personality: str, backstory: str, new_setting: str, language: str) -> str:
    """'Character Resurrection' subset: writes a new short story placing
    an existing (saved) character into a new setting/universe, briefed on
    their established personality and backstory so it stays recognizably
    them. Not verified against any other story the character may have
    appeared in beyond what's passed here."""
    lang_name = LANGUAGE_NAMES.get(language, language)
    system_prompt = (
        f"You are a creative writer placing an established character into a "
        f"new story, in {lang_name} (native script). Keep their personality "
        f"consistent with what's described. Write a short, vivid scene "
        f"suitable for audio narration. Keep it under {TARGET_CHAR_BUDGET} "
        f"characters. No title, no markdown, no notes."
    )
    user_content = (
        f"CHARACTER: {name}\nPERSONALITY: {personality}\nBACKSTORY: {backstory}\n\n"
        f"NEW SETTING: {new_setting}"
    )
    story = _chat_completion(system_prompt, user_content, max_tokens=500, temperature=0.9)
    return story[:950]


def generate_villain(fears: str, language: str) -> tuple[str, str]:
    """'Personalized Villain' subset: given USER-SPECIFIED fears/
    motivations (not learned from listener history - no such history is
    tracked anywhere in this system), generates a matching villain and a
    short scene introducing them. Returns (villain_name, scene_text)."""
    lang_name = LANGUAGE_NAMES.get(language, language)
    system_prompt = (
        f"Design a villain character whose menace specifically targets the "
        f"fears/anxieties described by the user, then write a short, vivid "
        f"scene in {lang_name} (native script) introducing this villain to a "
        f"protagonist. Respond as JSON only, no markdown fences: "
        f'{{"name": "villain name", "scene": "the scene text, under '
        f"{TARGET_CHAR_BUDGET} characters, no title, no notes\"}}."
    )
    raw = _chat_completion(system_prompt, fears, max_tokens=600, temperature=0.9)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
        name, scene = data["name"], data["scene"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise StoryGenError(f"Villain generation returned unparseable output: {e}")
    return name, scene[:950]
