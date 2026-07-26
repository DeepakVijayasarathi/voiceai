"""
Scene visual-prompt generation and image synthesis for narrated stories.

Pipeline: story text -> GPT splits it into a handful of visual "beats" (not
one per TTS chunk - that would be too many expensive image-generation calls
for a single story) -> each beat's excerpt is later located in the raw
synthesis chunks (see app.py's _map_beats_to_time) to give it an approximate
time range in the narrated audio -> OpenAI Images generates one picture per
beat, in English regardless of the story's own language (image models are
most reliable there) -> saved to disk, served by URL.

Best-effort throughout: a failure at any step (prompt generation, a single
image generation) drops that one beat rather than failing the whole
request - images are an enhancement layered on top of the narration, not
the point of the request that triggers them (same posture as
story_gen.extract_characters).
"""
import base64
import json
import logging
import os

import httpx

from story_gen import LANGUAGE_NAMES, StoryGenError, _chat_completion

logger = logging.getLogger("indic_tts.images")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_IMAGES_URL = "https://api.openai.com/v1/images/generations"

# Deliberately not one image per TTS chunk (a multi-minute story can have
# dozens of chunks) - a handful of illustrated beats covering the whole
# story, each holding across a stretch of narration, reads like a picture
# book / storyboard rather than needing per-sentence images.
MAX_BEATS = 6

STATIC_DIR = os.environ.get("INDIC_TTS_STATIC_DIR", "/root/indic-tts-fast/api/static")

_client = httpx.Client(timeout=60.0)


class ImageGenError(Exception):
    pass


def generate_visual_prompts(
    story_text: str, language: str, character_names: list[str] | None = None, max_beats: int = MAX_BEATS,
) -> list[dict]:
    """Splits story_text into at most max_beats visual scenes/beats, in
    reading order. Returns [{"excerpt": "...", "visual_prompt": "..."}].
    `excerpt` is a short, roughly-verbatim snippet from story_text, used
    downstream to locate where in the narration each beat begins (see
    app.py's _map_beats_to_time); `visual_prompt` is always written in
    English regardless of the story's language. Returns [] on any failure
    (non-fatal, mirrors story_gen.extract_characters)."""
    character_names = character_names or []
    lang_name = LANGUAGE_NAMES.get(language, language)
    names_clause = f" Known characters: {', '.join(character_names)}." if character_names else ""
    system_prompt = (
        f"This is a short story written in {lang_name}. Break it into at most "
        f"{max_beats} distinct visual scenes/beats, in reading order, covering "
        f"the whole story from start to end. For each beat, give a short "
        f"\"excerpt\" (a few words taken verbatim from the story, marking "
        f"roughly where that beat begins) and a vivid English-language "
        f"\"visual_prompt\" describing the scene visually (setting, who's "
        f"present, mood, lighting) for an image-generation model - cinematic "
        f"illustration style, no text/writing/captions rendered in the image "
        f"itself, keep character appearance consistent across "
        f"prompts.{names_clause} Respond with ONLY a JSON array, no markdown "
        f'fences, no other text: [{{"excerpt": "...", "visual_prompt": "..."}}]'
    )
    try:
        raw = _chat_completion(system_prompt, story_text, max_tokens=900, temperature=0.7)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        beats = json.loads(raw)
        if not isinstance(beats, list):
            return []
        results = []
        for b in beats[:max_beats]:
            if isinstance(b, dict) and b.get("excerpt") and b.get("visual_prompt"):
                results.append({"excerpt": str(b["excerpt"]), "visual_prompt": str(b["visual_prompt"])})
        return results
    except (StoryGenError, json.JSONDecodeError, AttributeError, TypeError) as e:
        logger.warning("visual prompt generation failed (non-fatal): %s", e)
        return []


def generate_scene_image(prompt: str) -> bytes:
    """Returns PNG bytes for `prompt` via OpenAI Images (OPENAI_IMAGE_MODEL,
    default gpt-image-1, which always returns b64-encoded image data).
    Raises ImageGenError on failure - callers should catch this per-image
    so one failed beat doesn't take the others down with it."""
    if not OPENAI_API_KEY:
        raise ImageGenError("OPENAI_API_KEY not configured on this service")
    try:
        resp = _client.post(
            OPENAI_IMAGES_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": OPENAI_IMAGE_MODEL, "prompt": prompt[:4000], "size": "1024x1024", "n": 1},
        )
        resp.raise_for_status()
        b64 = resp.json()["data"][0]["b64_json"]
        return base64.b64decode(b64)
    except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as e:
        raise ImageGenError(f"Image generation failed: {e}")


def save_scene_image(story_id: int, index: int, image_bytes: bytes) -> str:
    """Writes the image to STATIC_DIR/images/{story_id}/{index}.png and
    returns the URL path it's served at under app.py's /static mount."""
    story_dir = os.path.join(STATIC_DIR, "images", str(story_id))
    os.makedirs(story_dir, exist_ok=True)
    path = os.path.join(story_dir, f"{index}.png")
    with open(path, "wb") as f:
        f.write(image_bytes)
    return f"/static/images/{story_id}/{index}.png"
