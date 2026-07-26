"""
Story -> cover/scene image, via OpenAI's image generation API.

Two-step process, not one call: the story text is often in a native
Indic script and is prose, not an image prompt. A cheap chat completion
first distills the story's single strongest visual moment into a vivid,
concrete ENGLISH image-generation prompt (image models follow English
prompts far more reliably than being fed raw native-script narrative
text directly), then that prompt goes to the actual image model.

Opt-in only (a real cost/latency commitment per image, unlike the cheap
text classification calls elsewhere in this service) - see app.py's
`visual` request field. Failure is NOT silently swallowed the way
character extraction is: the caller explicitly asked for a visual, so a
failure should be visible in the response, not silently dropped.
"""
import base64
import logging
import os

import httpx

logger = logging.getLogger("indic_tts.image_gen")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_PROMPT_MODEL = os.environ.get("OPENAI_STORY_MODEL", "gpt-4o")
OPENAI_IMAGES_URL = "https://api.openai.com/v1/images/generations"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

IMAGE_SIZE = "1024x1024"


class ImageGenError(Exception):
    pass


def _derive_visual_prompt(story_text: str) -> str:
    if not OPENAI_API_KEY:
        raise ImageGenError("OPENAI_API_KEY not configured on this service")
    system_prompt = (
        "Read this short story (it may be written in a non-English "
        "script). Write ONE vivid, concrete image-generation prompt in "
        "English describing its single strongest visual moment - the "
        "scene, characters' appearance and action, mood, lighting, art "
        "style (cinematic, painterly - your choice, whatever fits the "
        "story's tone). Keep it under 300 characters. Output ONLY the "
        "prompt text itself, nothing else - no preamble, no quotes."
    )
    try:
        resp = httpx.post(
            OPENAI_CHAT_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": OPENAI_PROMPT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": story_text[:2000]},
                ],
                "max_tokens": 150,
                "temperature": 0.7,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        prompt = resp.json()["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, IndexError) as e:
        logger.error("visual prompt derivation failed: %s", e)
        raise ImageGenError(f"Visual prompt derivation failed: {e}")
    if not prompt:
        raise ImageGenError("Visual prompt derivation returned empty text")
    return prompt


def _generate_image_from_prompt(prompt: str) -> bytes:
    """Shared OpenAI Images call - takes an already-English prompt and
    returns PNG bytes. Raises ImageGenError on failure."""
    if not OPENAI_API_KEY:
        raise ImageGenError("OPENAI_API_KEY not configured on this service")
    try:
        resp = httpx.post(
            OPENAI_IMAGES_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": OPENAI_IMAGE_MODEL,
                "prompt": prompt,
                "size": IMAGE_SIZE,
                "n": 1,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        b64 = resp.json()["data"][0]["b64_json"]
    except (httpx.HTTPError, KeyError, IndexError) as e:
        logger.error("image generation failed: %s", e)
        raise ImageGenError(f"Image generation failed: {e}")

    try:
        return base64.b64decode(b64)
    except (ValueError, TypeError) as e:
        raise ImageGenError(f"Image response was not valid base64: {e}")


def generate_story_image(story_text: str, title: str | None = None) -> bytes:
    """Returns PNG bytes depicting the story's strongest visual moment.
    Raises ImageGenError on failure (missing key, network error,
    unexpected response) - this is opt-in, so the caller should see the
    failure rather than have it silently dropped.

    When `title` is given, the prompt asks the image model to bake it in
    as poster-style typography (see the instruction appended below) -
    like a real movie/audiobook cover, not just an illustration. This is
    best-effort: image-model text rendering, especially for non-Latin
    Indic scripts, is imprecise and sometimes garbled, unlike a
    guaranteed-legible HTML/CSS overlay. That's why the UI still draws
    the same title as a scrim over the image itself (see ui/index.html's
    CoverArt component) regardless of how this turns out - this is a
    visual flourish on top of that guarantee, not a replacement for it."""
    visual_prompt = _derive_visual_prompt(story_text)
    if title:
        visual_prompt = (
            f"{visual_prompt} Compose it like a movie/audiobook cover: "
            f"near the bottom, render the exact title text \"{title}\" in "
            f"bold, elegant poster typography (in its own script) that "
            f"fits the scene's mood and lighting."
        )
    return _generate_image_from_prompt(visual_prompt)


def generate_character_avatar(name: str, personality: str, backstory: str) -> bytes:
    """Returns PNG bytes: a portrait-style avatar for a saved character,
    built directly from their name/personality/backstory. No extra
    distillation call first (unlike generate_story_image) - these fields
    are already short and structured (see story_gen.extract_characters),
    not raw prose that needs summarizing into a visual moment. Raises
    ImageGenError on failure - opt-in, see app.py's `avatars` request
    field."""
    prompt = (
        f"A character portrait, head-and-shoulders, cinematic "
        f"illustration style, no text or writing in the image. "
        f"Character: {name}. Personality: {personality}. "
        f"Context: {backstory}"
    )
    return _generate_image_from_prompt(prompt[:900])
