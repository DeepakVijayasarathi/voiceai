"""
One representative image per language's cultural context, via OpenAI's
image generation API - lazily generated on first request, then cached to
disk for the life of the process (and across restarts, since the cache
lives on disk under STATIC_DIR). Reuses the same real, specific landmarks
already curated in story_gen.LANGUAGE_CULTURAL_CONTEXT rather than an
invented generic scene per language - a Tamil image should land on
Madurai's Meenakshi Amman Temple, not a placeless "Indian temple."
"""
import os

from image_gen import _generate_image_from_prompt
from story_gen import LANGUAGE_CULTURAL_CONTEXT, LANGUAGE_NAMES

STATIC_DIR = os.environ.get("INDIC_TTS_STATIC_DIR", "/root/indic-tts-fast/api/static")


def _cache_path(language: str) -> str:
    return os.path.join(STATIC_DIR, "culture", f"{language}.png")


def get_culture_image(language: str) -> bytes:
    """Returns PNG bytes for `language`'s representative cultural image -
    served from an on-disk cache after the first successful generation, so
    repeat requests (and every future process restart) don't re-spend an
    OpenAI image call. Raises KeyError for an unknown language code,
    image_gen.ImageGenError if generation fails and no cached copy exists
    yet (a transient failure just means the next request tries again -
    nothing partial is cached)."""
    path = _cache_path(language)
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return f.read()

    ctx = LANGUAGE_CULTURAL_CONTEXT[language]  # KeyError propagates for an unknown language
    lang_name = LANGUAGE_NAMES.get(language, language)
    prompt = (
        f"A vivid, cinematic illustration representing {lang_name} culture in India, "
        f"depicting {ctx['places']}. No text or writing in the image."
    )[:900]
    image_bytes = _generate_image_from_prompt(prompt)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(image_bytes)
    return image_bytes
