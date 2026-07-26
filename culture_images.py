"""
One representative image per language's cultural context, via OpenAI's
image generation API - lazily generated on first request, then cached to
disk for the life of the process (and across restarts, since the cache
lives on disk under STATIC_DIR). Built from the EFFECTIVE cultural context
(see story_gen.get_effective_cultural_context) - the built-in landmarks in
story_gen.LANGUAGE_CULTURAL_CONTEXT, or an admin's edited override if one
exists (see db.save_culture_pack_override, PUT /culture/{language}) - so a
Tamil image lands on Madurai's Meenakshi Amman Temple by default, or
whatever an editor has since replaced that with, not a placeless "Indian
temple." Editing a language's `places` invalidates its cached image (see
invalidate_culture_image) since the image is derived from that field.
"""
import os

from image_gen import _generate_image_from_prompt
from story_gen import LANGUAGE_NAMES, get_effective_cultural_context

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

    ctx = get_effective_cultural_context(language)
    if ctx is None:
        raise KeyError(language)
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


def invalidate_culture_image(language: str) -> None:
    """Deletes `language`'s cached image, if any, so the next
    get_culture_image call regenerates it from the current effective
    context - call this after an edit that changes `places` (see PUT
    /culture/{language})."""
    path = _cache_path(language)
    if os.path.isfile(path):
        os.remove(path)
