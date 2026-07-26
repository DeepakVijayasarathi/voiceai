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

# Real, specific regional cultural touchstones per language, across five
# dimensions - not mandatory set-dressing for every story, but something
# authentic for the model to reach for instead of generic invented
# stand-ins. A Tamil story set in a temple town should land on Madurai's
# Meenakshi Amman Temple, not an invented one; a Tamil horror story
# should be able to reach for actual Yakshi/Muni folklore rather than a
# generic "demon"; a new character introduced in a Bengali story should
# plausibly be named "Ananya", not "Anna". Religious content (devotional
# deities, Sikh Golden Temple reverence) is listed factually, for a
# writer to reference respectfully in context - not for parody or
# irreverent treatment.
#
#   places      - real landmarks, for scenes that need a setting
#   festivals   - real festivals, for scenes that need a time/occasion
#   folklore    - real regional folk-spirit/legend motifs, for
#                 horror/mystery genres specifically (see emotion.py's
#                 genre classification) instead of generic invented ones
#   deities     - real devotional figures/sites, for devotional-genre
#                 stories specifically
#   names       - common authentic first names, for new characters a
#                 story introduces
LANGUAGE_CULTURAL_CONTEXT = {
    "tam": {
        "places": "Madurai and the Meenakshi Amman Temple, Rameswaram, Thanjavur's Brihadeeswarar Temple, Chennai, the hills of Ooty",
        "festivals": "Pongal, Puthandu (Tamil New Year), Karthigai Deepam",
        "folklore": "Yakshi/spirit lore, guardian-spirit (Muni) shrines, the Madurai Veeran legend",
        "deities": "Murugan (Palani, Tiruchendur), Meenakshi Amman, Mariamman",
        "names": "Karthik, Arun, Vikram, Kannan, Meena, Priya, Lakshmi",
    },
    "hin": {
        "places": "Varanasi's ghats on the Ganga, Delhi's Red Fort, Agra's Taj Mahal, Mathura-Vrindavan, Rishikesh",
        "festivals": "Diwali, Holi, Navratri, Chhath Puja",
        "folklore": "Vikram-Betaal tales, Chudail/Bhoot ghost lore",
        "deities": "Krishna (Mathura-Vrindavan), Shiva (Varanasi), Rama (Ayodhya)",
        "names": "Rahul, Arjun, Rohan, Priya, Meera, Anjali",
    },
    "tel": {
        "places": "Tirupati's Venkateswara Temple, Hyderabad's Charminar, Vijayawada, Amaravati",
        "festivals": "Ugadi, Sankranti, Bathukamma",
        "folklore": "Katamaraju Kathalu legends, local Bhoothu ghost lore",
        "deities": "Venkateswara (Tirupati), Kanaka Durga (Vijayawada)",
        "names": "Ravi, Krishna, Venkat, Lakshmi, Padma",
    },
    "mal": {
        "places": "the Kerala backwaters near Alleppey, Kochi, Guruvayur Temple, the hills of Munnar, Sabarimala",
        "festivals": "Onam, Vishu, Thrissur Pooram",
        "folklore": "Yakshi legends, Theyyam ritual-spirit performances, Kadamattathu Kathanar tales",
        "deities": "Ayyappan (Sabarimala), Guruvayurappan",
        "names": "Arjun, Krishnan, Nandu, Anjali, Devika",
    },
    "kan": {
        "places": "Mysore Palace, the ruins of Hampi, Bengaluru, Udupi's Krishna Temple, Coorg's hills",
        "festivals": "Ugadi, Mysuru Dasara",
        "folklore": "Hampi ruin legends, local spirit lore",
        "deities": "Krishna (Udupi), Chamundeshwari (Mysore)",
        "names": "Vikram, Raju, Lakshmi, Radha",
    },
    "mar": {
        "places": "Shirdi, the Ajanta-Ellora cave temples, Pune, Mumbai, the Konkan coast",
        "festivals": "Ganesh Chaturthi, Gudi Padwa",
        "folklore": "Vetal tales, Konkan coast Zoting/Munjya ghost lore",
        "deities": "Vitthal (Pandharpur), Sai Baba (Shirdi)",
        "names": "Sai, Aditya, Priya, Sneha",
    },
    "guj": {
        "places": "Dwarka's Dwarkadhish Temple, the Somnath Temple, the Rann of Kutch, Ahmedabad",
        "festivals": "Navratri (Garba), Uttarayan (kite festival)",
        "folklore": "Rann of Kutch desert legends",
        "deities": "Krishna (Dwarka), Shiva (Somnath)",
        "names": "Krishna, Jay, Meera, Kavya",
    },
    "ben": {
        "places": "Kolkata's Howrah Bridge, the Sundarbans, Dakshineswar Kali Temple, Shantiniketan",
        "festivals": "Durga Puja, Poila Boishakh (Bengali New Year), Kali Puja",
        "folklore": "Petni/Shakchunni ghost lore, Thakurmar Jhuli folk tales",
        "deities": "Kali (Dakshineswar), Durga",
        "names": "Arjun, Rohan, Priya, Ananya",
    },
    "pan": {
        "places": "Amritsar's Golden Temple, Punjab's wheat fields, Anandpur Sahib",
        "festivals": "Baisakhi, Lohri",
        "folklore": "the Heer-Ranjha legend",
        "deities": "",  # Sikh tradition centers on the Guru Granth Sahib/gurdwara reverence, not a devotional-deity-genre framing - left out rather than forced.
        "names": "Simran, Gurpreet, Harpreet, Arjun",
    },
    "ory": {
        "places": "Puri's Jagannath Temple, the Konark Sun Temple, Chilika Lake, Bhubaneswar",
        "festivals": "Rath Yatra, Nuakhai",
        "folklore": "local coastal/river spirit lore",
        "deities": "Jagannath (Puri)",
        "names": "Subhash, Debasish, Priya",
    },
    "asm": {
        "places": "Guwahati's Kamakhya Temple, the Brahmaputra river, Kaziranga, Assam's tea gardens",
        "festivals": "Bihu",
        "folklore": "Brahmaputra river legends",
        "deities": "Kamakhya (Guwahati)",
        "names": "Bikram, Dipankar, Priya",
    },
}


def _cultural_clause(language: str, *, include_names: bool = True) -> str:
    """Builds a soft, contextual cultural-grounding instruction from
    LANGUAGE_CULTURAL_CONTEXT - places for any story that needs a
    setting, folklore for horror/mystery, deities for devotional, plus
    (optionally) authentic names for new characters. Genre isn't known
    in advance here (classification happens after generation - see
    emotion.py), so all dimensions are offered together and the model is
    told explicitly to only use what fits, not force every category in."""
    ctx = LANGUAGE_CULTURAL_CONTEXT.get(language)
    if not ctx:
        return ""
    parts = [
        f"places ({ctx['places']})",
        f"festivals, if the scene calls for one ({ctx['festivals']})",
        f"regional folklore/spirit motifs, specifically for horror or mystery content, instead of a generic invented spirit ({ctx['folklore']})",
    ]
    if ctx["deities"]:
        parts.append(f"devotional figures/sites, specifically for devotional content, referenced respectfully ({ctx['deities']})")
    if include_names:
        parts.append(f"common authentic first names for any NEW character the story introduces, instead of a generic or foreign-sounding invented one ({ctx['names']})")
    options = "; ".join(parts)
    return (
        f" Where it genuinely fits, prefer real, specific regional touchstones over invented or generic "
        f"ones - {options}. Use only what fits the story's actual content and genre; "
        f"don't force any category in, and don't relocate or reshape an abstract, fantastical, "
        f"or otherwise-located story just to include one."
    )


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


# A scaled-down three-act shape for short-form narration: establish
# scene/stakes fast, build to a real turn (the emotional/narrative peak,
# not just the next event in sequence), land on a resolution or a
# resonant final image. This isn't just prose quality for its own sake -
# it has a direct downstream effect: app.py's scene classification
# (emotion.py's detect_scenes_batch) tags each chunk's genre/bgm_intensity
# independently, so a story that actually HAS a dramatic arc gives that
# classifier real shifts to detect (calm setup -> rising tension ->
# climax -> resolution), which is what makes the scene-aware music/SFX
# meaningfully dynamic instead of one flat mood for a flatly-recounted
# sequence of events.
_ARC_CLAUSE = (
    " Shape it with a real dramatic arc even at this short length: "
    "establish the scene and stakes quickly, build toward a genuine turn "
    "or complication - the emotional peak, not just the next thing that "
    "happens - and land on a resolution or a resonant final image. Don't "
    "flatly recount a sequence of events with no rising tension."
)


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
        f"no markdown, no quotation marks around it.{_ARC_CLAUSE}{_cultural_clause(language)}"
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
        f"characters. No title, no markdown, no notes. Build this continuation "
        f"toward its own real turn and resolution rather than just narrating "
        f"what happens next in flat sequence.{_cultural_clause(language)}"
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
        f"characters. No title, no markdown, no notes. The user's NEW SETTING "
        f"below is the actual setting to use - only reach for a regional "
        f"landmark instead if NEW SETTING is vague/unspecified about "
        f"place.{_cultural_clause(language)}"
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
        f"protagonist.{_cultural_clause(language)} Respond as JSON only, no "
        f"markdown fences: "
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
