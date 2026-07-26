"""
Dream/memory description -> short narrative, via OpenAI. Also: character
extraction, story branching/continuation, character revival into a new
setting, personalized-villain generation, and language adaptation - the
real, scoped pieces of the "AI native storytelling" wishlist that don't
need infrastructure this service doesn't have (see db.py's module
docstring for what's explicitly NOT attempted here and why).

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

# Real, specific regional cultural touchstones per language, across six
# dimensions - not mandatory set-dressing for every story, but something
# authentic for the model to reach for instead of generic invented
# stand-ins. A Tamil story set in a temple town should land on Madurai's
# Meenakshi Amman Temple, not an invented one; a Tamil horror story
# should be able to reach for actual Yakshi/Muni folklore rather than a
# generic "demon"; a new character introduced in a Bengali story should
# plausibly be named "Ananya", not "Anna"; dialogue should use the
# register a real speaker of that language would use, not a translated
# one. Religious content (devotional deities, Sikh Golden Temple
# reverence) is listed factually, for a writer to reference respectfully
# in context - not for parody or irreverent treatment.
#
#   places        - real landmarks, for scenes that need a setting
#   festivals     - real festivals, for scenes that need a time/occasion
#   folklore      - real regional folk-spirit/legend motifs, for
#                   horror/mystery genres specifically (see emotion.py's
#                   genre classification) instead of generic invented ones
#   deities       - real devotional figures/sites, for devotional-genre
#                   stories specifically
#   names         - common authentic first names, for new characters a
#                   story introduces
#   register_note - how address/formality actually works in that
#                   language (kinship terms, tu/tum/aap-style distinctions),
#                   so dialogue reads local rather than translated
LANGUAGE_CULTURAL_CONTEXT = {
    "tam": {
        "places": "Madurai and the Meenakshi Amman Temple, Rameswaram, Thanjavur's Brihadeeswarar Temple, Chennai, the hills of Ooty",
        "festivals": "Pongal, Puthandu (Tamil New Year), Karthigai Deepam",
        "folklore": "Yakshi/spirit lore, guardian-spirit (Muni) shrines, the Madurai Veeran legend",
        "deities": "Murugan (Palani, Tiruchendur), Meenakshi Amman, Mariamman",
        "names": "Karthik, Arun, Vikram, Kannan, Meena, Priya, Lakshmi",
        "register_note": "everyday address leans on kinship terms even for non-relatives - anna/akka for someone elder, machaan/thozhi for a close friend - rather than formal titles",
    },
    "hin": {
        "places": "Varanasi's ghats on the Ganga, Delhi's Red Fort, Agra's Taj Mahal, Mathura-Vrindavan, Rishikesh",
        "festivals": "Diwali, Holi, Navratri, Chhath Puja",
        "folklore": "Vikram-Betaal tales, Chudail/Bhoot ghost lore",
        "deities": "Krishna (Mathura-Vrindavan), Shiva (Varanasi), Rama (Ayodhya)",
        "names": "Rahul, Arjun, Rohan, Priya, Meera, Anjali",
        "register_note": "address shifts by relationship - aap for respect or elders, tum for peers/closeness, tu only for real intimacy or as a deliberate insult",
    },
    "tel": {
        "places": "Tirupati's Venkateswara Temple, Hyderabad's Charminar, Vijayawada, Amaravati",
        "festivals": "Ugadi, Sankranti, Bathukamma",
        "folklore": "Katamaraju Kathalu legends, local Bhoothu ghost lore",
        "deities": "Venkateswara (Tirupati), Kanaka Durga (Vijayawada)",
        "names": "Ravi, Krishna, Venkat, Lakshmi, Padma",
        "register_note": "polite requests often close with -andi, and elders are addressed with garu appended to their name",
    },
    "mal": {
        "places": "the Kerala backwaters near Alleppey, Kochi, Guruvayur Temple, the hills of Munnar, Sabarimala",
        "festivals": "Onam, Vishu, Thrissur Pooram",
        "folklore": "Yakshi legends, Theyyam ritual-spirit performances, Kadamattathu Kathanar tales",
        "deities": "Ayyappan (Sabarimala), Guruvayurappan",
        "names": "Arjun, Krishnan, Nandu, Anjali, Devika",
        "register_note": "kinship terms (chettan/chechi for an elder sibling) are extended to non-relatives too, as a mark of warmth rather than literal family",
    },
    "kan": {
        "places": "Mysore Palace, the ruins of Hampi, Bengaluru, Udupi's Krishna Temple, Coorg's hills",
        "festivals": "Ugadi, Mysuru Dasara",
        "folklore": "Hampi ruin legends, local spirit lore",
        "deities": "Krishna (Udupi), Chamundeshwari (Mysore)",
        "names": "Vikram, Raju, Lakshmi, Radha",
        "register_note": "-avaru appended to a name is the respectful register for elders or strangers",
    },
    "mar": {
        "places": "Shirdi, the Ajanta-Ellora cave temples, Pune, Mumbai, the Konkan coast",
        "festivals": "Ganesh Chaturthi, Gudi Padwa",
        "folklore": "Vetal tales, Konkan coast Zoting/Munjya ghost lore",
        "deities": "Vitthal (Pandharpur), Sai Baba (Shirdi)",
        "names": "Sai, Aditya, Priya, Sneha",
        "register_note": "tumhi is the respectful/plural you, tu reserved for close intimacy or addressing children",
    },
    "guj": {
        "places": "Dwarka's Dwarkadhish Temple, the Somnath Temple, the Rann of Kutch, Ahmedabad",
        "festivals": "Navratri (Garba), Uttarayan (kite festival)",
        "folklore": "Rann of Kutch desert legends",
        "deities": "Krishna (Dwarka), Shiva (Somnath)",
        "names": "Krishna, Jay, Meera, Kavya",
        "register_note": "tame is the respectful you, tu reserved for close family or friends",
    },
    "ben": {
        "places": "Kolkata's Howrah Bridge, the Sundarbans, Dakshineswar Kali Temple, Shantiniketan",
        "festivals": "Durga Puja, Poila Boishakh (Bengali New Year), Kali Puja",
        "folklore": "Petni/Shakchunni ghost lore, Thakurmar Jhuli folk tales",
        "deities": "Kali (Dakshineswar), Durga",
        "names": "Arjun, Rohan, Priya, Ananya",
        "register_note": "apni is formal/respectful, tumi is familiar, tui reserved for very close intimacy or addressing children",
    },
    "pan": {
        "places": "Amritsar's Golden Temple, Punjab's wheat fields, Anandpur Sahib",
        "festivals": "Baisakhi, Lohri",
        "folklore": "the Heer-Ranjha legend",
        "deities": "",  # Sikh tradition centers on the Guru Granth Sahib/gurdwara reverence, not a devotional-deity-genre framing - left out rather than forced.
        "names": "Simran, Gurpreet, Harpreet, Arjun",
        "register_note": "tussi is the respectful you, tu reserved for close intimacy",
    },
    "ory": {
        "places": "Puri's Jagannath Temple, the Konark Sun Temple, Chilika Lake, Bhubaneswar",
        "festivals": "Rath Yatra, Nuakhai",
        "folklore": "local coastal/river spirit lore",
        "deities": "Jagannath (Puri)",
        "names": "Subhash, Debasish, Priya",
        "register_note": "apana is the respectful address, tume the familiar one",
    },
    "asm": {
        "places": "Guwahati's Kamakhya Temple, the Brahmaputra river, Kaziranga, Assam's tea gardens",
        "festivals": "Bihu",
        "folklore": "Brahmaputra river legends",
        "deities": "Kamakhya (Guwahati)",
        "names": "Bikram, Dipankar, Priya",
        "register_note": "apuni is respectful address, tumi familiar, tai reserved for close intimacy or addressing children",
    },
}


def _cultural_clause(language: str, *, include_names: bool = True) -> str:
    """Builds a soft, contextual cultural-grounding instruction from
    LANGUAGE_CULTURAL_CONTEXT - places for any story that needs a
    setting, folklore for horror/mystery, deities for devotional, address
    register for dialogue, plus (optionally) authentic names for new
    characters. Genre isn't known in advance here (classification happens
    after generation - see emotion.py), so all dimensions are offered
    together and the model is told explicitly to only use what fits, not
    force every category in."""
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
        f"or otherwise-located story just to include one. Address between characters should feel "
        f"natural for the culture, not translated: {ctx['register_note']}."
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
        f"no markdown, no quotation marks around it.{_cultural_clause(language)}"
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


def generate_continuation(
    prior_text: str, changed_decision: str, language: str, ancestor_summaries: list[str] | None = None,
) -> str:
    """'Story Time Machine' subset: generates a new continuation from a
    user-specified changed decision, given the prior story text as
    context. When `ancestor_summaries` is given (earlier branch-point
    decisions from this story's full lineage, root-first - see db.
    get_story_lineage), they're passed as condensed context ahead of the
    direct parent's full text, so a branch-of-a-branch stays aware of the
    whole chain of changes that led here, not just the one hop directly
    above it - deliberately condensed to branch notes only (not each
    ancestor's full text) so token usage doesn't grow with lineage depth.
    This still does NOT verify consistency against anything OUTSIDE this
    lineage - a different story/branch elsewhere in the system isn't
    considered, and nothing here re-checks the ancestor summaries against
    the generated output beyond passing them as context (see db.py's
    docstring)."""
    lang_name = LANGUAGE_NAMES.get(language, language)
    system_prompt = (
        f"You are continuing an existing story in {lang_name} (native script). "
        f"The user will give you the story so far, then a changed decision at "
        f"some point in it. Write ONLY the new continuation from that changed "
        f"decision onward - do not repeat the original text, do not include "
        f"the parts before the change. Keep it under {TARGET_CHAR_BUDGET} "
        f"characters. No title, no markdown, no notes.{_cultural_clause(language)}"
    )
    lineage_note = ""
    if ancestor_summaries:
        bullets = "\n".join(f"- {s}" for s in ancestor_summaries if s)
        if bullets:
            lineage_note = (
                f"EARLIER IN THIS STORY'S HISTORY (already happened, for "
                f"continuity only - don't repeat or re-narrate these):\n{bullets}\n\n"
            )
    user_content = f"{lineage_note}STORY SO FAR:\n{prior_text}\n\nCHANGED DECISION:\n{changed_decision}"
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
        f"place.{_cultural_clause(language, include_names=False)}"
    )
    user_content = (
        f"CHARACTER: {name}\nPERSONALITY: {personality}\nBACKSTORY: {backstory}\n\n"
        f"NEW SETTING: {new_setting}"
    )
    story = _chat_completion(system_prompt, user_content, max_tokens=500, temperature=0.9)
    return story[:950]


def adapt_story_language(text: str, source_language: str, target_language: str) -> str:
    """'Switch language' subset: retells `text` naturally in
    target_language - NOT a literal/mechanical translation call. Same
    plot, same characters, same tone; character names come out phonetically
    natural for the target language/script rather than transliterated
    letter-for-letter. Scope note: this regenerates a full narration in the
    new language rather than splicing one audio stream mid-sentence - VITS
    (this service's TTS engine) is a separate model per language, so an
    actual mid-utterance language splice isn't a meaningful operation here.
    "Switching mid-story" in this system means resuming the same story,
    same characters, in a new language from a saved variant (see
    db.get_variants) - not one audio file that changes language partway
    through."""
    source_name = LANGUAGE_NAMES.get(source_language, source_language)
    target_name = LANGUAGE_NAMES.get(target_language, target_language)
    system_prompt = (
        f"The following is a short story written in {source_name}. Retell "
        f"the exact same story - same plot, same characters, same events, "
        f"same tone - naturally in {target_name} (native script), as if it "
        f"had originally been written in {target_name} rather than "
        f"translated. Character names should sound phonetically natural in "
        f"{target_name} rather than being transliterated letter-for-letter. "
        f"Keep it under {TARGET_CHAR_BUDGET} characters. Write only the "
        f"retold narrative itself - no title, no notes, no markdown, no "
        f"quotation marks around it.{_cultural_clause(target_language, include_names=False)}"
    )
    adapted = _chat_completion(system_prompt, text, max_tokens=500, temperature=0.7)
    return adapted[:950]


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
