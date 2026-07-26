"""
Fast multilingual Indic TTS API.

Engine: facebook/mms-tts-* (VITS, ~36M params/language) - chosen specifically
for CPU-only, low-latency serving. It does NOT itself support voice cloning
or learned emotion conditioning (both need a much heavier reference-audio-
conditioned model, which conflicts directly with the speed/CPU-only
requirements this service is built for). /clone proxies cloning requests
to the separate IndicF5 service instead of reimplementing it here. Emotion
presets are a documented, approximate speed/expressiveness knob, not true
emotional TTS; emotion='auto' classifies the text's tone via OpenAI.
"""
import base64
import inspect
import io
import logging
import os
import re
import time
import wave
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

import db
from bgm import GENRES, _generate_track, get_bgm_loop, mix_with_bgm, mix_with_scene_bgm
from clone import CloneError, clone_speech
from emotion import detect_emotion, detect_scenes_batch
from engine import (
    TTSEngine,
    _DEFAULT_NOISE_SCALE,
    _DEFAULT_NOISE_SCALE_DURATION,
    _MAX_SYNTH_RETRIES,
    _MIN_CHARS_PER_SEC,
)
from lang_detect import detect_language
from mixed_lang import resolve_mixed_language
from normalize import normalize_text
from ratelimit import RateLimitMiddleware
from sfx import SFX_TYPES, get_sfx_loop
from voice_profile import PROFILE_NAMES, _PROFILES, assign_profiles, apply_voice_profile, get_speed_multiplier
from story_gen import (
    LANGUAGE_CULTURAL_CONTEXT,
    TARGET_CHAR_BUDGET,
    StoryGenError,
    extract_characters,
    generate_continuation,
    generate_story,
    generate_villain,
    revive_character,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("indic_tts.app")

API_KEY = os.environ.get("INDIC_TTS_API_KEY")
MAX_TEXT_LEN = 6000  # raised from 1000 for full-length story narration; long
# text is auto-chunked by sentence (see _synthesize_long_form) rather than
# sent to the model as one giant sequence.
CPU_THREADS = int(os.environ.get("INDIC_TTS_THREADS", "4"))

_CHUNK_CHAR_BUDGET = 300  # per-chunk synthesis size, keeps each VITS forward pass fast/reliable
# Smaller budget used specifically when known_characters is active: a
# back-and-forth exchange between two characters can otherwise land in
# the same 300-char chunk, forcing one classification (and one voice
# treatment) onto lines that belong to different speakers.
_DIALOGUE_CHUNK_CHAR_BUDGET = 120
_INTER_CHUNK_SILENCE_S = 0.3
# Matches a sentence terminator optionally followed by a closing quote
# mark, then whitespace - quoted dialogue ends like `...रहा।" राहुल...`
# where the terminator isn't immediately before the whitespace (the quote
# mark is), so a plain `(?<=[.!?।॥])\s+` lookbehind never fires there and
# a whole back-and-forth exchange silently stays one unsplit "sentence".
_SENTENCE_BOUNDARY_RE = re.compile(r'([.!?।॥][\'"’”]*)\s+')


def _split_sentences(text: str) -> list[str]:
    marked = _SENTENCE_BOUNDARY_RE.sub(r"\1\n", text)
    return [s.strip() for s in marked.split("\n") if s.strip()]

engine = TTSEngine(num_threads=CPU_THREADS)

# Crude, documented-approximate emotion/genre -> (speed, noise_scale)
# mapping. noise_scale controls the model's stochastic expressiveness -
# lower is clearer/steadier, higher is more varied/energetic. All values
# here sit at or below VITS's own default (0.667): general narration
# clarity was a specific ask, so nothing in this table trades clarity for
# expressiveness beyond the model's own baseline. This is NOT learned
# emotion conditioning, just a cheap prosody nudge. The same keys double
# as bgm.py's genre selector (see emotion.py).
EMOTION_PRESETS = {
    "neutral": (1.0, 0.55),
    "happy": (1.08, 0.65),
    "sad": (0.9, 0.5),
    "excited": (1.15, 0.65),
    "calm": (0.92, 0.4),
    "horror": (0.85, 0.55),
    "romance": (0.92, 0.5),
    "action": (1.15, 0.6),
    "comedy": (1.05, 0.6),
    "devotional": (0.85, 0.35),
    "mystery": (0.90, 0.5),
}

_metrics = {"requests_total": 0, "requests_failed": 0, "synth_seconds_total": 0.0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    logger.info("database initialized at %s", db.DB_PATH)

    t0 = time.time()
    engine.load_all()
    logger.info("all models loaded in %.2fs", time.time() - t0)

    t0 = time.time()
    for genre in GENRES:
        get_bgm_loop(genre)
    logger.info("all %d bgm genre loops pre-generated in %.2fs", len(GENRES), time.time() - t0)

    yield


app = FastAPI(title="Indic Fast TTS API", lifespan=lifespan)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=int(os.environ.get("INDIC_TTS_RATE_LIMIT_PER_MIN", "60")),
)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
    language: str | None = Field(default=None, description="ISO 639-3 code, e.g. 'hin', 'tam'. Auto-detected from script if omitted.")
    speed: float = Field(default=1.0, gt=0, le=2.5)
    emotion: str | None = Field(default=None, description="One of: " + ", ".join(EMOTION_PRESETS) + ", or 'auto' to classify from the text via OpenAI.")
    bgm: bool = Field(default=False, description="Mix in a procedurally-generated, mood-matched ambient background pad (not licensed/produced music - see /health).")
    speaker: str | None = Field(default=None, description="Reserved, unused - use POST /clone for voice cloning instead.")


class CloneRequest(BaseModel):
    # Capped much lower than MAX_TEXT_LEN - cloning proxies to IndicF5,
    # which takes 30-70s+ for even a short sentence; a 6000-char request
    # there could run for many minutes.
    text: str = Field(..., min_length=1, max_length=500)
    ref_audio_b64: str = Field(..., description="Base64-encoded reference audio clip of the voice to clone.")
    ref_text: str = Field(..., min_length=1, description="Transcript of ref_audio_b64.")


class DreamToStoryRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=2000, description="Describe a dream, memory, or scene in your own words.")
    language: str = Field(default="hin", description="ISO 639-3 code for the generated story's language, e.g. 'hin', 'tam'.")
    emotion: str | None = Field(default="auto", description="One of: " + ", ".join(EMOTION_PRESETS) + ", or 'auto' to classify from the generated story via OpenAI (default).")
    bgm: bool = Field(default=True, description="Mix in genre-matched background ambience (on by default for this endpoint - it's built for dramatic narration).")


def _check_auth(x_api_key: str | None) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _resolve_language(text: str, requested_lang: str | None) -> str:
    lang = requested_lang or detect_language(text)
    if lang not in engine.languages():
        raise HTTPException(status_code=400, detail=f"Unsupported language '{lang}'. Supported: {list(engine.languages())}")
    return lang


def _resolve_emotion_and_speed(req: TTSRequest) -> tuple[str | None, float, float | None]:
    """Returns (resolved_emotion, speed, noise_scale). 'auto' is resolved
    via OpenAI classification here so the speed/clarity heuristic and BGM
    mood selection downstream all use the same resolved value.
    noise_scale is None (engine.py's own default applies) when no emotion
    is set at all."""
    speed = req.speed
    emotion = req.emotion
    noise_scale = None

    if emotion and emotion.lower() == "auto":
        emotion = detect_emotion(req.text)
    elif emotion and emotion.lower() not in EMOTION_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown emotion '{req.emotion}'. Choices: {list(EMOTION_PRESETS)} or 'auto'")
    else:
        emotion = emotion.lower() if emotion else None

    if emotion:
        emotion_speed, noise_scale = EMOTION_PRESETS[emotion]
        speed = req.speed * emotion_speed if req.speed != 1.0 else emotion_speed

    return emotion, speed, noise_scale


def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current = ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > max_chars:
            chunks.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        chunks.append(current)
    return chunks or [text]


def _synthesize_long_form(
    text: str, lang: str, speed: float, noise_scale: float | None = None,
    bgm: bool = False, bgm_fallback: str = "neutral",
    known_characters: list[str] | None = None,
    personality_hints: dict[str, str] | None = None,
) -> tuple[np.ndarray, int]:
    """Splits long text into sentence-grouped chunks (~_CHUNK_CHAR_BUDGET
    chars each), synthesizes each separately, and concatenates with a
    short silence gap - keeps each individual VITS forward pass fast and
    avoids the reliability/memory risk of one very long single sequence,
    while still handling full-length story narration end to end.

    When bgm=True, each chunk's mood/genre is classified independently
    (one batched OpenAI call covering every chunk) and the background
    music shifts scene-by-scene to match - a single flat genre tag for an
    entire multi-scene story is a blunt instrument; this tracks the story
    as it actually moves through different beats. Narration speed stays
    a single value for the whole piece regardless (see caller) - only the
    music adapts per chunk, since variable narration pace would sound
    jarring in a way variable background music doesn't.

    When `known_characters` is given (non-empty), the same classification
    call also attributes each chunk to a speaker - the narrator, or one of
    these characters if a chunk is clearly their direct dialogue - and any
    chunk attributed to a named character gets a per-character pitch/EQ
    voice treatment AND a slight speaking-pace offset (see
    voice_profile.py) applied, so recurring characters are distinguishable
    from the narrator and from each other. This is a treatment on the same
    single underlying voice, not true separate speakers - see
    voice_profile.py's docstring. Classification runs BEFORE synthesis in
    this path (unlike the bgm-only path) specifically so the pace offset
    can be passed into engine.synthesize() itself, rather than adjusted
    after audio already exists."""
    known_characters = known_characters or []
    sr = engine.sample_rate(lang)
    chunk_budget = _DIALOGUE_CHUNK_CHAR_BUDGET if known_characters else _CHUNK_CHAR_BUDGET
    chunks = _split_into_chunks(text, chunk_budget)
    silence = np.zeros(int(sr * _INTER_CHUNK_SILENCE_S), dtype=np.float32)

    included: list[tuple[str, str]] = []  # (raw chunk, normalized chunk_text)
    for chunk in chunks:
        chunk_text = normalize_text(resolve_mixed_language(chunk, lang), lang)
        if chunk_text:
            included.append((chunk, chunk_text))

    if not included:
        raise ValueError("Text normalized to empty string")

    included_chunk_texts = [raw for raw, _ in included]
    need_classification = (bgm or known_characters) and len(included) > 1
    scenes = detect_scenes_batch(included_chunk_texts, known_characters=known_characters) if need_classification else None

    profile_assignment = assign_profiles(known_characters, personality_hints=personality_hints) if (scenes and known_characters) else {}
    chunk_audios: list[np.ndarray] = []
    for i, (_raw, chunk_text) in enumerate(included):
        chunk_speed = speed
        profile = None
        if scenes:
            speaker = scenes[i]["speaker"]
            if speaker != "narrator" and speaker in profile_assignment:
                profile = profile_assignment[speaker]
                chunk_speed = speed * get_speed_multiplier(profile)
        audio, _ = engine.synthesize(chunk_text, lang, speed=chunk_speed, noise_scale=noise_scale)
        if profile:
            audio = apply_voice_profile(audio, sr, profile)
        chunk_audios.append(audio)

    parts: list[np.ndarray] = []
    chunk_offsets: list[tuple[int, int]] = []
    cursor = 0
    for audio in chunk_audios:
        if parts:
            parts.append(silence)
            cursor += len(silence)
        start = cursor
        parts.append(audio)
        cursor += len(audio)
        chunk_offsets.append((start, cursor))
    combined = np.concatenate(parts).astype(np.float32)

    if not bgm:
        return combined, sr

    if scenes is None:
        return mix_with_bgm(combined, sr, mood=bgm_fallback), sr

    segments = [(start, end, (scene["genre"], scene["bgm_intensity"])) for (start, end), scene in zip(chunk_offsets, scenes)]
    sfx_segments = [(start, end, (scene["sfx"], scene["sfx_intensity"])) for (start, end), scene in zip(chunk_offsets, scenes)]
    return mix_with_scene_bgm(combined, sr, segments, sfx_segments=sfx_segments), sr


def _pcm16_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype("<i2").tobytes()


def _wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(_pcm16_bytes(audio))
    return buf.getvalue()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "facebook/mms-tts (VITS)",
        "languages_loaded": list(engine.languages()),
        "cpu_threads": CPU_THREADS,
        "voice_cloning_supported": True,
        "voice_cloning_note": "This engine (VITS) can't clone voices itself - POST /clone proxies to the separate IndicF5 service (reference-audio-conditioned), which is far slower (30-70s) since that's inherent to that model architecture, not this proxy.",
        "emotion_note": "Emotion presets are an approximate speed/expressiveness heuristic, not learned emotion conditioning. emotion='auto' classifies the text's tone via OpenAI.",
        "emotion_auto_available": bool(os.environ.get("OPENAI_API_KEY")),
        "bgm_note": "bgm=true mixes in a procedurally-generated, mood-matched chord progression + rhythm track, scene-classified per-chunk on multi-scene stories, auto-ducked against the speech envelope - not licensed/produced music.",
        "sfx_note": "Multi-scene bgm requests also layer environmental sound effects (rain, wind, thunder, lightning, fire, footsteps, car, traffic, crowd, door_creak, glass_break, birds, water, dog_bark) with a per-scene intensity (0-1, how prominent the sound should be) plus a touch of reverb for spatial blend, when the text describes them, classified in the same batched call as genre - which now also gets its own independent bgm_intensity (0-1) so the music itself swells for dramatic beats and recedes for quiet ones, not just switches genre. This is a broad hand-built palette OpenAI picks from freely with dynamic intensity, not arbitrary open-ended sound generation - that would need a dedicated generative-audio model this service doesn't have.",
        "clarity_note": "noise_scale (VITS's stochastic-expressiveness knob) is now tuned per emotion, all at or below the model's own default, for more consistent pronunciation - previously computed but silently discarded, now actually applied.",
        "voice_note": "This engine has ONE fixed voice per language - no true multi-speaker model backs it. On /dream-to-story, /stories/{id}/branch, and /characters/{id}/revive, dialogue lines attributed to a known named character get a deterministic per-character pitch/EQ treatment plus a slight speaking-pace offset on that same base voice (see voice_profile.py), and the specific treatment is chosen to fit that character's personality (a menacing villain leans deep and speaks a touch slower, a cheerful/young character leans bright and speaks a touch faster) rather than assigned arbitrarily - a real, audible differentiation, but still one underlying voice, not separate speakers.",
    }


@app.get("/warmup")
def warmup():
    results = {}
    for lang in engine.languages():
        t0 = time.time()
        engine.synthesize("नमस्ते" if lang in ("hin", "mar") else "test", lang)
        results[lang] = round(time.time() - t0, 3)
    return {"warmup_seconds_per_language": results}


@app.get("/languages")
def languages():
    return {"languages": engine.languages()}


@app.get("/config")
def config():
    """Live runtime configuration - every env-derived setting (secrets
    reported as set/unset, never as values) plus the in-code tunable
    constants that actually shape behavior (chunk sizes, BGM/SFX levels
    and duck amounts, voice profile ranges, etc.). A JSON mirror of
    CONFIG.md that can't go stale the way a static doc can, since it
    reads the values directly from the running process."""
    bgm_defaults = inspect.signature(mix_with_scene_bgm).parameters
    return {
        "env": {
            "INDIC_TTS_API_KEY_set": bool(API_KEY),
            "INDIC_TTS_THREADS": CPU_THREADS,
            "INDIC_TTS_RATE_LIMIT_PER_MIN": int(os.environ.get("INDIC_TTS_RATE_LIMIT_PER_MIN", "60")),
            "INDIC_TTS_DB_PATH": db.DB_PATH,
            "OPENAI_API_KEY_set": bool(os.environ.get("OPENAI_API_KEY")),
            "OPENAI_STORY_MODEL": os.environ.get("OPENAI_STORY_MODEL", "gpt-4o"),
            "OPENAI_EMOTION_MODEL": os.environ.get("OPENAI_EMOTION_MODEL", "gpt-4o"),
            "XLIT_BASE_URL": os.environ.get("XLIT_BASE_URL", "http://127.0.0.1:8501"),
            "XLIT_API_KEY_set": bool(os.environ.get("XLIT_API_KEY")),
            "INDICF5_BASE_URL": os.environ.get("INDICF5_BASE_URL", "http://127.0.0.1:8500"),
            "INDICF5_API_KEY_set": bool(os.environ.get("INDICF5_API_KEY")),
        },
        "synthesis": {
            "max_text_len": MAX_TEXT_LEN,
            "chunk_char_budget": _CHUNK_CHAR_BUDGET,
            "dialogue_chunk_char_budget": _DIALOGUE_CHUNK_CHAR_BUDGET,
            "inter_chunk_silence_s": _INTER_CHUNK_SILENCE_S,
            "min_chars_per_sec_before_retry": _MIN_CHARS_PER_SEC,
            "max_synth_retries": _MAX_SYNTH_RETRIES,
            "default_noise_scale": _DEFAULT_NOISE_SCALE,
            "default_noise_scale_duration": _DEFAULT_NOISE_SCALE_DURATION,
            "emotion_presets": {k: {"speed": v[0], "noise_scale": v[1]} for k, v in EMOTION_PRESETS.items()},
        },
        "bgm_sfx_mix": {
            "bgm_level": bgm_defaults["bgm_level"].default,
            "sfx_level": bgm_defaults["sfx_level"].default,
            "bgm_duck_amount": bgm_defaults["bgm_duck_amount"].default,
            "sfx_duck_amount": bgm_defaults["sfx_duck_amount"].default,
            "genre_loop_duration_s": inspect.signature(_generate_track).parameters["duration_s"].default,
            "sfx_loop_duration_s": inspect.signature(get_sfx_loop).parameters["duration_s"].default,
            "sfx_types": list(SFX_TYPES),
            "genres": list(GENRES.keys()),
        },
        "voice_profiles": {
            "profile_names": PROFILE_NAMES,
            "profiles": {
                name: {"pitch_shift_semitones": v[0], "shelf_eq_tilt_db": v[1], "speed_multiplier": v[2]}
                for name, v in _PROFILES.items()
            },
        },
        "story_gen": {
            "target_char_budget": TARGET_CHAR_BUDGET,
            "languages_with_cultural_context": list(LANGUAGE_CULTURAL_CONTEXT.keys()),
        },
    }


@app.get("/metrics")
def metrics():
    lines = [
        "# TYPE indic_tts_requests_total counter",
        f"indic_tts_requests_total {_metrics['requests_total']}",
        "# TYPE indic_tts_requests_failed_total counter",
        f"indic_tts_requests_failed_total {_metrics['requests_failed']}",
        "# TYPE indic_tts_synth_seconds_total counter",
        f"indic_tts_synth_seconds_total {_metrics['synth_seconds_total']:.4f}",
    ]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain")


@app.post("/tts")
def tts(req: TTSRequest, x_api_key: str | None = Header(default=None)):
    _check_auth(x_api_key)
    if req.speaker:
        raise HTTPException(status_code=400, detail="'speaker' is not used by /tts - use POST /clone for voice cloning instead.")

    lang = _resolve_language(req.text, req.language)
    emotion, speed, noise_scale = _resolve_emotion_and_speed(req)

    _metrics["requests_total"] += 1
    t0 = time.time()
    try:
        audio, sr = _synthesize_long_form(req.text, lang, speed, noise_scale=noise_scale, bgm=req.bgm, bgm_fallback=emotion or "neutral")
    except ValueError as e:
        _metrics["requests_failed"] += 1
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        _metrics["requests_failed"] += 1
        logger.exception("synthesis failed for lang=%s", lang)
        raise HTTPException(status_code=500, detail="Speech generation failed")
    _metrics["synth_seconds_total"] += time.time() - t0

    return Response(content=_wav_bytes(audio, sr), media_type="audio/wav")


@app.post("/clone")
def clone(req: CloneRequest, x_api_key: str | None = Header(default=None)):
    """Voice cloning via the existing IndicF5 service - see /health's
    voice_cloning_note for why this isn't handled by this service's own
    (fast, non-cloning-capable) engine."""
    _check_auth(x_api_key)
    _metrics["requests_total"] += 1
    t0 = time.time()
    try:
        wav_bytes = clone_speech(req.text, req.ref_audio_b64, req.ref_text)
    except CloneError as e:
        _metrics["requests_failed"] += 1
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    _metrics["synth_seconds_total"] += time.time() - t0
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/dream-to-story")
def dream_to_story(req: DreamToStoryRequest, x_api_key: str | None = Header(default=None)):
    """Turns a described dream/memory into a short narrative (via OpenAI)
    and narrates it (via this service's own TTS pipeline). The generated
    story text comes back as a base64 header (X-Story-Text-B64) alongside
    the audio body, since HTTP headers are ASCII-only and the story is in
    a native Indic script."""
    _check_auth(x_api_key)
    if req.language not in engine.languages():
        raise HTTPException(status_code=400, detail=f"Unsupported language '{req.language}'. Supported: {list(engine.languages())}")

    try:
        story_text = generate_story(req.description, req.language)
    except StoryGenError as e:
        raise HTTPException(status_code=502, detail=str(e))

    story_id = db.save_story(title=req.description[:80], language=req.language, text=story_text)
    character_ids = []
    character_names = []
    personality_hints = {}
    for char in extract_characters(story_text, req.language):
        cid = db.save_character(
            name=char["name"], personality=char["personality"], backstory=char["backstory"],
            language=req.language, origin_story_id=story_id, voice_profile=char.get("voice_profile"),
        )
        db.link_character_to_story(story_id, cid)
        character_ids.append(cid)
        character_names.append(char["name"])
        if char.get("voice_profile"):
            personality_hints[char["name"]] = char["voice_profile"]

    tts_req = TTSRequest(text=story_text, language=req.language, emotion=req.emotion, bgm=req.bgm)
    lang = req.language
    emotion, speed, noise_scale = _resolve_emotion_and_speed(tts_req)

    _metrics["requests_total"] += 1
    t0 = time.time()
    try:
        audio, sr = _synthesize_long_form(
            story_text, lang, speed, noise_scale=noise_scale, bgm=req.bgm,
            bgm_fallback=emotion or "neutral", known_characters=character_names,
            personality_hints=personality_hints,
        )
    except Exception:
        _metrics["requests_failed"] += 1
        logger.exception("dream-to-story synthesis failed for lang=%s", lang)
        raise HTTPException(status_code=500, detail="Speech generation failed")
    _metrics["synth_seconds_total"] += time.time() - t0

    story_b64 = base64.b64encode(story_text.encode("utf-8")).decode("ascii")
    return Response(
        content=_wav_bytes(audio, sr),
        media_type="audio/wav",
        headers={
            "X-Story-Text-B64": story_b64,
            "X-Resolved-Emotion": emotion or "none",
            "X-Story-Id": str(story_id),
            "X-Character-Ids": ",".join(str(c) for c in character_ids),
        },
    )


@app.get("/stories")
def list_stories(limit: int = 50):
    return {"stories": db.list_stories(limit=limit)}


@app.get("/stories/{story_id}")
def get_story(story_id: int):
    story = db.get_story(story_id)
    if not story:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")
    return story


@app.get("/characters")
def list_characters(limit: int = 50):
    return {"characters": db.list_characters(limit=limit)}


@app.get("/characters/{character_id}")
def get_character(character_id: int):
    character = db.get_character(character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character {character_id} not found")
    return character


class BranchRequest(BaseModel):
    changed_decision: str = Field(..., min_length=1, max_length=1000, description="Describe what changes at some point in the story - the continuation is generated from there.")
    emotion: str | None = Field(default="auto")
    bgm: bool = Field(default=True)


@app.post("/stories/{story_id}/branch")
def branch_story(story_id: int, req: BranchRequest, x_api_key: str | None = Header(default=None)):
    """'Story Time Machine' subset - see story_gen.generate_continuation's
    docstring for exactly what this does and doesn't guarantee about
    consistency."""
    _check_auth(x_api_key)
    parent = db.get_story(story_id)
    if not parent:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    try:
        continuation = generate_continuation(parent["text"], req.changed_decision, parent["language"])
    except StoryGenError as e:
        raise HTTPException(status_code=502, detail=str(e))

    new_story_id = db.save_story(
        title=f"Branch of #{story_id}", language=parent["language"], text=continuation,
        parent_story_id=story_id, branch_note=req.changed_decision,
    )
    story_characters = db.get_characters_for_story(story_id)
    character_names = [c["name"] for c in story_characters]
    personality_hints = {c["name"]: c["voice_profile"] for c in story_characters if c.get("voice_profile")}
    for c in story_characters:
        db.link_character_to_story(new_story_id, c["id"])

    tts_req = TTSRequest(text=continuation, language=parent["language"], emotion=req.emotion, bgm=req.bgm)
    emotion, speed, noise_scale = _resolve_emotion_and_speed(tts_req)

    _metrics["requests_total"] += 1
    t0 = time.time()
    try:
        audio, sr = _synthesize_long_form(
            continuation, parent["language"], speed, noise_scale=noise_scale, bgm=req.bgm,
            bgm_fallback=emotion or "neutral", known_characters=character_names,
            personality_hints=personality_hints,
        )
    except Exception:
        _metrics["requests_failed"] += 1
        logger.exception("branch synthesis failed for story_id=%s", story_id)
        raise HTTPException(status_code=500, detail="Speech generation failed")
    _metrics["synth_seconds_total"] += time.time() - t0

    story_b64 = base64.b64encode(continuation.encode("utf-8")).decode("ascii")
    return Response(
        content=_wav_bytes(audio, sr),
        media_type="audio/wav",
        headers={"X-Story-Text-B64": story_b64, "X-Story-Id": str(new_story_id), "X-Parent-Story-Id": str(story_id)},
    )


class ReviveRequest(BaseModel):
    new_setting: str = Field(..., min_length=1, max_length=1000, description="Describe the new story/setting/universe to place this character into.")
    emotion: str | None = Field(default="auto")
    bgm: bool = Field(default=True)


@app.post("/characters/{character_id}/revive")
def revive(character_id: int, req: ReviveRequest, x_api_key: str | None = Header(default=None)):
    """'Character Resurrection' subset - see story_gen.revive_character's
    docstring for scope."""
    _check_auth(x_api_key)
    character = db.get_character(character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character {character_id} not found")

    try:
        story_text = revive_character(
            character["name"], character["personality"], character["backstory"],
            req.new_setting, character["language"],
        )
    except StoryGenError as e:
        raise HTTPException(status_code=502, detail=str(e))

    new_story_id = db.save_story(
        title=f"{character['name']} revived", language=character["language"], text=story_text,
    )
    db.link_character_to_story(new_story_id, character_id)

    tts_req = TTSRequest(text=story_text, language=character["language"], emotion=req.emotion, bgm=req.bgm)
    emotion, speed, noise_scale = _resolve_emotion_and_speed(tts_req)

    _metrics["requests_total"] += 1
    t0 = time.time()
    try:
        audio, sr = _synthesize_long_form(
            story_text, character["language"], speed, noise_scale=noise_scale, bgm=req.bgm,
            bgm_fallback=emotion or "neutral", known_characters=[character["name"]],
            personality_hints={character["name"]: character["voice_profile"]} if character.get("voice_profile") else None,
        )
    except Exception:
        _metrics["requests_failed"] += 1
        logger.exception("revive synthesis failed for character_id=%s", character_id)
        raise HTTPException(status_code=500, detail="Speech generation failed")
    _metrics["synth_seconds_total"] += time.time() - t0

    story_b64 = base64.b64encode(story_text.encode("utf-8")).decode("ascii")
    return Response(
        content=_wav_bytes(audio, sr),
        media_type="audio/wav",
        headers={"X-Story-Text-B64": story_b64, "X-Story-Id": str(new_story_id)},
    )


class VillainRequest(BaseModel):
    fears: str = Field(..., min_length=1, max_length=1000, description="Describe the listener's fears/anxieties/motivations - user-specified, not learned from any history (none is tracked).")
    language: str = Field(default="hin")
    emotion: str | None = Field(default="horror")
    bgm: bool = Field(default=True)


@app.post("/villain")
def villain(req: VillainRequest, x_api_key: str | None = Header(default=None)):
    """'Personalized Villain' subset - see story_gen.generate_villain's
    docstring: personalized to USER-STATED fears, not learned from
    listener behavior, since no listener history exists in this system."""
    _check_auth(x_api_key)
    if req.language not in engine.languages():
        raise HTTPException(status_code=400, detail=f"Unsupported language '{req.language}'. Supported: {list(engine.languages())}")

    try:
        villain_name, scene_text = generate_villain(req.fears, req.language)
    except StoryGenError as e:
        raise HTTPException(status_code=502, detail=str(e))

    story_id = db.save_story(title=f"Villain: {villain_name}", language=req.language, text=scene_text)
    character_id = db.save_character(
        name=villain_name, personality=f"Villain designed around fears: {req.fears[:200]}",
        backstory=scene_text[:500], language=req.language, origin_story_id=story_id,
    )
    db.link_character_to_story(story_id, character_id)

    tts_req = TTSRequest(text=scene_text, language=req.language, emotion=req.emotion, bgm=req.bgm)
    emotion, speed, noise_scale = _resolve_emotion_and_speed(tts_req)

    _metrics["requests_total"] += 1
    t0 = time.time()
    try:
        audio, sr = _synthesize_long_form(scene_text, req.language, speed, noise_scale=noise_scale, bgm=req.bgm, bgm_fallback=emotion or "horror")
    except Exception:
        _metrics["requests_failed"] += 1
        logger.exception("villain synthesis failed")
        raise HTTPException(status_code=500, detail="Speech generation failed")
    _metrics["synth_seconds_total"] += time.time() - t0

    story_b64 = base64.b64encode(scene_text.encode("utf-8")).decode("ascii")
    return Response(
        content=_wav_bytes(audio, sr),
        media_type="audio/wav",
        headers={
            "X-Story-Text-B64": story_b64,
            "X-Villain-Name-B64": base64.b64encode(villain_name.encode("utf-8")).decode("ascii"),
            "X-Story-Id": str(story_id),
            "X-Character-Id": str(character_id),
        },
    )


@app.post("/stream")
def stream(req: TTSRequest, x_api_key: str | None = Header(default=None)):
    """Streams audio sentence-by-sentence as it's synthesized, so the first
    chunk of a long passage is playable well before the rest finishes.
    VITS is non-autoregressive (whole-sentence-at-once), so true sub-
    sentence chunk streaming isn't applicable here - sentence granularity
    is the real unit of "as soon as it's ready" for this architecture."""
    _check_auth(x_api_key)
    if req.speaker:
        raise HTTPException(status_code=400, detail="'speaker' is not used by /stream - use POST /clone for voice cloning instead.")

    lang = req.language or detect_language(req.text)
    if lang not in engine.languages():
        raise HTTPException(status_code=400, detail=f"Unsupported language '{lang}'")
    _, speed, noise_scale = _resolve_emotion_and_speed(req)
    sr = engine.sample_rate(lang)

    sentences = _split_sentences(req.text) or [req.text]

    def generate():
        # Streaming raw 16-bit PCM mono at the model's native sample rate
        # (no WAV container, since a valid WAV header needs the total size
        # up front) - documented via the response headers below.
        for sentence in sentences:
            text = normalize_text(resolve_mixed_language(sentence, lang), lang)
            if not text:
                continue
            audio, _ = engine.synthesize(text, lang, speed=speed, noise_scale=noise_scale)
            yield _pcm16_bytes(audio)

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Format": "pcm_s16le",
            "X-Sample-Rate": str(sr),
            "X-Channels": "1",
        },
    )
