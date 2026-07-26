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
from fastapi.staticfiles import StaticFiles
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
from export import ExportError, build_comic_panels, build_game_tree, render_video
from images import STATIC_DIR, ImageGenError, generate_scene_image, generate_visual_prompts, save_scene_image
from lang_detect import detect_language
from mixed_lang import resolve_mixed_language
from normalize import normalize_text
from quality import QUALITY_THRESHOLD, score_and_maybe_rewrite
from ratelimit import RateLimitMiddleware
# image_gen/video_gen power the single-cover-image "visual"/"video" feature
# on /dream-to-story - distinct from images.py's multi-scene, time-synced
# "images" feature (see db.py's save_story_image vs save_scene_image
# docstrings). image_gen.ImageGenError is aliased to avoid shadowing
# images.ImageGenError, imported just above under the same bare name.
from image_gen import ImageGenError as CoverImageGenError, generate_story_image
from video_gen import VideoGenError, compose_video
from sfx import SFX_TYPES, get_sfx_loop
from voice_profile import PROFILE_NAMES, _PROFILES, assign_profiles, apply_voice_profile, get_speed_multiplier
from story_gen import (
    LANGUAGE_CULTURAL_CONTEXT,
    TARGET_CHAR_BUDGET,
    StoryGenError,
    adapt_story_language,
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

    # Serves generated scene images (and saved narration audio for video/
    # comic export) - directory must exist before StaticFiles mounts it.
    # Deferred to startup (not module level) for the same reason db.init_db()
    # and engine.load_all() are: importing app.py (e.g. from tests) must
    # stay a cheap, side-effect-free operation.
    os.makedirs(STATIC_DIR, exist_ok=True)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    yield


app = FastAPI(title="Indic Fast TTS API", lifespan=lifespan)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=int(os.environ.get("INDIC_TTS_RATE_LIMIT_PER_MIN", "60")),
)
# CORS (including Access-Control-Expose-Headers, without which custom
# X-Story-Id/X-Quality-*/etc. response headers are invisible to browser JS
# on a cross-origin request - e.g. this API's own web UI) is handled at the
# nginx layer in front of this service, not here - see indic-tts-fast.nginx.
# Adding a second CORS layer in FastAPI risks emitting duplicate
# Access-Control-* headers (nginx's plus this app's), which browsers treat
# as an invalid response and reject entirely - worse than doing nothing.


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
    language: str | None = Field(default=None, description="ISO 639-3 code, e.g. 'hin', 'tam'. Auto-detected from script if omitted.")
    speed: float = Field(default=1.0, gt=0, le=2.5)
    emotion: str | None = Field(default=None, description="One of: " + ", ".join(EMOTION_PRESETS) + ", or 'auto' to classify from the text via OpenAI.")
    bgm: bool = Field(default=False, description="Mix in a procedurally-generated, mood-matched ambient background pad (not licensed/produced music - see /health).")
    speaker: str | None = Field(default=None, description="Reserved, unused - use POST /clone for voice cloning instead.")
    images: bool = Field(default=False, description="Also generate scene illustrations via OpenAI Images, mapped to approximate time ranges - see GET /stories/{id}/images. Saves this text as a story (title = first 80 chars) so images have somewhere to attach. Off by default: adds real per-image latency and OpenAI cost.")
    max_images: int = Field(default=4, ge=1, le=6, description="Max number of scene images to generate when images=true.")


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
    visual: bool = Field(default=False, description="Also generate a single cover/scene image via OpenAI's image API depicting the story's strongest visual moment. Off by default - a real cost/latency commitment per image, unlike the cheap text calls elsewhere. Retrieve via GET /stories/{id}/image. Distinct from images=true below - this is one cover image, not a time-synced storyboard.")
    video: bool = Field(default=False, description="Also compose a downloadable/shareable MP4 (the cover image held for the full narration) via ffmpeg. Implies visual=true (video needs a cover image) even if visual wasn't separately set. Off by default. Retrieve via GET /stories/{id}/video.")
    images: bool = Field(default=False, description="Also generate several scene illustrations via OpenAI Images, mapped to approximate time ranges in the narration - see GET /stories/{id}/images. Off by default: adds real per-image latency and OpenAI cost on top of narration. Distinct from visual=true above - this is a multi-image, time-synced storyboard, not one cover image.")
    max_images: int = Field(default=4, ge=1, le=6, description="Max number of scene images to generate when images=true.")


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
) -> tuple[np.ndarray, int, list[tuple[int, int]], list[str]]:
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
    raw_chunk_texts = [raw for raw, _ in included]

    if not bgm:
        return combined, sr, chunk_offsets, raw_chunk_texts

    if scenes is None:
        return mix_with_bgm(combined, sr, mood=bgm_fallback), sr, chunk_offsets, raw_chunk_texts

    segments = [(start, end, (scene["genre"], scene["bgm_intensity"])) for (start, end), scene in zip(chunk_offsets, scenes)]
    sfx_segments = [(start, end, (scene["sfx"], scene["sfx_intensity"])) for (start, end), scene in zip(chunk_offsets, scenes)]
    return mix_with_scene_bgm(combined, sr, segments, sfx_segments=sfx_segments), sr, chunk_offsets, raw_chunk_texts


def _map_beats_to_time(
    beats: list[dict], raw_chunks: list[str], chunk_offsets: list[tuple[int, int]], sr: int,
) -> list[dict]:
    """Maps each visual beat (from images.generate_visual_prompts) to an
    approximate time range in the synthesized audio, by locating its
    "excerpt" within the raw sentence-grouped chunks used for synthesis
    (see _synthesize_long_form's chunk_offsets/raw_chunk_texts). A beat
    whose excerpt can't be located verbatim (GPT sometimes paraphrases
    rather than quoting exactly) falls back to spreading it evenly across
    the chunk range in beat order - same "use what's real, don't discard
    on a partial match failure" posture as emotion._reconcile_length."""
    n_chunks = len(raw_chunks)
    if n_chunks == 0 or not beats:
        return []

    def _locate(excerpt: str) -> int:
        needle = excerpt.strip()[:30].strip()
        if not needle:
            return -1
        for i, chunk in enumerate(raw_chunks):
            if needle in chunk:
                return i
        return -1

    def _fallback_index(i: int) -> int:
        return min(n_chunks - 1, int(i * n_chunks / len(beats)))

    located = [_locate(b.get("excerpt", "")) for b in beats]
    results = []
    for i, beat in enumerate(beats):
        chunk_idx = located[i] if located[i] != -1 else _fallback_index(i)
        start_sample, _ = chunk_offsets[chunk_idx]
        if i + 1 < len(beats):
            next_idx = located[i + 1] if located[i + 1] != -1 else _fallback_index(i + 1)
            next_idx = max(next_idx, chunk_idx)
            end_sample = chunk_offsets[next_idx][0]
        else:
            end_sample = chunk_offsets[-1][1]
        start_s = start_sample / sr
        end_s = max(end_sample / sr, start_s + 0.1)
        results.append({
            "excerpt": beat.get("excerpt", ""),
            "visual_prompt": beat.get("visual_prompt", ""),
            "start_time_s": start_s,
            "end_time_s": end_s,
        })
    return results


def _quality_headers(quality_scores: dict) -> dict:
    """Small set of ASCII-safe headers surfacing the quality-scoring result
    (see quality.score_and_maybe_rewrite) alongside a story response, so a
    client can show it without a separate request."""
    return {
        "X-Quality-Scored": str(bool(quality_scores.get("scored"))).lower(),
        "X-Quality-Overall": str(quality_scores.get("overall", "")),
        "X-Quality-Rewritten": str(bool(quality_scores.get("rewrite_attempted"))).lower(),
    }


def _maybe_generate_scene_images(
    max_images: int, story_id: int, story_text: str, language: str,
    character_names: list[str], chunk_offsets: list[tuple[int, int]], raw_chunks: list[str],
    audio: np.ndarray, sr: int,
) -> int:
    """Shared scene-image pipeline used by both /tts (images=true) and
    /dream-to-story: generates up to max_images scene illustrations,
    persists them, and - only if at least one succeeded - saves the
    narration audio to disk (needed later by video export). Returns the
    number of images actually generated. Best-effort: any failure here is
    logged and swallowed, never fails the request that's already produced
    valid narration audio."""
    image_count = 0
    try:
        beats = generate_visual_prompts(story_text, language, character_names, max_beats=max_images)
        mapped_beats = _map_beats_to_time(beats, raw_chunks, chunk_offsets, sr)
        for idx, beat in enumerate(mapped_beats):
            try:
                image_bytes = generate_scene_image(beat["visual_prompt"])
                url = save_scene_image(story_id, idx, image_bytes)
                db.save_scene_image(
                    story_id, idx, beat["start_time_s"], beat["end_time_s"],
                    beat["excerpt"], beat["visual_prompt"], url,
                )
                image_count += 1
            except ImageGenError as e:
                logger.warning("scene image %d failed for story_id=%s (non-fatal): %s", idx, story_id, e)
        if image_count:
            _save_story_audio(story_id, audio, sr)
    except Exception:
        logger.exception("scene image pipeline failed for story_id=%s (non-fatal)", story_id)
    return image_count


def _save_story_audio(story_id: int, audio: np.ndarray, sr: int) -> None:
    """Persists narration audio to STATIC_DIR/audio/{story_id}.wav so a
    later video export (see export.render_video) can mux it against scene
    images without re-synthesizing. Only called when images were actually
    generated for a story - other endpoints don't need audio on disk."""
    audio_dir = os.path.join(STATIC_DIR, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    with open(os.path.join(audio_dir, f"{story_id}.wav"), "wb") as f:
        f.write(_wav_bytes(audio, sr))


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
        "sfx_note": "Multi-scene bgm requests also layer environmental sound effects (rain, wind, thunder, lightning, fire, footsteps, car, traffic, crowd, door_creak, glass_break, birds, water, dog_bark) with a per-scene intensity (0-1, how prominent the sound should be) plus a touch of reverb for spatial blend, when the text describes them, classified in the same batched call as genre - which now also gets its own independent bgm_intensity (0-1) so the music itself swells for dramatic beats and recedes for quiet ones, not just switches genre. Up to 2 SFX cues can layer together per scene (a storm gets rain AND thunder together, not one or the other). This is a broad hand-built palette OpenAI picks from freely with dynamic intensity, not arbitrary open-ended sound generation - that would need a dedicated generative-audio model this service doesn't have.",
        "cinematic_note": "BGM genres now also carry a melodic hook (a short, genre-distinctive motif an octave above the harmony - what actually makes procedural music sound composed rather than ambient wash) and, for dramatic genres (horror, mystery, action, excited), a felt-more-than-heard sub-bass layer for cinematic weight.",
        "clarity_note": "noise_scale (VITS's stochastic-expressiveness knob) is now tuned per emotion, all at or below the model's own default, for more consistent pronunciation - previously computed but silently discarded, now actually applied.",
        "quality_note": "Story text from /dream-to-story, /stories/{id}/branch, /characters/{id}/revive, and /villain is scored by an independent second OpenAI call on 4 axes (native-sounding, tone-appropriateness, cultural-accuracy, emotional-delivery) before narration. A low score triggers ONE automatic rewrite targeting the weak axes; whichever version scores higher is what gets narrated. This judges the generated TEXT only, not the synthesized audio's pronunciation - nothing on this box can evaluate audio quality. Stories still below threshold after the rewrite attempt are visible at GET /quality/backlog.",
        "export_note": "GET /stories/{id}/game-tree exposes the existing branch feature as a navigable choice-tree (interactive fiction, not a game engine). GET /stories/{id}/comic returns scene images + native-script captions as JSON panels for a client to render (no server-side speech-bubble compositing - that would need a bundled Unicode font per script). POST /stories/{id}/render/video builds an ffmpeg slideshow from scene images + saved narration audio - requires the story to have been generated with images=true AND the ffmpeg binary to be installed on this host (checked at call time, returns 501 if missing rather than crashing).",
        "images_note": "images=true on /tts or /dream-to-story generates up to max_images scene illustrations via OpenAI's image API (gpt-image-1 by default), mapped to approximate time ranges in the narration via GET /stories/{id}/images - off by default since it adds real per-image latency and OpenAI cost on top of narration. On /tts this also saves the input text as a story (needed so images have somewhere to attach). Best-effort: a failed prompt/image generation drops that one scene rather than failing the whole request.",
        "voice_note": "This engine has ONE fixed voice per language - no true multi-speaker model backs it. On /dream-to-story, /stories/{id}/branch, and /characters/{id}/revive, dialogue lines attributed to a known named character get a deterministic per-character pitch/EQ treatment plus a slight speaking-pace offset on that same base voice (see voice_profile.py), and the specific treatment is chosen to fit that character's personality (a menacing villain leans deep and speaks a touch slower, a cheerful/young character leans bright and speaks a touch faster) rather than assigned arbitrarily - a real, audible differentiation, but still one underlying voice, not separate speakers.",
        "visual_note": "POST /dream-to-story accepts visual=true (off by default) to also generate a cover/scene image via OpenAI's image API, depicting the story's single strongest visual moment - a two-step process (derive an English scene prompt from the native-script story, then generate from that). Retrieve via GET /stories/{id}/image. Off by default: a real per-image cost/latency commitment, unlike the cheap text classification calls elsewhere. Generation failure is non-fatal to the request (audio still returns) and reported via X-Has-Image/X-Image-Error-B64 headers.",
        "video_note": "POST /dream-to-story also accepts video=true (implies visual=true) to compose a real, shareable MP4 (the cover image held for the full narration) via ffmpeg - retrieve via GET /stories/{id}/video. This is NOT AI-generated moving video (that needs a fundamentally different, much slower/costlier generative-video model this service doesn't have access to) - it's a static-image-plus-audio mux, honest about what it is.",
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
        audio, sr, chunk_offsets, raw_chunks = _synthesize_long_form(req.text, lang, speed, noise_scale=noise_scale, bgm=req.bgm, bgm_fallback=emotion or "neutral")
    except ValueError as e:
        _metrics["requests_failed"] += 1
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        _metrics["requests_failed"] += 1
        logger.exception("synthesis failed for lang=%s", lang)
        raise HTTPException(status_code=500, detail="Speech generation failed")
    _metrics["synth_seconds_total"] += time.time() - t0

    headers = {"X-Resolved-Emotion": emotion or "none"}
    if req.images:
        story_id = db.save_story(title=req.text[:80], language=lang, text=req.text)
        image_count = _maybe_generate_scene_images(
            req.max_images, story_id, req.text, lang, [], chunk_offsets, raw_chunks, audio, sr,
        )
        headers["X-Story-Id"] = str(story_id)
        headers["X-Image-Count"] = str(image_count)

    return Response(content=_wav_bytes(audio, sr), media_type="audio/wav", headers=headers)


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
    story_text, quality_scores = score_and_maybe_rewrite(story_text, req.language)

    story_id = db.save_story(title=req.description[:80], language=req.language, text=story_text)
    if quality_scores.get("scored") and quality_scores["overall"] < QUALITY_THRESHOLD:
        db.save_quality_entry(story_id, quality_scores, rewritten=quality_scores.get("rewrite_attempted", False))
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

    image_error = None
    if req.visual or req.video:
        try:
            image_bytes = generate_story_image(story_text)
            db.save_story_image(story_id, image_bytes)
        except CoverImageGenError as e:
            logger.warning("visual generation failed for story_id=%s (non-fatal): %s", story_id, e)
            image_error = str(e)

    tts_req = TTSRequest(text=story_text, language=req.language, emotion=req.emotion, bgm=req.bgm)
    lang = req.language
    emotion, speed, noise_scale = _resolve_emotion_and_speed(tts_req)

    _metrics["requests_total"] += 1
    t0 = time.time()
    try:
        audio, sr, chunk_offsets, raw_chunks = _synthesize_long_form(
            story_text, lang, speed, noise_scale=noise_scale, bgm=req.bgm,
            bgm_fallback=emotion or "neutral", known_characters=character_names,
            personality_hints=personality_hints,
        )
    except Exception:
        _metrics["requests_failed"] += 1
        logger.exception("dream-to-story synthesis failed for lang=%s", lang)
        raise HTTPException(status_code=500, detail="Speech generation failed")
    _metrics["synth_seconds_total"] += time.time() - t0
    wav_bytes = _wav_bytes(audio, sr)

    if req.video:
        # Only persisted when video was requested - routine narration-
        # only requests don't carry this hundred-KB-to-MB blob into the DB.
        db.save_story_audio(story_id, wav_bytes)

    image_count = 0
    if req.images:
        image_count = _maybe_generate_scene_images(
            req.max_images, story_id, story_text, req.language, character_names,
            chunk_offsets, raw_chunks, audio, sr,
        )

    story_b64 = base64.b64encode(story_text.encode("utf-8")).decode("ascii")
    headers = {
        "X-Story-Text-B64": story_b64,
        "X-Resolved-Emotion": emotion or "none",
        "X-Story-Id": str(story_id),
        "X-Character-Ids": ",".join(str(c) for c in character_ids),
        "X-Image-Count": str(image_count),
        **_quality_headers(quality_scores),
    }
    if req.visual or req.video:
        headers["X-Has-Image"] = "true" if image_error is None else "false"
        if image_error:
            headers["X-Image-Error-B64"] = base64.b64encode(image_error.encode("utf-8")).decode("ascii")
    if req.video:
        headers["X-Video-Available"] = "true" if image_error is None else "false"
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers=headers,
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


@app.get("/stories/{story_id}/image")
def get_story_image(story_id: int):
    """PNG bytes for a story's generated cover/scene image - only present
    if the story was created with visual=true (or had one generated some
    other way) and generation succeeded. 404 if the story has no image,
    whether because visual was never requested or generation failed."""
    story = db.get_story(story_id)
    if not story:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")
    image_bytes = db.get_story_image(story_id)
    if image_bytes is None:
        raise HTTPException(status_code=404, detail=f"Story {story_id} has no generated image")
    return Response(content=image_bytes, media_type="image/png")


@app.get("/stories/{story_id}/video")
def get_story_video(story_id: int):
    """Composes (on demand, not cached - ffmpeg muxing a still image
    against an existing audio track is fast, a few seconds at most) an
    MP4 from a story's stored cover image and narration audio. Both only
    exist if the story was created with video=true. 404 if either is
    missing, 500 if ffmpeg itself fails."""
    story = db.get_story(story_id)
    if not story:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")
    image_bytes = db.get_story_image(story_id)
    audio_bytes = db.get_story_audio(story_id)
    if image_bytes is None or audio_bytes is None:
        raise HTTPException(
            status_code=404,
            detail=f"Story {story_id} is missing {'an image' if image_bytes is None else 'audio'} "
                   f"- video requires the story to have been created with video=true",
        )
    try:
        video_bytes = compose_video(image_bytes, audio_bytes)
    except VideoGenError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=video_bytes, media_type="video/mp4")


@app.get("/stories/{story_id}/images")
def get_story_images(story_id: int):
    """Scene illustrations generated for this story (only present if it was
    created with images=true) - each mapped to an approximate time range in
    the narration, see images.py/_map_beats_to_time for how that mapping
    works and its limits. Distinct from GET /stories/{id}/image (singular)
    above, which is a single cover image."""
    story = db.get_story(story_id)
    if not story:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")
    images = db.get_images_for_story(story_id)
    return {
        "images": [
            {
                "start_time_s": img["start_time_s"],
                "end_time_s": img["end_time_s"],
                "url": img["url"],
                "excerpt": img["excerpt"],
            }
            for img in images
        ]
    }


@app.get("/quality/backlog")
def quality_backlog(limit: int = 50):
    """Stories whose generated text scored below quality.QUALITY_THRESHOLD
    on at least one axis even after the one automatic rewrite attempt (see
    quality.score_and_maybe_rewrite) - a visible backlog of weak spots
    rather than a log of every generation."""
    return {"backlog": db.list_quality_backlog(limit=limit)}


@app.get("/stories/{story_id}/game-tree")
def get_game_tree(story_id: int):
    """This story's branches (see POST /stories/{id}/branch), recursively,
    as a navigable choice-tree - see export.build_game_tree's docstring.
    "Game" here means interactive fiction (walk the tree, each node's
    children are the available choices), not a game engine."""
    try:
        return build_game_tree(story_id)
    except ExportError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.get("/stories/{story_id}/comic")
def get_comic(story_id: int):
    """This story's scene images as comic panels - see
    export.build_comic_panels's docstring. Empty panels (not an error) for
    a story that wasn't generated with images=true."""
    try:
        return build_comic_panels(story_id)
    except ExportError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.post("/stories/{story_id}/render/video")
def render_story_video(story_id: int, x_api_key: str | None = Header(default=None)):
    """Renders an MP4 slideshow from this story's scene images and saved
    narration audio - see export.render_video's docstring, including why
    it requires images=true to have been used and the `ffmpeg` binary to
    be installed on this host. Distinct from GET /stories/{id}/video above,
    which composes from a single cover image instead."""
    _check_auth(x_api_key)
    try:
        video_bytes = render_video(story_id)
    except ExportError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    return Response(content=video_bytes, media_type="video/mp4")


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
    """'Story Time Machine' subset - passes this branch's full ancestor
    lineage (see db.get_story_lineage), not just its direct parent, so a
    branch-of-a-branch stays consistent with the whole chain of prior
    changes. See story_gen.generate_continuation's docstring for exactly
    what this does and doesn't guarantee about consistency."""
    _check_auth(x_api_key)
    parent = db.get_story(story_id)
    if not parent:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    lineage = db.get_story_lineage(story_id)
    ancestor_summaries = [a["branch_note"] for a in lineage if a.get("branch_note")]
    try:
        continuation = generate_continuation(
            parent["text"], req.changed_decision, parent["language"], ancestor_summaries=ancestor_summaries,
        )
    except StoryGenError as e:
        raise HTTPException(status_code=502, detail=str(e))
    continuation, quality_scores = score_and_maybe_rewrite(continuation, parent["language"])

    new_story_id = db.save_story(
        title=f"Branch of #{story_id}", language=parent["language"], text=continuation,
        parent_story_id=story_id, branch_note=req.changed_decision,
    )
    if quality_scores.get("scored") and quality_scores["overall"] < QUALITY_THRESHOLD:
        db.save_quality_entry(new_story_id, quality_scores, rewritten=quality_scores.get("rewrite_attempted", False))
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
        audio, sr, _chunk_offsets, _raw_chunks = _synthesize_long_form(
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
        headers={
            "X-Story-Text-B64": story_b64, "X-Story-Id": str(new_story_id), "X-Parent-Story-Id": str(story_id),
            **_quality_headers(quality_scores),
        },
    )


@app.get("/stories/{story_id}/variants")
def get_story_variants(story_id: int):
    """Other-language adaptations of this story (see POST
    /stories/{id}/switch-language), so a client can offer 'listen in...'
    switching between them."""
    story = db.get_story(story_id)
    if not story:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")
    return {"variants": db.get_variants(story_id)}


class SwitchLanguageRequest(BaseModel):
    language: str = Field(..., description="ISO 639-3 code to adapt this story into, e.g. 'tam', 'ben'.")
    emotion: str | None = Field(default="auto")
    bgm: bool = Field(default=True)


@app.post("/stories/{story_id}/switch-language")
def switch_language(story_id: int, req: SwitchLanguageRequest, x_api_key: str | None = Header(default=None)):
    """'Switch language' subset - see story_gen.adapt_story_language's
    docstring for what this actually does: retells the same story, same
    characters, in a new language, saved as a new story linked back via
    variant_of_story_id (see GET /stories/{id}/variants). This is NOT one
    audio stream that changes language mid-sentence - VITS is a separate
    model per language, so that's not a meaningful operation here.
    "Switching mid-story" means resuming the same story, same characters,
    from a saved variant in the new language."""
    _check_auth(x_api_key)
    original = db.get_story(story_id)
    if not original:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")
    if req.language not in engine.languages():
        raise HTTPException(status_code=400, detail=f"Unsupported language '{req.language}'. Supported: {list(engine.languages())}")
    if req.language == original["language"]:
        raise HTTPException(status_code=400, detail=f"Story {story_id} is already in language '{req.language}'")

    try:
        adapted_text = adapt_story_language(original["text"], original["language"], req.language)
    except StoryGenError as e:
        raise HTTPException(status_code=502, detail=str(e))
    adapted_text, quality_scores = score_and_maybe_rewrite(adapted_text, req.language)

    new_story_id = db.save_story(
        title=f"{original['title'] or 'Story #' + str(story_id)} ({req.language})",
        language=req.language, text=adapted_text, variant_of_story_id=story_id,
    )
    if quality_scores.get("scored") and quality_scores["overall"] < QUALITY_THRESHOLD:
        db.save_quality_entry(new_story_id, quality_scores, rewritten=quality_scores.get("rewrite_attempted", False))
    story_characters = db.get_characters_for_story(story_id)
    character_names = [c["name"] for c in story_characters]
    personality_hints = {c["name"]: c["voice_profile"] for c in story_characters if c.get("voice_profile")}
    for c in story_characters:
        db.link_character_to_story(new_story_id, c["id"])

    tts_req = TTSRequest(text=adapted_text, language=req.language, emotion=req.emotion, bgm=req.bgm)
    emotion, speed, noise_scale = _resolve_emotion_and_speed(tts_req)

    _metrics["requests_total"] += 1
    t0 = time.time()
    try:
        audio, sr, _chunk_offsets, _raw_chunks = _synthesize_long_form(
            adapted_text, req.language, speed, noise_scale=noise_scale, bgm=req.bgm,
            bgm_fallback=emotion or "neutral", known_characters=character_names,
            personality_hints=personality_hints,
        )
    except Exception:
        _metrics["requests_failed"] += 1
        logger.exception("switch-language synthesis failed for story_id=%s -> %s", story_id, req.language)
        raise HTTPException(status_code=500, detail="Speech generation failed")
    _metrics["synth_seconds_total"] += time.time() - t0

    story_b64 = base64.b64encode(adapted_text.encode("utf-8")).decode("ascii")
    return Response(
        content=_wav_bytes(audio, sr),
        media_type="audio/wav",
        headers={
            "X-Story-Text-B64": story_b64, "X-Story-Id": str(new_story_id), "X-Variant-Of-Story-Id": str(story_id),
            **_quality_headers(quality_scores),
        },
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
    story_text, quality_scores = score_and_maybe_rewrite(story_text, character["language"])

    new_story_id = db.save_story(
        title=f"{character['name']} revived", language=character["language"], text=story_text,
    )
    if quality_scores.get("scored") and quality_scores["overall"] < QUALITY_THRESHOLD:
        db.save_quality_entry(new_story_id, quality_scores, rewritten=quality_scores.get("rewrite_attempted", False))
    db.link_character_to_story(new_story_id, character_id)

    tts_req = TTSRequest(text=story_text, language=character["language"], emotion=req.emotion, bgm=req.bgm)
    emotion, speed, noise_scale = _resolve_emotion_and_speed(tts_req)

    _metrics["requests_total"] += 1
    t0 = time.time()
    try:
        audio, sr, _chunk_offsets, _raw_chunks = _synthesize_long_form(
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
        headers={"X-Story-Text-B64": story_b64, "X-Story-Id": str(new_story_id), **_quality_headers(quality_scores)},
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
    scene_text, quality_scores = score_and_maybe_rewrite(scene_text, req.language)

    story_id = db.save_story(title=f"Villain: {villain_name}", language=req.language, text=scene_text)
    if quality_scores.get("scored") and quality_scores["overall"] < QUALITY_THRESHOLD:
        db.save_quality_entry(story_id, quality_scores, rewritten=quality_scores.get("rewrite_attempted", False))
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
        audio, sr, _chunk_offsets, _raw_chunks = _synthesize_long_form(scene_text, req.language, speed, noise_scale=noise_scale, bgm=req.bgm, bgm_fallback=emotion or "horror")
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
            **_quality_headers(quality_scores),
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
