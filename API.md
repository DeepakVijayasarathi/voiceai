# Indic Fast TTS — API Reference

Base URL: `http://178.104.214.191:8604`
Auth: every endpoint except `/health`, `/warmup`, `/languages`, `/metrics` requires header `x-api-key: <key>`.
Rate limit: 60 requests/min per IP by default (`429` with `Retry-After` header when exceeded).

All audio-returning endpoints respond `audio/wav` (mono, 16-bit PCM) unless noted. Story text that comes back alongside audio is delivered as a **base64-encoded UTF-8 response header** (not JSON), since HTTP headers are ASCII-only and story text is in native Indic script — decode with `atob()` + `TextDecoder('utf-8')` in JS, or `base64.b64decode(...).decode('utf-8')` in Python.

Supported language codes (ISO 639-3): `tam` `hin` `tel` `mal` `kan` `mar` `guj` `ben` `pan` `ory` `asm`

---

## Core TTS

### `POST /tts`
Synthesize speech. Long text (multi-sentence) is automatically chunked and stitched.

```json
{
  "text": "string, required, max 6000 chars",
  "language": "hin",            // optional, auto-detected from script if omitted
  "speed": 1.0,                 // optional, 0 < speed <= 2.5
  "emotion": "auto",            // optional: neutral|happy|sad|excited|calm|horror|romance|action|comedy|devotional|mystery|auto
  "bgm": false                  // optional: mix in procedurally-generated background music
}
```
Response: `audio/wav` bytes.

`bgm: true` on multi-sentence text classifies **each chunk's mood independently** (one batched OpenAI call) and shifts the music scene-by-scene, crossfaded at boundaries. Single-chunk text gets one flat genre.

### `POST /stream`
Same request body as `/tts`. Returns audio progressively, sentence-by-sentence, as `application/octet-stream` (raw PCM16 mono, no WAV container — format given in response headers `X-Audio-Format`, `X-Sample-Rate`, `X-Channels`). No BGM support (streaming + scene classification isn't compatible with real-time delivery).

### `POST /clone`
Voice cloning — proxies to the separate **IndicF5** service (this engine can't clone voices itself; see `/health`). **Slow: 30-70s+.**

```json
{
  "text": "string, required, max 500 chars",
  "ref_audio_b64": "base64-encoded reference audio clip",
  "ref_text": "exact transcript of the reference clip"
}
```
Response: `audio/wav` bytes.

---

## AI Storytelling

### `POST /dream-to-story`
Turns a description into a short narrative (OpenAI) and narrates it. Named characters are automatically extracted and saved for reuse.

```json
{
  "description": "string, required, max 2000 chars",
  "language": "hin",            // default "hin"
  "emotion": "auto",            // default "auto"
  "bgm": true                   // default true
}
```
Response: `audio/wav` bytes, plus headers:
- `X-Story-Text-B64` — the generated story (base64 UTF-8)
- `X-Resolved-Emotion` — the classified emotion/genre used
- `X-Story-Id` — saved story's database ID
- `X-Character-Ids` — comma-separated IDs of any characters extracted (empty if none)

### `GET /stories?limit=50`
List saved stories (newest first). Returns metadata only (no full text).
```json
{"stories": [{"id": 2, "title": "...", "language": "hin", "parent_story_id": null, "branch_note": null, "created_at": "..."}]}
```

### `GET /stories/{story_id}`
Full story record including text.

### `POST /stories/{story_id}/branch`
**Story Time Machine (scoped)**: generates a new continuation from a changed decision at some point in an existing story, using the story's own text as context. Does *not* verify consistency against a wider "universe" beyond the given story.

```json
{
  "changed_decision": "string, required, max 1000 chars",
  "emotion": "auto",
  "bgm": true
}
```
Response: `audio/wav` + headers `X-Story-Text-B64`, `X-Story-Id` (new branch), `X-Parent-Story-Id`.

### `GET /characters?limit=50`
List saved characters (newest first).
```json
{"characters": [{"id": 1, "name": "...", "personality": "...", "language": "hin", "origin_story_id": 2, "created_at": "..."}]}
```

### `GET /characters/{character_id}`
Full character record (name, personality, backstory, language, origin story).

### `POST /characters/{character_id}/revive`
**Character Resurrection (scoped)**: writes a new scene placing an existing saved character into a new setting, briefed on their established personality/backstory.

```json
{
  "new_setting": "string, required, max 1000 chars",
  "emotion": "auto",
  "bgm": true
}
```
Response: `audio/wav` + headers `X-Story-Text-B64`, `X-Story-Id` (new story).

### `POST /villain`
**Personalized Villain (scoped)**: designs a villain around **user-stated** fears/motivations (not learned from any listener history — none is tracked in this system) and narrates an introduction scene.

```json
{
  "fears": "string, required, max 1000 chars",
  "language": "hin",
  "emotion": "horror",          // default "horror"
  "bgm": true
}
```
Response: `audio/wav` + headers `X-Story-Text-B64`, `X-Villain-Name-B64` (base64 UTF-8), `X-Story-Id`, `X-Character-Id`.

---

## Service info

### `GET /health` (no auth)
Engine status, capability notes (voice cloning, emotion, BGM — what each actually does and doesn't do).

### `GET /warmup` (no auth)
Forces one synthesis per language; returns per-language timing. Useful post-deploy.

### `GET /languages` (no auth)
`{"languages": {"tam": "Tamil", ...}}`

### `GET /metrics` (no auth)
Prometheus text format: `indic_tts_requests_total`, `indic_tts_requests_failed_total`, `indic_tts_synth_seconds_total`.

### `GET /config` (no auth)
Live runtime configuration, read directly from the running process (so it can't go stale the way a static doc can) — every env-derived setting (secrets reported as `*_set: true/false`, never as values) plus the in-code tunable constants that actually shape behavior: chunk sizes, retry thresholds, BGM/SFX levels and duck amounts, all 7 voice profiles' pitch/EQ/speed values, emotion presets, and which languages have cultural-context grounding. See `CONFIG.md` for the human-readable walkthrough of what each value means and why.

---

## What's deliberately NOT here

- **Story Genome** (auto-extracting "the DNA" of successful stories) — no known implementation approach exists; not attempted.
- **AI Co-Author at scale** (thousands of concurrent readers shaping a live story) — needs real-time multi-user infrastructure that doesn't exist here.
- **True adaptive Personalized Villains** (learned from listener behavior over time) — no listener history is tracked anywhere in this system; `/villain` uses what you tell it, not what it's inferred.
- **Cross-story consistency enforcement** — `/branch` and `/revive` generate against the specific story/character record passed in, not a full checked "canon" spanning every story ever generated.

See `README.md` for architecture rationale and `db.py`'s module docstring for the persistence-layer scope notes.
