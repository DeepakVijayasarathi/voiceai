# Indic Fast TTS — Features & USP

## What makes this different (USP)

- **Characters persist.** Named characters from any generated story are automatically saved with their personality and backstory, and can be pulled into brand-new settings later.
- **Stories branch.** Any saved story can be re-forked from a changed decision at any point, with a consistent AI-generated continuation.
- **Villains are personal.** Not generic antagonists — built around fears/motivations you actually describe.
- **Music and sound follow the story scene-by-scene**, not one flat mood for the whole narration — verified to correctly shift from calm → horror → action as a story moves through its beats.
- **CPU-only, low-latency**, 11 Indian languages, no GPU required.
- Honest about its limits — every feature below says what it actually does, not an inflated claim.

---

## Core TTS

- **11 languages**: Tamil, Hindi, Telugu, Malayalam, Kannada, Marathi, Gujarati, Bengali, Punjabi, Odia, Assamese
- **Auto language detection** from script if not specified
- **Mixed-language handling** — English words inside Indic-script text get transliterated rather than dropped
- **Long-form narration** — text of any length is auto-chunked by sentence and stitched, not capped to one short line
- **Self-healing synthesis** — rare stochastic "stuck repeating" glitches are detected (abnormal chars/sec) and automatically retried
- **Speed control** and **11 emotion/genre presets** (neutral, happy, sad, excited, calm, horror, romance, action, comedy, devotional, mystery), plus `auto` mode that classifies mood from the text itself via OpenAI
- **Tuned voice clarity** — stochastic-expressiveness parameter tuned per emotion for more consistent pronunciation
- **Streaming endpoint** — audio starts playing sentence-by-sentence instead of waiting for the whole thing
- **Voice cloning** (proxied to a separate reference-audio-conditioned model; slower by nature, not a bug)

## Background Music & Sound

- **Procedurally generated**, genre-matched music (11 genres) — chords, rhythm, reverb, no licensing risk since nothing is sampled from external sources
- **Auto-ducking** — music swells in pauses, recedes under speech, so it never fights intelligibility
- **Scene-aware** — a single long story gets its mood re-classified chunk-by-chunk (one batched AI call, not one per chunk) so the music actually shifts as the scene changes, cross-faded smoothly at transitions
- **Environmental sound effects** (14 types): rain, wind, thunder, lightning, fire, footsteps, car, traffic, crowd, door creak, glass break, birds, water, dog bark — layered under the music when the story describes them
- Honest scope: this is a broad hand-built palette an AI picks from, not unlimited generative sound (that would need a different, heavier model)

## AI Storytelling

- **Dream to Story** — describe a dream/memory/scene, get a short cinematic narrative, narrated with matching music/SFX. Named characters are auto-extracted and saved.
- **Story Time Machine (scoped)** — branch any saved story from a changed decision; get a new, consistent continuation from that point
- **Character Resurrection (scoped)** — revive a saved character into a brand-new setting, personality/backstory preserved
- **Personalized Villains (scoped)** — describe a listener's fears/motivations, get a villain and scene built around them specifically
- **Persistent storage** — every story and character is saved (SQLite) and browsable/reusable, not thrown away after one request

## Platform

- Full REST API (15 endpoints) + web UI (Generate / Clone Voice / Story Tools)
- API key auth + rate limiting
- Health/metrics endpoints for monitoring

## What's deliberately NOT included (and why)

- **Story Genome** (auto-extracting "the DNA" of successful stories) — no known implementation approach exists
- **AI Co-Author at scale** (thousands of concurrent readers shaping one live story) — needs real-time multi-user infrastructure not built here
- **True multi-voice/gender casting** — the underlying TTS model is one fixed voice per language; a real fix needs a different (much slower) voice-cloning backend per character
- **Cross-story consistency enforcement** — branching/revival generate against the specific story/character passed in, not a fully checked canon across every story ever made
- **"Undetectable as AI"** — not promised; it's a synthetic voice, just a clearer and more produced-sounding one
