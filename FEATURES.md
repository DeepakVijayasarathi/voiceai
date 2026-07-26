# Indic Fast TTS — Features & USP

## What makes this different (USP)

- **Characters persist.** Named characters from any generated story are automatically saved with their personality and backstory, and can be pulled into brand-new settings later.
- **Stories branch, with real history.** Any saved story can be re-forked from a changed decision at any point, with a consistent AI-generated continuation - and a branch-of-a-branch is generated aware of its FULL lineage of prior changes, not just the one step directly above it.
- **Villains are personal.** Not generic antagonists — built around fears/motivations you actually describe.
- **Music and sound follow the story scene-by-scene**, not one flat mood for the whole narration — verified to correctly shift from calm → horror → action as a story moves through its beats.
- **Stories switch language**, not just get translated. A saved story can be adapted into any of the 11 supported languages - same plot, same characters, natural phrasing and names for that language - and switched between later without losing the story's identity.
- **Generated text is judged, not just generated.** A second, independent AI call scores every story on 4 axes real listeners notice (native-sounding phrasing, tone fit, cultural accuracy, emotional delivery) before it's narrated; a low score triggers one automatic rewrite, and anything still weak stays visible in a backlog instead of quietly shipping.
- **One story, four formats.** The same generated story can be listened to as narrated audio, viewed as scene-illustrated "images that follow the story," read as comic panels, exported as an MP4 slideshow, or walked as a branching choice-tree - all from data this service already generates, no format-specific re-authoring.
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
- **Voice cloning** (proxied to a separate reference-audio-conditioned model; slower by nature, not a bug). Scope note: this is a per-request proxy only - there's no persistent "voice bank" of saved, consent-gated reference voices reusable across requests/languages yet; each /clone call needs its own reference clip.

## Background Music & Sound

- **Procedurally generated**, genre-matched music (11 genres) — chords, rhythm, reverb, no licensing risk since nothing is sampled from external sources
- **Auto-ducking** — music swells in pauses, recedes under speech, so it never fights intelligibility
- **Scene-aware** — a single long story gets its mood re-classified chunk-by-chunk (one batched AI call, not one per chunk) so the music actually shifts as the scene changes, cross-faded smoothly at transitions
- **Environmental sound effects** (14 types): rain, wind, thunder, lightning, fire, footsteps, car, traffic, crowd, door creak, glass break, birds, water, dog bark — layered under the music when the story describes them
- Honest scope: this is a broad hand-built palette an AI picks from, not unlimited generative sound (that would need a different, heavier model)

## AI Storytelling

- **Dream to Story** — describe a dream/memory/scene, get a short cinematic narrative, narrated with matching music/SFX. Named characters are auto-extracted and saved.
- **Story Time Machine (scoped, now lineage-aware)** — branch any saved story from a changed decision; get a new, consistent continuation from that point. When branching a branch, the generation is given condensed context of every earlier changed-decision in that story's history, not just its one direct parent.
- **Character Resurrection (scoped)** — revive a saved character into a brand-new setting, personality/backstory preserved
- **Personalized Villains (scoped)** — describe a listener's fears/motivations, get a villain and scene built around them specifically
- **Language switching (scoped)** — adapt any saved story into another supported language: same plot, same characters, phrasing and names natural for the new language rather than a literal translation. Saved as a linked variant (`GET /stories/{id}/variants`), not a single audio file that changes language mid-sentence - the underlying TTS engine is a separate model per language, so that's not a meaningful operation here.
- **Culture packs across six dimensions** — each language's guidance now covers real places, festivals, regional folklore/spirit motifs (for horror/mystery), devotional figures (for devotional content, left out entirely for languages like Punjabi where that framing doesn't fit rather than forced), authentic first names for new characters, and a natural address-register note (formality/kinship terms) — not just place names, so dialogue and scene-setting can feel local rather than translated. Still soft guidance, never forced onto a story that doesn't call for it.
- **Quality scoring + auto-rewrite (scoped to text)** — every generated story/continuation/scene is scored 0-100 on native-sounding phrasing, tone fit, cultural accuracy, and emotional delivery by an independent second AI call. A low score triggers one automatic rewrite targeting the weak axes; whichever version scores higher is what gets narrated. This judges the generated TEXT only - nothing on this CPU-only box evaluates the synthesized AUDIO's pronunciation/prosody. Stories still weak after the rewrite attempt are listed at `GET /quality/backlog`.
- **Persistent storage** — every story and character is saved (SQLite) and browsable/reusable, not thrown away after one request

## Scene Illustrations & Multi-Format Export

Two distinct, independently-opt-in image features on Dream to Story - not one feature two ways:

- **Cover image (`visual=true`)** — a single AI-generated image depicting the story's strongest visual moment, via OpenAI's image API. Retrieve via `GET /stories/{id}/image`. Failure is non-fatal (audio still returns), reported via `X-Has-Image`/`X-Image-Error-B64` headers.
- **Cover video (`video=true`, implies `visual=true`)** — that same cover image held for the full narration, composed into a real downloadable/shareable MP4 via `ffmpeg`. Retrieve via `GET /stories/{id}/video`. Deliberately NOT AI-generated moving video (a fundamentally different, much slower/costlier model this service doesn't have) - an honest static-image-plus-audio mux.
- **Scene storyboard (`images=true`, also available on plain `/tts`)** — a handful (up to `max_images`) of AI illustrations, each mapped to an approximate time range in the narration (`GET /stories/{id}/images`), so a player can show the right image as playback reaches that point. Off by default: real per-image latency and OpenAI cost on top of narration. Best-effort - a failed prompt/image drops that one scene rather than failing the whole request.
- **Comic panels** — `GET /stories/{id}/comic` returns each storyboard image paired with the native-script story excerpt it came from, as JSON panels for a client to lay out. No server-side speech-bubble compositing (that would need a bundled Unicode font covering all 11 Indic scripts) - the image and its real story text are handed over as data, not a baked-in picture.
- **Storyboard video export** — `POST /stories/{id}/render/video` builds an MP4 slideshow from the storyboard images (each held for its time range) muxed with the story's saved narration audio, via `ffmpeg`. Requires the story to have been generated with `images=true` and the `ffmpeg` binary to be present on the host - checked at call time (returns a clear error rather than crashing if missing). Distinct from the single-cover-image video above.
- **Game tree** — `GET /stories/{id}/game-tree` exposes the existing branch feature as a navigable, recursive choice-tree (each node's children are the branches made from it). "Game" here means interactive fiction you can walk through, not a game engine.

## Platform

- Full REST API (25 endpoints) + web UI (Overview / Narrate / Clone Voice / Dream to Story / Villain / Stories & Branches / Characters / Quality backlog / Services / Config)
- **`GET /config`** — live runtime configuration (env-derived settings with secrets reported as set/unset only, chunk sizes, BGM/SFX mix levels, all voice profile values, emotion presets, cultural-context language coverage) read directly from the running process, so it can't drift out of date the way a static doc can.
- API key auth + rate limiting
- Health/metrics endpoints for monitoring

## What's deliberately NOT included (and why)

- **Voice Bank / consent-gated cloned voices reusable across languages** — on hold. `/clone` remains a per-request proxy to a separate cloning service; there's no persistent, consent-recorded reference voice a listener sets up once and reuses.
- **Story Genome** (auto-extracting "the DNA" of successful stories) — no known implementation approach exists
- **AI Co-Author at scale** (thousands of concurrent readers shaping one live story) — needs real-time multi-user infrastructure not built here
- **True multi-voice/gender casting** — the underlying TTS model is one fixed voice per language; a real fix needs a different (much slower) voice-cloning backend per character
- **Full cross-story-universe consistency enforcement** — branching now considers a story's full lineage of prior changed-decisions (not just its direct parent), and revival/villain generation is seeded with the specific character/story passed in, but nothing here checks generated output against every story ever made as one shared canon - that would need a much larger context-management system than a handful of OpenAI calls.
- **Audio-level quality judging** — the quality-scoring pass judges generated TEXT only; nothing on this CPU-only box evaluates the synthesized audio's actual pronunciation or prosody.
- **Server-side comic speech-bubble compositing** — comic export hands over images + real caption text as JSON; baking native-script text into the image itself would need a bundled Unicode font per script, which isn't included.
- **"Undetectable as AI"** — not promised; it's a synthetic voice, just a clearer and more produced-sounding one
