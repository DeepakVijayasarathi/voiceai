# Indic Fast TTS — Complete Configuration Reference

## Environment variables

Set in `/root/indic-tts-fast/env` on the server (loaded by systemd via `EnvironmentFile=`). Copy `.env.example` as a starting point for a fresh deploy.

| Variable | Default | Currently set (live) | Purpose |
|---|---|---|---|
| `INDIC_TTS_API_KEY` | *(none — auth disabled if unset)* | `hqC3S-oePGWbMot_HuwmvI-YqrrGoa58lKqJh8nnFhg` | Required `x-api-key` header value for all protected endpoints |
| `INDIC_TTS_THREADS` | `4` | `4` | PyTorch CPU thread count for VITS inference |
| `INDIC_TTS_RATE_LIMIT_PER_MIN` | `60` | `60` | Per-IP request cap before `429` |
| `INDIC_TTS_DB_PATH` | `/root/indic-tts-fast/api/stories.db` | *(unset, using default)* | SQLite file location for stories/characters |
| `OPENAI_API_KEY` | *(none — story/emotion features degrade or fail)* | set | Powers story generation, character extraction, scene classification, villain generation |
| `OPENAI_STORY_MODEL` | `gpt-4o` | *(unset, using default)* | Model for story gen, branching, revival, villain — creative/generation calls |
| `OPENAI_EMOTION_MODEL` | `gpt-4o` | *(unset, using default)* | Model for genre/SFX/speaker classification — was `gpt-4o-mini` before the clarity-focused upgrade |
| `XLIT_BASE_URL` | `http://127.0.0.1:8501` | `http://127.0.0.1:8501` | IndicXlit transliteration service (mixed-language handling) |
| `XLIT_API_KEY` | *(none)* | set | Auth for the transliteration service, if it requires one |
| `INDICF5_BASE_URL` | `http://127.0.0.1:8500` | `http://127.0.0.1:8500` | IndicF5 voice-cloning service (used by `/clone`) |
| `INDICF5_API_KEY` | *(none)* | set | Auth for the IndicF5 service, if it requires one |

## Per-request parameters

These are already documented in `API.md` (request body fields), not repeated here — `speed`, `emotion`, `bgm`, `language`, etc.

## In-code tunable constants (not env vars — require a code change + redeploy)

These are the actual "knobs" behind the tuning work done this session. Listed so future tuning starts from the right number instead of guessing.

### `app.py`
| Constant | Value | Meaning |
|---|---|---|
| `MAX_TEXT_LEN` | `6000` | Max `/tts` request text length (chunked automatically above one sentence) |
| `_CHUNK_CHAR_BUDGET` | `300` | Per-chunk synthesis size for normal long-form text |
| `_DIALOGUE_CHUNK_CHAR_BUDGET` | `120` | Smaller budget used when `known_characters` is active, so back-and-forth dialogue lands in separate chunks for correct speaker attribution |
| `_INTER_CHUNK_SILENCE_S` | `0.3` | Silence gap inserted between stitched chunks |
| `EMOTION_PRESETS` | per-emotion `(speed, noise_scale)` | Speed multiplier and VITS clarity/expressiveness tuning per emotion — all noise_scale values sit at or below VITS's own default (0.667) |

### `engine.py`
| Constant | Value | Meaning |
|---|---|---|
| `_MIN_CHARS_PER_SEC` | `4.0` | Below this, a synthesis attempt is flagged as the rare VITS "stuck repeating" failure and retried |
| `_MAX_SYNTH_RETRIES` | `3` | Retry cap for the above |
| `_DEFAULT_NOISE_SCALE` | `0.6` | Fallback when no emotion-specific value applies |
| `_DEFAULT_NOISE_SCALE_DURATION` | `0.6` | Below VITS's default (0.8) — reduces the duration-predictor variance that causes the runaway-repetition bug in the first place |

### `bgm.py`
| Constant | Value | Meaning |
|---|---|---|
| `bgm_level` | `1.75` | BGM base level (recalibrated from `0.55` after measuring it was ~29dB below speech, effectively inaudible) |
| `sfx_level` | `0.19` | SFX base level (recalibrated from `0.35` — was backwards, louder than BGM) |
| `bgm_duck_amount` | `0.85` | How hard music ducks under speech (mood cue — should mostly get out of the way) |
| `sfx_duck_amount` | `0.55` | How hard SFX ducks under speech (ambiance — should stay present as room tone) |
| loop `duration_s` (genre tracks) | `32.0` | Base loop length before tiling — doubled from `16.0` to reduce audible repetition on long stories |
| bar-to-bar gain jitter | `±6%` | Humanization so sequenced music doesn't feel mechanically identical bar-to-bar |

### `sfx.py`
| Constant | Value | Meaning |
|---|---|---|
| loop `duration_s` | `24.0` | Base SFX loop length before tiling (up from `16.0`) |
| SFX reverb mix | `0.18` | Applied in `bgm.py`'s scene mixer, not in `sfx.py` itself — spatial blend for the SFX layer |

### `voice_profile.py`
| Constant | Value | Meaning |
|---|---|---|
| `_PROFILES` | 7 named profiles | Each: `(pitch_shift_semitones, shelf_eq_tilt_db, speed_multiplier)` — e.g. `"deepest": (-3.5, -3.5, 0.92)` |
| Max pitch shift | `±3.5` semitones | Deliberately capped — larger shifts sound artificial on a single-speaker model never trained for this |
| Max speed multiplier | `0.92`–`1.08` | ±8% pace variation, deliberately modest |

### `story_gen.py`
| Constant | Value | Meaning |
|---|---|---|
| `TARGET_CHAR_BUDGET` | `700` | Requested story length (soft target for the model, hard-truncated to 950 after) |
| `LANGUAGE_CULTURAL_CONTEXT` | per-language landmarks | Real regional places the model is told to prefer over generic/invented settings |

## Deployment files (not runtime config, but part of "all config")

| File | Purpose |
|---|---|
| `indic-tts-fast.service` | systemd unit — runs uvicorn, loads `env` |
| `indic-tts-fast.nginx` | Reverse proxy for the API (port 8604 → 8503) |
| `indic-tts-ui.nginx` | Reverse proxy for the static UI (port 8605) |
| `docker-compose.yml` / `Dockerfile` | Containerized alternative deployment (built but not what's currently running — the live deploy is bare systemd + nginx) |
| `pytest.ini` | Points pytest at `tests/` |
| `requirements.txt` / `requirements-dev.txt` | Runtime vs. test dependencies |
