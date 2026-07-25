# Indic Fast TTS

Low-latency multilingual text-to-speech for 11 Indian languages, purpose-built
for CPU-only serving (no GPU on the target box).

## Why this exists, and why it's a separate service from IndicF5

This box also runs `indicf5-api`, a voice-cloning TTS service built on
IndicF5 (F5-TTS, flow-matching, ~330M params). That architecture needs
dozens of iterative denoising steps per request - on this CPU-only
16-core box it takes 30-70+ seconds per sentence. It's the right tool if
you need to clone an arbitrary voice from a reference clip, and the wrong
tool if you need low latency.

This service trades voice cloning for speed: [MMS-TTS](https://huggingface.co/facebook/mms-tts-hin)
(VITS architecture, ~36M params/language, non-autoregressive single-pass
synthesis) hits **226-633ms per sentence** on the same hardware - see
`benchmark/latency_report.json` for the full per-language breakdown from
the last benchmark run. If you need both a cloned voice AND sub-second
latency on CPU, that combination isn't achievable with today's open
models - you'd need a GPU.

## Supported languages

Tamil, Hindi, Telugu, Malayalam, Kannada, Marathi, Gujarati, Bengali,
Punjabi, Odia, Assamese (ISO 639-3 codes: tam, hin, tel, mal, kan, mar,
guj, ben, pan, ory, asm).

## What it does and does not do

| Capability | Status |
|---|---|
| Text-to-speech, 11 languages | Yes |
| Language auto-detection (script-based) | Yes |
| Mixed-language text (e.g. Tamil + embedded English) | Yes, via the existing `xlit-api` transliteration service |
| Number normalization | Yes (digit-by-digit, not natural "twenty-five" grammar - see `api/normalize.py` for why) |
| Speed control | Yes (native VITS `speaking_rate`) |
| "Emotion" control | Approximate speed/expressiveness heuristic only - **not** learned emotion conditioning |
| Voice cloning / custom speakers | **No** - architectural tradeoff for speed, see above. Requests with a `speaker` field get a clear 501, not silent ignoring. |
| Streaming | Sentence-level progressive streaming (`/stream`) - VITS is non-autoregressive so sub-sentence chunk streaming isn't meaningful for this architecture |

## API

All endpoints except `/health` and `/metrics` require `x-api-key`.

- `POST /tts` - `{"text": "...", "language": "hin", "speed": 1.0, "emotion": "happy"}` -> WAV bytes
- `POST /stream` - same body -> chunked raw PCM16 mono (`X-Sample-Rate`, `X-Channels` response headers tell you the format; no WAV container since a valid one needs the total size up front)
- `GET /health` - engine status, capability notes
- `GET /warmup` - forces one synthesis per language, returns timing (useful post-deploy to pre-warm caches)
- `GET /languages` - supported language list
- `GET /metrics` - Prometheus-text-format counters (request count, failures, total synth time)

`language` is optional - omitted, it's auto-detected from the dominant
Unicode script. Devanagari (Hindi/Marathi) and Bengali-script
(Bengali/Assamese) pairs are disambiguated with a small function-word
list; pass `language` explicitly if you need certainty for those four.

## Running it

**Already deployed on this box** as systemd unit `indic-tts-fast`,
internal port 8503, public via nginx on port 8604
(`http://<server-ip>:8604`). Env/secrets in `/root/indic-tts-fast/env`.

**Via Docker**, elsewhere:
```
cp .env.example .env   # fill in INDIC_TTS_API_KEY (and XLIT_API_KEY if you have an xlit-api instance)
docker compose up -d
```
The image bakes all 11 language models in at build time, so a fresh
container needs no network access to start serving.

**Important - tested on this box, doesn't fit on this box.** A build was
run here to validate the Dockerfile: all layers built correctly (deps
installed, all 11 models downloaded and loaded inside the build step)
and it failed only at the final image-export step with "no space left on
device" - this VPS has 38GB total, already at ~29GB used by the existing
services/models, and a build needs roughly 10GB of headroom to export the
final image on top of its build cache. It briefly filled the disk to 100%
before cleanup (`docker system prune -af --volumes`, which recovered it
back to 7.7GB free with no impact on the other running services - verified
by health-checking all of them afterward). **Don't retry this build on
this box** without first attaching more disk; build it on a machine/CI
runner with at least ~12GB free instead.

## Known limitations / next steps if this needs to scale further

- **Rate limiting is per-process, in-memory.** Fine for one instance; if
  this ever runs as multiple replicas, each gets its own budget rather
  than a shared global one - would need a Redis-backed limiter at that
  point (not built now since there's no multi-replica deployment yet).
- **`xlit-api` cold-starts per language** (~10-15s on first use after its
  own restart), which the first mixed-language request per language pays.
  Worth adding a startup warmup loop to `xlit-api` itself if mixed-language
  traffic is common.
- **No real HTTPS.** No domain is pointed at this box (nginx configs are
  all `server_name _`), so Let's Encrypt isn't usable as-is - it needs
  domain validation. Get a domain pointed here first if TLS matters.
- **ONNX/OpenVINO/INT8 quantization were deliberately not pursued** - the
  latency target is already met with plain PyTorch for most languages, so
  the added complexity wasn't justified. Revisit if real traffic shows a
  need (Punjabi, the current slowest at ~633ms, would be the first
  candidate).
- **Number normalization is digit-by-digit**, not natural place-value
  grammar (e.g. Indian lakh/crore convention). Correct and unambiguous
  today; a native-speaker-reviewed per-language grammar would sound more
  natural but wasn't in scope to get right for 11 languages without
  verification.
