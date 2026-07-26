# Infinity Studio — Cultural Story Operating System

> One story. Infinite cultures.

Built for the listener the internet still dubs over — the same character,
the same moment, told with someone's own festivals, landmarks, and voice,
in whichever of 11 languages they reach for. Switch languages mid-story
and nothing breaks: not the plot, not the character, not the feeling.

| | | |
|---|---|---|
| Tamil · Madurai | Hindi · Lucknow | Bengali · Kolkata |

> "நான் அந்த கடிதத்தை படித்துவிட்டேன்," என்று அர்ஜுன் மெதுவாகச் சொன்னான்,
> மீனாக்ஷி அம்மன் கோவில் முற்றத்தில் நிலவொளியில்.
>
> "I read the letter," Arjun said quietly, in the moonlit courtyard of the
> Meenakshi Amman Temple.

## Problem — why storytelling breaks at scale

**Every story either speaks one language, or stops feeling real.**

Today's tools force a choice. Write once and translate, and the culture
goes flat — a temple becomes "a temple," a festival becomes a footnote. Or
hand-localize per region, and cost climbs in a straight line with every
language you add. Either way, a listener who switches languages mid-story
loses the thread entirely, because the script and the state are the same
file.

1. **Culture becomes a word-swap.** Translation keeps the sentence and
   loses the place, the festival, the register — technically correct,
   quietly foreign.
2. **A language switch breaks the story.** Script and state live in the
   same artifact, so changing language usually means starting over, not
   continuing.
3. **Cost scales with every region.** Hand-localizing means a new voice
   cast, a new studio pass, a new QA cycle — every single time.

## Solution — what Infinity Studio does instead

**Split the story from its skin.**

The plot, the characters, the emotional beats live in one place — a state
graph, in no particular language. A separate layer renders that state into
a specific culture's language, idiom, and references, fresh, every time
someone asks for it. Change the culture pack, and the same story comes out
sounding like it was always theirs.

- **State, not script.** The plot is data. Prose in any language is
  generated from it on demand — never the source of truth, never the
  thing that has to be rewritten.
- **A language switch is a re-render.** Not a restart, not a
  re-authoring request — the same moment in the story, rendered again
  through a different cultural lens.
- **Quality that's checked, not assumed.** Every render is scored for how
  authentic it actually sounds before it reaches a listener, and weak
  spots get caught automatically.

## Reach — social & cultural impact

Hindi alone outnumbers every Romance language combined. It's still an
afterthought. Add Tamil, Telugu, Malayalam, Kannada, Marathi, Gujarati,
Bengali, Punjabi, Odia, and Assamese, and you have hundreds of millions of
listeners most storytelling platforms treat as a translation checkbox —
one generic temple, one generic festival, one flattened voice, no matter
which of them is listening.

- **11** Indian languages, each with its own culture pack — not a shared
  script with words swapped out.
- **1** Meenakshi Amman Temple for a Madurai listener, a different
  landmark entirely for Lucknow — never the same generic stand-in.
- **0** listeners asked to wait for "their" language to get a real
  release, months after the first one.

## Cost — market impact

**One performance. Every region. The cost doesn't climb with it.**

Traditional dubbing scales linearly — a new region means a new studio
booking, a new voice cast, a new QA pass. Infinity Studio scales by adding
a culture pack: a document, not a production.

| Translation-first pipeline | Infinity Studio |
|---|---|
| Re-cast a voice actor for the new language | A Voice Bank entry per locale — a reference clip and signed consent, not a studio session *(roadmap: reuse across languages)* |
| Re-record every line in a studio | A new culture pack: idiom, festivals, register, landmarks |
| Re-QA the whole thing for that region | The story re-renders itself against it, automatically |
| Repeat, in full, for the next region | Add the next region the same way — no re-shoot |

## Experience — what the listener actually feels

**The story remembers them back.**

1. **Characters persist.** A named character from any story is saved with
   their personality and backstory — ready to walk into a brand-new
   setting later.
2. **Villains are personal.** Not a generic antagonist — built around the
   fears and motivations a listener actually describes.
3. **Stories branch.** Re-fork any saved story from a changed decision,
   with a continuation that stays consistent with everything before it.
4. **Audiobook mode** *(building now)*. Narration layered with a
   generated soundscape that turns with the scene — atmosphere with no
   video track required.

## Quality — the part that keeps getting better

**Cultural authenticity, measured — not a one-time guess.**

Every render gets scored by a second, independent model on four things a
listener actually notices: does it sound native, is the tone right for
who's speaking, are the local references correct, does the emotion land.
Low scores get one automatic rewrite before a human ever sees it — and the
weak spots become a visible backlog instead of a quiet, compounding
problem.

## Compare — why this wins, plainly

| | Translation-first | Infinity Studio |
|---|---|---|
| New region | Re-shoot, re-record, re-QA | Add a culture pack |
| Mid-story language switch | Breaks continuity, often restarts | Same character, same moment, new render |
| Cultural accuracy | Fixed at launch, decays as coverage grows | Continuously scored, self-correcting |
| Voice | A different actor per dub, re-cast each time | A consent-gated cloned voice per locale — no studio re-cast, no re-shoot *(roadmap)* |
| Reach | Usually one medium | Audio, video, game, and comic from one story |

## The ask

**Reach more listeners. Spend less doing it. Make them feel remembered.**

That's the pitch in one line — everything above is one of those three
promises, kept:

- **Reach** — watch a story switch from Tamil to Hindi mid-scene, live,
  with nothing lost.
- **Cost** — see a new region come online from a culture pack, no studio,
  no re-cast.
- **Experience** — meet a villain built from a listener's own fears, and
  a story that remembers them next time.

---

# Indic Fast TTS

The engine underneath the pitch above. Low-latency multilingual
text-to-speech for 11 Indian languages, purpose-built for CPU-only
serving (no GPU on the target box), plus the AI storytelling layer
(story/branch/revive/villain generation, scene illustrations, quality
scoring, multi-format export) built on top of it. See
[FEATURES.md](FEATURES.md) for the full, honestly-scoped feature list —
including what's still roadmap (like the Voice Bank above) versus what's
actually running today.

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
| Voice cloning / custom speakers | Proxied to a separate reference-audio-conditioned service (`/clone`) - not a persistent, consent-gated Voice Bank yet (see the Cost section above for that roadmap item) |
| Streaming | Sentence-level progressive streaming (`/stream`) - VITS is non-autoregressive so sub-sentence chunk streaming isn't meaningful for this architecture |
| AI storytelling (Dream to Story, branching, character revival, villains, language switching, quality scoring, scene images, comic/video/game export) | Yes - see [FEATURES.md](FEATURES.md) |

## API

All endpoints except `/health` and `/metrics` require `x-api-key`.

- `POST /tts` - `{"text": "...", "language": "hin", "speed": 1.0, "emotion": "happy"}` -> WAV bytes
- `POST /stream` - same body -> chunked raw PCM16 mono (`X-Sample-Rate`, `X-Channels` response headers tell you the format; no WAV container since a valid one needs the total size up front)
- `GET /health` - engine status, capability notes
- `GET /warmup` - forces one synthesis per language, returns timing (useful post-deploy to pre-warm caches)
- `GET /languages` - supported language list
- `GET /metrics` - Prometheus-text-format counters (request count, failures, total synth time)

See [FEATURES.md](FEATURES.md) for the full AI-storytelling and
multi-format-export endpoint list (Dream to Story, branching, character
revival, villains, language switching, scene images, comic, video,
game-tree, quality backlog).

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
- **Voice Bank (consent-gated, reusable cloned voices per locale) is
  still roadmap**, not implemented - see the Cost/Compare sections above
  and [FEATURES.md](FEATURES.md)'s "What's deliberately NOT included"
  for current scope.
