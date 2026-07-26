"""
Per-character voice differentiation.

Scope note (important): this engine (MMS-TTS/VITS) has exactly one voice
per language - there is no multi-speaker model backing this. What's here
is NOT true separate voices; it's a DSP transform applied to that same
base voice, deterministically assigned per character name, so a
recurring character sounds consistently different from the narrator and
from other characters every time they speak. A real multi-voice cast
would need a different (and much slower) voice-cloning backend per
character; see app.py's /health voice_note.

Implementation: pitch and formants are shifted INDEPENDENTLY, not
together. A naive pitch shift (resample-and-restretch, what a single
`librosa.pitch_shift` call does on its own) moves the fundamental
frequency AND the vocal-tract resonances (formants) by the same ratio -
which is exactly what happens when you speed up or slow down a
recording, and it's why that alone reads as "chipmunk" or "monster"
rather than "a different person". Real voices don't work that way: a
child's voice isn't an adult's pitched up by a fixed ratio, its formants
sit relatively higher too (smaller vocal tract), and a deep authoritative
voice has formants pulled down along with the pitch, not independent of
it.

So here: _formant_shift warps the spectral ENVELOPE only (via frequency-
axis smoothing to separate the slowly-varying formant shape from the
fast pitch-harmonic detail, then re-imposing a frequency-warped version
of that envelope while leaving the harmonic detail's frequency positions
untouched) - this changes vocal-tract-size impression without touching
pitch at all. `librosa.effects.pitch_shift` is then applied on top for
the actual pitch target, which also carries the already-warped envelope
along with it proportionally - so a profile's formant_ratio is really
"how much the formants move BEYOND what the pitch shift alone would
carry them", which is what lets two profiles share a similar pitch but
still sound like different-sized speakers, or vice versa.

Plus a light shelf-EQ tilt for a bit more timbral separation, plus a
slight speaking-pace offset (VITS's native speaking_rate - the same
clean, artifact-free knob already used for emotion-based speed, see
app.py's EMOTION_PRESETS) so differentiation doesn't rest on pitch/
formants alone. Assignment is a stable hash of the character's name, so
"Priya" gets the same profile in every story she appears in, this
session or a future one.
"""
import hashlib

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt

# (pitch shift in semitones, shelf-EQ tilt in dB at the shelf frequency -
# positive brightens/thins, negative darkens/warms, a speaking-pace
# multiplier applied at synthesis time, and a formant_ratio - how much
# EXTRA the vocal-tract envelope shifts on top of what the pitch shift
# alone would carry it, see module docstring). Most shifts are still
# deliberately modest (large ones start to sound artificial on a
# single-speaker model never trained for this) - "child" is the one
# deliberate exception, and only viable at all because formant
# decoupling keeps it from reading as a simple pitched-up adult; even
# so it's still a DSP trick on one adult voice, not a real child speaker.
# Eight profiles so a typical small story cast (2-4 named characters) is
# unlikely to collide onto the same one.
_PROFILES = {
    "deepest": (-3.5, -3.5, 0.92, 0.93),
    "deep": (-2.0, -2.0, 0.96, 0.96),
    "warm": (-1.0, -1.0, 1.00, 0.99),
    "neutral_shift": (0.5, 0.5, 1.00, 1.00),
    "bright": (1.5, 1.5, 1.03, 1.02),
    "high": (2.5, 2.5, 1.06, 1.04),
    "highest": (3.5, 3.5, 1.08, 1.06),
    "child": (4.0, 2.0, 1.10, 1.12),
}
PROFILE_NAMES = list(_PROFILES.keys())
_SHELF_FREQ_HZ = 1500

# Spectral envelope extraction/warping params for formant shifting.
_STFT_N_FFT = 1024
_STFT_HOP = 256
# Frequency-bin width of the moving-average smoothing that separates the
# slowly-varying formant envelope from the fast pitch-harmonic detail -
# wide enough to average out harmonic spacing at typical speech F0
# (~85-300Hz => harmonics every few bins at this FFT size/sample rate),
# narrow enough to still track real formant peaks (a few hundred Hz wide).
_ENVELOPE_SMOOTH_BINS = 15


def get_voice_profile(character_name: str) -> str:
    """Deterministic: the same name always maps to the same profile (used
    standalone when only one character's name is known, e.g. /revive).
    For a full story cast, prefer assign_profiles instead - it resolves
    collisions so characters appearing together don't end up identical."""
    digest = hashlib.sha1(character_name.strip().lower().encode("utf-8")).hexdigest()
    return PROFILE_NAMES[int(digest, 16) % len(PROFILE_NAMES)]


def assign_profiles(character_names: list[str], personality_hints: dict[str, str] | None = None) -> dict[str, str]:
    """Assigns each name in `character_names` a profile. `personality_hints`
    (name -> profile, from story_gen.extract_characters's personality-based
    suggestion) is preferred when given and valid; otherwise falls back to
    the name's hash-based profile. Either way, reassigns to the next
    unused profile (in a stable, deterministic order) whenever two names
    in the SAME list would otherwise collide - guarantees every character
    in a given story's cast is distinguishable from the others, as long as
    the cast is no larger than len(PROFILE_NAMES). A personality hint
    takes priority over hash-collision-avoidance for ITS OWN name, but a
    later name can still bump an earlier hint-based assignment if they'd
    otherwise collide - first-come-first-served in `character_names`
    order, same as the hash-only path."""
    personality_hints = personality_hints or {}
    assigned: dict[str, str] = {}
    used: set[str] = set()
    for name in character_names:
        hint = personality_hints.get(name)
        preferred = hint if hint in PROFILE_NAMES else get_voice_profile(name)
        if preferred not in used:
            assigned[name] = preferred
            used.add(preferred)
            continue
        for candidate in PROFILE_NAMES:
            if candidate not in used:
                assigned[name] = candidate
                used.add(candidate)
                break
        else:
            assigned[name] = preferred  # cast bigger than palette - reuse is unavoidable
    return assigned


def _shelf_eq(audio: np.ndarray, sr: int, tilt_db: float) -> np.ndarray:
    if tilt_db == 0:
        return audio
    nyquist = sr / 2
    normalized_freq = min(_SHELF_FREQ_HZ / nyquist, 0.99)
    sos = butter(2, normalized_freq, btype="high" if tilt_db > 0 else "low", output="sos")
    filtered = sosfilt(sos, audio)
    gain = 10 ** (abs(tilt_db) / 20) - 1
    return (audio + filtered * gain).astype(np.float32)


def _warp_envelope(envelope: np.ndarray, ratio: float) -> np.ndarray:
    """envelope: (n_bins, n_frames) smoothed log-magnitude. Stretches/
    compresses it along the frequency axis by `ratio` (>1 moves content
    to higher bins - smaller-vocal-tract impression; <1 moves it lower),
    holding the edge value flat past the input's range rather than
    extrapolating or wrapping."""
    n_bins = envelope.shape[0]
    orig_idx = np.arange(n_bins)
    src_idx = orig_idx / ratio
    warped = np.empty_like(envelope)
    for f in range(envelope.shape[1]):
        warped[:, f] = np.interp(src_idx, orig_idx, envelope[:, f])
    return warped


def _formant_shift(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Shifts the spectral envelope (formants) by `ratio` WITHOUT moving
    pitch - the fine harmonic structure (which encodes F0) is carried
    over unchanged; only its amplitude envelope (which encodes vocal-
    tract shape) is re-warped. See module docstring for why this has to
    be separate from a plain pitch shift."""
    if abs(ratio - 1.0) < 1e-3:
        return audio
    stft = librosa.stft(audio.astype(np.float32), n_fft=_STFT_N_FFT, hop_length=_STFT_HOP)
    mag = np.abs(stft)
    phase = np.angle(stft)
    log_mag = np.log(mag + 1e-6)
    envelope = uniform_filter1d(log_mag, size=_ENVELOPE_SMOOTH_BINS, axis=0)
    warped_envelope = _warp_envelope(envelope, ratio)
    new_log_mag = log_mag - envelope + warped_envelope
    new_mag = np.exp(new_log_mag)
    new_stft = new_mag * np.exp(1j * phase)
    out = librosa.istft(new_stft, hop_length=_STFT_HOP, length=len(audio))
    return out.astype(np.float32)


def get_speed_multiplier(profile: str) -> float:
    """1.0 (no change) for unrecognized profiles (e.g. 'narrator')."""
    return _PROFILES.get(profile, (0, 0, 1.0, 1.0))[2]


def apply_voice_profile(audio: np.ndarray, sr: int, profile: str) -> np.ndarray:
    """No-op if `profile` is unrecognized (e.g. 'narrator') - returns
    `audio` unchanged. Only handles the pitch/formant/EQ part of the
    treatment - the speaking-pace part (get_speed_multiplier) is applied
    separately, at synthesis time, since it needs to be passed into
    engine.synthesize before audio exists rather than adjusted after the
    fact. Formant warp runs BEFORE the pitch shift - see module
    docstring for why the order matters."""
    if profile not in _PROFILES:
        return audio
    semitones, tilt_db, _speed, formant_ratio = _PROFILES[profile]
    shifted = _formant_shift(audio.astype(np.float32), sr, formant_ratio)
    shifted = librosa.effects.pitch_shift(shifted, sr=sr, n_steps=semitones)
    shifted = _shelf_eq(shifted, sr, tilt_db)
    peak = np.abs(shifted).max()
    if peak > 0.98:
        shifted = shifted / peak * 0.98
    return shifted.astype(np.float32)
