"""
Per-character voice differentiation.

Scope note (important): this engine (MMS-TTS/VITS) has exactly one voice
per language - there is no multi-speaker model backing this. What's here
is NOT true separate voices; it's a pitch/timbre transform applied to
that same base voice, deterministically assigned per character name, so
a recurring character sounds consistently different from the narrator and
from other characters every time they speak - closer to "a distinct
character voice" than one flat voice for everyone, but still one
underlying synthesis engine underneath. A real multi-voice cast would
need a different (and much slower) voice-cloning backend per character;
see app.py's /health voice_note.

Implementation: pitch-shift via librosa's phase vocoder (preserves
duration/tempo, unlike naive resampling) plus a light shelf-EQ tilt for a
bit of timbral separation beyond pitch alone. Assignment is a stable hash
of the character's name, so "Priya" gets the same profile in every story
she appears in, this session or a future one.
"""
import hashlib

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

# (pitch shift in semitones, shelf-EQ tilt in dB at the shelf frequency -
# positive brightens/thins, negative darkens/warms). Deliberately modest
# shifts (<=3.5 semitones) - large shifts start to sound artificial/
# robotic on a single-speaker model never trained for this, which would
# cut against "more attractive" rather than help it. Seven profiles
# (rather than a handful) so a typical small story cast (2-4 named
# characters) is unlikely to collide onto the same one.
_PROFILES = {
    "deepest": (-3.5, -3.5),
    "deep": (-2.0, -2.0),
    "warm": (-1.0, -1.0),
    "neutral_shift": (0.5, 0.5),
    "bright": (1.5, 1.5),
    "high": (2.5, 2.5),
    "highest": (3.5, 3.5),
}
PROFILE_NAMES = list(_PROFILES.keys())
_SHELF_FREQ_HZ = 1500


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


def apply_voice_profile(audio: np.ndarray, sr: int, profile: str) -> np.ndarray:
    """No-op if `profile` is unrecognized (e.g. 'narrator') - returns
    `audio` unchanged."""
    if profile not in _PROFILES:
        return audio
    semitones, tilt_db = _PROFILES[profile]
    shifted = librosa.effects.pitch_shift(audio.astype(np.float32), sr=sr, n_steps=semitones)
    shifted = _shelf_eq(shifted, sr, tilt_db)
    peak = np.abs(shifted).max()
    if peak > 0.98:
        shifted = shifted / peak * 0.98
    return shifted.astype(np.float32)
