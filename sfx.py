"""
Environmental sound effects, procedurally synthesized - same reasoning as
bgm.py: no verifiably-licensed sample library was available, so these are
generated locally with numpy/scipy rather than pulled from an unverified
source. Layered alongside (not instead of) the genre background music, at
a lower level, and auto-ducked against speech the same way.

Scope note on "dynamic": true open-ended sound generation (any sound
described in arbitrary text, synthesized fresh) needs a dedicated
generative-audio model (e.g. AudioGen-class) - that's a different, much
heavier model than anything else in this service and doesn't fit the
CPU-only, low-latency design goal. What's here instead is a broad but
still finite hand-built palette (14 types); OpenAI picks freely from
this set per scene rather than being forced into a narrow handful, which
covers most real narration scenes without pretending to synthesize
anything imaginable.

Per-instance variation: each generator used to seed its own RNG with a
hardcoded literal, which meant "rain" (or any other type) sounded
byte-identical every single time it was used, in every story, forever -
and since a scene's SFX is a short loop TILED to fill however long the
scene runs, a scene longer than one loop repeated it verbatim, which is
an obvious, robotic-sounding tell for anything with a distinct transient
pattern (footsteps landing on an exact 0.6s grid, the same thunder
rumble at the same relative position, etc). Generators now take an
external `rng` instead of constructing their own, and get_sfx_loop/
tile_sfx_to_length accept a `variation_index` that offsets the seed -
index 0 reproduces the exact original output (nothing that depends on
today's default behavior breaks), any other index produces a genuinely
different-sounding instance of the same type. bgm.py's scene mixer
passes each story segment's own index through, so a story with rain in
scene 1 and rain again in scene 4 gets two distinct-sounding instances
instead of the same loop twice.
"""
import numpy as np
from scipy.signal import lfilter

_SR = 24000
_loop_cache: dict[tuple[str, int], np.ndarray] = {}

SFX_TYPES = (
    "none", "rain", "wind", "thunder", "lightning", "fire", "footsteps",
    "car", "traffic", "crowd", "door_creak", "glass_break", "birds",
    "water", "dog_bark",
)

# Per-type base seed - variation_index=0 reproduces this exactly (see
# _seed_for below), so the original, always-used-until-now sound for
# each type is still reachable and still what a first/only use gets.
_BASE_SEEDS = {
    "rain": 1, "wind": 2, "thunder": 3, "fire": 4, "footsteps": 5,
    "lightning": 6, "car": 7, "traffic": 8, "crowd": 9, "door_creak": 10,
    "glass_break": 11, "birds": 12, "water": 13, "dog_bark": 14,
}


def _seed_for(sfx_type: str, variation_index: int) -> int:
    return _BASE_SEEDS[sfx_type] + variation_index * 97


def _lowpass_noise(n: int, rng: np.random.Generator, cutoff_alpha: float) -> np.ndarray:
    """Cheap one-pole low-pass filtered white noise - cutoff_alpha closer
    to 1.0 = darker/heavier filtering, closer to 0 = closer to raw hiss."""
    noise = rng.normal(0, 1, size=n)
    return lfilter([1 - cutoff_alpha], [1, -cutoff_alpha], noise)


def _generate_rain(duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(sr * duration_s)
    # Steady hiss bed (many overlapping droplets averages into filtered
    # noise) plus sparser sharper droplet transients on top for texture.
    hiss = _lowpass_noise(n, rng, cutoff_alpha=0.7) * 0.5
    droplets = np.zeros(n)
    n_drops = int(duration_s * 40)  # ~40 droplet transients/sec
    drop_positions = rng.integers(0, n, size=n_drops)
    drop_len = int(sr * 0.008)
    for pos in drop_positions:
        end = min(n, pos + drop_len)
        length = end - pos
        if length <= 0:
            continue
        droplets[pos:end] += rng.normal(0, 1, length) * np.linspace(1, 0, length) * 0.3
    return (hiss + droplets).astype(np.float32)


def _generate_wind(duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    base = _lowpass_noise(n, rng, cutoff_alpha=0.85)
    # Slow gust swells - a couple of independent low-rate LFOs summed for
    # an irregular (not perfectly periodic) gusting feel.
    gust = (0.5 + 0.3 * np.sin(2 * np.pi * 0.07 * t) + 0.2 * np.sin(2 * np.pi * 0.23 * t + 1.3))
    return (base * gust).astype(np.float32)


def _generate_thunder(duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(sr * duration_s)
    audio = np.zeros(n)
    n_rumbles = max(1, int(duration_s / 6))  # one rumble roughly every 6s
    for _ in range(n_rumbles):
        pos = rng.integers(0, max(1, n - sr * 2))
        rumble_len = int(sr * rng.uniform(1.5, 3.0))
        rumble_len = min(rumble_len, n - pos)
        if rumble_len <= 0:
            continue
        rumble = _lowpass_noise(rumble_len, rng, cutoff_alpha=0.95)
        env = np.concatenate([
            np.linspace(0, 1, int(rumble_len * 0.1)),
            np.linspace(1, 0, rumble_len - int(rumble_len * 0.1)) ** 1.5,
        ])
        audio[pos:pos + rumble_len] += rumble * env * 0.8
    return audio.astype(np.float32)


def _generate_fire(duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(sr * duration_s)
    base = _lowpass_noise(n, rng, cutoff_alpha=0.6) * 0.35
    crackles = np.zeros(n)
    n_crackles = int(duration_s * 8)
    for pos in rng.integers(0, n, size=n_crackles):
        crackle_len = int(sr * rng.uniform(0.01, 0.04))
        end = min(n, pos + crackle_len)
        length = end - pos
        if length <= 0:
            continue
        crackles[pos:end] += rng.normal(0, 1, length) * np.linspace(1, 0, length) * 0.4
    return (base + crackles).astype(np.float32)


def _generate_footsteps(duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    audio = np.zeros(n)
    # Interval jittered per-instance (was a fixed 0.6s grid) - a person's
    # gait isn't a metronome, and a fixed grid is exactly the kind of
    # thing that makes a tiled loop's repeat seam audible.
    step_interval_s = rng.uniform(0.5, 0.7)
    step_positions = np.arange(0, duration_s, step_interval_s)
    for step_t in step_positions:
        pos = int(step_t * sr)
        step_len = int(sr * 0.08)
        end = min(n, pos + step_len)
        length = end - pos
        if length <= 0:
            continue
        thud_freq = rng.uniform(80, 100)
        thud = np.sin(2 * np.pi * thud_freq * t[pos:end]) * np.linspace(1, 0, length) ** 2
        audio[pos:end] += thud * rng.uniform(0.42, 0.58)
    return audio.astype(np.float32)


def _generate_lightning(duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Distinct from thunder's low rumble: a sharp, near-instant broadband
    crack with a very fast decay, occurring once or twice."""
    n = int(sr * duration_s)
    audio = np.zeros(n)
    n_cracks = max(1, int(duration_s / 8))
    for _ in range(n_cracks):
        pos = rng.integers(0, max(1, n - sr))
        crack_len = int(sr * rng.uniform(0.15, 0.35))
        crack_len = min(crack_len, n - pos)
        if crack_len <= 0:
            continue
        crack = rng.normal(0, 1, crack_len)
        env = np.linspace(1, 0, crack_len) ** 0.5  # fast decay, sharp attack
        audio[pos:pos + crack_len] += crack * env * 0.9
    return audio.astype(np.float32)


def _generate_car(duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Engine idle/drive drone: a low fundamental plus harmonics (cheap
    approximation of combustion-cycle pulses) with filtered noise for
    road/exhaust texture."""
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    fundamental = 45.0
    engine = sum(np.sin(2 * np.pi * fundamental * h * t) / h for h in (1, 2, 3, 4)) * 0.15
    engine *= 0.85 + 0.15 * np.sin(2 * np.pi * 4 * t)  # cylinder-firing-ish flutter
    road_noise = _lowpass_noise(n, rng, cutoff_alpha=0.75) * 0.2
    return (engine + road_noise).astype(np.float32)


def _generate_traffic(duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Denser street ambience: a couple of car drones at different pitches
    plus occasional horn honks."""
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    base = _generate_car(duration_s, sr, rng) * 0.7
    base += 0.6 * np.roll(base, int(sr * 1.3))  # a second, offset "vehicle"
    honks = np.zeros(n)
    n_honks = max(1, int(duration_s / 5))
    for _ in range(n_honks):
        pos = rng.integers(0, max(1, n - sr))
        honk_len = int(sr * rng.uniform(0.3, 0.6))
        honk_len = min(honk_len, n - pos)
        if honk_len <= 0:
            continue
        honk_t = t[pos:pos + honk_len] - t[pos]
        honk = (np.sin(2 * np.pi * 400 * honk_t) + 0.5 * np.sin(2 * np.pi * 500 * honk_t))
        env = np.clip(np.minimum(honk_t * 30, 1) * np.minimum((honk_len / sr - honk_t) * 15, 1), 0, 1)
        honks[pos:pos + honk_len] += honk * env * 0.3
    return (base + honks).astype(np.float32)


def _generate_crowd(duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Indistinct chatter ('walla') - several independently, erratically
    amplitude-modulated band-passed noise voices summed together, none
    individually intelligible."""
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    audio = np.zeros(n)
    for i in range(5):
        voice = _lowpass_noise(n, rng, cutoff_alpha=0.5)
        rate = rng.uniform(1.5, 3.5)
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * rate * t + rng.uniform(0, 6.28))
        audio += voice * mod / 5
    return audio.astype(np.float32) * 0.6


def _generate_door_creak(duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A resonant pitch-bending groan, occurring once or twice."""
    n = int(sr * duration_s)
    audio = np.zeros(n)
    n_creaks = max(1, int(duration_s / 7))
    for _ in range(n_creaks):
        pos = rng.integers(0, max(1, n - sr))
        creak_len = int(sr * rng.uniform(0.8, 1.5))
        creak_len = min(creak_len, n - pos)
        if creak_len <= 0:
            continue
        ct = np.linspace(0, 1, creak_len)
        freq_sweep = 180 + 60 * np.sin(2 * np.pi * 1.5 * ct) - 40 * ct
        phase = 2 * np.pi * np.cumsum(freq_sweep) / sr
        creak = np.sin(phase) * (1 - np.abs(2 * ct - 1)) ** 0.5
        audio[pos:pos + creak_len] += creak * 0.5
    return audio.astype(np.float32)


def _generate_glass_break(duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A sharp high-frequency burst plus a handful of overlapping ringing
    'shard' resonances, occurring once."""
    n = int(sr * duration_s)
    audio = np.zeros(n)
    pos = rng.integers(0, max(1, n - sr))
    burst_len = int(sr * 0.15)
    burst_len = min(burst_len, n - pos)
    if burst_len > 0:
        burst = rng.normal(0, 1, burst_len) * np.linspace(1, 0, burst_len) ** 0.3
        audio[pos:pos + burst_len] += burst * 0.8
    for _ in range(6):
        shard_freq = rng.uniform(1800, 4500)
        shard_len = int(sr * rng.uniform(0.2, 0.5))
        shard_len = min(shard_len, n - pos)
        if shard_len <= 0:
            continue
        st = np.linspace(0, shard_len / sr, shard_len)
        shard = np.sin(2 * np.pi * shard_freq * st) * np.exp(-st * 8)
        audio[pos:pos + shard_len] += shard * 0.25
    return audio.astype(np.float32)


def _generate_birds(duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Staggered chirp bursts at a few different pitches, simulating
    several birds rather than one repeating call."""
    n = int(sr * duration_s)
    audio = np.zeros(n)
    n_chirps = int(duration_s * 1.2)
    for _ in range(n_chirps):
        pos = rng.integers(0, max(1, n - int(sr * 0.3)))
        chirp_len = int(sr * rng.uniform(0.08, 0.2))
        chirp_len = min(chirp_len, n - pos)
        if chirp_len <= 0:
            continue
        ct = np.linspace(0, chirp_len / sr, chirp_len)
        freq = rng.uniform(2000, 3500) + rng.uniform(500, 1500) * ct / (chirp_len / sr)
        chirp = np.sin(2 * np.pi * freq * ct) * np.sin(np.pi * ct / (chirp_len / sr))
        audio[pos:pos + chirp_len] += chirp * 0.3
    return audio.astype(np.float32)


def _generate_water(duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Continuous flowing texture (river/stream) - similar filtered-noise
    approach to rain but smoother modulation, no discrete droplets."""
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    base = _lowpass_noise(n, rng, cutoff_alpha=0.6)
    flow = 0.6 + 0.4 * np.sin(2 * np.pi * 0.15 * t) * np.sin(2 * np.pi * 0.4 * t + 0.7)
    return (base * flow * 0.5).astype(np.float32)


def _generate_dog_bark(duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Short sharp barks (noise burst + low tonal growl) at irregular
    intervals."""
    n = int(sr * duration_s)
    audio = np.zeros(n)
    n_barks = max(1, int(duration_s / 3))
    for _ in range(n_barks):
        pos = rng.integers(0, max(1, n - int(sr * 0.3)))
        bark_len = int(sr * rng.uniform(0.1, 0.18))
        bark_len = min(bark_len, n - pos)
        if bark_len <= 0:
            continue
        bt = np.linspace(0, bark_len / sr, bark_len)
        tone = np.sin(2 * np.pi * 220 * bt) * np.exp(-bt * 15)
        noise = rng.normal(0, 1, bark_len) * np.exp(-bt * 25) * 0.4
        audio[pos:pos + bark_len] += (tone + noise) * 0.6
    return audio.astype(np.float32)


_GENERATORS = {
    "rain": _generate_rain,
    "wind": _generate_wind,
    "thunder": _generate_thunder,
    "lightning": _generate_lightning,
    "fire": _generate_fire,
    "footsteps": _generate_footsteps,
    "car": _generate_car,
    "traffic": _generate_traffic,
    "crowd": _generate_crowd,
    "door_creak": _generate_door_creak,
    "glass_break": _generate_glass_break,
    "birds": _generate_birds,
    "water": _generate_water,
    "dog_bark": _generate_dog_bark,
}


def get_sfx_loop(sfx_type: str, duration_s: float = 24.0, variation_index: int = 0) -> np.ndarray | None:
    """Returns None for 'none' or an unrecognized type (silence - no SFX
    layer added), otherwise the (cached) generated loop. `variation_index`
    picks which instance of this type to generate - 0 is the original,
    always-cached default; any other value produces a distinct-sounding
    instance (own cache slot, generated once then reused for the rest of
    this process's life)."""
    if sfx_type not in _GENERATORS:
        return None
    cache_key = (sfx_type, variation_index)
    if cache_key not in _loop_cache:
        rng = np.random.default_rng(_seed_for(sfx_type, variation_index))
        audio = _GENERATORS[sfx_type](duration_s, _SR, rng)
        # Generators aren't individually peak-tuned (some layer several
        # transient sources whose worst-case overlap isn't hand-checked) -
        # normalize defensively so every generator returns a comparable,
        # bounded level regardless of internal mixing choices.
        peak = np.abs(audio).max()
        if peak > 0.9:
            audio = audio / peak * 0.9
        fade_len = int(_SR * 0.3)
        fade = np.linspace(0, 1, fade_len)
        audio[:fade_len] *= fade
        audio[-fade_len:] *= fade[::-1]
        _loop_cache[cache_key] = audio.astype(np.float32)
    return _loop_cache[cache_key]


def tile_sfx_to_length(sfx_type: str, sr: int, n: int, variation_index: int = 0) -> np.ndarray | None:
    loop = get_sfx_loop(sfx_type, variation_index=variation_index)
    if loop is None:
        return None
    if sr != _SR:
        x_old = np.linspace(0, 1, num=len(loop), endpoint=False)
        x_new = np.linspace(0, 1, num=int(len(loop) * sr / _SR), endpoint=False)
        loop = np.interp(x_new, x_old, loop).astype(np.float32)
    reps = n // len(loop) + 1
    return np.tile(loop, reps)[:n].astype(np.float32)
