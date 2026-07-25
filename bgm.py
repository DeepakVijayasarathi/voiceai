"""
Genre-based background music, with auto-ducking (sidechain compression)
and algorithmic reverb.

Deliberately NOT sourced from a downloaded/licensed track library - no
verifiably-licensed music source was available to pull from, and using
unverified audio off the internet is a real copyright risk. Instead, each
genre is procedurally synthesized: a chord PROGRESSION, a soft rhythmic
pulse at a genre-appropriate tempo, an optional arpeggio layer, and a
Schroeder-style reverb for space - built locally with numpy/scipy. This
is honest about what it is: generative texture tuned per genre, not
produced music with real instruments - documented in the README.

Auto-ducking is what makes this usable rather than a gimmick: the music's
gain is modulated by an inverse, smoothed envelope of the speech track
(the same sidechain-compression technique podcasts and video production
use), so it swells during pauses and recedes under speech instead of
needing to be mixed so quiet it's inaudible everywhere just to avoid
masking words.
"""
import numpy as np
from scipy.signal import lfilter

from sfx import tile_sfx_to_length

# Each genre: chord progression (list of chords, each a list of freqs in
# Hz), tempo (beats/sec), background-noise level, brightness (overall
# gain multiplier for the tonal layers), whether to add an arpeggio,
# detune spread (wider = more dissonant/uneasy beating between voices),
# and reverb wetness (0-1, spaciousness).
GENRES = {
    "neutral": dict(
        chords=[[130.81, 164.81, 196.00], [146.83, 174.61, 220.00]],  # C -> Dm
        tempo=1.0, noise=0.012, brightness=1.0, arp=False, detune=0.001, reverb=0.20,
    ),
    "happy": dict(
        chords=[[261.63, 329.63, 392.00], [293.66, 369.99, 440.00], [329.63, 415.30, 493.88]],  # C -> D -> E
        tempo=2.2, noise=0.010, brightness=1.5, arp=True, detune=0.001, reverb=0.15,
    ),
    "sad": dict(
        chords=[[220.00, 261.63, 329.63], [196.00, 233.08, 293.66]],  # Am -> Gm
        tempo=0.7, noise=0.018, brightness=0.65, arp=False, detune=0.002, reverb=0.35,
    ),
    "excited": dict(
        chords=[[293.66, 369.99, 440.00], [329.63, 415.30, 493.88], [349.23, 440.00, 523.25]],  # D -> E -> F
        tempo=3.0, noise=0.014, brightness=1.7, arp=True, detune=0.001, reverb=0.15,
    ),
    "calm": dict(
        chords=[[130.81, 196.00], [146.83, 220.00]],  # root+fifth only
        tempo=0.5, noise=0.008, brightness=0.55, arp=False, detune=0.001, reverb=0.30,
    ),
    "horror": dict(
        # Tritone + minor 2nd clusters - intentionally dissonant.
        chords=[[110.00, 155.56, 207.65], [98.00, 138.59, 185.00]],  # A-Eb-Ab, G-C#-F#
        tempo=0.3, noise=0.030, brightness=0.9, arp=False, detune=0.012, reverb=0.55,
    ),
    "romance": dict(
        # Major 7th chords for a warm, lush character.
        chords=[[261.63, 329.63, 392.00, 493.88], [174.61, 220.00, 261.63, 329.63]],  # Cmaj7 -> Fmaj7
        tempo=0.6, noise=0.008, brightness=1.1, arp=False, detune=0.0015, reverb=0.40,
    ),
    "action": dict(
        chords=[[293.66, 349.23, 440.00], [220.00, 261.63, 329.63]],  # Dm -> Am
        tempo=3.5, noise=0.016, brightness=1.6, arp=True, detune=0.001, reverb=0.12,
    ),
    "comedy": dict(
        chords=[[261.63, 329.63, 392.00], [392.00, 493.88, 587.33]],  # C -> G
        tempo=2.5, noise=0.008, brightness=1.4, arp=True, detune=0.001, reverb=0.10,
    ),
    "devotional": dict(
        # Drone on a single sustained root+fifth (~136.1 Hz, the
        # traditionally-cited "Om" pitch) - sparse and meditative.
        chords=[[136.10, 204.15]],
        tempo=0.2, noise=0.006, brightness=0.5, arp=False, detune=0.001, reverb=0.50,
    ),
    "mystery": dict(
        chords=[[164.81, 196.00, 246.94], [123.47, 146.83, 185.00]],  # Em -> Bm
        tempo=0.8, noise=0.022, brightness=0.8, arp=False, detune=0.004, reverb=0.45,
    ),
}

_SR = 24000
_loop_cache: dict[str, np.ndarray] = {}


def _schroeder_reverb(signal: np.ndarray, sr: int, mix: float) -> np.ndarray:
    """Classic 4-comb-filter reverb (Schroeder, 1962) - cheap, no external
    impulse-response file needed, and gives a genuine decaying-tail sense
    of space rather than the dry, flat quality of raw synthesized tones."""
    if mix <= 0:
        return signal
    comb_delays_ms = [29.7, 37.1, 41.1, 43.7]
    feedback = 0.78
    wet = np.zeros_like(signal)
    for d_ms in comb_delays_ms:
        delay_samples = max(1, int(sr * d_ms / 1000))
        a = np.zeros(delay_samples + 1)
        a[0] = 1.0
        a[delay_samples] = -feedback
        wet += lfilter([1.0], a, signal)
    wet /= len(comb_delays_ms)
    return signal * (1 - mix) + wet * mix


def _generate_track(genre: str, duration_s: float = 32.0, sr: int = _SR) -> np.ndarray:
    """duration_s defaults to 32s (not the original 16s) specifically for
    long-form narration: this loop gets tiled to cover the full speech
    length (see _tile_loop_to_length), so a short base loop repeats
    audibly often across a multi-minute story - doubling it halves how
    often the exact same pattern recurs."""
    g = GENRES.get(genre, GENRES["neutral"])
    chords, tempo, noise_level, brightness, use_arp, detune_spread, reverb_mix = (
        g["chords"], g["tempo"], g["noise"], g["brightness"], g["arp"], g["detune"], g["reverb"]
    )
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)

    bar_len_s = 4 / tempo
    # Humanization: real sequenced-but-live-feeling music doesn't repeat
    # each bar at a perfectly identical level - a small deterministic
    # (per-genre-seeded, so caching/reproducibility is unaffected)
    # bar-to-bar gain variation avoids the "obviously on a mechanical
    # loop" quality a perfectly uniform pattern has.
    humanize_rng = np.random.default_rng((abs(hash(genre)) + 1) % (2**32))
    n_bars = int(duration_s / bar_len_s) + 2
    bar_gains = 1.0 + humanize_rng.uniform(-0.06, 0.06, size=n_bars)
    bar_idx_f = np.clip((t // bar_len_s).astype(int), 0, n_bars - 1)
    bar_gain_curve = bar_gains[bar_idx_f]

    chord_idx = (t // bar_len_s).astype(int) % len(chords)

    pad = np.zeros(n, dtype=np.float64)
    for chord_i, freqs in enumerate(chords):
        mask = chord_idx == chord_i
        if not mask.any():
            continue
        for j, f in enumerate(freqs):
            # Two slightly-detuned voices per note for a fuller unison;
            # detune_spread controls how far apart (wider = more beating/
            # dissonant tension, used for horror/mystery).
            for det in (1.0 - detune_spread, 1.0 + detune_spread):
                pad[mask] += np.sin(2 * np.pi * f * det * t[mask]) / (len(freqs) * 2)

    swell = 0.65 + 0.35 * np.sin(2 * np.pi * 0.12 * t)
    pad *= swell * brightness * 0.14 * bar_gain_curve

    beat_phase = (t * tempo) % 1.0
    pulse_env = np.clip(1.0 - beat_phase * 6.0, 0, 1) ** 2
    pulse = np.sin(2 * np.pi * 80 * t) * pulse_env * 0.08 * bar_gain_curve

    arp = np.zeros(n, dtype=np.float64)
    if use_arp:
        step_len_s = 1 / (tempo * 4)
        step_idx = (t // step_len_s).astype(int)
        for chord_i, freqs in enumerate(chords):
            mask = chord_idx == chord_i
            if not mask.any():
                continue
            note_freqs = np.array(freqs)[step_idx[mask] % len(freqs)]
            step_phase = (t[mask] % step_len_s) / step_len_s
            note_env = np.clip(1.0 - step_phase * 3.0, 0, 1)
            arp[mask] += (
                np.sin(2 * np.pi * note_freqs * t[mask])
                + 0.3 * np.sin(2 * np.pi * note_freqs * 3 * t[mask])
            ) * note_env * 0.05

    rng = np.random.default_rng(abs(hash(genre)) % (2**32))
    noise = rng.normal(0, 1, size=n)
    kernel = np.ones(64) / 64
    noise = np.convolve(noise, kernel, mode="same") * noise_level

    dry = (pad + pulse + arp + noise).astype(np.float64)
    track = _schroeder_reverb(dry, sr, reverb_mix).astype(np.float32)

    fade_len = int(sr * 0.5)
    fade = np.linspace(0, 1, fade_len)
    track[:fade_len] *= fade
    track[-fade_len:] *= fade[::-1]
    return track


def get_bgm_loop(genre: str) -> np.ndarray:
    """Generates (if not already cached) and returns the loop for `genre`.
    Public mainly so app.py's startup can pre-warm all genres - without
    this, the first real request for any given genre pays the ~1.2s
    reverb-filtering cost that later requests for the same genre skip."""
    if genre not in _loop_cache:
        _loop_cache[genre] = _generate_track(genre)
    return _loop_cache[genre]


def _speech_envelope(speech: np.ndarray, sr: int, smooth_s: float = 0.15) -> np.ndarray:
    """Smoothed absolute-value envelope of the speech signal, normalized to
    [0, 1], used to drive the ducking gain curve."""
    env = np.abs(speech)
    kernel_len = max(1, int(sr * smooth_s))
    kernel = np.ones(kernel_len) / kernel_len
    env = np.convolve(env, kernel, mode="same")
    peak = env.max()
    return env / peak if peak > 0 else env


def mix_with_bgm(
    speech: np.ndarray,
    sr: int,
    mood: str = "neutral",
    bgm_level: float = 1.75,
    duck_amount: float = 0.85,
) -> np.ndarray:
    """Mixes a genre-matched generative track under `speech`, auto-ducked
    against speech's own envelope: `duck_amount` is how much the music's
    gain drops at the loudest point of speech (0 = no ducking, 1 = fully
    silent under speech). `bgm_level` is the track's gain during silence/
    pauses, where it isn't ducked - this can be much higher than a static
    mix's level precisely because ducking protects intelligibility.
    `mood` accepts any GENRES key (the name is kept for API compatibility
    with the emotion presets, which now double as genre selectors)."""
    n = len(speech)
    if n == 0:
        return speech
    bgm = _tile_loop_to_length(mood, sr, n)

    envelope = _speech_envelope(speech, sr)
    duck_gain = 1.0 - duck_amount * envelope
    bgm = bgm * duck_gain * bgm_level

    fade_len = min(int(sr * 0.3), n // 4)
    if fade_len > 0:
        fade = np.linspace(0, 1, fade_len)
        bgm[:fade_len] *= fade
        bgm[-fade_len:] *= fade[::-1]

    mixed = speech + bgm
    peak = np.abs(mixed).max()
    if peak > 0.98:
        mixed = mixed / peak * 0.98
    return mixed.astype(np.float32)


def _tile_loop_to_length(genre: str, sr: int, n: int) -> np.ndarray:
    loop = get_bgm_loop(genre)
    if sr != _SR:
        x_old = np.linspace(0, 1, num=len(loop), endpoint=False)
        x_new = np.linspace(0, 1, num=int(len(loop) * sr / _SR), endpoint=False)
        loop = np.interp(x_new, x_old, loop).astype(np.float32)
    reps = n // len(loop) + 1
    return np.tile(loop, reps)[:n].astype(np.float32)


def _build_crossfaded_track(
    n: int, sr: int, segments: list[tuple[int, int, str]], tile_fn,
) -> np.ndarray:
    """Shared crossfade-blend logic for both the genre-music layer and the
    SFX layer: each segment (except the last) generates extra audio past
    its official boundary, overlapping into the next segment's region, so
    there's a real overlap zone where both fade against each other rather
    than one cutting off before the next begins. `tile_fn(label, sr, len)`
    returns the audio for a segment's label, or None to contribute silence
    (used for SFX segments labeled 'none')."""
    track = np.zeros(n, dtype=np.float32)
    crossfade_len = int(sr * 0.5)
    for i, (start, end, label) in enumerate(segments):
        start = max(0, start)
        end = min(n, end)
        if end <= start:
            continue
        extend = crossfade_len if i < len(segments) - 1 else 0
        gen_end = min(n, end + extend)
        segment_audio = tile_fn(label, sr, gen_end - start)
        if segment_audio is None:
            continue

        actual_len = gen_end - start
        fade_out_len = min(extend, actual_len) if extend else 0
        if fade_out_len > 0:
            fade_out = np.concatenate([
                np.ones(actual_len - fade_out_len),
                np.linspace(1, 0, fade_out_len),
            ])
            segment_audio = segment_audio * fade_out

        fade_in_len = min(crossfade_len, actual_len) if i > 0 else 0
        if fade_in_len > 0:
            fade_in = np.concatenate([
                np.linspace(0, 1, fade_in_len),
                np.ones(actual_len - fade_in_len),
            ])
            segment_audio = segment_audio * fade_in

        track[start:gen_end] += segment_audio
    return track


def mix_with_scene_bgm(
    speech: np.ndarray,
    sr: int,
    segments: list[tuple[int, int, tuple[str, float]]],
    sfx_segments: list[tuple[int, int, tuple[str, float]]] | None = None,
    bgm_level: float = 1.75,
    sfx_level: float = 0.19,
    bgm_duck_amount: float = 0.85,
    sfx_duck_amount: float = 0.55,
) -> np.ndarray:
    """Scene-aware version of mix_with_bgm: `segments` is a list of
    (start_sample, end_sample, (genre, bgm_intensity)) covering `speech`
    contiguously (as produced by grouping chunked-synthesis output with
    per-chunk genre labels - see app.py's _synthesize_long_form). Each
    segment gets its own genre's music, cross-faded into the next
    segment's genre at the boundary so a mood shift sounds like a
    transition rather than a splice; bgm_intensity (0-1) additionally
    scales how prominent the music sits within that segment, so the same
    genre can swell for a dramatic beat and recede for a quiet one rather
    than playing at one flat level throughout. `sfx_segments` optionally
    layers environmental sound effects (rain, wind, etc.) alongside the
    music at a lower level with a touch of reverb for spatial blend
    (rather than sounding like a dry, pasted-in loop), each entry's label
    being an (sfx_type, sfx_intensity) pair.

    BGM and SFX are ducked INDEPENDENTLY, not as one combined layer:
    bgm_duck_amount is high (music mostly gets out of the way for
    dialogue, then swells back in pauses - it's a mood cue, not meant to
    compete with speech), while sfx_duck_amount is deliberately much
    lower (ambiance like rain or a crowd realistically stays present at
    a reduced level under dialogue rather than vanishing - it's room
    tone, not a foreground element). Measured and rebalanced from an
    earlier pass where music was ~29dB below speech (near-inaudible) and
    louder than that by SFX, which had it backwards - see
    tests/mix_level_probe.py for the measurement approach."""
    n = len(speech)
    if n == 0:
        return speech
    if not segments:
        return mix_with_bgm(speech, sr, mood="neutral", bgm_level=bgm_level, duck_amount=bgm_duck_amount)

    envelope = _speech_envelope(speech, sr)

    def _tile_genre_with_intensity(label, sr_, ln):
        genre, intensity = label
        return _tile_loop_to_length(genre, sr_, ln) * intensity

    bgm = _build_crossfaded_track(n, sr, segments, _tile_genre_with_intensity)
    bgm_duck_gain = 1.0 - bgm_duck_amount * envelope
    bgm = bgm * bgm_duck_gain * bgm_level

    if sfx_segments:
        def _tile_sfx_with_intensity(label, sr_, ln):
            sfx_type, intensity = label
            audio = tile_sfx_to_length(sfx_type, sr_, ln)
            return None if audio is None else audio * intensity

        sfx_track = _build_crossfaded_track(n, sr, sfx_segments, _tile_sfx_with_intensity)
        sfx_track = _schroeder_reverb(sfx_track, sr, mix=0.18)
        sfx_duck_gain = 1.0 - sfx_duck_amount * envelope
        bgm = bgm + sfx_track * sfx_duck_gain * sfx_level

    fade_len = min(int(sr * 0.3), n // 4)
    if fade_len > 0:
        fade = np.linspace(0, 1, fade_len)
        bgm[:fade_len] *= fade
        bgm[-fade_len:] *= fade[::-1]

    mixed = speech + bgm
    peak = np.abs(mixed).max()
    if peak > 0.98:
        mixed = mixed / peak * 0.98
    return mixed.astype(np.float32)
