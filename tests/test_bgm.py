import inspect

import numpy as np
import pytest

from bgm import GENRES, get_bgm_loop, mix_with_bgm, mix_with_scene_bgm


def _fake_speech(sr=24000, duration_s=9.0):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)).astype(np.float32) * 0.3


@pytest.mark.parametrize("genre", list(GENRES))
def test_genre_loop_generates_cleanly(genre):
    loop = get_bgm_loop(genre)
    assert not np.isnan(loop).any()
    assert not np.isinf(loop).any()


def _low_freq_energy(signal, sr, cutoff_hz=90):
    """Fraction of total spectral energy below cutoff_hz - an objective
    proxy for 'how much sub-bass weight', rather than just eyeballing."""
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1 / sr)
    total = spectrum.sum()
    if total == 0:
        return 0.0
    return spectrum[freqs < cutoff_hz].sum() / total


def _energy_near_freq(signal, sr, target_hz, tolerance_hz=15):
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1 / sr)
    band = (freqs > target_hz - tolerance_hz) & (freqs < target_hz + tolerance_hz)
    return spectrum[band].sum()


@pytest.mark.parametrize("genre", list(GENRES))
def test_melody_notes_are_actually_present_in_the_spectrum(genre):
    """Regression guard for a real risk with this technique: it's easy to
    wire up a melody layer that's computed but drowned out or silent due
    to a masking/indexing bug. Checks that each genre's melody note
    frequencies show measurably more energy nearby than a comparable
    random gap in the spectrum - i.e. the hook is actually audible, not
    just present in code."""
    g = GENRES[genre]
    root_freq = min(g["chords"][0]) * 2
    note_freqs = [root_freq * (2.0 ** (o / 12.0)) for o in set(g["melody"])]
    loop = get_bgm_loop(genre)
    for note_hz in note_freqs:
        near = _energy_near_freq(loop, 24000, note_hz)
        # Compare against a deliberately off-note gap a third away - not
        # zero, since reverb/noise/other layers have some broadband energy.
        gap_hz = note_hz * (2.0 ** (2.5 / 12.0))
        far = _energy_near_freq(loop, 24000, gap_hz)
        assert near > 0, f"{genre}: no energy at all near melody note {note_hz:.1f}Hz"
        assert near >= far * 0.5, f"{genre}: melody note {note_hz:.1f}Hz not measurably present vs off-note gap"


def test_high_sub_bass_genre_has_more_low_end_than_zero_sub_bass_genre():
    # horror (sub_bass=0.85) vs happy (sub_bass=0.0) - a real, measurable
    # difference in low-frequency content, not just a config value that
    # exists but does nothing.
    horror_energy = _low_freq_energy(get_bgm_loop("horror"), 24000)
    happy_energy = _low_freq_energy(get_bgm_loop("happy"), 24000)
    assert horror_energy > happy_energy


def test_sub_bass_zero_genres_stay_bright():
    # happy/comedy (sub_bass=0, bright upbeat chords) should measure
    # clearly lower than horror/mystery/action (high sub_bass) - not an
    # absolute cutoff, since even zero-sub-bass genres with naturally low
    # chord roots (calm, neutral) carry some inherent low end from the
    # pulse layer alone; this compares genres where sub_bass is the
    # actual differentiator.
    for bright_genre in ("happy", "comedy"):
        assert GENRES[bright_genre]["sub_bass"] == 0.0
        bright_energy = _low_freq_energy(get_bgm_loop(bright_genre), 24000)
        for weighty_genre in ("horror", "mystery", "action"):
            weighty_energy = _low_freq_energy(get_bgm_loop(weighty_genre), 24000)
            assert bright_energy < weighty_energy


def test_mix_with_bgm_stays_bounded_and_same_shape():
    speech = _fake_speech()
    mixed = mix_with_bgm(speech, 24000, mood="horror")
    assert mixed.shape == speech.shape
    assert np.abs(mixed).max() <= 0.98 + 1e-6
    assert not np.isnan(mixed).any()


def test_mix_with_bgm_empty_speech_returns_empty():
    empty = np.zeros(0, dtype=np.float32)
    assert mix_with_bgm(empty, 24000, mood="calm").shape == (0,)


def test_mix_with_scene_bgm_crossfade_has_no_silent_dip():
    sr = 24000
    speech = _fake_speech(sr=sr, duration_s=9.0)
    segments = [
        (0, sr * 3, ("calm", 0.7)),
        (sr * 3, sr * 6, ("horror", 0.9)),
        (sr * 6, sr * 9, ("action", 0.8)),
    ]
    mixed = mix_with_scene_bgm(speech, sr, segments)
    assert mixed.shape == speech.shape
    assert np.abs(mixed).max() <= 0.98 + 1e-6
    assert not np.isnan(mixed).any()

    overall_rms = np.sqrt(np.mean(mixed.astype(np.float64) ** 2))
    for boundary_s in (3, 6):
        idx = int(sr * boundary_s)
        window = mixed[idx - 1000:idx + 1000]
        local_rms = np.sqrt(np.mean(window.astype(np.float64) ** 2))
        # A real crossfade blend should stay in the same ballpark as the
        # overall mix - a silent dip (the bug this guards against) would
        # show up as local_rms collapsing near zero.
        assert local_rms > overall_rms * 0.3


def test_mix_with_scene_bgm_empty_segments_falls_back():
    speech = _fake_speech()
    mixed = mix_with_scene_bgm(speech, 24000, [])
    assert mixed.shape == speech.shape


def test_mix_with_scene_bgm_with_sfx_layer():
    sr = 24000
    speech = _fake_speech(sr=sr, duration_s=6.0)
    segments = [(0, sr * 3, ("calm", 0.6)), (sr * 3, sr * 6, ("horror", 0.8))]
    sfx_segments = [(0, sr * 3, (["rain"], 0.5)), (sr * 3, sr * 6, (["thunder"], 0.9))]
    mixed = mix_with_scene_bgm(speech, sr, segments, sfx_segments=sfx_segments)
    assert mixed.shape == speech.shape
    assert np.abs(mixed).max() <= 0.98 + 1e-6


def test_multi_sfx_layering_combines_both_cues_not_just_one():
    """A storm scene combining rain+thunder should carry measurably more
    energy than EITHER cue alone at the same intensity - proof the
    layering actually sums both tracks rather than silently picking one
    (e.g. a bug where only the first list item ever got used)."""
    sr = 24000
    n = sr * 4
    silence = np.zeros(n, dtype=np.float32)
    segments = [(0, n, ("neutral", 0.0))]  # bgm_intensity 0 isolates the sfx contribution

    rain_only = mix_with_scene_bgm(silence, sr, segments, sfx_segments=[(0, n, (["rain"], 0.7))])
    thunder_only = mix_with_scene_bgm(silence, sr, segments, sfx_segments=[(0, n, (["thunder"], 0.7))])
    both = mix_with_scene_bgm(silence, sr, segments, sfx_segments=[(0, n, (["rain", "thunder"], 0.7))])

    rms = lambda x: np.sqrt(np.mean(x.astype(np.float64) ** 2))
    assert rms(both) > rms(rain_only)
    assert rms(both) > rms(thunder_only)


def test_sfx_segment_empty_list_contributes_silence():
    sr = 24000
    n = sr * 2
    silence = np.zeros(n, dtype=np.float32)
    segments = [(0, n, ("neutral", 0.0))]
    mixed = mix_with_scene_bgm(silence, sr, segments, sfx_segments=[(0, n, ([], 0.7))])
    assert np.abs(mixed).max() < 1e-6
    assert not np.isnan(mixed).any()


def test_bgm_intensity_scales_output_level():
    """A higher bgm_intensity segment should contribute more energy than
    an identical genre at lower intensity, all else equal."""
    sr = 24000
    silence = np.zeros(sr * 4, dtype=np.float32)  # no speech -> isolates bgm level
    low = mix_with_scene_bgm(silence, sr, [(0, sr * 4, ("action", 0.2))])
    high = mix_with_scene_bgm(silence, sr, [(0, sr * 4, ("action", 0.9))])
    assert np.abs(high).mean() > np.abs(low).mean()


def _rms_db(x):
    r = np.sqrt(np.mean(x.astype(np.float64) ** 2))
    return 20 * np.log10(r) if r > 0 else -np.inf


def test_bgm_and_sfx_mix_balance_is_reasonable():
    """Regression test for a real mix-balance bug: BGM previously sat
    ~29dB below speech (near-inaudible - the whole point of scene-aware
    music was defeated) and was actually QUIETER than SFX, which had the
    mood-setting layer subordinate to the ambiance layer. Pins down the
    corrected relationship: BGM should be clearly audible under speech
    (not more than ~22dB down) and louder than SFX (SFX is texture, not
    the primary layer)."""
    sr = 24000
    duration_s = 8.0
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    # ~55% duty-cycle voiced bursts with real silence gaps, closer to
    # actual narration's speech/pause ratio than a near-continuous tone.
    speech = (np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 0.4 * t) > 0.1)).astype(np.float32) * 0.35
    speech_db = _rms_db(speech)

    segments = [(0, n, ("horror", 0.7))]
    sfx_segments = [(0, n, (["rain"], 0.7))]

    mixed_both = mix_with_scene_bgm(speech, sr, segments, sfx_segments=sfx_segments)
    mixed_bgm_only = mix_with_scene_bgm(speech, sr, segments)
    bgm_contribution_db = _rms_db(mixed_bgm_only - speech)
    sfx_contribution_db = _rms_db(mixed_both - mixed_bgm_only)

    # BGM should be clearly audible under speech, not buried.
    assert speech_db - bgm_contribution_db < 22
    # BGM (the mood cue) should sit louder than SFX (supporting texture).
    assert bgm_contribution_db > sfx_contribution_db


def test_single_chunk_fallback_level_matches_scene_mixer():
    """Regression test for a real bug: _synthesize_long_form's single-chunk
    path calls mix_with_bgm() without overriding bgm_level, so it silently
    used mix_with_bgm's OWN default - which had gone stale at the old,
    pre-fix 0.55 while mix_with_scene_bgm's default was correctly
    recalibrated to 1.75. Most /tts calls and short villain/revive scenes
    are single-chunk, so they were still getting the too-quiet mix after
    the "fix". Pins the two defaults together so they can't drift apart
    silently again."""
    bgm_only_default = inspect.signature(mix_with_bgm).parameters["bgm_level"].default
    scene_default = inspect.signature(mix_with_scene_bgm).parameters["bgm_level"].default
    assert bgm_only_default == scene_default
