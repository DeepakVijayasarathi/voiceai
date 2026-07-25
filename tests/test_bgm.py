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
    sfx_segments = [(0, sr * 3, ("rain", 0.5)), (sr * 3, sr * 6, ("thunder", 0.9))]
    mixed = mix_with_scene_bgm(speech, sr, segments, sfx_segments=sfx_segments)
    assert mixed.shape == speech.shape
    assert np.abs(mixed).max() <= 0.98 + 1e-6
    assert not np.isnan(mixed).any()


def test_bgm_intensity_scales_output_level():
    """A higher bgm_intensity segment should contribute more energy than
    an identical genre at lower intensity, all else equal."""
    sr = 24000
    silence = np.zeros(sr * 4, dtype=np.float32)  # no speech -> isolates bgm level
    low = mix_with_scene_bgm(silence, sr, [(0, sr * 4, ("action", 0.2))])
    high = mix_with_scene_bgm(silence, sr, [(0, sr * 4, ("action", 0.9))])
    assert np.abs(high).mean() > np.abs(low).mean()
