import librosa
import numpy as np
import pytest

from voice_profile import (
    PROFILE_NAMES, _PROFILES, _formant_shift, apply_voice_profile,
    assign_profiles, get_speed_multiplier, get_voice_profile,
)


def _harmonic_signal(sr, duration, f0, n_harmonics=8, formant_center_harmonic=3):
    # A vowel-like signal: several harmonics of f0, with one band boosted
    # to act as a synthetic "formant" the envelope warp can actually move
    # - a pure sine tone has no envelope shape for formant shifting to
    # act on.
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        amp = 0.15 if abs(h - formant_center_harmonic) > 1 else 0.5
        audio += amp * np.sin(2 * np.pi * f0 * h * t)
    audio = audio / np.abs(audio).max() * 0.5
    return audio.astype(np.float32)


def _estimate_f0(audio, sr, fmin=60, fmax=400):
    # Autocorrelation pitch estimate - periodicity-based, so it's a fair
    # check that formant shifting (which only touches per-bin amplitude,
    # not where harmonic peaks sit) leaves the fundamental where it was.
    corr = np.correlate(audio, audio, mode="full")[len(audio) - 1:]
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    peak_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
    return sr / peak_lag


def test_get_voice_profile_is_deterministic_and_case_insensitive():
    assert get_voice_profile("Priya") == get_voice_profile("priya") == get_voice_profile(" Priya ")


def test_get_voice_profile_returns_valid_profile():
    assert get_voice_profile("AnyName") in PROFILE_NAMES


def test_assign_profiles_avoids_collisions_within_cast():
    cast = ["Priya", "Vikram", "Ravi", "Meera"]
    assignment = assign_profiles(cast)
    assert len(assignment) == len(cast)
    assert len(set(assignment.values())) == len(cast)


def test_assign_profiles_is_stable_across_calls():
    cast = ["Priya", "Vikram", "Ravi", "Meera"]
    assert assign_profiles(cast) == assign_profiles(cast)


def test_assign_profiles_honors_personality_hint():
    assignment = assign_profiles(["Vaal"], personality_hints={"Vaal": "deepest"})
    assert assignment["Vaal"] == "deepest"


def test_assign_profiles_ignores_invalid_hint():
    # An out-of-vocabulary hint shouldn't crash or get used verbatim.
    assignment = assign_profiles(["X"], personality_hints={"X": "not_a_real_profile"})
    assert assignment["X"] in PROFILE_NAMES


def test_assign_profiles_resolves_hint_collision():
    # Both hinted to the same profile - second must be bumped to a
    # different one rather than silently colliding.
    assignment = assign_profiles(
        ["A", "B"], personality_hints={"A": "deepest", "B": "deepest"},
    )
    assert assignment["A"] != assignment["B"]


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_apply_voice_profile_shifts_and_stays_bounded(profile):
    sr = 24000
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    audio = (np.sin(2 * np.pi * 180 * t) * 0.4).astype(np.float32)
    shifted = apply_voice_profile(audio, sr, profile)
    assert shifted.shape == audio.shape
    assert not np.isnan(shifted).any()
    assert not np.isinf(shifted).any()
    assert np.abs(shifted).max() <= 0.98 + 1e-6


def test_apply_voice_profile_narrator_is_noop():
    sr = 24000
    audio = np.random.default_rng(0).normal(0, 0.1, sr).astype(np.float32)
    assert np.array_equal(apply_voice_profile(audio, sr, "narrator"), audio)


def test_get_speed_multiplier_narrator_is_unchanged():
    assert get_speed_multiplier("narrator") == 1.0


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_get_speed_multiplier_within_modest_bounds(profile):
    # Large pace swings would sound unnatural on a model never trained
    # for this - keep the multiplier close to 1.0.
    assert 0.85 <= get_speed_multiplier(profile) <= 1.15


def test_deeper_profiles_speak_no_faster_than_brighter_ones():
    assert get_speed_multiplier("deepest") < get_speed_multiplier("highest")


def test_formant_shift_is_noop_for_ratio_one():
    sr = 24000
    audio = _harmonic_signal(sr, 1.0, 150)
    assert np.array_equal(_formant_shift(audio, sr, 1.0), audio)


def test_formant_shift_preserves_fundamental_frequency():
    # The whole point: formants move, pitch doesn't - unlike a plain
    # pitch_shift call, which would move both together.
    sr = 24000
    audio = _harmonic_signal(sr, 1.0, 150)
    shifted = _formant_shift(audio, sr, 1.3)
    assert abs(_estimate_f0(shifted, sr) - _estimate_f0(audio, sr)) < 5


def test_formant_shift_moves_spectral_centroid_up_for_ratio_above_one():
    sr = 24000
    audio = _harmonic_signal(sr, 1.0, 150)
    shifted = _formant_shift(audio, sr, 1.3)
    centroid_before = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
    centroid_after = librosa.feature.spectral_centroid(y=shifted, sr=sr).mean()
    assert centroid_after > centroid_before


def test_formant_shift_moves_spectral_centroid_down_for_ratio_below_one():
    sr = 24000
    audio = _harmonic_signal(sr, 1.0, 150)
    shifted = _formant_shift(audio, sr, 0.75)
    centroid_before = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
    centroid_after = librosa.feature.spectral_centroid(y=shifted, sr=sr).mean()
    assert centroid_after < centroid_before


def test_formant_shift_preserves_audio_length():
    sr = 24000
    audio = _harmonic_signal(sr, 0.73, 180)  # deliberately not a clean multiple of the hop size
    shifted = _formant_shift(audio, sr, 1.2)
    assert shifted.shape == audio.shape


def test_child_profile_pitches_and_formants_higher_than_highest():
    child_pitch, _, _, child_formant = _PROFILES["child"]
    highest_pitch, _, _, highest_formant = _PROFILES["highest"]
    assert child_pitch > highest_pitch
    assert child_formant > highest_formant


def test_apply_voice_profile_with_formant_shift_stays_bounded_on_harmonic_signal():
    # The parametrized sine-wave test above covers every profile on a
    # pure tone (no envelope shape); this covers the harmonic-signal case
    # the formant warp is actually designed to act on.
    sr = 24000
    audio = _harmonic_signal(sr, 1.5, 160)
    for profile in PROFILE_NAMES:
        shifted = apply_voice_profile(audio, sr, profile)
        assert shifted.shape == audio.shape
        assert not np.isnan(shifted).any()
        assert not np.isinf(shifted).any()
