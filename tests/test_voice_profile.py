import numpy as np
import pytest

from voice_profile import PROFILE_NAMES, apply_voice_profile, assign_profiles, get_speed_multiplier, get_voice_profile


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
