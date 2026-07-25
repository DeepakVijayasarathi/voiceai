import numpy as np

from voice_profile import get_voice_profile, apply_voice_profile, assign_profiles, _PROFILE_NAMES

# Determinism check.
for name in ["Priya", "priya", " Priya ", "Vikram", "Ravi", "Meera"]:
    print(f"{name!r} -> {get_voice_profile(name)}")

assert get_voice_profile("Priya") == get_voice_profile("priya") == get_voice_profile(" Priya ")
print("determinism OK (case/whitespace-insensitive)")

cast = ["Priya", "Vikram", "Ravi", "Meera"]
assignment = assign_profiles(cast)
print("cast assignment:", assignment)
assert len(set(assignment.values())) == len(cast), "collision in small cast!"
print("collision-avoidance OK (all distinct within cast)")

# Re-running with the same cast should give the same assignment.
assert assign_profiles(cast) == assignment
print("stable-reassignment OK")

sr = 24000
t = np.linspace(0, 3, sr * 3, endpoint=False)
fake_speech = (np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.float32) * 0.4

for profile in _PROFILE_NAMES + ["narrator"]:
    shifted = apply_voice_profile(fake_speech, sr, profile)
    assert shifted.shape == fake_speech.shape, f"{profile}: shape mismatch {shifted.shape} vs {fake_speech.shape}"
    assert not np.isnan(shifted).any(), f"{profile}: NaN"
    assert not np.isinf(shifted).any(), f"{profile}: Inf"
    assert np.abs(shifted).max() <= 0.98 + 1e-6, f"{profile}: clipping"
    print(f"{profile}: OK peak={np.abs(shifted).max():.4f} rms={np.sqrt(np.mean(shifted.astype(np.float64)**2)):.4f}")

print("DONE")
