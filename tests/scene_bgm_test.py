import numpy as np

from bgm import mix_with_scene_bgm, mix_with_bgm

sr = 24000
t = np.linspace(0, 9, sr * 9, endpoint=False)
fake_speech = (np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)).astype(np.float32) * 0.3

# Three 3-second segments with different genres.
segments = [
    (0, sr * 3, "calm"),
    (sr * 3, sr * 6, "horror"),
    (sr * 6, sr * 9, "excited"),
]

mixed = mix_with_scene_bgm(fake_speech, sr, segments)
assert mixed.shape == fake_speech.shape, "shape mismatch"
assert np.abs(mixed).max() <= 0.98 + 1e-6, "clipping guard failed"
assert not np.isnan(mixed).any(), "NaN in output"
assert not np.isinf(mixed).any(), "Inf in output"
print(f"scene mix OK: peak={np.abs(mixed).max():.4f} rms={np.sqrt(np.mean(mixed.astype(np.float64)**2)):.4f}")

# Single-segment edge case (should behave sanely, same as mix_with_bgm).
single = mix_with_scene_bgm(fake_speech, sr, [(0, sr * 9, "romance")])
assert single.shape == fake_speech.shape
print(f"single-segment OK: peak={np.abs(single).max():.4f}")

# Empty segments -> fallback.
empty = mix_with_scene_bgm(fake_speech, sr, [])
assert empty.shape == fake_speech.shape
print(f"empty-segments fallback OK: peak={np.abs(empty).max():.4f}")

# Check the crossfade region (around 3s and 6s boundaries) has energy
# from BOTH neighboring genres rather than a silent dip.
for boundary_s in (3, 6):
    idx = int(sr * boundary_s)
    window = mixed[idx - 1000: idx + 1000]
    rms = np.sqrt(np.mean(window.astype(np.float64) ** 2))
    print(f"boundary at {boundary_s}s: local rms={rms:.4f} (should not be near-zero)")

print("DONE")
