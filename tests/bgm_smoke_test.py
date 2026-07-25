import os

from bgm import mix_with_bgm
import numpy as np

# Fake 4s speech-like signal (sine bursts + silence) to run all moods
# through the mixer without needing a real synthesis call.
sr = 24000
t = np.linspace(0, 4, sr * 4, endpoint=False)
fake_speech = (np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)).astype(np.float32) * 0.3

for mood in ["neutral", "happy", "sad", "excited", "calm", "unknown_mood_fallback"]:
    mixed = mix_with_bgm(fake_speech, sr, mood=mood)
    assert mixed.shape == fake_speech.shape, f"{mood}: shape mismatch"
    assert np.abs(mixed).max() <= 0.98 + 1e-6, f"{mood}: clipping guard failed"
    assert not np.isnan(mixed).any(), f"{mood}: NaN in output"
    print(f"{mood}: OK, peak={np.abs(mixed).max():.4f}, rms={np.sqrt(np.mean(mixed.astype(np.float64)**2)):.4f}")

print("ALL MOODS OK")
