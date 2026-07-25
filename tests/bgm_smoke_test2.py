import time

import numpy as np

from bgm import GENRES, mix_with_bgm

sr = 24000
t = np.linspace(0, 4, sr * 4, endpoint=False)
fake_speech = (np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)).astype(np.float32) * 0.3

for genre in list(GENRES.keys()) + ["unknown_fallback"]:
    t0 = time.time()
    mixed = mix_with_bgm(fake_speech, sr, mood=genre)
    dt = time.time() - t0
    assert mixed.shape == fake_speech.shape, f"{genre}: shape mismatch"
    assert np.abs(mixed).max() <= 0.98 + 1e-6, f"{genre}: clipping guard failed"
    assert not np.isnan(mixed).any(), f"{genre}: NaN in output"
    assert not np.isinf(mixed).any(), f"{genre}: Inf in output"
    print(f"{genre:12s}: OK  gen_time={dt:.3f}s peak={np.abs(mixed).max():.4f} rms={np.sqrt(np.mean(mixed.astype(np.float64)**2)):.4f}")

print("ALL GENRES OK")
