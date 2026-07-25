import numpy as np

from bgm import mix_with_scene_bgm
from sfx import SFX_TYPES, get_sfx_loop

sr = 24000
t = np.linspace(0, 9, sr * 9, endpoint=False)
fake_speech = (np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)).astype(np.float32) * 0.3

# Verify every SFX type generates cleanly.
for sfx_type in SFX_TYPES:
    loop = get_sfx_loop(sfx_type)
    if sfx_type == "none":
        assert loop is None, "none should produce no loop"
        print("none: OK (no loop, as expected)")
        continue
    assert loop is not None
    assert not np.isnan(loop).any() and not np.isinf(loop).any()
    print(f"{sfx_type}: OK peak={np.abs(loop).max():.4f} rms={np.sqrt(np.mean(loop.astype(np.float64)**2)):.4f}")

# Full scene + sfx mix.
segments = [(0, sr * 3, "calm"), (sr * 3, sr * 6, "horror"), (sr * 6, sr * 9, "action")]
sfx_segments = [(0, sr * 3, "rain"), (sr * 3, sr * 6, "thunder"), (sr * 6, sr * 9, "footsteps")]
mixed = mix_with_scene_bgm(fake_speech, sr, segments, sfx_segments=sfx_segments)
assert mixed.shape == fake_speech.shape
assert np.abs(mixed).max() <= 0.98 + 1e-6, "clipping guard failed"
assert not np.isnan(mixed).any()
print(f"scene+sfx mix: OK peak={np.abs(mixed).max():.4f} rms={np.sqrt(np.mean(mixed.astype(np.float64)**2)):.4f}")

print("DONE")
