import numpy as np

from sfx import SFX_TYPES, get_sfx_loop

for sfx_type in SFX_TYPES:
    loop = get_sfx_loop(sfx_type)
    if sfx_type == "none":
        assert loop is None
        print("none: OK (no loop)")
        continue
    assert loop is not None, f"{sfx_type} produced no loop"
    assert not np.isnan(loop).any(), f"{sfx_type}: NaN"
    assert not np.isinf(loop).any(), f"{sfx_type}: Inf"
    peak = np.abs(loop).max()
    assert peak <= 0.95, f"{sfx_type}: peak too high ({peak})"
    print(f"{sfx_type}: OK peak={peak:.4f} rms={np.sqrt(np.mean(loop.astype(np.float64)**2)):.4f}")

print(f"ALL {len(SFX_TYPES)} SFX TYPES OK")
