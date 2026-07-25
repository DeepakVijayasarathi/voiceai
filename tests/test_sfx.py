import numpy as np
import pytest

from sfx import SFX_TYPES, get_sfx_loop, tile_sfx_to_length


@pytest.mark.parametrize("sfx_type", SFX_TYPES)
def test_sfx_loop_generates_cleanly(sfx_type):
    loop = get_sfx_loop(sfx_type)
    if sfx_type == "none":
        assert loop is None
        return
    assert loop is not None
    assert not np.isnan(loop).any()
    assert not np.isinf(loop).any()
    assert np.abs(loop).max() <= 0.95


def test_sfx_loop_is_cached():
    first = get_sfx_loop("rain")
    second = get_sfx_loop("rain")
    assert first is second


def test_tile_sfx_to_length_matches_requested_length():
    sr = 24000
    n = sr * 5
    tiled = tile_sfx_to_length("rain", sr, n)
    assert tiled.shape == (n,)
    assert not np.isnan(tiled).any()


def test_tile_sfx_to_length_none_returns_none():
    assert tile_sfx_to_length("none", 24000, 24000) is None


def test_tile_sfx_to_length_resamples_for_different_sample_rate():
    tiled = tile_sfx_to_length("wind", 16000, 16000 * 3)
    assert tiled.shape == (16000 * 3,)
