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


def test_variation_index_zero_matches_default_call():
    # variation_index=0 must reproduce exactly what a plain get_sfx_loop()
    # call already produced before variation existed - nothing that
    # depended on the original default output should see it change.
    assert np.array_equal(get_sfx_loop("rain"), get_sfx_loop("rain", variation_index=0))


@pytest.mark.parametrize("sfx_type", [t for t in SFX_TYPES if t != "none"])
def test_different_variation_index_produces_different_audio(sfx_type):
    default = get_sfx_loop(sfx_type, variation_index=0)
    other = get_sfx_loop(sfx_type, variation_index=1)
    assert not np.array_equal(default, other)
    # still a valid loop, not just noise-shaped garbage
    assert not np.isnan(other).any()
    assert not np.isinf(other).any()
    assert np.abs(other).max() <= 0.95


def test_same_variation_index_is_cached_and_stable():
    first = get_sfx_loop("thunder", variation_index=3)
    second = get_sfx_loop("thunder", variation_index=3)
    assert first is second


def test_tile_sfx_to_length_honors_variation_index():
    sr = 24000
    n = sr * 3
    a = tile_sfx_to_length("footsteps", sr, n, variation_index=0)
    b = tile_sfx_to_length("footsteps", sr, n, variation_index=5)
    assert not np.array_equal(a, b)
    assert a.shape == b.shape == (n,)
