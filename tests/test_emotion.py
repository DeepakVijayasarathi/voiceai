from emotion import VALID_EMOTIONS, _normalize_item, _normalize_sfx_list, _reconcile_length
from sfx import SFX_TYPES


def test_normalize_item_passes_through_valid_values():
    item = {"genre": "horror", "bgm_intensity": 0.8, "sfx": ["rain"], "sfx_intensity": 0.4, "speaker": "Priya"}
    result = _normalize_item(item, known_characters=["Priya"])
    assert result == {"genre": "horror", "bgm_intensity": 0.8, "sfx": ["rain"], "sfx_intensity": 0.4, "speaker": "Priya"}


def test_normalize_item_rejects_invalid_genre_and_sfx():
    item = {"genre": "not_a_genre", "sfx": ["not_a_sfx"], "speaker": "narrator"}
    result = _normalize_item(item, known_characters=[])
    assert result["genre"] == "neutral"
    assert result["sfx"] == []


def test_normalize_item_rejects_unknown_speaker():
    item = {"genre": "neutral", "sfx": [], "speaker": "SomeoneNotInCast"}
    result = _normalize_item(item, known_characters=["Priya", "Vikram"])
    assert result["speaker"] == "narrator"


def test_normalize_item_clamps_intensity_to_0_1():
    item = {"genre": "neutral", "sfx": [], "bgm_intensity": 5.0, "sfx_intensity": -3.0}
    result = _normalize_item(item, known_characters=[])
    assert result["bgm_intensity"] == 1.0
    assert result["sfx_intensity"] == 0.0


def test_normalize_item_non_dict_returns_default():
    result = _normalize_item("not a dict", known_characters=[])
    assert result["genre"] == "neutral"
    assert result["sfx"] == []
    assert result["speaker"] == "narrator"


def test_normalize_item_missing_intensity_falls_back_to_default():
    item = {"genre": "happy", "sfx": []}
    result = _normalize_item(item, known_characters=[])
    assert result["bgm_intensity"] == 0.6
    assert result["sfx_intensity"] == 0.6


def test_normalize_sfx_list_caps_at_two_and_dedupes():
    assert _normalize_sfx_list(["rain", "rain", "thunder", "wind"]) == ["rain", "thunder"]


def test_normalize_sfx_list_filters_invalid_and_none():
    assert _normalize_sfx_list(["rain", "not_a_real_sfx", "none"]) == ["rain"]


def test_normalize_sfx_list_tolerates_bare_string():
    # The model occasionally returns "sfx": "rain" instead of ["rain"]
    # despite the schema - should still be usable, not discarded.
    assert _normalize_sfx_list("rain") == ["rain"]


def test_normalize_sfx_list_handles_empty_and_garbage():
    assert _normalize_sfx_list([]) == []
    assert _normalize_sfx_list(None) == []
    assert _normalize_sfx_list(123) == []


def test_normalize_item_multi_sfx_passes_through():
    item = {"genre": "horror", "sfx": ["rain", "thunder"], "sfx_intensity": 0.8}
    result = _normalize_item(item, known_characters=[])
    assert result["sfx"] == ["rain", "thunder"]


def test_reconcile_length_more_items_than_expected():
    # Reproduces the observed case: model returns 3 sub-scene items for
    # what was sent as a single input element.
    items = [
        {"genre": "neutral", "sfx": ["traffic"], "sfx_intensity": 0.5, "speaker": "narrator"},
        {"genre": "excited", "sfx": ["car"], "sfx_intensity": 0.6, "speaker": "narrator"},
        {"genre": "calm", "sfx": ["door_creak"], "sfx_intensity": 0.3, "speaker": "narrator"},
    ]
    result = _reconcile_length(items, n_expected=1, known_characters=[])
    assert len(result) == 1
    # Should surface real (non-fallback) genre/sfx rather than discarding everything.
    assert result[0]["genre"] in VALID_EMOTIONS
    assert all(s in SFX_TYPES for s in result[0]["sfx"])
    assert result[0]["sfx"]  # merged from the bucket, not empty
    assert len(result[0]["sfx"]) <= 2


def test_reconcile_length_fewer_items_than_expected():
    items = [{"genre": "horror", "sfx": ["rain"], "sfx_intensity": 0.7, "speaker": "narrator"}]
    result = _reconcile_length(items, n_expected=3, known_characters=[])
    assert len(result) == 3


def test_reconcile_length_empty_items_returns_defaults():
    result = _reconcile_length([], n_expected=2, known_characters=[])
    assert len(result) == 2
    assert all(r["genre"] == "neutral" and r["sfx"] == [] for r in result)


def test_reconcile_length_preserves_speaker_attribution():
    items = [
        {"genre": "neutral", "sfx": [], "speaker": "narrator"},
        {"genre": "horror", "sfx": [], "speaker": "Priya"},
    ]
    result = _reconcile_length(items, n_expected=1, known_characters=["Priya"])
    assert result[0]["speaker"] == "Priya"
