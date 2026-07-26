from normalize import normalize_numbers, normalize_text


def test_normalize_numbers_spells_digits_individually():
    assert normalize_numbers("42", "hin") == "चार दो"


def test_normalize_numbers_unknown_language_falls_back_to_hindi():
    assert normalize_numbers("5", "xx") == normalize_numbers("5", "hin")


def test_normalize_text_strips_markdown_symbols():
    text = "## **Bold Title**\n> a quote\n_italic_ `code` [link](url) {x} <tag>"
    result = normalize_text(text, "hin")
    for ch in "#*_`^[]{}<>|\\@":
        assert ch not in result


def test_normalize_text_strips_emoji_without_corrupting_surrounding_text():
    # Regression test: a story containing an emoji (a real, live example -
    # "## **💔 கடைசி காத்திருப்பு**") must not lose or garble the native-
    # script text around it.
    text = "💔 கடைசி காத்திருப்பு"
    result = normalize_text(text, "tam")
    assert "கடைசி" in result
    assert "காத்திருப்பு" in result
    assert "💔" not in result


def test_normalize_text_emoji_in_middle_collapses_whitespace_cleanly():
    text = "கடைசி 💔 காத்திருப்பு"
    result = normalize_text(text, "tam")
    assert "  " not in result  # no leftover double space where the emoji was
    assert result == "கடைசி காத்திருப்பு"


def test_normalize_text_handles_compound_emoji_with_zwj_and_variation_selectors():
    # Family emoji (person+person+child, joined with zero-width joiners)
    # and a variation-selector-16 heart - multi-codepoint sequences the
    # tokenizer's implicit drop-behavior was never explicitly verified
    # against.
    text = "வணக்கம் 👨‍👩‍👧 மற்றும் ❤️ முடிவு"
    result = normalize_text(text, "tam")
    assert "வணக்கம்" in result
    assert "மற்றும்" in result
    assert "முடிவு" in result
    assert "‍" not in result  # zero-width joiner
    assert "️" not in result  # variation selector-16


def test_normalize_text_collapses_whitespace():
    assert normalize_text("hello    world\n\n\ttest", "hin") == "hello world test"


def test_normalize_text_pure_native_script_is_unaffected():
    text = "நமஸ்தே, இன்று வானிலை மிகவும் அழகாக உள்ளது."
    assert normalize_text(text, "tam") == text
