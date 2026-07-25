from app import _split_into_chunks, _split_sentences


def test_split_sentences_basic_hindi_dandas():
    text = "यह पहला वाक्य है। यह दूसरा वाक्य है। यह तीसरा वाक्य है।"
    sentences = _split_sentences(text)
    assert len(sentences) == 3


def test_split_sentences_handles_quoted_dialogue_boundary():
    # Regression test: a sentence ending inside a closing quote mark
    # (terminator immediately followed by a quote, not whitespace) must
    # still split - this silently failed before and merged whole
    # back-and-forth dialogue exchanges into one "sentence".
    text = (
        'प्रिया ने कहा, "मुझे डर लग रहा है।" '
        'राहुल ने जवाब दिया, "चिंता मत करो।"'
    )
    sentences = _split_sentences(text)
    assert len(sentences) == 2
    assert "प्रिया" in sentences[0]
    assert "राहुल" in sentences[1]


def test_split_sentences_empty_string():
    assert _split_sentences("") == []


def test_split_sentences_no_terminator_returns_whole_text():
    assert _split_sentences("no terminator here") == ["no terminator here"]


def test_split_into_chunks_respects_char_budget():
    text = "एक। " * 200  # many tiny sentences
    chunks = _split_into_chunks(text, max_chars=50)
    assert all(len(c) <= 60 for c in chunks)  # small slack for the trailing sentence


def test_split_into_chunks_never_drops_content():
    text = "पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य। चौथा वाक्य।"
    chunks = _split_into_chunks(text, max_chars=20)
    rejoined_words = " ".join(chunks).split()
    original_words = text.split()
    assert sorted(rejoined_words) == sorted(original_words)


def test_split_into_chunks_single_short_text_is_one_chunk():
    chunks = _split_into_chunks("छोटा वाक्य।", max_chars=300)
    assert len(chunks) == 1
