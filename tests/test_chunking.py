import numpy as np

import app
from app import _map_beats_to_time, _split_into_chunks, _split_sentences


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


def test_map_beats_to_time_locates_verbatim_excerpt():
    raw_chunks = ["once upon a time in a village", "a storm arrived suddenly", "the hero returned home"]
    chunk_offsets = [(0, 1000), (1000, 2500), (2500, 4000)]
    beats = [
        {"excerpt": "once upon a time", "visual_prompt": "a village at dawn"},
        {"excerpt": "a storm arrived", "visual_prompt": "a dark storm"},
        {"excerpt": "the hero returned", "visual_prompt": "a hero walking home"},
    ]
    mapped = _map_beats_to_time(beats, raw_chunks, chunk_offsets, sr=1000)
    assert len(mapped) == 3
    assert mapped[0]["start_time_s"] == 0.0
    assert mapped[1]["start_time_s"] == 1.0
    assert mapped[2]["start_time_s"] == 2.5
    assert mapped[2]["end_time_s"] == 4.0


def test_map_beats_to_time_falls_back_when_excerpt_not_found():
    raw_chunks = ["chunk one text", "chunk two text", "chunk three text"]
    chunk_offsets = [(0, 1000), (1000, 2000), (2000, 3000)]
    beats = [{"excerpt": "not present anywhere", "visual_prompt": "p"}]
    mapped = _map_beats_to_time(beats, raw_chunks, chunk_offsets, sr=1000)
    assert len(mapped) == 1
    assert mapped[0]["start_time_s"] == 0.0


def test_map_beats_to_time_empty_inputs():
    assert _map_beats_to_time([], ["chunk"], [(0, 100)], sr=100) == []
    assert _map_beats_to_time([{"excerpt": "x", "visual_prompt": "y"}], [], [], sr=100) == []


def test_maybe_generate_scene_images_saves_audio_when_at_least_one_succeeds(monkeypatch):
    monkeypatch.setattr(app, "generate_visual_prompts", lambda *a, **k: [{"excerpt": "hello", "visual_prompt": "a prompt"}])
    monkeypatch.setattr(app, "generate_scene_image", lambda prompt: b"fake-png")
    monkeypatch.setattr(app, "save_scene_image", lambda story_id, idx, data: f"/static/images/{story_id}/{idx}.png")
    save_image_calls = []
    monkeypatch.setattr(app.db, "save_scene_image", lambda *a, **k: save_image_calls.append(a))
    monkeypatch.setattr(app.db, "get_story_image", lambda story_id: None)
    cover_save_calls = []
    monkeypatch.setattr(app.db, "save_story_image", lambda *a, **k: cover_save_calls.append(a))
    audio_save_calls = []
    monkeypatch.setattr(app, "_save_story_audio", lambda *a, **k: audio_save_calls.append(a))

    count = app._maybe_generate_scene_images(
        4, 1, "hello world", "hin", [], [(0, 1000)], ["hello world"], np.zeros(1000, dtype=np.float32), 1000,
    )
    assert count == 1
    assert len(save_image_calls) == 1
    assert len(audio_save_calls) == 1
    # First scene image also becomes the story's cover so it shows up
    # anywhere has_image is checked (Stories & Branches, Dub Studio, ...),
    # not just inside its own storyboard.
    assert cover_save_calls == [(1, b"fake-png")]


def test_maybe_generate_scene_images_does_not_overwrite_existing_cover(monkeypatch):
    monkeypatch.setattr(app, "generate_visual_prompts", lambda *a, **k: [{"excerpt": "hello", "visual_prompt": "a prompt"}])
    monkeypatch.setattr(app, "generate_scene_image", lambda prompt: b"fake-png")
    monkeypatch.setattr(app, "save_scene_image", lambda story_id, idx, data: f"/static/images/{story_id}/{idx}.png")
    monkeypatch.setattr(app.db, "save_scene_image", lambda *a, **k: None)
    monkeypatch.setattr(app.db, "get_story_image", lambda story_id: b"existing-cover-bytes")
    cover_save_calls = []
    monkeypatch.setattr(app.db, "save_story_image", lambda *a, **k: cover_save_calls.append(a))
    monkeypatch.setattr(app, "_save_story_audio", lambda *a, **k: None)

    app._maybe_generate_scene_images(
        4, 1, "hello world", "hin", [], [(0, 1000)], ["hello world"], np.zeros(1000, dtype=np.float32), 1000,
    )
    assert cover_save_calls == []


def test_maybe_generate_scene_images_skips_audio_save_when_all_images_fail(monkeypatch):
    monkeypatch.setattr(app, "generate_visual_prompts", lambda *a, **k: [{"excerpt": "hello", "visual_prompt": "a prompt"}])

    def _raise(prompt):
        raise app.ImageGenError("boom")

    monkeypatch.setattr(app, "generate_scene_image", _raise)
    audio_save_calls = []
    monkeypatch.setattr(app, "_save_story_audio", lambda *a, **k: audio_save_calls.append(a))

    count = app._maybe_generate_scene_images(
        4, 1, "hello world", "hin", [], [(0, 1000)], ["hello world"], np.zeros(1000, dtype=np.float32), 1000,
    )
    assert count == 0
    assert audio_save_calls == []


def test_maybe_generate_scene_images_returns_zero_on_unexpected_error(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(app, "generate_visual_prompts", _raise)
    count = app._maybe_generate_scene_images(
        4, 1, "hello world", "hin", [], [(0, 1000)], ["hello world"], np.zeros(1000, dtype=np.float32), 1000,
    )
    assert count == 0
