import pytest

import db


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(db, "_conn", None)
    db.init_db()
    yield db
    monkeypatch.setattr(db, "_conn", None)


def test_save_and_get_story(fresh_db):
    story_id = fresh_db.save_story(title="A Story", language="hin", text="कहानी")
    story = fresh_db.get_story(story_id)
    assert story["title"] == "A Story"
    assert story["text"] == "कहानी"
    assert story["has_image"] == 0


def test_story_image_round_trip(fresh_db):
    story_id = fresh_db.save_story(title="T", language="hin", text="text")
    assert fresh_db.get_story_image(story_id) is None
    assert fresh_db.get_story(story_id)["has_image"] == 0

    fake_png_bytes = b"\x89PNG\r\n\x1a\nfakeimagedata"
    fresh_db.save_story_image(story_id, fake_png_bytes)

    assert fresh_db.get_story_image(story_id) == fake_png_bytes
    assert fresh_db.get_story(story_id)["has_image"] == 1


def test_get_story_excludes_raw_image_blob(fresh_db):
    story_id = fresh_db.save_story(title="T", language="hin", text="text")
    fresh_db.save_story_image(story_id, b"some image bytes")
    story = fresh_db.get_story(story_id)
    assert "image" not in story


def test_list_stories_reports_has_image(fresh_db):
    with_image = fresh_db.save_story(title="A", language="hin", text="a")
    without_image = fresh_db.save_story(title="B", language="hin", text="b")
    fresh_db.save_story_image(with_image, b"bytes")

    stories = {s["id"]: s for s in fresh_db.list_stories()}
    assert stories[with_image]["has_image"] == 1
    assert stories[without_image]["has_image"] == 0


def test_get_story_image_missing_story_returns_none(fresh_db):
    assert fresh_db.get_story_image(99999) is None


def test_story_audio_round_trip(fresh_db):
    story_id = fresh_db.save_story(title="T", language="hin", text="text")
    assert fresh_db.get_story_audio(story_id) is None
    assert fresh_db.get_story(story_id)["has_audio"] == 0

    fake_wav_bytes = b"RIFF....WAVEfmt fakeaudiodata"
    fresh_db.save_story_audio(story_id, fake_wav_bytes)

    assert fresh_db.get_story_audio(story_id) == fake_wav_bytes
    assert fresh_db.get_story(story_id)["has_audio"] == 1


def test_get_story_excludes_raw_audio_blob(fresh_db):
    story_id = fresh_db.save_story(title="T", language="hin", text="text")
    fresh_db.save_story_audio(story_id, b"some audio bytes")
    story = fresh_db.get_story(story_id)
    assert "audio" not in story


def test_list_stories_reports_has_audio(fresh_db):
    with_audio = fresh_db.save_story(title="A", language="hin", text="a")
    without_audio = fresh_db.save_story(title="B", language="hin", text="b")
    fresh_db.save_story_audio(with_audio, b"bytes")

    stories = {s["id"]: s for s in fresh_db.list_stories()}
    assert stories[with_audio]["has_audio"] == 1
    assert stories[without_audio]["has_audio"] == 0


def test_save_and_get_character_with_voice_profile(fresh_db):
    story_id = fresh_db.save_story(title="T", language="hin", text="text")
    char_id = fresh_db.save_character(
        name="Vaal", personality="menacing", backstory="a villain",
        language="hin", origin_story_id=story_id, voice_profile="deepest",
    )
    character = fresh_db.get_character(char_id)
    assert character["name"] == "Vaal"
    assert character["voice_profile"] == "deepest"
