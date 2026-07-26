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
    assert character["has_avatar"] == 0


def test_character_avatar_round_trip(fresh_db):
    story_id = fresh_db.save_story(title="T", language="hin", text="text")
    char_id = fresh_db.save_character(
        name="Vaal", personality="menacing", backstory="a villain",
        language="hin", origin_story_id=story_id,
    )
    assert fresh_db.get_character_avatar(char_id) is None
    assert fresh_db.get_character(char_id)["has_avatar"] == 0

    fake_png_bytes = b"\x89PNG\r\n\x1a\nfakeavatardata"
    fresh_db.save_character_avatar(char_id, fake_png_bytes)

    assert fresh_db.get_character_avatar(char_id) == fake_png_bytes
    assert fresh_db.get_character(char_id)["has_avatar"] == 1


def test_get_character_excludes_raw_avatar_blob(fresh_db):
    story_id = fresh_db.save_story(title="T", language="hin", text="text")
    char_id = fresh_db.save_character(
        name="Vaal", personality="menacing", backstory="a villain",
        language="hin", origin_story_id=story_id,
    )
    fresh_db.save_character_avatar(char_id, b"some avatar bytes")
    character = fresh_db.get_character(char_id)
    assert "avatar" not in character


def test_list_characters_reports_has_avatar(fresh_db):
    story_id = fresh_db.save_story(title="T", language="hin", text="text")
    with_avatar = fresh_db.save_character(name="A", personality="p", backstory="b", language="hin", origin_story_id=story_id)
    without_avatar = fresh_db.save_character(name="B", personality="p", backstory="b", language="hin", origin_story_id=story_id)
    fresh_db.save_character_avatar(with_avatar, b"bytes")

    characters = {c["id"]: c for c in fresh_db.list_characters()}
    assert characters[with_avatar]["has_avatar"] == 1
    assert characters[without_avatar]["has_avatar"] == 0


def test_get_story_lineage_empty_for_root_story(fresh_db):
    story_id = fresh_db.save_story(title="root", language="hin", text="root text")
    assert fresh_db.get_story_lineage(story_id) == []


def test_get_story_lineage_walks_full_chain_root_first(fresh_db):
    root_id = fresh_db.save_story(title="root", language="hin", text="root text")
    mid_id = fresh_db.save_story(
        title="mid", language="hin", text="mid text", parent_story_id=root_id, branch_note="decision A",
    )
    leaf_id = fresh_db.save_story(
        title="leaf", language="hin", text="leaf text", parent_story_id=mid_id, branch_note="decision B",
    )

    lineage = fresh_db.get_story_lineage(leaf_id)
    assert [a["id"] for a in lineage] == [root_id, mid_id]
    assert lineage[1]["branch_note"] == "decision A"


def test_get_child_branches(fresh_db):
    root_id = fresh_db.save_story(title="root", language="hin", text="root text")
    child_id = fresh_db.save_story(
        title="child", language="hin", text="child text", parent_story_id=root_id, branch_note="a choice",
    )
    children = fresh_db.get_child_branches(root_id)
    assert [c["id"] for c in children] == [child_id]
    assert children[0]["branch_note"] == "a choice"


def test_save_quality_entry_and_list_backlog(fresh_db):
    story_id = fresh_db.save_story(title="weak", language="hin", text="weak text")
    fresh_db.save_quality_entry(story_id, {"overall": 40, "scored": True}, rewritten=True)
    backlog = fresh_db.list_quality_backlog()
    assert len(backlog) == 1
    assert backlog[0]["story_id"] == story_id
    assert backlog[0]["rewritten"] is True
    assert backlog[0]["axis_scores"]["overall"] == 40


def test_save_scene_image_and_get_images_for_story(fresh_db):
    # Distinct from save_story_image/get_story_image above (single cover
    # image BLOB) - this is a time-mapped storyboard entry (see images.py).
    story_id = fresh_db.save_story(title="illustrated", language="hin", text="story text")
    fresh_db.save_scene_image(story_id, 0, 0.0, 5.0, "एक बार की बात है", "a cinematic prompt", "/static/images/1/0.png")
    images = fresh_db.get_images_for_story(story_id)
    assert len(images) == 1
    assert images[0]["url"] == "/static/images/1/0.png"
    assert images[0]["excerpt"] == "एक बार की बात है"


def test_get_variants_empty_when_none_exist(fresh_db):
    story_id = fresh_db.save_story(title="original", language="hin", text="story text")
    assert fresh_db.get_variants(story_id) == []


def test_get_variants_returns_linked_language_variants(fresh_db):
    original_id = fresh_db.save_story(title="original", language="hin", text="story text")
    variant_id = fresh_db.save_story(
        title="original (tam)", language="tam", text="translated text", variant_of_story_id=original_id,
    )
    variants = fresh_db.get_variants(original_id)
    assert [v["id"] for v in variants] == [variant_id]
    assert variants[0]["language"] == "tam"
