import shutil

import pytest

import db
import export


def _fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(db, "_conn", None)
    db.init_db()


def test_build_game_tree_raises_for_unknown_story(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    with pytest.raises(export.ExportError) as exc_info:
        export.build_game_tree(999)
    assert exc_info.value.status_code == 404


def test_build_game_tree_includes_branches_as_children(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    root_id = db.save_story(title="root", language="hin", text="root text")
    child_id = db.save_story(
        title="child", language="hin", text="child text", parent_story_id=root_id, branch_note="a choice",
    )
    tree = export.build_game_tree(root_id)
    assert tree["id"] == root_id
    assert len(tree["children"]) == 1
    assert tree["children"][0]["id"] == child_id
    assert tree["children"][0]["branch_note"] == "a choice"
    assert tree["children"][0]["children"] == []


def test_build_comic_panels_empty_without_images(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    story_id = db.save_story(title="no images", language="hin", text="story text")
    result = export.build_comic_panels(story_id)
    assert result == {"panels": []}


def test_build_comic_panels_uses_native_excerpt_as_caption(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    story_id = db.save_story(title="illustrated", language="hin", text="story text")
    db.save_scene_image(
        story_id, 0, 0.0, 5.0, "एक बार की बात है", "an English visual prompt", "/static/images/1/0.png",
    )
    result = export.build_comic_panels(story_id)
    assert result["panels"] == [
        {"image_url": "/static/images/1/0.png", "caption": "एक बार की बात है", "start_time_s": 0.0}
    ]


def test_render_video_raises_404_for_unknown_story(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    with pytest.raises(export.ExportError) as exc_info:
        export.render_video(999)
    assert exc_info.value.status_code == 404


def test_render_video_raises_501_when_ffmpeg_missing(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    story_id = db.save_story(title="a story", language="hin", text="story text")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(export.ExportError) as exc_info:
        export.render_video(story_id)
    assert exc_info.value.status_code == 501


def test_render_video_raises_409_without_saved_audio(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    monkeypatch.setattr(export, "STATIC_DIR", str(tmp_path / "static"))
    story_id = db.save_story(title="a story", language="hin", text="story text")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")
    with pytest.raises(export.ExportError) as exc_info:
        export.render_video(story_id)
    assert exc_info.value.status_code == 409
