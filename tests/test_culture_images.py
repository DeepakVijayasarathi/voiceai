import pytest

import culture_images
import db


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    # get_culture_image -> story_gen.get_effective_cultural_context now
    # reads db.get_culture_pack_override - isolate DB_PATH in every test
    # here so none of them touch the real default location.
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(db, "_conn", None)
    db.init_db()
    yield
    monkeypatch.setattr(db, "_conn", None)


def test_get_culture_image_unknown_language_raises_keyerror(tmp_path, monkeypatch):
    monkeypatch.setattr(culture_images, "STATIC_DIR", str(tmp_path))
    with pytest.raises(KeyError):
        culture_images.get_culture_image("xx")


def test_get_culture_image_generates_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(culture_images, "STATIC_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(
        culture_images, "_generate_image_from_prompt",
        lambda prompt: calls.append(prompt) or b"fake-png-bytes",
    )

    result = culture_images.get_culture_image("tam")
    assert result == b"fake-png-bytes"
    assert len(calls) == 1
    assert "Tamil" in calls[0]

    cache_path = tmp_path / "culture" / "tam.png"
    assert cache_path.read_bytes() == b"fake-png-bytes"


def test_get_culture_image_uses_cache_without_regenerating(tmp_path, monkeypatch):
    monkeypatch.setattr(culture_images, "STATIC_DIR", str(tmp_path))
    cache_dir = tmp_path / "culture"
    cache_dir.mkdir()
    (cache_dir / "hin.png").write_bytes(b"cached-bytes")

    def _should_not_be_called(prompt):
        raise AssertionError("should not regenerate when a cached copy exists")

    monkeypatch.setattr(culture_images, "_generate_image_from_prompt", _should_not_be_called)
    result = culture_images.get_culture_image("hin")
    assert result == b"cached-bytes"


def test_get_culture_image_uses_edited_places(tmp_path, monkeypatch):
    monkeypatch.setattr(culture_images, "STATIC_DIR", str(tmp_path))
    db.save_culture_pack_override("tam", {"places": "a custom test landmark"})
    calls = []
    monkeypatch.setattr(
        culture_images, "_generate_image_from_prompt",
        lambda prompt: calls.append(prompt) or b"fake-png-bytes",
    )
    culture_images.get_culture_image("tam")
    assert "a custom test landmark" in calls[0]
    assert "Madurai" not in calls[0]


def test_invalidate_culture_image_removes_cache_file(tmp_path, monkeypatch):
    monkeypatch.setattr(culture_images, "STATIC_DIR", str(tmp_path))
    cache_dir = tmp_path / "culture"
    cache_dir.mkdir()
    (cache_dir / "hin.png").write_bytes(b"stale-bytes")

    culture_images.invalidate_culture_image("hin")
    assert not (cache_dir / "hin.png").exists()


def test_invalidate_culture_image_noop_when_nothing_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(culture_images, "STATIC_DIR", str(tmp_path))
    culture_images.invalidate_culture_image("hin")  # should not raise
