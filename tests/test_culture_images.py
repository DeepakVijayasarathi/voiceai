import pytest

import culture_images


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
