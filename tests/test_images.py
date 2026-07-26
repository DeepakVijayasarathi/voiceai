import pytest

import images
import story_gen


def test_generate_scene_image_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(images, "OPENAI_API_KEY", None)
    with pytest.raises(images.ImageGenError):
        images.generate_scene_image("a cinematic prompt")


def test_generate_visual_prompts_returns_empty_without_api_key(monkeypatch):
    # story_gen._chat_completion raises StoryGenError when OPENAI_API_KEY is
    # unset - generate_visual_prompts must swallow that (non-fatal) and
    # return [] rather than propagate, same posture as extract_characters.
    monkeypatch.setattr(story_gen, "OPENAI_API_KEY", None)
    result = images.generate_visual_prompts("एक बार की बात है...", "hin", ["Arjun"])
    assert result == []


def test_generate_visual_prompts_caps_at_max_beats(monkeypatch):
    monkeypatch.setattr(
        images, "_chat_completion",
        lambda *a, **k: '[{"excerpt": "a", "visual_prompt": "1"}, {"excerpt": "b", "visual_prompt": "2"}, {"excerpt": "c", "visual_prompt": "3"}]',
    )
    result = images.generate_visual_prompts("some story", "hin", [], max_beats=2)
    assert len(result) == 2


def test_generate_visual_prompts_skips_malformed_entries(monkeypatch):
    monkeypatch.setattr(
        images, "_chat_completion",
        lambda *a, **k: '[{"excerpt": "a"}, {"visual_prompt": "no excerpt"}, {"excerpt": "ok", "visual_prompt": "ok prompt"}]',
    )
    result = images.generate_visual_prompts("some story", "hin", [])
    assert result == [{"excerpt": "ok", "visual_prompt": "ok prompt"}]


def test_save_scene_image_writes_file_and_returns_url(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "STATIC_DIR", str(tmp_path))
    url = images.save_scene_image(42, 0, b"fake-png-bytes")
    assert url == "/static/images/42/0.png"
    saved_path = tmp_path / "images" / "42" / "0.png"
    assert saved_path.read_bytes() == b"fake-png-bytes"
