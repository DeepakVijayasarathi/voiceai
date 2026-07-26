import pytest

import image_gen


def test_generate_story_image_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(image_gen, "OPENAI_API_KEY", None)
    with pytest.raises(image_gen.ImageGenError, match="OPENAI_API_KEY"):
        image_gen.generate_story_image("a story")


def test_derive_visual_prompt_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(image_gen, "OPENAI_API_KEY", None)
    with pytest.raises(image_gen.ImageGenError, match="OPENAI_API_KEY"):
        image_gen._derive_visual_prompt("a story")


def test_generate_character_avatar_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(image_gen, "OPENAI_API_KEY", None)
    with pytest.raises(image_gen.ImageGenError, match="OPENAI_API_KEY"):
        image_gen.generate_character_avatar("Vikram", "sharp-witted detective", "solves locked-room mysteries")


def test_generate_character_avatar_builds_prompt_from_fields(monkeypatch):
    captured = {}

    def _fake_generate_image_from_prompt(prompt):
        captured["prompt"] = prompt
        return b"fake-png-bytes"

    monkeypatch.setattr(image_gen, "_generate_image_from_prompt", _fake_generate_image_from_prompt)
    result = image_gen.generate_character_avatar("Vikram", "sharp-witted detective", "solves locked-room mysteries")
    assert result == b"fake-png-bytes"
    assert "Vikram" in captured["prompt"]
    assert "sharp-witted detective" in captured["prompt"]


def test_generate_story_image_still_uses_two_step_prompt_derivation(monkeypatch):
    calls = []
    monkeypatch.setattr(image_gen, "_derive_visual_prompt", lambda text: calls.append(text) or "a derived prompt")
    monkeypatch.setattr(image_gen, "_generate_image_from_prompt", lambda prompt: (calls.append(prompt), b"fake-png")[1])
    result = image_gen.generate_story_image("a story about a detective")
    assert result == b"fake-png"
    assert calls == ["a story about a detective", "a derived prompt"]
