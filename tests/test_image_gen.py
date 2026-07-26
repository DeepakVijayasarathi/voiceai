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
