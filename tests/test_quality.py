import quality


def test_score_story_returns_unscored_fallback_without_api_key(monkeypatch):
    monkeypatch.setattr(quality, "OPENAI_API_KEY", None)
    scores = quality.score_story("some story text", "hin")
    assert scores["scored"] is False
    assert scores["overall"] == 100
    assert all(scores[axis] == 100 for axis in quality.QUALITY_AXES)


def test_score_and_maybe_rewrite_skips_rewrite_when_unscored(monkeypatch):
    monkeypatch.setattr(quality, "OPENAI_API_KEY", None)
    text, scores = quality.score_and_maybe_rewrite("some story text", "hin")
    assert text == "some story text"
    assert scores["rewrite_attempted"] is False


def test_score_and_maybe_rewrite_keeps_good_first_draft(monkeypatch):
    monkeypatch.setattr(quality, "score_story", lambda text, lang: {
        "native_sounding": 90, "tone_appropriateness": 85, "cultural_accuracy": 88,
        "emotional_delivery": 92, "overall": 85, "scored": True,
    })
    called = {"rewrite": False}

    def _rewrite(*a, **k):
        called["rewrite"] = True
        return "should not be used"

    monkeypatch.setattr(quality, "rewrite_story", _rewrite)
    text, scores = quality.score_and_maybe_rewrite("a solid story", "hin")
    assert text == "a solid story"
    assert called["rewrite"] is False
    assert scores["rewrite_attempted"] is False


def test_score_and_maybe_rewrite_uses_rewrite_when_it_scores_higher(monkeypatch):
    call_count = {"n": 0}

    def _score(text, lang):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"native_sounding": 40, "tone_appropriateness": 50, "cultural_accuracy": 45,
                     "emotional_delivery": 50, "overall": 40, "scored": True}
        return {"native_sounding": 90, "tone_appropriateness": 88, "cultural_accuracy": 85,
                "emotional_delivery": 90, "overall": 85, "scored": True}

    monkeypatch.setattr(quality, "score_story", _score)
    monkeypatch.setattr(quality, "rewrite_story", lambda text, scores, lang: "the improved rewrite")

    text, scores = quality.score_and_maybe_rewrite("a weak story", "hin")
    assert text == "the improved rewrite"
    assert scores["overall"] == 85
    assert scores["rewrite_attempted"] is True


def test_score_and_maybe_rewrite_keeps_original_when_rewrite_fails(monkeypatch):
    monkeypatch.setattr(quality, "score_story", lambda text, lang: {
        "native_sounding": 40, "tone_appropriateness": 40, "cultural_accuracy": 40,
        "emotional_delivery": 40, "overall": 40, "scored": True,
    })

    def _raise(*a, **k):
        raise quality.StoryGenError("boom")

    monkeypatch.setattr(quality, "rewrite_story", _raise)
    text, scores = quality.score_and_maybe_rewrite("a weak story", "hin")
    assert text == "a weak story"
    assert scores["overall"] == 40
    assert scores["rewrite_attempted"] is True
