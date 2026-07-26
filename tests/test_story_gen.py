import story_gen
from story_gen import LANGUAGE_CULTURAL_CONTEXT, LANGUAGE_NAMES, _cultural_clause

REQUIRED_DIMENSIONS = {"places", "festivals", "folklore", "deities", "names", "register_note"}


def test_every_language_has_cultural_context():
    assert set(LANGUAGE_CULTURAL_CONTEXT.keys()) == set(LANGUAGE_NAMES.keys())


def test_every_language_has_all_dimensions():
    for lang, ctx in LANGUAGE_CULTURAL_CONTEXT.items():
        assert set(ctx.keys()) == REQUIRED_DIMENSIONS, f"{lang} missing/extra dimensions"


def test_every_language_has_nonempty_places_festivals_folklore_names():
    # deities is allowed to be empty (e.g. Punjabi, where Sikh tradition
    # doesn't fit a devotional-deity-genre framing) - the other five
    # dimensions should always have real content.
    for lang, ctx in LANGUAGE_CULTURAL_CONTEXT.items():
        for dim in ("places", "festivals", "folklore", "names", "register_note"):
            assert ctx[dim], f"{lang}.{dim} is empty"


def test_cultural_clause_nonempty_for_known_language():
    clause = _cultural_clause("tam")
    assert clause
    assert "Madurai" in clause or "Meenakshi" in clause


def test_cultural_clause_includes_all_dimensions_when_present():
    clause = _cultural_clause("tam")
    assert "Pongal" in clause  # festivals
    assert "Yakshi" in clause  # folklore
    assert "Murugan" in clause  # deities
    assert "Karthik" in clause  # names


def test_cultural_clause_omits_empty_deities():
    # Punjabi's deities field is deliberately empty - shouldn't produce a
    # dangling/empty deities mention in the prompt.
    clause = _cultural_clause("pan")
    assert "devotional figures/sites, specifically for devotional content" not in clause


def test_cultural_clause_can_exclude_names():
    with_names = _cultural_clause("hin", include_names=True)
    without_names = _cultural_clause("hin", include_names=False)
    assert "Rahul" in with_names
    assert "Rahul" not in without_names


def test_cultural_clause_empty_for_unknown_language():
    assert _cultural_clause("xx") == ""


def test_cultural_clause_is_soft_not_mandatory():
    # The clause must explicitly avoid forcing every story into real
    # touchstones - abstract/fantastical stories shouldn't be reshaped.
    clause = _cultural_clause("hin")
    assert "don't force" in clause.lower() or "shouldn't be" in clause.lower()


def test_cultural_clause_includes_festival_and_register_hints():
    clause = _cultural_clause("ben")
    assert "Durga Puja" in clause
    assert "apni" in clause or "tumi" in clause


def test_generate_continuation_includes_ancestor_summaries(monkeypatch):
    captured = {}

    def _fake_chat_completion(system_prompt, user_content, **kwargs):
        captured["user_content"] = user_content
        return "the continuation"

    monkeypatch.setattr(story_gen, "_chat_completion", _fake_chat_completion)
    result = story_gen.generate_continuation(
        "prior story text", "a door opens", "hin",
        ancestor_summaries=["earlier decision one", "earlier decision two"],
    )
    assert result == "the continuation"
    assert "earlier decision one" in captured["user_content"]
    assert "earlier decision two" in captured["user_content"]
    assert "prior story text" in captured["user_content"]


def test_generate_continuation_without_ancestor_summaries_omits_lineage_note(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        story_gen, "_chat_completion",
        lambda sp, uc, **k: captured.setdefault("uc", uc) or "x",
    )
    story_gen.generate_continuation("prior text", "change", "hin")
    assert "EARLIER IN THIS STORY" not in captured["uc"]


def test_adapt_story_language_targets_correct_language_names(monkeypatch):
    captured = {}

    def _fake_chat_completion(system_prompt, user_content, **kwargs):
        captured["system_prompt"] = system_prompt
        captured["user_content"] = user_content
        return "தமிழில் மொழிபெயர்க்கப்பட்ட கதை"

    monkeypatch.setattr(story_gen, "_chat_completion", _fake_chat_completion)
    result = story_gen.adapt_story_language("मूल हिंदी कहानी", "hin", "tam")
    assert result == "தமிழில் மொழிபெயர்க்கப்பட்ட கதை"
    assert "Hindi" in captured["system_prompt"]
    assert "Tamil" in captured["system_prompt"]
    assert captured["user_content"] == "मूल हिंदी कहानी"
