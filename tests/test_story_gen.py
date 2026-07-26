import pytest

import db
import story_gen
from story_gen import LANGUAGE_CULTURAL_CONTEXT, LANGUAGE_NAMES, _ARC_CLAUSE, _cultural_clause, get_effective_cultural_context

REQUIRED_DIMENSIONS = {"places", "festivals", "folklore", "deities", "names", "register_note"}


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    # _cultural_clause/get_effective_cultural_context now read
    # db.get_culture_pack_override on every call - autouse so every test
    # in this file gets an isolated DB, never the real default path,
    # whether or not the individual test cares about overrides.
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(db, "_conn", None)
    db.init_db()
    yield db
    monkeypatch.setattr(db, "_conn", None)


def test_arc_clause_asks_for_a_real_turn_and_resolution():
    assert "turn" in _ARC_CLAUSE.lower()
    assert "resolution" in _ARC_CLAUSE.lower()


def test_arc_clause_rejects_flat_sequencing():
    assert "flat" in _ARC_CLAUSE.lower() or "sequence" in _ARC_CLAUSE.lower()


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


def test_culturally_adapt_text_stays_in_the_same_language(monkeypatch):
    captured = {}

    def _fake_chat_completion(system_prompt, user_content, **kwargs):
        captured["system_prompt"] = system_prompt
        captured["user_content"] = user_content
        return "revised text"

    monkeypatch.setattr(story_gen, "_chat_completion", _fake_chat_completion)
    result = story_gen.culturally_adapt_text("original text", "tam")
    assert result == "revised text"
    assert "Tamil" in captured["system_prompt"]
    assert captured["user_content"] == "original text"
    # Same-language framing, not adapt_story_language's "different language" one.
    assert "already written in Tamil" in captured["system_prompt"]


def test_culturally_adapt_text_includes_cultural_clause(monkeypatch, fresh_db):
    captured = {}
    monkeypatch.setattr(
        story_gen, "_chat_completion",
        lambda sp, uc, **k: captured.setdefault("sp", sp) or "x",
    )
    story_gen.culturally_adapt_text("some text", "tam")
    assert "Madurai" in captured["sp"] or "Meenakshi" in captured["sp"]


def test_get_effective_cultural_context_returns_default_without_override(fresh_db):
    ctx = get_effective_cultural_context("tam")
    assert ctx == LANGUAGE_CULTURAL_CONTEXT["tam"]


def test_get_effective_cultural_context_unknown_language_returns_none(fresh_db):
    assert get_effective_cultural_context("xx") is None


def test_get_effective_cultural_context_merges_override_over_default(fresh_db):
    db.save_culture_pack_override("tam", {"places": "an edited place"})
    ctx = get_effective_cultural_context("tam")
    assert ctx["places"] == "an edited place"
    # Untouched dimensions still fall through to the built-in default.
    assert ctx["festivals"] == LANGUAGE_CULTURAL_CONTEXT["tam"]["festivals"]
    assert ctx["names"] == LANGUAGE_CULTURAL_CONTEXT["tam"]["names"]


def test_cultural_clause_reflects_edited_places(fresh_db):
    db.save_culture_pack_override("tam", {"places": "a completely invented test landmark"})
    clause = _cultural_clause("tam")
    assert "a completely invented test landmark" in clause
    # Rameswaram only ever appears in the (now-overridden) places field -
    # unlike "Madurai", which also legitimately appears in the untouched
    # folklore field ("the Madurai Veeran legend") and shouldn't vanish.
    assert "Rameswaram" not in clause
