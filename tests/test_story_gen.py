from story_gen import LANGUAGE_CULTURAL_CONTEXT, LANGUAGE_NAMES, _setting_clause


def test_every_language_has_cultural_context():
    assert set(LANGUAGE_CULTURAL_CONTEXT.keys()) == set(LANGUAGE_NAMES.keys())


def test_setting_clause_nonempty_for_known_language():
    clause = _setting_clause("tam")
    assert clause
    assert "Madurai" in clause or "Meenakshi" in clause


def test_setting_clause_empty_for_unknown_language():
    assert _setting_clause("xx") == ""


def test_setting_clause_is_soft_not_mandatory():
    # The clause must explicitly avoid forcing every story into a real
    # place - abstract/fantastical stories shouldn't be relocated.
    clause = _setting_clause("hin")
    assert "don't force" in clause.lower() or "shouldn't be" in clause.lower()
