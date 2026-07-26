from story_gen import LANGUAGE_CULTURAL_CONTEXT, LANGUAGE_NAMES, _cultural_clause

REQUIRED_DIMENSIONS = {"places", "festivals", "folklore", "deities", "names"}


def test_every_language_has_cultural_context():
    assert set(LANGUAGE_CULTURAL_CONTEXT.keys()) == set(LANGUAGE_NAMES.keys())


def test_every_language_has_all_dimensions():
    for lang, ctx in LANGUAGE_CULTURAL_CONTEXT.items():
        assert set(ctx.keys()) == REQUIRED_DIMENSIONS, f"{lang} missing/extra dimensions"


def test_every_language_has_nonempty_places_festivals_folklore_names():
    # deities is allowed to be empty (e.g. Punjabi, where Sikh tradition
    # doesn't fit a devotional-deity-genre framing) - the other four
    # dimensions should always have real content.
    for lang, ctx in LANGUAGE_CULTURAL_CONTEXT.items():
        for dim in ("places", "festivals", "folklore", "names"):
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
