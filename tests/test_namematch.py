"""Normalisation and the brute-force matcher on the cases embeddings get wrong."""
from __future__ import annotations

from sanctions_rag.namematch import NameMatcher, normalise
from sanctions_rag.store import Entity


def _matcher():
    ents = [
        Entity("gpb", "Company", "Gazprombank Joint Stock Company", properties={"alias": ["Bank GPB JSC"]}),
        Entity("gpn", "Company", "Gazprom Neft PJSC"),
        Entity("pe", "Company", "LIMITED LIABILITY COMPANY PROMELEKTRO ENGINEERING",
               properties={"alias": ["Promelektro Engineering OOO"]}),
        Entity("js", "Person", "John Smith"),
    ]
    return NameMatcher().index(ents)


def test_normalise_strips_legal_forms_and_transliterates():
    assert normalise("Gazprombank Joint Stock Company") == "gazprombank"
    assert normalise("Газпромбанк") == "gazprombank"
    assert normalise("OOO Promelektro-Engineering") == "promelektro engineering"


def test_normalise_keeps_a_name_made_only_of_legal_words():
    assert normalise("The Company") == "the company"


def test_typo_finds_the_right_entity_not_the_related_one():
    top = _matcher().match("Gazpronbank", k=2)
    assert top[0].entity_id == "gpb" and top[0].score > top[1].score + 10


def test_cyrillic_and_spelling_variants():
    m = _matcher()
    assert m.match("Газпромбанк", k=1)[0].entity_id == "gpb"
    assert m.match("Promelectro Engineering", k=1)[0].entity_id == "pe"


def test_word_order_and_explanation():
    top = _matcher().match("Smith John", k=1)[0]
    assert top.entity_id == "js" and top.measures["token_sort"] == 100.0 and top.matched_name == "John Smith"


def test_one_result_per_entity_and_min_score():
    m = _matcher()
    ids = [x.entity_id for x in m.match("Gazprombank", k=10)]
    assert len(ids) == len(set(ids))
    assert all(x.score >= 90 for x in m.match("Gazprombank", k=10, min_score=90))
