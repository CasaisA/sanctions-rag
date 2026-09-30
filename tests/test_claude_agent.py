"""The tool layer and the citation check are code; Claude is not needed to test them."""
from __future__ import annotations

from types import SimpleNamespace

from sanctions_rag.claude_agent import ToolBox, _count_hits, check_citations


class _Store:
    def __init__(self):
        self.ents = {i: SimpleNamespace(id=i, caption=c, schema="Company", countries=["ru"],
                                        topics=["sanction"], datasets=["us_ofac_sdn"],
                                        properties={"name": [c]}, search_text=c)
                     for i, c in [("A1", "Alpha Bank"), ("B2", "Beta Holding"), ("C3", "Gamma LLC")]}

    def get(self, i):
        return self.ents.get(i)

    def neighbours(self, i):
        return [("B2", "Ownership", "out")] if i == "A1" else []


def _box():
    store = _Store()
    pipe = SimpleNamespace(store=store,
                           search=lambda q, k=10: [("A1", 0.9), ("missing", 0.5)][:k],
                           graph=SimpleNamespace(path=lambda a, b, max_hops=3: [a, b]))
    return ToolBox(pipe)


def test_search_skips_unknown_ids_and_records_what_was_seen():
    box = _box()
    hits = box.search("alpha")
    assert [h["id"] for h in hits] == ["A1"]
    assert box.seen == {"A1"}


def test_search_clamps_k():
    assert len(_box().search("alpha", k=500)) == 1


def test_neighbours_and_path_extend_seen():
    box = _box()
    assert box.neighbours("A1")[0]["relation"] == "Ownership"
    box.path("A1", "C3")
    assert box.seen == {"A1", "B2", "C3"}


def test_unknown_entity_is_none():
    assert _box().entity("nope") is None


def test_citations_split_into_retrieved_and_unsupported():
    ok, bad = check_citations("Listed [NK-a1]; owner [Q4211] and [NK-a1], see [1].", {"NK-a1"})
    assert ok == ["NK-a1"] and bad == ["Q4211"]


def test_hit_counting():
    txt = lambda s: [{"type": "text", "text": s}]
    assert _count_hits(txt('[{"id": 1}, {"id": 2}]')) == 2
    assert _count_hits(txt('{"hits": [], "note": "x"}')) == 0
    assert _count_hits(txt('{"found": false}')) == 0
    assert _count_hits(txt("unknown entity id")) == 0
