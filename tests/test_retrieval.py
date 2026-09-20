"""Unit tests that do not need the full corpus: build a tiny store in memory."""
from __future__ import annotations

import pytest

from sanctions_rag.bm25 import BM25
from sanctions_rag.hybrid import rrf
from sanctions_rag.store import Entity, Relation, Store
from sanctions_rag.text import fold, ngrams, tokenize

DOCS = [
    ("e1", "Gazprombank Joint Stock Company ru sanction bank"),
    ("e2", "GPB INTERNATIONAL SA lu sanction subsidiary"),
    ("e3", "Yahia Abdul Aziz AL-ABADSAH ps sanction person 1958"),
    ("e4", "Central Bank of the Russian Federation ru sanction"),
]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    s.upsert_entities([Entity(i, "Organization", t.split(" ru ")[0], ["ru"], ["sanction"], ["test"], {}, t)
                       for i, t in DOCS])
    s.upsert_relations([Relation("r1", "Ownership", "e1", "e2", "100%")])
    return s


def test_fold_strips_accents_and_case():
    assert fold("Müller  ÁÑO") == "muller ano"


def test_tokenize_and_ngrams():
    assert tokenize("AL-ABADSAH, 1958") == ["al", "abadsah", "1958"]
    assert "gaz" in ngrams("Gazprom", 3)


def test_bm25_ranks_exact_name_first():
    bm = BM25().index(DOCS)
    top = bm.search("Gazprombank", k=3)
    assert top[0][0] == "e1"


def test_bm25_empty_query_returns_nothing():
    assert BM25().index(DOCS).search("   ") == []


def test_rrf_rewards_agreement_between_runs():
    run_a = [("x", 9.0), ("y", 8.0)]
    run_b = [("y", 0.9), ("x", 0.1)]
    fused = dict(rrf([run_a, run_b]))
    # y is 2nd and 1st; x is 1st and 2nd -> equal fused score, both above a singleton
    assert pytest.approx(fused["x"]) == pytest.approx(fused["y"])
    assert set(fused) == {"x", "y"}


def test_store_roundtrip_and_neighbours(store):
    assert store.counts() == {"entities": 4, "relations": 1}
    assert store.get("e1").caption.startswith("Gazprombank")
    nb = store.neighbours("e1")
    assert ("e2", "Ownership", "out") in nb
    assert store.get("missing") is None
