"""Prompt building, reordering and the decision cache; Claude itself is not called."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from sanctions_rag.adjudicate import Adjudicator, Decision, apply, build_prompt


def _ent(i, cap, **props):
    return SimpleNamespace(id=i, caption=cap, schema="Company", countries=["ru"], datasets=["us_ofac_sdn"],
                           properties=props)


def test_apply_promotes_match_and_keeps_the_rest_in_order():
    assert apply(["a", "b", "c"], Decision("c", "high", "")) == ["c", "a", "b"]


def test_apply_leaves_order_alone_on_none_or_unknown():
    assert apply(["a", "b"], Decision(None, "low", "")) == ["a", "b"]
    assert apply(["a", "b"], Decision("zz", "high", "")) == ["a", "b"]


def test_prompt_numbers_candidates_and_shows_identifiers():
    p = build_prompt("Gazpronbank", [_ent("x", "Gazprombank", registrationNumber=["123"], alias=["GPB"])])
    assert "QUERY: Gazpronbank" in p and "[1] Gazprombank" in p
    assert "registrationNumber=123" in p and "aliases=GPB" in p


def test_cached_decision_skips_the_model(tmp_path):
    store = SimpleNamespace(get=lambda i: _ent(i, "Alpha"))
    adj = Adjudicator(store, model="m", cache=tmp_path / "c.jsonl")
    key = adj._key(build_prompt("alpha", [store.get("a1")]))
    (tmp_path / "c.jsonl").write_text(json.dumps({"key": key, "query": "alpha", "match_id": "a1",
                                                  "confidence": "high", "reason": "same"}) + "\n")
    d = asyncio.run(Adjudicator(store, model="m", cache=tmp_path / "c.jsonl").judge("alpha", ["a1"]))
    assert d.cached and d.match_id == "a1"
