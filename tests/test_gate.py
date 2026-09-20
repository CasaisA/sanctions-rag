"""The quality gate is the thing standing between a bad ranking change and main."""
from __future__ import annotations

from sanctions_rag.gate import check
from sanctions_rag.observability import Metrics

BASE = {"overall": {"bm25": {"recall@10": 0.850}, "hybrid+rerank": {"recall@10": 0.907}}}


def _res(bm25: float, hybrid: float) -> dict:
    return {"overall": {"bm25": {"recall@10": bm25}, "hybrid+rerank": {"recall@10": hybrid}}}


def test_passes_when_metrics_hold():
    assert check(_res(0.850, 0.907), BASE) == []


def test_small_drop_is_inside_tolerance():
    assert check(_res(0.845, 0.900), BASE) == []


def test_real_regression_fails():
    fails = check(_res(0.850, 0.820), BASE)
    assert any("hybrid+rerank.recall@10" in f for f in fails)


def test_missing_system_fails():
    assert check({"overall": {"bm25": {"recall@10": 0.85}}}, BASE)


def test_best_system_must_stay_best():
    # hybrid inside tolerance but now below bm25 -> the ordering claim in the README breaks
    fails = check(_res(0.95, 0.90), BASE)
    assert any("no longer the best" in f for f in fails)


def test_metrics_render_prometheus_text():
    m = Metrics()
    m.count("http_request", "GET /search")
    m.observe("retrieve", 0.012)
    text = m.render()
    assert 'sanctions_rag_http_request_total{stage="GET /search"} 1' in text
    assert "sanctions_rag_retrieve_seconds_count 1" in text
    assert 'le="+Inf"' in text
