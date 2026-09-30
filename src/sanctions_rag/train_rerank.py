"""Fit the reranker's feature weights on training queries, never on the test set.

Training queries are generated like the evaluation set but from entities that do not
appear in it, so the reported metrics stay out-of-sample. The objective is listwise:
for each query, a softmax over its candidate pool, maximising the probability of the
relevant entity (ListNet-style). Five weights, L-BFGS, a few seconds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .bm25 import BM25
from .build_eval import build
from .embed import get_encoder
from .hybrid import HybridRetriever
from .namematch import NameMatcher
from .rerank import FEATURES, WEIGHTS_PATH, FeatureReranker
from .store import Store
from .vector import VectorIndex


def fit(groups: list[tuple[np.ndarray, int]], l2: float = 1e-3) -> np.ndarray:
    """groups: (features [n_candidates, n_features], index of the relevant candidate)."""
    def loss(w):
        total, grad = 0.0, np.zeros_like(w)
        for X, y in groups:
            z = X @ w
            z = z - z.max()
            p = np.exp(z) / np.exp(z).sum()
            total -= np.log(p[y] + 1e-12)
            grad -= X[y] - p @ X
        n = len(groups)
        return total / n + l2 * w @ w, grad / n + 2 * l2 * w

    res = minimize(loss, np.ones(len(FEATURES)), jac=True, method="L-BFGS-B")
    return res.x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sanctions.db")
    ap.add_argument("--test-queries", default="data/eval/queries.jsonl")
    ap.add_argument("--train-out", default="data/eval/train.jsonl")
    ap.add_argument("--per-family", type=int, default=150)
    ap.add_argument("--pool", type=int, default=100)
    args = ap.parse_args()

    store = Store(args.db)
    test = [json.loads(line) for line in Path(args.test_queries).read_text().splitlines() if line.strip()]
    held_out = {r for q in test for r in q["relevant"]}
    train = build(store, args.per_family, seed=7, exclude=held_out)
    Path(args.train_out).write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in train))

    docs = [(e.id, e.search_text) for e in store.iter_entities()]
    names = NameMatcher().index(store.iter_entities())
    hybrid = HybridRetriever(BM25().index(docs), VectorIndex(get_encoder()).index(docs), names=names)
    feat = FeatureReranker(names)

    groups, missed = [], 0
    for q in train:
        pool = hybrid.search(q["query"], k=args.pool, pool=args.pool)
        ids = [d for d, _ in pool]
        rel = next((i for i, d in enumerate(ids) if d in q["relevant"]), None)
        if rel is None:
            missed += 1  # nothing to learn from: the retrievers never surfaced it
            continue
        X = np.array([feat.features(q["query"], store.get(d), r) for r, d in enumerate(ids)])
        groups.append((X, rel))

    w = fit(groups)
    w = w / np.abs(w).sum()  # only the ordering matters; normalise for readability
    out = {"features": list(FEATURES), "weights": dict(zip(FEATURES, map(float, np.round(w, 4)))),
           "train_queries": len(train), "used": len(groups), "not_in_pool": missed,
           "held_out_entities": len(held_out)}
    WEIGHTS_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
