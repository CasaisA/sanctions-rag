"""Reciprocal rank fusion of lexical and dense results.

RRF is used instead of score interpolation because BM25 and cosine live on different
scales and RRF needs no per-corpus tuning.
"""
from __future__ import annotations


def rrf(runs: list[list[tuple[str, float]]], k: int = 60, top: int = 50,
        weights: list[float] | None = None) -> list[tuple[str, float]]:
    weights = weights or [1.0] * len(runs)
    fused: dict[str, float] = {}
    for run, w in zip(runs, weights):
        for rank, (doc_id, _score) in enumerate(run, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])[:top]


class HybridRetriever:
    def __init__(self, bm25, vectors, weights=(1.0, 1.0), rrf_k: int = 60) -> None:
        self.bm25, self.vectors = bm25, vectors
        self.weights, self.rrf_k = list(weights), rrf_k

    def search(self, query: str, k: int = 50, pool: int = 100) -> list[tuple[str, float]]:
        lex = self.bm25.search(query, k=pool)
        dense = self.vectors.search(query, k=pool)
        return rrf([lex, dense], k=self.rrf_k, top=k, weights=self.weights)
