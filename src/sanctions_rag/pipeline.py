"""Assembles the retrieval stack once and hands it to the API, CLI and agent."""
from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25
from .embed import get_encoder
from .graph import GraphExpander
from .hybrid import HybridRetriever
from .rerank import get_reranker
from .store import Store
from .vector import VectorIndex


@dataclass
class Pipeline:
    store: Store
    bm25: BM25
    vectors: VectorIndex
    hybrid: HybridRetriever
    reranker: object
    graph: GraphExpander

    @classmethod
    def build(cls, db: str = "data/sanctions.db", dsn: str | None = None) -> "Pipeline":
        store = Store(db, dsn)
        docs = [(e.id, e.search_text) for e in store.iter_entities()]
        bm25 = BM25().index(docs)
        vectors = VectorIndex(get_encoder()).index(docs)
        hybrid = HybridRetriever(bm25, vectors)
        return cls(store, bm25, vectors, hybrid, get_reranker(), GraphExpander(store))

    def search(self, query: str, k: int = 10, pool: int = 100,
               expand: bool = False, hops: int = 1) -> list[tuple[str, float]]:
        hits = self.reranker.rerank(query, self.hybrid.search(query, k=pool, pool=pool), self.store, k=k)
        if expand:
            hits = self.graph.expand(hits, hops=hops)
        return hits
