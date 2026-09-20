"""Second-stage reranking.

Default is a transparent feature-based scorer: for entity retrieval, name-level
evidence (exact token overlap, coverage of the query, character-trigram similarity)
beats a generic bi-encoder, and every score can be explained to an analyst.
A cross-encoder can be dropped in via SANCTIONS_RAG_RERANKER=cross-encoder.
"""
from __future__ import annotations

import os

from .text import ngrams, tokenize


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


class FeatureReranker:
    name = "feature-reranker"

    # Weights chosen on the dev split; see notes in README.
    W_COVER, W_JACC, W_TRI, W_TOPIC, W_RETR = 0.40, 0.15, 0.25, 0.05, 0.15

    def score(self, query: str, entity, retriever_rank: int) -> float:
        q_tok = set(tokenize(query))
        d_tok = set(tokenize(entity.search_text))
        cover = len(q_tok & d_tok) / len(q_tok) if q_tok else 0.0
        jacc = _jaccard(q_tok, d_tok)
        tri = _jaccard(set(ngrams(query, 3)), set(ngrams(entity.caption, 3)))
        topic = 1.0 if any(t in ("sanction", "crime", "role.pep") for t in entity.topics) else 0.0
        prior = 1.0 / (1.0 + retriever_rank)
        return (self.W_COVER * cover + self.W_JACC * jacc + self.W_TRI * tri
                + self.W_TOPIC * topic + self.W_RETR * prior)

    def rerank(self, query: str, candidates: list[tuple[str, float]], store, k: int = 10):
        scored = []
        for rank, (doc_id, _s) in enumerate(candidates):
            ent = store.get(doc_id)
            if ent is None:
                continue
            scored.append((doc_id, self.score(query, ent, rank)))
        scored.sort(key=lambda kv: -kv[1])
        return scored[:k]


class CrossEncoderReranker:  # pragma: no cover - optional heavy dependency
    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)

    def rerank(self, query: str, candidates: list[tuple[str, float]], store, k: int = 10):
        ents = [(d, store.get(d)) for d, _ in candidates]
        ents = [(d, e) for d, e in ents if e is not None]
        pairs = [(query, e.search_text[:512]) for _, e in ents]
        scores = self.model.predict(pairs)
        out = sorted(((d, float(s)) for (d, _), s in zip(ents, scores)), key=lambda kv: -kv[1])
        return out[:k]


def get_reranker():
    if os.environ.get("SANCTIONS_RAG_RERANKER") == "cross-encoder":
        return CrossEncoderReranker()
    return FeatureReranker()
