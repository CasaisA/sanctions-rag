"""Second-stage reranking.

Default is a transparent feature-based scorer: for entity retrieval, name-level
evidence (exact token overlap, coverage of the query, character-trigram similarity)
beats a generic bi-encoder, and every score can be explained to an analyst.
A cross-encoder can be dropped in via SANCTIONS_RAG_RERANKER=cross-encoder.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .text import ngrams, tokenize

FEATURES = ("coverage", "jaccard", "name_similarity", "topic", "retriever_prior")
WEIGHTS_PATH = Path("data/eval/reranker_weights.json")


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


class FeatureReranker:
    name = "feature-reranker"

    # Hand-set defaults. train_rerank.py fits weights on training queries drawn from
    # entities outside the evaluation set; those are used when the file exists.
    DEFAULT_WEIGHTS = (0.40, 0.15, 0.25, 0.05, 0.15)

    def __init__(self, names=None, weights=None) -> None:
        # With a name matcher, the name-similarity feature is its best score over every
        # listed name and alias; without one, trigram overlap with the caption.
        self.names = names
        self.weights = tuple(weights) if weights is not None else self.DEFAULT_WEIGHTS

    def features(self, query: str, entity, retriever_rank: int) -> list[float]:
        q_tok = set(tokenize(query))
        d_tok = set(tokenize(entity.search_text))
        cover = len(q_tok & d_tok) / len(q_tok) if q_tok else 0.0
        jacc = _jaccard(q_tok, d_tok)
        if self.names is not None:
            tri = self.names.score_entity(query, entity.id) / 100
        else:
            tri = _jaccard(set(ngrams(query, 3)), set(ngrams(entity.caption, 3)))
        topic = 1.0 if any(t in ("sanction", "crime", "role.pep") for t in entity.topics) else 0.0
        prior = 1.0 / (1.0 + retriever_rank)
        return [cover, jacc, tri, topic, prior]

    def score(self, query: str, entity, retriever_rank: int) -> float:
        return sum(w * f for w, f in zip(self.weights, self.features(query, entity, retriever_rank)))

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


def load_weights(path: Path = WEIGHTS_PATH) -> list[float] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return [data["weights"][f] for f in FEATURES]


def get_reranker(names=None, learned: bool = False):
    if os.environ.get("SANCTIONS_RAG_RERANKER") == "cross-encoder":
        return CrossEncoderReranker()
    return FeatureReranker(names, load_weights() if learned else None)
