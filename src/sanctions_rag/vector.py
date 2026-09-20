"""Brute-force cosine index. At 20k entities an ANN index buys nothing but a dependency."""
from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self, encoder) -> None:
        self.encoder = encoder
        self.doc_ids: list[str] = []
        self.matrix: np.ndarray | None = None

    def index(self, docs: list[tuple[str, str]]) -> "VectorIndex":
        self.doc_ids = [d for d, _ in docs]
        texts = [t for _, t in docs]
        self.encoder.fit(texts)
        self.matrix = self.encoder.encode(texts)
        return self

    def search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        assert self.matrix is not None, "index() first"
        q = self.encoder.encode([query])[0]
        sims = self.matrix @ q
        if k >= len(sims):
            order = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, k)[:k]
            order = part[np.argsort(-sims[part])]
        return [(self.doc_ids[i], float(sims[i])) for i in order[:k]]
