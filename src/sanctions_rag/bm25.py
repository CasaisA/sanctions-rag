"""Okapi BM25 over the entity corpus. Pure standard library, no index server."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from .text import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.avgdl = 0.0

    def index(self, docs: list[tuple[str, str]]) -> "BM25":
        """docs: list of (entity_id, searchable_text)."""
        for doc_id, text in docs:
            toks = tokenize(text)
            tf = Counter(toks)
            i = len(self.doc_ids)
            self.doc_ids.append(doc_id)
            self.doc_len.append(len(toks))
            self.tf.append(tf)
            for term in tf:
                self.df[term] += 1
                self.postings[term].append(i)
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        return self

    def _idf(self, term: str) -> float:
        n, df = len(self.doc_ids), self.df.get(term, 0)
        # BM25+ style floor keeps very common terms from going negative.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        q = tokenize(query)
        if not q:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(q):
            if term not in self.postings:
                continue
            idf = self._idf(term)
            for i in self.postings[term]:
                tf = self.tf[i][term]
                dl = self.doc_len[i] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.doc_ids[i], s) for i, s in top]
