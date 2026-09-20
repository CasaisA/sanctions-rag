"""Pluggable text encoder.

Default: TF-IDF over word and character n-grams, reduced with a truncated SVD (LSA).
Runs offline on numpy/scipy alone, which keeps the repo installable anywhere.
Set SANCTIONS_RAG_ENCODER=sentence-transformers to use a dense neural encoder instead;
the retrieval and evaluation code does not change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .text import ngrams, tokenize


@dataclass
class EncoderInfo:
    name: str
    dim: int


class LSAEncoder:
    """TF-IDF (words + char 3-grams) -> truncated SVD -> L2-normalised vectors."""

    def __init__(self, dim: int = 256, min_df: int = 2) -> None:
        self.dim, self.min_df = dim, min_df
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.components: np.ndarray | None = None

    @staticmethod
    def _features(text: str) -> list[str]:
        return tokenize(text) + ngrams(text, 3)

    def _counts(self, texts: list[str]):
        from scipy import sparse

        rows, cols, vals = [], [], []
        for i, t in enumerate(texts):
            c: dict[int, int] = {}
            for f in self._features(t):
                j = self.vocab.get(f)
                if j is not None:
                    c[j] = c.get(j, 0) + 1
            for j, v in c.items():
                rows.append(i); cols.append(j); vals.append(v)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(texts), len(self.vocab)), dtype=np.float32)

    def fit(self, texts: list[str]) -> LSAEncoder:
        from scipy.sparse.linalg import svds

        df: dict[str, int] = {}
        for t in texts:
            for f in set(self._features(t)):
                df[f] = df.get(f, 0) + 1
        kept = [f for f, c in sorted(df.items()) if c >= self.min_df]
        self.vocab = {f: i for i, f in enumerate(kept)}
        n = len(texts)
        idf = np.zeros(len(self.vocab), dtype=np.float32)
        for f, i in self.vocab.items():
            idf[i] = np.log((1 + n) / (1 + df[f])) + 1.0
        self.idf = idf

        X = self._counts(texts)
        X = X.multiply(idf[None, :]).tocsr()
        X = _l2(X)
        k = min(self.dim, min(X.shape) - 1)
        # svds returns singular triplets in ascending order; flip to descending.
        _, _, vt = svds(X.astype(np.float32), k=k)
        self.components = np.asarray(vt[::-1], dtype=np.float32)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        assert self.idf is not None and self.components is not None, "fit() first"
        X = self._counts(texts).multiply(self.idf[None, :]).tocsr()
        X = _l2(X)
        V = np.asarray(X @ self.components.T, dtype=np.float32)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        return V / np.clip(norms, 1e-9, None)

    @property
    def info(self) -> EncoderInfo:
        return EncoderInfo("tfidf-svd(word+char3)", self.components.shape[0] if self.components is not None else 0)


class SentenceTransformerEncoder:  # pragma: no cover - optional heavy dependency
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.model = SentenceTransformer(model)

    def fit(self, texts: list[str]):
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    @property
    def info(self) -> EncoderInfo:
        return EncoderInfo(self.model_name, int(self.model.get_sentence_embedding_dimension()))


def _l2(X):
    from scipy import sparse

    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    inv = 1.0 / np.clip(norms, 1e-9, None)
    return sparse.diags(inv.astype(np.float32)) @ X


def get_encoder():
    if os.environ.get("SANCTIONS_RAG_ENCODER") == "sentence-transformers":
        return SentenceTransformerEncoder()
    return LSAEncoder()
