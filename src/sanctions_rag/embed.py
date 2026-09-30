"""Pluggable text encoder.

Default: TF-IDF over word and character n-grams, reduced with a truncated SVD (LSA).
Runs offline on numpy/scipy alone, which keeps the repo installable anywhere.
Set SANCTIONS_RAG_ENCODER=fastembed for a small neural encoder that runs on ONNX without
torch (default model BAAI/bge-small-en-v1.5, 67 MB, downloaded once), or
SANCTIONS_RAG_ENCODER=sentence-transformers for the torch stack. The retrieval and
evaluation code does not change.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

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


def _cached(model_name: str, texts: list[str], cache_dir: Path, compute) -> np.ndarray:
    """Document vectors are the slow part; cache them keyed by model and a hash of the texts."""
    h = hashlib.sha1(model_name.encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore")); h.update(b"\0")
    path = cache_dir / f"emb-{h.hexdigest()[:16]}.npy"
    if path.exists():
        return np.load(path)
    V = compute(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, V)
    return V


class SentenceTransformerEncoder:  # pragma: no cover - optional heavy dependency
    """Any sentence-transformers model from Hugging Face, e.g. voyageai/voyage-4-nano.

    Uses the model's own query/document prompts when it defines them (encode_query /
    encode_document). Models that ship custom code need SANCTIONS_RAG_TRUST_REMOTE_CODE=1,
    an explicit opt-in because that code runs locally. SANCTIONS_RAG_EMBED_DIM truncates
    Matryoshka models to fewer dimensions; documents are cut at SANCTIONS_RAG_EMBED_MAX_TOKENS
    (default 256: names and aliases come first in the search text).
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model: str | None = None, cache_dir: str = "data/cache") -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model or os.environ.get("SANCTIONS_RAG_EMBED_MODEL", self.DEFAULT_MODEL)
        dim = os.environ.get("SANCTIONS_RAG_EMBED_DIM")
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            self.model_name, device=device,
            trust_remote_code=os.environ.get("SANCTIONS_RAG_TRUST_REMOTE_CODE") == "1",
            truncate_dim=int(dim) if dim else None)
        self.model.max_seq_length = int(os.environ.get("SANCTIONS_RAG_EMBED_MAX_TOKENS", "256"))
        self.cache_dir = Path(cache_dir)

    def fit(self, texts: list[str]):
        return self

    def _run(self, texts: list[str], kind: str) -> np.ndarray:
        fn = getattr(self.model, f"encode_{kind}", None) or self.model.encode
        V = fn(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=len(texts) > 1000)
        return np.asarray(V, dtype=np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Documents (the corpus)."""
        if len(texts) < 64:
            return self._run(texts, "document")
        key = f"{self.model_name}|{self.model.truncate_dim}|{self.model.max_seq_length}"
        return _cached(key, texts, self.cache_dir, lambda t: self._run(t, "document"))

    def encode_query(self, texts: list[str]) -> np.ndarray:
        return self._run(texts, "query")

    @property
    def info(self) -> EncoderInfo:
        dim = self.model.truncate_dim or self.model.get_sentence_embedding_dimension()
        return EncoderInfo(self.model_name, int(dim))


class FastEmbedEncoder:
    """Neural sentence embeddings via fastembed (ONNX runtime, no torch).

    Encoding the corpus is the slow part (about 14 minutes on a laptop CPU), so document
    vectors are cached on disk, keyed by model and a hash of the texts.
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model: str | None = None, cache_dir: str = "data/cache") -> None:
        from fastembed import TextEmbedding

        self.model_name = model or os.environ.get("SANCTIONS_RAG_EMBED_MODEL", self.DEFAULT_MODEL)
        self.model = TextEmbedding(model_name=self.model_name)
        self.cache_dir = Path(cache_dir)
        self._dim = 0

    def fit(self, texts: list[str]) -> FastEmbedEncoder:
        return self

    def _embed(self, texts: list[str]) -> np.ndarray:
        V = np.asarray(list(self.model.embed(texts, batch_size=256)), dtype=np.float32)
        self._dim = V.shape[1]
        return V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)

    def encode(self, texts: list[str]) -> np.ndarray:
        if len(texts) < 64:
            return self._embed(texts)
        V = _cached(self.model_name, texts, self.cache_dir, self._embed)
        self._dim = V.shape[1]
        return V

    @property
    def info(self) -> EncoderInfo:
        return EncoderInfo(self.model_name, self._dim)


def _l2(X):
    from scipy import sparse

    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    inv = 1.0 / np.clip(norms, 1e-9, None)
    return sparse.diags(inv.astype(np.float32)) @ X


def get_encoder():
    kind = os.environ.get("SANCTIONS_RAG_ENCODER", "lsa")
    if kind == "fastembed":
        return FastEmbedEncoder()
    if kind == "sentence-transformers":
        return SentenceTransformerEncoder()
    return LSAEncoder()
