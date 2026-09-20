"""Text normalisation shared by every retriever, so comparisons are apples to apples."""
from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[a-z0-9]+")

# Latin-script transliteration is deliberately crude: sanctions names arrive in many
# scripts and a full transliteration library would be a dependency for little gain here.
def fold(s: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.lower()).strip()


def tokenize(s: str) -> list[str]:
    return _TOKEN.findall(fold(s))


def ngrams(s: str, n: int = 3) -> list[str]:
    """Character n-grams over the folded string; catches transliteration variants."""
    f = fold(s).replace(" ", "_")
    return [f[i : i + n] for i in range(max(0, len(f) - n + 1))]
