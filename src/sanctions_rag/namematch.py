"""Brute-force name matching: every query against every listed name and alias.

No index and no embeddings. Each name is normalised the same way (accents folded,
Cyrillic transliterated, legal-form words such as LLC / OOO / JSC removed) and scored
against the query with three character-level measures. An entity's score is its best
name. At ~70k names a full scan is a few milliseconds with rapidfuzz, so there is nothing
to gain from blocking or an ANN index, and nothing can be missed because of one.

The measures, each 0-100:
  jaro_winkler  per-character agreement, weighted to the start of the string: robust to a
                dropped or swapped letter ("Gazpronbank" / "Gazprombank")
  token_sort    edit similarity after sorting words: robust to word order
                ("Smith John" / "John Smith")
  token_set     compares shared words against the rest: a query that is a clean subset of a
                longer listed name scores high ("Promelektro Engineering" / "LLC Promelektro
                Engineering Plant")

Every match reports which listed name it hit and the per-measure scores, so a reviewer
can see why it matched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from .text import fold

NAME_PROPS = ("name", "alias", "weakAlias", "previousName")

_CYR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюяіїєґ",
                ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o",
                 "p", "r", "s", "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e",
                 "yu", "ya", "i", "yi", "ye", "g"]))

# Legal forms and filler words that say what kind of company it is, not which one.
LEGAL = {
    "llc", "ltd", "limited", "inc", "incorporated", "co", "corp", "corporation", "company",
    "plc", "sa", "sas", "sarl", "srl", "spa", "bv", "nv", "gmbh", "ag", "kg", "ab", "as", "oy",
    "ooo", "oao", "zao", "pao", "ao", "jsc", "pjsc", "ojsc", "cjsc", "fzco", "fze", "fzc",
    "tov", "pte", "pty", "sdn", "bhd", "llp", "lp", "joint", "stock", "public", "open", "closed",
    "liability", "obshchestvo", "ogranichennoy", "ogranichennoi", "otvetstvennostyu",
    "aktsionernoe", "publichnoe", "zakrytoe", "otkrytoe", "the", "of", "and",
}

_NONWORD = re.compile(r"[^a-z0-9 ]+")


def translit(s: str) -> str:
    return "".join(_CYR.get(c, c) for c in s.lower())


def normalise(name: str) -> str:
    s = _NONWORD.sub(" ", fold(translit(name)))
    toks = [t for t in s.split() if t not in LEGAL]
    return " ".join(toks) or s.strip()


@dataclass
class Match:
    entity_id: str
    score: float
    matched_name: str
    measures: dict[str, float]


class NameMatcher:
    WEIGHTS: ClassVar[dict[str, float]] = {"jaro_winkler": 0.45, "token_sort": 0.30, "token_set": 0.25}

    def __init__(self) -> None:
        self.entity_ids: list[str] = []
        self.raw: list[str] = []
        self.norm: list[str] = []

    def index(self, entities) -> NameMatcher:
        self.by_entity: dict[str, list[int]] = {}
        for e in entities:
            names = [e.caption] + [n for k in NAME_PROPS for n in (e.properties or {}).get(k, [])]
            for n in dict.fromkeys(str(x) for x in names if x):
                self.by_entity.setdefault(e.id, []).append(len(self.raw))
                self.entity_ids.append(e.id)
                self.raw.append(n)
                self.norm.append(normalise(n))
        return self

    def _scores(self, q: str):
        import numpy as np
        from rapidfuzz import fuzz, process
        from rapidfuzz.distance import JaroWinkler

        jw = process.cdist([q], self.norm, scorer=JaroWinkler.normalized_similarity, workers=-1)[0] * 100
        ts = process.cdist([q], self.norm, scorer=fuzz.token_sort_ratio, workers=-1)[0]
        st = process.cdist([q], self.norm, scorer=fuzz.token_set_ratio, workers=-1)[0]
        w = self.WEIGHTS
        total = w["jaro_winkler"] * jw + w["token_sort"] * ts + w["token_set"] * st
        return np.asarray(total), jw, ts, st

    def match(self, query: str, k: int = 10, min_score: float = 0.0) -> list[Match]:
        q = normalise(query)
        if not q or not self.norm:
            return []
        total, jw, ts, st = self._scores(q)
        order = total.argsort()[::-1]
        best: dict[str, Match] = {}
        for i in order:
            if total[i] < min_score or len(best) >= k:
                break
            eid = self.entity_ids[i]
            if eid in best:  # an entity scores by its best-matching name
                continue
            best[eid] = Match(eid, round(float(total[i]), 1), self.raw[i],
                              {"jaro_winkler": round(float(jw[i]), 1), "token_sort": round(float(ts[i]), 1),
                               "token_set": round(float(st[i]), 1)})
        return list(best.values())

    def score_entity(self, query: str, entity_id: str) -> float:
        """Best name score (0-100) for one entity; used as a reranking feature."""
        from rapidfuzz import fuzz
        from rapidfuzz.distance import JaroWinkler

        q, w, best = normalise(query), self.WEIGHTS, 0.0
        for i in self.by_entity.get(entity_id, ()):
            n = self.norm[i]
            s = (w["jaro_winkler"] * JaroWinkler.normalized_similarity(q, n) * 100
                 + w["token_sort"] * fuzz.token_sort_ratio(q, n) + w["token_set"] * fuzz.token_set_ratio(q, n))
            best = max(best, s)
        return best

    def search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        """Same shape as the other retrievers, so it slots into fusion and evaluation."""
        return [(m.entity_id, m.score) for m in self.match(query, k=k)]
