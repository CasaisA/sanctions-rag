"""Build labelled queries from the corpus itself.

Relevance judgements come from the data rather than from human annotation: each query
is derived from one entity, and that entity is the single relevant result. Four
families stress different failure modes:

  alias       query is an alias the entity is also known by  -> name variation
  partial     surname plus country                           -> low-specificity queries
  attribute   name plus birth year or registration number    -> mixed text and structured
  noisy       name with a character dropped and two swapped  -> transliteration/typos

This is weaker than editorial judgements, and the README says so. It is reproducible,
it costs nothing, and it is enough to rank retrieval strategies against each other.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .store import Store
from .text import fold


def _noise(name: str, rng: random.Random) -> str:
    chars = list(name)
    if len(chars) > 6:
        del chars[rng.randrange(1, len(chars) - 1)]
    for _ in range(2):
        if len(chars) > 4:
            i = rng.randrange(0, len(chars) - 1)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def build(store: Store, per_family: int = 75, seed: int = 20260920) -> list[dict]:
    rng = random.Random(seed)
    ents = [e for e in store.iter_entities() if e.schema in ("Person", "Organization", "Company")]
    rng.shuffle(ents)
    out: list[dict] = []

    def add(family: str, query: str, ent) -> None:
        q = query.strip()
        if len(q) >= 4:
            out.append({"id": f"{family}-{len(out):04d}", "family": family,
                        "query": q, "relevant": [ent.id], "entity_caption": ent.caption})

    pool = iter(ents)
    for family in ("alias", "partial", "attribute", "noisy"):
        made = 0
        for ent in pool:
            props = ent.properties or {}
            name = ent.caption
            if family == "alias":
                aliases = [a for a in (props.get("alias") or []) if fold(a) != fold(name)]
                if not aliases:
                    continue
                add(family, str(rng.choice(aliases)), ent)
            elif family == "partial":
                toks = name.split()
                if len(toks) < 2 or not ent.countries:
                    continue
                add(family, f"{toks[-1]} {ent.countries[0]}", ent)
            elif family == "attribute":
                extra = (props.get("birthDate") or props.get("registrationNumber")
                         or props.get("idNumber") or props.get("incorporationDate") or [])
                if not extra:
                    continue
                add(family, f"{name} {extra[0]}", ent)
            else:
                if len(name) < 8:
                    continue
                add(family, _noise(name, rng), ent)
            made += 1
            if made >= per_family:
                break
        pool = iter(ents)  # families may exhaust different subsets
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sanctions.db")
    ap.add_argument("--out", default="data/eval/queries.jsonl")
    ap.add_argument("--per-family", type=int, default=75)
    args = ap.parse_args()
    rows = build(Store(args.db), args.per_family)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} queries -> {args.out}")


if __name__ == "__main__":
    main()
