"""Claude as the final judge of identity on a retrieval shortlist.

Retrieval (BM25, vectors, the name matcher, the reranker) finds candidates that look
*similar*. Screening needs the one that is *the same* entity, or none. That is a
judgement over names, aliases, dates, identifiers and countries at once, which is where
a language model is strong and a weighted sum of string features is not.

The adjudicator sees the query and the top-k candidates with their attributes, and returns
the number of the candidate it judges to be the same entity (0 for none), a confidence and
a one-line reason. It only reorders: the chosen candidate moves to rank 1, the rest keep
their retrieval order. Because a call takes seconds and costs money, `margin` gates it to
ambiguous shortlists, where the top two reranker scores are close.

Decisions are cached on disk by (model, prompt), so an evaluation can be re-scored under
different gates without calling Claude again.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5-5"
CACHE = Path("data/eval/adjudications.jsonl")
SHOWN = ("birthDate", "birthPlace", "nationality", "incorporationDate", "jurisdiction",
         "registrationNumber", "idNumber", "taxNumber", "passportNumber", "imoNumber", "position")

SYSTEM = """You resolve identity for sanctions screening. You get a query, which may be a \
name, a misspelled or transliterated name, an alias, a surname with a country code, or a \
name followed by a date or an identifier, and a numbered shortlist of records from \
sanctions lists.

Decide which record is the same real-world person, company, vessel or aircraft the query \
refers to. Judge identity, not topical similarity: a subsidiary, a parent, a relative or a \
company with a similar name is a different entity. Use every signal: spelling and \
transliteration variants, aliases, word order, legal-form words (LLC, OOO, JSC), dates, \
registration and ID numbers, countries. A date or identifier in the query that contradicts \
a record rules it out.

Answer with the candidate number, or 0 if none of them is the same entity. Confidence is \
high when the evidence is specific (an identifier, a date, a distinctive full name), low \
when several candidates fit equally well."""

SCHEMA = {
    "type": "object",
    "properties": {
        "match": {"type": "integer", "minimum": 0},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
    "required": ["match", "confidence", "reason"],
}


@dataclass
class Decision:
    match_id: str | None
    confidence: str
    reason: str
    cached: bool = False


def describe(e, idx: int) -> str:
    props = e.properties or {}
    aliases = [a for k in ("alias", "weakAlias", "previousName") for a in props.get(k, []) if a != e.caption]
    parts = [f"[{idx}] {e.caption}", f"type={e.schema}"]
    if e.countries:
        parts.append("countries=" + ",".join(e.countries[:4]))
    if aliases:
        parts.append("aliases=" + "; ".join(aliases[:6]))
    for k in SHOWN:
        if props.get(k):
            parts.append(f"{k}=" + "; ".join(str(v) for v in props[k][:3]))
    parts.append("lists=" + ",".join(e.datasets[:3]))
    return " | ".join(parts)


def build_prompt(query: str, entities) -> str:
    lines = [f"QUERY: {query}", "", "CANDIDATES (in retrieval order):"]
    lines += [describe(e, i) for i, e in enumerate(entities, start=1)]
    return "\n".join(lines)


class Adjudicator:
    def __init__(self, store, model: str | None = None, effort: str = "low",
                 cache: Path = CACHE, concurrency: int = 6) -> None:
        self.store = store
        self.model = model or os.environ.get("SANCTIONS_RAG_ADJUDICATOR_MODEL", DEFAULT_MODEL)
        self.effort, self.cache_path = effort, cache
        self.sem = asyncio.Semaphore(concurrency)
        self.cache: dict[str, dict] = {}
        if cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row

    def _key(self, prompt: str) -> str:
        return hashlib.sha1(f"{self.model}|{self.effort}|{SYSTEM}|{prompt}".encode()).hexdigest()

    async def judge(self, query: str, candidate_ids: list[str]) -> Decision:
        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage
        from claude_agent_sdk import query as sdk_query

        ents = [e for e in (self.store.get(i) for i in candidate_ids) if e is not None]
        if not ents:
            return Decision(None, "high", "no candidates")
        prompt = build_prompt(query, ents)
        key = self._key(prompt)
        if key in self.cache:
            row = self.cache[key]
            return Decision(row["match_id"], row["confidence"], row["reason"], cached=True)

        options = ClaudeAgentOptions(system_prompt=SYSTEM, tools=[], setting_sources=[],
                                     max_turns=3, model=self.model, effort=self.effort,
                                     output_format={"type": "json_schema", "schema": SCHEMA})
        out = None
        async with self.sem:
            async for msg in sdk_query(prompt=prompt, options=options):
                if isinstance(msg, ResultMessage) and msg.subtype == "success":
                    out = msg.structured_output
        if not out:
            return Decision(None, "low", "adjudicator returned no decision")
        n = int(out.get("match", 0))
        match_id = ents[n - 1].id if 1 <= n <= len(ents) else None
        row = {"key": key, "query": query, "match_id": match_id, "confidence": out.get("confidence", "low"),
               "reason": out.get("reason", "")}
        self.cache[key] = row
        with self.cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return Decision(match_id, row["confidence"], row["reason"])


def apply(ranked: list[str], decision: Decision) -> list[str]:
    """Move the adjudicated match to the top; a 'none' leaves retrieval order unchanged."""
    if decision.match_id is None or decision.match_id not in ranked:
        return ranked
    return [decision.match_id] + [d for d in ranked if d != decision.match_id]


def evaluate(db: str = "data/sanctions.db", queries_path: str = "data/eval/queries.jsonl",
             k: int = 10, margins=(0.02, 0.05, 0.1, 0.2, float("inf")), model: str | None = None) -> dict:
    """Score retrieval alone vs retrieval + Claude on the test set, for several gates."""
    from .pipeline import Pipeline

    p = Pipeline.build(db)
    adj = Adjudicator(p.store, model=model)
    queries = [json.loads(line) for line in Path(queries_path).read_text().splitlines() if line.strip()]
    shortlists = [p.search(q["query"], k=k) for q in queries]

    async def run():
        return await asyncio.gather(*(adj.judge(q["query"], [d for d, _ in sl])
                                      for q, sl in zip(queries, shortlists)))
    decisions = asyncio.run(run())

    def metrics(rankings):
        r1 = mrr = 0.0
        for q, ranked in zip(queries, rankings):
            rel = set(q["relevant"])
            r1 += ranked[:1] and ranked[0] in rel
            mrr += next((1 / i for i, d in enumerate(ranked[:10], 1) if d in rel), 0.0)
        n = len(queries)
        return {"recall@1": round(r1 / n, 3), "mrr@10": round(mrr / n, 3)}

    base = [[d for d, _ in sl] for sl in shortlists]
    out = {"model": adj.model, "queries": len(queries), "in_shortlist": 0,
           "retrieval_only": metrics(base), "gated": {}}
    out["in_shortlist"] = sum(bool(set(q["relevant"]) & set(b)) for q, b in zip(queries, base))
    for m in margins:
        ranks, calls = [], 0
        for sl, b, dec in zip(shortlists, base, decisions):
            gap = sl[0][1] - sl[1][1] if len(sl) > 1 else float("inf")
            if gap < m:
                calls += 1
                ranks.append(apply(b, dec))
            else:
                ranks.append(b)
        label = "always" if m == float("inf") else f"margin<{m}"
        out["gated"][label] = {**metrics(ranks), "claude_calls": calls}

    fixed = broke = abstain = wrong_pick = 0
    by_family: dict[str, list[int]] = {}
    for q, b, dec in zip(queries, base, decisions):
        rel = set(q["relevant"])
        before, after = b[0] in rel, apply(b, dec)[0] in rel
        fixed += after and not before
        broke += before and not after
        abstain += dec.match_id is None
        wrong_pick += dec.match_id is not None and dec.match_id not in rel
        fam = by_family.setdefault(q["family"], [0, 0, 0])
        fam[0] += 1; fam[1] += before; fam[2] += after
    out["always_detail"] = {"fixed": fixed, "broke": broke, "abstained": abstain, "picked_wrong": wrong_pick}
    out["by_family_recall@1"] = {f: {"before": round(v[1] / v[0], 2), "after": round(v[2] / v[0], 2)}
                                 for f, v in sorted(by_family.items())}
    conf = {}
    for q, dec in zip(queries, decisions):
        if dec.match_id is not None:
            c = conf.setdefault(dec.confidence, [0, 0])
            c[0] += 1; c[1] += dec.match_id in q["relevant"]
    out["precision_by_confidence"] = {c: f"{v[1]}/{v[0]}" for c, v in conf.items()}
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sanctions.db")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="data/eval/results-adjudicator.json")
    a = ap.parse_args()
    res = evaluate(a.db, model=a.model)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
