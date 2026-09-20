"""Investigation agent: plan, retrieve, expand, critique, repeat.

The loop is deliberately small and legible. What makes it an agent rather than a
single RAG call is that it decomposes the question, decides when the evidence is thin,
issues its own follow-up queries, and stops when a critic says the report is covered
or the budget is spent. Every step is recorded in a trace so a failure can be read
afterwards rather than guessed at.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .rag import Evidence, gather
from .rag import answer as rag_answer

STOPWORDS = {
    "who", "what", "which", "where", "when", "why", "how", "is", "are", "the", "a", "an",
    "of", "to", "and", "or", "in", "on", "for", "with", "by", "any", "there", "does", "do",
    "linked", "related", "connected", "about", "tell", "me", "show", "list", "find",
}


@dataclass
class Step:
    kind: str
    detail: str
    hits: int = 0


@dataclass
class Report:
    question: str
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    trace: list[Step] = field(default_factory=list)
    iterations: int = 0
    stopped_because: str = ""

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "report": self.text,
            "iterations": self.iterations,
            "stopped_because": self.stopped_because,
            "evidence": [{"id": e.entity_id, "caption": e.caption, "schema": e.schema,
                          "datasets": e.datasets, "score": round(e.score, 4)} for e in self.evidence],
            "trace": [{"kind": s.kind, "detail": s.detail, "hits": s.hits} for s in self.trace],
        }


def plan(question: str, max_subqueries: int = 4) -> list[str]:
    """Split a question into retrieval-sized pieces.

    Quoted spans and capitalised runs are treated as entity mentions and searched
    verbatim; the remaining content words become one catch-all query.
    """
    subs: list[str] = []
    subs += [m.strip() for m in re.findall(r'"([^"]+)"', question) if len(m.strip()) > 2]
    for m in re.findall(r"\b([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,3})", question):
        # A capitalised run may start with the sentence's first word ("Which Russian
        # banks..."), so trim stopwords from both ends before treating it as a mention.
        words = m.split()
        while words and words[0].lower() in STOPWORDS:
            words.pop(0)
        while words and words[-1].lower() in STOPWORDS:
            words.pop()
        cand = " ".join(words).strip()
        if len(cand) > 3 and cand.lower() not in STOPWORDS and cand not in subs:
            subs.append(cand)
    rest = [w for w in re.findall(r"[\w'-]+", question.lower()) if w not in STOPWORDS and len(w) > 2]
    if rest:
        subs.append(" ".join(rest[:10]))
    seen, out = set(), []
    for s in subs:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out[:max_subqueries]


def critique(report_text: str, evidence: list[Evidence], subqueries: list[str]) -> tuple[bool, str, list[str]]:
    """Return (is_good_enough, reason, follow_up_queries)."""
    if not evidence:
        return False, "no evidence retrieved", []
    covered = {e.caption.lower() for e in evidence}
    missing = [q for q in subqueries
               if not any(tok in cap for cap in covered for tok in q.lower().split() if len(tok) > 3)]
    if missing:
        return False, f"no evidence for: {', '.join(missing[:3])}", missing[:2]
    if len(evidence) < 3:
        return False, "thin evidence (<3 entities)", subqueries[:1]
    return True, "all sub-questions have supporting evidence", []


def investigate(pipeline, question: str, max_iterations: int = 3, k: int = 6,
                expand: bool = True) -> Report:
    trace: list[Step] = []
    subqueries = plan(question)
    trace.append(Step("plan", " | ".join(subqueries), len(subqueries)))

    evidence: dict[str, Evidence] = {}
    reason = ""
    queries = list(subqueries)
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        for q in queries:
            hits = pipeline.search(q, k=k, expand=expand)
            trace.append(Step("search", q, len(hits)))
            for ev in gather(q, hits, pipeline.store, limit=k):
                # keep the best score seen for an entity across sub-queries
                prev = evidence.get(ev.entity_id)
                if prev is None or ev.score > prev.score:
                    evidence[ev.entity_id] = ev

        ordered = sorted(evidence.values(), key=lambda e: -e.score)[:12]
        draft = rag_answer(question, [(e.entity_id, e.score) for e in ordered], pipeline.store, limit=12)
        ok, reason, follow_ups = critique(draft.text, ordered, subqueries)
        trace.append(Step("critique", reason, len(ordered)))
        if ok or not follow_ups:
            break
        queries = follow_ups
        trace.append(Step("replan", " | ".join(queries), len(queries)))

    ordered = sorted(evidence.values(), key=lambda e: -e.score)[:12]
    final = rag_answer(question, [(e.entity_id, e.score) for e in ordered], pipeline.store, limit=12)
    return Report(question, final.text, ordered, trace, iterations, reason)


def link(pipeline, name_a: str, name_b: str, max_hops: int = 3) -> dict:
    """Is there a path of ownership/control between two named entities?"""
    a = pipeline.search(name_a, k=1)
    b = pipeline.search(name_b, k=1)
    if not a or not b:
        return {"found": False, "reason": "one or both entities not found"}
    path = pipeline.graph.path(a[0][0], b[0][0], max_hops=max_hops)
    if not path:
        return {"found": False, "from": a[0][0], "to": b[0][0],
                "reason": f"no path within {max_hops} hops"}
    return {"found": True, "hops": len(path) - 1,
            "path": [{"id": pid, "caption": (pipeline.store.get(pid).caption
                                             if pipeline.store.get(pid) else pid)} for pid in path]}
