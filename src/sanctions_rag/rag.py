"""Grounded answers over retrieved entities.

Every sentence in an answer must point at an entity id. With no LLM configured the
generator is extractive and deterministic, which keeps the pipeline testable; set
ANTHROPIC_API_KEY to have a model write the prose over the same evidence.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Evidence:
    entity_id: str
    caption: str
    schema: str
    countries: list[str]
    topics: list[str]
    datasets: list[str]
    score: float
    snippet: str

    def cite(self) -> str:
        return f"[{self.entity_id}]"


@dataclass
class Answer:
    question: str
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    generator: str = "extractive"

    def uncited_sentences(self) -> list[str]:
        out = []
        for sent in self.text.split("\n"):
            s = sent.strip("- ").strip()
            if s and "[" not in s and not s.endswith(":"):
                out.append(s)
        return out


def gather(query: str, hits: list[tuple[str, float]], store, limit: int = 8) -> list[Evidence]:
    ev: list[Evidence] = []
    for doc_id, score in hits[:limit]:
        e = store.get(doc_id)
        if e is None:
            continue
        ev.append(Evidence(e.id, e.caption, e.schema, e.countries, e.topics,
                           e.datasets, float(score), e.search_text[:280]))
    return ev


def _extractive(question: str, ev: list[Evidence]) -> str:
    if not ev:
        return "No matching entity in the indexed sanctions data."
    lines = [f"{len(ev)} candidate entities for: {question}"]
    for e in ev:
        bits = [e.schema]
        if e.countries:
            bits.append("/".join(e.countries[:3]))
        if e.topics:
            bits.append(", ".join(e.topics[:3]))
        src = ", ".join(e.datasets[:2]) or "unknown source"
        lines.append(f"- {e.caption} ({'; '.join(bits)}) — listed in {src} {e.cite()}")
    return "\n".join(lines)


def _llm(question: str, ev: list[Evidence]) -> str | None:  # pragma: no cover - needs network
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    ctx = "\n".join(
        f"[{e.entity_id}] {e.caption} | {e.schema} | countries={e.countries} | "
        f"topics={e.topics} | datasets={e.datasets} | {e.snippet}" for e in ev)
    prompt = (
        "You are a sanctions research assistant. Answer the question using ONLY the records below.\n"
        "Cite every claim with the record id in square brackets. If the records do not support an "
        "answer, say so plainly.\n\n"
        f"RECORDS:\n{ctx}\n\nQUESTION: {question}\n")
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(model=os.environ.get("SANCTIONS_RAG_MODEL", "claude-sonnet-5"),
                                 max_tokens=700, messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def answer(question: str, hits: list[tuple[str, float]], store, limit: int = 8) -> Answer:
    ev = gather(question, hits, store, limit)
    text = _llm(question, ev)
    return Answer(question, text, ev, "llm") if text else Answer(question, _extractive(question, ev), ev)
