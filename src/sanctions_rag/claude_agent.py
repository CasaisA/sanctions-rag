"""Investigation agent driven by Claude through the Claude Agent SDK.

The rule-based loop in agent.py fixes the plan in code: regex sub-queries, a word-overlap
critic, a bounded replan. Here the model owns those decisions. It gets four read-only
tools over the same retrieval stack (search, entity lookup, graph neighbours, link path),
decides what to search, which hits are the same entity, which ownership edges to follow
and when the evidence is enough, then writes a report where every claim cites an entity id.

Two checks stay in code rather than in the prompt:
  * the tool set is closed: no shell, no files, no web, only the four tools below;
  * citations are verified afterwards. An id the report cites but no tool ever returned
    is flagged as unsupported, so a fabricated citation is visible, not silent.

The SDK runs the local Claude Code binary and uses whatever credentials it has (an
ANTHROPIC_API_KEY, or the machine's existing Claude Code login for personal use).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .agent import Report, Step
from .rag import Evidence

SERVER = "sanctions"
TOOL_NAMES = ("search_entities", "get_entity", "get_neighbours", "find_path")
DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are a sanctions research analyst working over an index of official \
sanctions lists (OFAC SDN, OFAC consolidated non-SDN, UK HMT investment bans), loaded from \
OpenSanctions. You can only see that index through your tools; you have no other source.

How to work:
- Search for each person, company or vessel the question mentions. Names vary across lists \
(aliases, transliterations, legal-form suffixes), so try variants when a search comes back thin.
- A name match is not an identity match. Before you treat a hit as the entity asked about, \
check it with get_entity: country, birth date, registration number, programme.
- Ownership and control matter as much as direct listing. Use get_neighbours to follow \
ownership, directorship and family links, and find_path to test whether two entities connect.
- Stop when the question is answered or when further searches stop adding evidence.

Report:
- Answer the question directly first, then the supporting facts.
- Cite every factual claim with the entity id in square brackets, e.g. [NK-abc123]. Only cite \
ids that a tool returned to you.
- Say plainly what you could not establish. "Not found in the indexed lists" is not the same \
as "not sanctioned": the index is a slice of a few lists."""


def _entity_brief(e, score: float | None = None) -> dict[str, Any]:
    out = {"id": e.id, "caption": e.caption, "schema": e.schema, "countries": e.countries[:4],
           "topics": e.topics[:4], "datasets": e.datasets[:3]}
    if score is not None:
        out["score"] = round(float(score), 4)
    return out


def _text(payload: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}


def _error(msg: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": msg}], "is_error": True}


@dataclass
class ToolBox:
    """Plain functions over the pipeline. Kept free of SDK types so they test without Claude."""

    pipeline: Any
    seen: set[str] = field(default_factory=set)

    def _mark(self, ids) -> None:
        self.seen.update(ids)

    def search(self, query: str, k: int = 8) -> list[dict]:
        k = max(1, min(int(k), 20))
        out = []
        for doc_id, score in self.pipeline.search(query, k=k):
            e = self.pipeline.store.get(doc_id)
            if e is not None:
                out.append(_entity_brief(e, score))
        self._mark(h["id"] for h in out)
        return out

    def entity(self, entity_id: str) -> dict | None:
        e = self.pipeline.store.get(entity_id)
        if e is None:
            return None
        self._mark([e.id])
        # search_text repeats the properties; the structured fields are what the model needs.
        return {**_entity_brief(e), "properties": {k: v[:6] for k, v in e.properties.items()}}

    def neighbours(self, entity_id: str) -> list[dict]:
        out = []
        for nid, schema, direction in self.pipeline.store.neighbours(entity_id):
            n = self.pipeline.store.get(nid)
            if n is not None:
                out.append({**_entity_brief(n), "relation": schema, "direction": direction})
        self._mark(o["id"] for o in out)
        return out[:40]

    def path(self, a: str, b: str, max_hops: int = 3) -> list[dict] | None:
        p = self.pipeline.graph.path(a, b, max_hops=max(1, min(int(max_hops), 4)))
        if not p:
            return None
        out = [_entity_brief(e) for e in (self.pipeline.store.get(i) for i in p) if e is not None]
        self._mark(o["id"] for o in out)
        return out


def build_server(box: ToolBox):
    from claude_agent_sdk import ToolAnnotations, create_sdk_mcp_server, tool

    ro = ToolAnnotations(readOnlyHint=True)

    @tool("search_entities",
          "Search the sanctions index by name or free text (aliases, identifiers and countries "
          "are indexed too). Returns ranked candidates with id, caption, schema, countries, "
          "topics and source lists. Optional 'k' (1-20, default 8).",
          {"query": str}, annotations=ro)
    async def search_entities(args: dict[str, Any]) -> dict[str, Any]:
        hits = box.search(args["query"], args.get("k", 8))
        return _text(hits) if hits else _text({"hits": [], "note": "no match; try a name variant"})

    @tool("get_entity",
          "Full record for one entity id: every listed name and alias, birth date, nationality, "
          "registration and ID numbers, programme, notes.",
          {"entity_id": str}, annotations=ro)
    async def get_entity(args: dict[str, Any]) -> dict[str, Any]:
        e = box.entity(args["entity_id"])
        return _text(e) if e else _error(f"unknown entity id {args['entity_id']!r}")

    @tool("get_neighbours",
          "Entities linked to this one by ownership, directorship, family, association, "
          "membership or employment, with the relation type and its direction.",
          {"entity_id": str}, annotations=ro)
    async def get_neighbours(args: dict[str, Any]) -> dict[str, Any]:
        return _text(box.neighbours(args["entity_id"]))

    @tool("find_path",
          "Shortest chain of relations between two entity ids, up to 'max_hops' (default 3, "
          "max 4). Use it to test whether two entities are connected.",
          {"from_id": str, "to_id": str}, annotations=ro)
    async def find_path(args: dict[str, Any]) -> dict[str, Any]:
        p = box.path(args["from_id"], args["to_id"], args.get("max_hops", 3))
        return _text(p) if p else _text({"found": False})

    return create_sdk_mcp_server(name=SERVER, version="1.0.0",
                                 tools=[search_entities, get_entity, get_neighbours, find_path])


_CITE = re.compile(r"\[([A-Za-z0-9][\w.:-]{2,})\]")


def check_citations(text: str, seen: set[str]) -> tuple[list[str], list[str]]:
    """Split the ids a report cites into (retrieved, unsupported)."""
    cited = list(dict.fromkeys(_CITE.findall(text)))
    return [c for c in cited if c in seen], [c for c in cited if c not in seen]


def _brief_input(inp: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in inp.items())


async def ainvestigate(pipeline, question: str, max_turns: int = 20,
                       model: str | None = None, effort: str = "medium") -> Report:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
        query,
    )

    box = ToolBox(pipeline)
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={SERVER: build_server(box)},
        tools=[],                                  # no built-ins: no shell, files or web
        allowed_tools=[f"mcp__{SERVER}__{t}" for t in TOOL_NAMES],
        setting_sources=[],                        # ignore the user's CLAUDE.md, skills, hooks
        max_turns=max_turns,
        model=model or os.environ.get("SANCTIONS_RAG_AGENT_MODEL", DEFAULT_MODEL),
        effort=effort,
        env={"ENABLE_TOOL_SEARCH": "false"},       # four tools: load them all up front
    )

    trace: list[Step] = []
    pending: dict[str, Step] = {}
    final, stop, turns, cost = "", "", 0, None
    async for msg in query(prompt=question, options=options):
        if isinstance(msg, AssistantMessage):
            for b in msg.content:
                if isinstance(b, ToolUseBlock):
                    st = Step(b.name.removeprefix(f"mcp__{SERVER}__"), _brief_input(b.input))
                    trace.append(st)
                    pending[b.id] = st
                elif isinstance(b, TextBlock) and b.text.strip():
                    final = b.text
        elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
            for b in msg.content:
                if isinstance(b, ToolResultBlock) and b.tool_use_id in pending:
                    pending[b.tool_use_id].hits = _count_hits(b.content)
        elif isinstance(msg, ResultMessage):
            turns, cost = msg.num_turns, msg.total_cost_usd
            stop = msg.subtype if msg.subtype != "success" else "model finished"
            if msg.result:
                final = msg.result

    ok_ids, bad_ids = check_citations(final, box.seen)
    trace.append(Step("verify", f"{len(ok_ids)} citations retrieved, {len(bad_ids)} unsupported"
                      + (f": {', '.join(bad_ids[:5])}" if bad_ids else ""), len(ok_ids)))
    evidence = []
    for eid in ok_ids:
        e = pipeline.store.get(eid)
        if e is not None:
            evidence.append(Evidence(e.id, e.caption, e.schema, e.countries, e.topics,
                                     e.datasets, 1.0, e.search_text[:280]))
    if cost is not None:
        stop += f" (api-equivalent cost ${cost:.3f})"
    return Report(question, final, evidence, trace, turns, stop)


def _count_hits(content) -> int:
    if not isinstance(content, list):
        return 0
    for c in content:
        if isinstance(c, dict) and c.get("type") == "text":
            try:
                data = json.loads(c["text"])
            except (json.JSONDecodeError, TypeError):
                return 0
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict) and "hits" in data:
                return len(data["hits"])
            return 0 if isinstance(data, dict) and data.get("found") is False else 1
    return 0


def investigate(pipeline, question: str, **kw) -> Report:
    return asyncio.run(ainvestigate(pipeline, question, **kw))
