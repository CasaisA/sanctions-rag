"""Export the entity graph as an Obsidian vault.

One note per entity, relations written as [[wikilinks]] so Obsidian's graph view shows
the ownership and control network. Frontmatter carries the structured fields, `aliases`
makes every listed name resolvable from Obsidian's search and link autocomplete, and
tags (schema/, topic/, list/) drive the graph colour groups.

By default only entities that take part in at least one relation are exported: an
isolated node adds nothing to a graph view and 20k of them make it unusable. `--all`
exports everything; `--seed NAME --hops N` exports the neighbourhood of one entity.
"""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter, deque
from pathlib import Path

from .store import Entity, Store

MARKER = ".sanctions-rag-export"

# (label on the source's note, label on the target's note), from FtM edge direction.
PHRASES = {
    "Ownership": ("owns", "owned by"),
    "Directorship": ("director of", "has director"),
    "Family": ("relative of", "relative of"),
    "Associate": ("associate of", "associate of"),
    "Membership": ("member of", "has member"),
    "Employment": ("employed by", "employs"),
    "Representation": ("represents", "represented by"),
    "Succession": ("succeeded by", "successor of"),
    "UnknownLink": ("linked to", "linked to"),
}

ALIAS_PROPS = ("alias", "weakAlias", "previousName")
SHOWN_PROPS = ("birthDate", "birthPlace", "nationality", "incorporationDate", "jurisdiction",
               "registrationNumber", "taxNumber", "idNumber", "passportNumber", "imoNumber",
               "position", "program", "programId", "sanctionStatus", "address", "notes")

_BAD = re.compile(r'[\\/:*?"<>|#^\[\]\n\r\t]+')


def safe_name(caption: str, limit: int = 100) -> str:
    """A filename Obsidian accepts and a wikilink can target."""
    s = _BAD.sub(" ", caption).strip(" .")
    s = re.sub(r"\s+", " ", s)[:limit].rstrip(" .")
    return s or "unnamed"


def _tag(prefix: str, value: str) -> str:
    return f"{prefix}/" + re.sub(r"[^\w/-]+", "-", value.lower()).strip("-")


def _yaml(key: str, value) -> str:
    # JSON scalars and arrays are valid YAML, which sidesteps quoting rules.
    return f"{key}: {json.dumps(value, ensure_ascii=False)}"


def load_relations(store: Store) -> list[tuple[str, str, str, dict]]:
    cur = store.conn.cursor()
    cur.execute("SELECT source_id, target_id, schema, properties FROM relations")
    return [(r[0], r[1], r[2], json.loads(r[3] or "{}")) for r in cur.fetchall()]


def select_ids(store: Store, rels, mode: str = "connected", seed: str | None = None,
               hops: int = 2) -> set[str]:
    if mode == "all":
        return {e.id for e in store.iter_entities()}
    adj: dict[str, set[str]] = {}
    for s, t, _schema, _p in rels:
        adj.setdefault(s, set()).add(t)
        adj.setdefault(t, set()).add(s)
    if seed is None:
        return set(adj)
    out, q = {seed}, deque([(seed, 0)])
    while q:
        node, d = q.popleft()
        if d >= hops:
            continue
        for nb in adj.get(node, ()):
            if nb not in out:
                out.add(nb)
                q.append((nb, d + 1))
    return out


def assign_names(entities: list[Entity]) -> dict[str, str]:
    """Caption-based note names; entities that share a caption get an id suffix."""
    base = {e.id: safe_name(e.caption) for e in entities}
    clashes = Counter(n.lower() for n in base.values())
    return {i: (n if clashes[n.lower()] == 1 else f"{n} ({i[-6:]})") for i, n in base.items()}


def render_note(e: Entity, links: list[tuple[str, str, str]]) -> str:
    """links: (phrase, target note name, detail) for this entity."""
    aliases = []
    for key in ALIAS_PROPS:
        aliases += [a for a in e.properties.get(key, []) if a and a != e.caption]
    aliases = list(dict.fromkeys(aliases))[:20]
    tags = [_tag("schema", e.schema)] + [_tag("topic", t) for t in e.topics] \
        + [_tag("list", d) for d in e.datasets]
    lines = ["---", _yaml("id", e.id), _yaml("schema", e.schema), _yaml("aliases", aliases),
             _yaml("countries", e.countries), _yaml("topics", e.topics),
             _yaml("datasets", e.datasets), _yaml("tags", tags), "---", "", f"# {e.caption}", ""]
    facts = [(k, e.properties.get(k)) for k in SHOWN_PROPS if e.properties.get(k)]
    if facts:
        lines += ["| field | value |", "|---|---|"]
        for k, v in facts:
            val = "; ".join(str(x) for x in v[:4]).replace("|", "/").replace("\n", " ")
            lines.append(f"| {k} | {val[:300]} |")
        lines.append("")
    if aliases:
        lines += ["**Also listed as:** " + "; ".join(aliases), ""]
    if links:
        lines += ["## Relations", ""]
        for phrase, target, detail in sorted(links):
            lines.append(f"- {phrase} [[{target}]]" + (f" ({detail})" if detail else ""))
        lines.append("")
    lines.append(f"Source: OpenSanctions entity `{e.id}` (CC BY-NC 4.0).")
    return "\n".join(lines) + "\n"


GRAPH_SETTINGS = {
    "colorGroups": [
        {"query": "tag:#topic/sanction", "color": {"a": 1, "rgb": 14701138}},
        {"query": "tag:#topic/role-pep", "color": {"a": 1, "rgb": 15105570}},
        {"query": "tag:#schema/person", "color": {"a": 1, "rgb": 3900150}},
        {"query": "tag:#schema/vessel", "color": {"a": 1, "rgb": 1356876}},
    ],
    "showTags": False,
    "showOrphans": False,
}


def export(store: Store, out: str | Path, mode: str = "connected", seed: str | None = None,
           hops: int = 2) -> dict[str, int]:
    out = Path(out)
    notes_dir = out / "Entities"
    if notes_dir.exists():
        if not (out / MARKER).exists():
            raise SystemExit(f"{notes_dir} exists and was not written by this exporter; refusing to overwrite")
        shutil.rmtree(notes_dir)

    rels = load_relations(store)
    ids = select_ids(store, rels, mode, seed, hops)
    entities = [e for e in (store.get(i) for i in sorted(ids)) if e is not None]
    names = assign_names(entities)

    links: dict[str, list[tuple[str, str, str]]] = {i: [] for i in names}
    degree: Counter = Counter()
    for s, t, schema, props in rels:
        if s not in names or t not in names:
            continue
        fwd, back = PHRASES.get(schema, ("linked to", "linked to"))
        detail = ", ".join(str(v[0]) for k, v in props.items() if k in ("percentage", "startDate") and v)
        links[s].append((fwd, names[t], detail))
        links[t].append((back, names[s], detail))
        degree[s] += 1
        degree[t] += 1

    for e in entities:
        folder = notes_dir / safe_name(e.schema)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{names[e.id]}.md").write_text(render_note(e, links[e.id]), encoding="utf-8")

    by_schema = Counter(e.schema for e in entities)
    hubs = [f"- [[{names[i]}]] ({n} links)" for i, n in degree.most_common(25)]
    index = ["# Sanctions graph", "",
             (f"{len(entities)} entities and {sum(degree.values()) // 2} relations exported from the "
              "sanctions-rag index. Open the graph view (Cmd/Ctrl+G): red = sanctioned, "
              "orange = politically exposed person, blue = person, green = vessel."), "",
             "## By type", ""] + [f"- {k}: {v}" for k, v in by_schema.most_common()] + \
            ["", "## Most connected", ""] + hubs + ["", "Regenerate with `sanctions-rag export-obsidian`."]
    (out / "Index.md").write_text("\n".join(index) + "\n", encoding="utf-8")

    cfg = out / ".obsidian"
    cfg.mkdir(parents=True, exist_ok=True)
    graph = cfg / "graph.json"
    if not graph.exists():  # keep any view settings the user has changed since
        graph.write_text(json.dumps(GRAPH_SETTINGS, indent=2), encoding="utf-8")
    (out / MARKER).write_text("generated by sanctions-rag export-obsidian\n", encoding="utf-8")
    return {"notes": len(entities), "relations": sum(degree.values()) // 2}
