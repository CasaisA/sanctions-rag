"""Load OpenSanctions FollowTheMoney JSONL into the store.

Entity rows carry a flattened `search_text`; interval schemata (Ownership, Family,
Directorship...) become edges in `relations`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .store import Entity, Relation, Store

# FtM schemata that describe a link between two entities rather than an entity.
EDGE_SCHEMATA = {
    "Ownership": ("owner", "asset"),
    "Directorship": ("director", "organization"),
    "Family": ("person", "relative"),
    "Associate": ("person", "associate"),
    "Membership": ("member", "organization"),
    "Employment": ("employee", "employer"),
    "UnknownLink": ("subject", "object"),
    "Representation": ("agent", "client"),
    "Succession": ("predecessor", "successor"),
}

# Properties worth putting in the searchable blob, in rough order of usefulness.
TEXT_PROPS = [
    "name", "alias", "weakAlias", "previousName", "fatherName", "motherName",
    "position", "notes", "summary", "program", "programId", "country", "nationality",
    "jurisdiction", "birthDate", "birthPlace", "incorporationDate", "address",
    "registrationNumber", "taxNumber", "idNumber", "passportNumber", "imoNumber",
    "sanctionStatus", "topics",
]


def _first(props: dict, key: str) -> str | None:
    v = props.get(key)
    return v[0] if isinstance(v, list) and v else None


def entity_from_ftm(rec: dict) -> Entity:
    props = rec.get("properties", {}) or {}
    parts: list[str] = [rec.get("caption") or ""]
    for key in TEXT_PROPS:
        vals = props.get(key)
        if not vals:
            continue
        parts.extend(str(v) for v in vals[:8])
    return Entity(
        id=rec["id"],
        schema=rec.get("schema", "Thing"),
        caption=rec.get("caption") or rec["id"],
        countries=[str(c) for c in (props.get("country") or props.get("jurisdiction") or [])][:8],
        topics=[str(t) for t in (props.get("topics") or [])][:8],
        datasets=list(rec.get("datasets") or [])[:8],
        properties={k: v for k, v in props.items() if k in TEXT_PROPS},
        search_text=" ".join(p for p in parts if p),
    )


def relation_from_ftm(rec: dict) -> Relation | None:
    schema = rec.get("schema")
    spec = EDGE_SCHEMATA.get(schema or "")
    if spec is None:
        return None
    props = rec.get("properties", {}) or {}
    src, tgt = _first(props, spec[0]), _first(props, spec[1])
    if not src or not tgt:
        return None
    return Relation(
        id=rec["id"], schema=schema, source_id=src, target_id=tgt,
        role=_first(props, "role") or _first(props, "percentage"),
        properties={k: v for k, v in props.items() if k in ("percentage", "role", "startDate", "endDate")},
    )


def load(paths: list[Path], store: Store, limit: int | None = None) -> dict[str, int]:
    store.init_schema()
    entities: list[Entity] = []
    relations: list[Relation] = []
    seen = 0
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated final line of a sliced dump
                if "id" not in rec:
                    continue
                edge = relation_from_ftm(rec)
                if edge is not None:
                    relations.append(edge)
                elif rec.get("schema") not in ("Address", "Sanction", "Identification"):
                    entities.append(entity_from_ftm(rec))
                seen += 1
                if limit and seen >= limit:
                    break
    n_e = store.upsert_entities(entities)
    n_r = store.upsert_relations(relations)
    return {"records": seen, "entities": n_e, "relations": n_r}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest OpenSanctions FtM JSONL")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--db", default="data/sanctions.db")
    ap.add_argument("--dsn", default=None, help="Postgres DSN; omit to use SQLite")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    store = Store(args.db, args.dsn)
    stats = load(args.paths, store, args.limit)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
