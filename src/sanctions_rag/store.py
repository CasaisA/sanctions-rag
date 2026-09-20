"""Storage for entities and relations.

SQLite by default so the project runs with zero setup; Postgres when a DSN is given,
which is what docker-compose brings up. Same schema either way (see schema.sql).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    schema      TEXT NOT NULL,
    caption     TEXT NOT NULL,
    countries   TEXT NOT NULL DEFAULT '[]',
    topics      TEXT NOT NULL DEFAULT '[]',
    datasets    TEXT NOT NULL DEFAULT '[]',
    properties  TEXT NOT NULL DEFAULT '{}',
    search_text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS entities_schema_idx ON entities(schema);

CREATE TABLE IF NOT EXISTS relations (
    id        TEXT PRIMARY KEY,
    schema    TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    role      TEXT,
    properties TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS relations_source_idx ON relations(source_id);
CREATE INDEX IF NOT EXISTS relations_target_idx ON relations(target_id);
"""


@dataclass
class Entity:
    id: str
    schema: str
    caption: str
    countries: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    search_text: str = ""


@dataclass
class Relation:
    id: str
    schema: str
    source_id: str
    target_id: str
    role: str | None = None
    properties: dict = field(default_factory=dict)


class Store:
    """Thin persistence layer. Swap the driver, keep the calls."""

    def __init__(self, path: str | Path = "data/sanctions.db", dsn: str | None = None) -> None:
        self.dsn = dsn
        if dsn:  # pragma: no cover - exercised via docker-compose, not in CI
            import psycopg  # type: ignore

            self.conn = psycopg.connect(dsn)
            self.ph = "%s"
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(path))
            self.conn.row_factory = sqlite3.Row
            self.ph = "?"

    def init_schema(self) -> None:
        if self.dsn:  # pragma: no cover
            ddl = (Path(__file__).parent / "schema.sql").read_text()
            with self.conn.cursor() as cur:
                cur.execute(ddl)
        else:
            self.conn.executescript(DDL_SQLITE)
        self.conn.commit()

    def upsert_entities(self, rows: Iterable[Entity]) -> int:
        p = self.ph
        sql = (
            f"INSERT INTO entities (id, schema, caption, countries, topics, datasets, properties, search_text) "
            f"VALUES ({p},{p},{p},{p},{p},{p},{p},{p}) "
            f"ON CONFLICT (id) DO UPDATE SET caption=EXCLUDED.caption, search_text=EXCLUDED.search_text"
        )
        n = 0
        cur = self.conn.cursor()
        for e in rows:
            cur.execute(sql, (e.id, e.schema, e.caption, json.dumps(e.countries),
                              json.dumps(e.topics), json.dumps(e.datasets),
                              json.dumps(e.properties), e.search_text))
            n += 1
        self.conn.commit()
        return n

    def upsert_relations(self, rows: Iterable[Relation]) -> int:
        p = self.ph
        sql = (f"INSERT INTO relations (id, schema, source_id, target_id, role, properties) "
               f"VALUES ({p},{p},{p},{p},{p},{p}) ON CONFLICT (id) DO NOTHING")
        n = 0
        cur = self.conn.cursor()
        for r in rows:
            cur.execute(sql, (r.id, r.schema, r.source_id, r.target_id, r.role, json.dumps(r.properties)))
            n += 1
        self.conn.commit()
        return n

    def iter_entities(self) -> Iterator[Entity]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, schema, caption, countries, topics, datasets, properties, search_text FROM entities")
        for row in cur:
            r = dict(row) if not self.dsn else dict(zip(
                ["id", "schema", "caption", "countries", "topics", "datasets", "properties", "search_text"], row))
            yield Entity(r["id"], r["schema"], r["caption"], json.loads(r["countries"]),
                         json.loads(r["topics"]), json.loads(r["datasets"]),
                         json.loads(r["properties"]), r["search_text"])

    def get(self, entity_id: str) -> Entity | None:
        cur = self.conn.cursor()
        cur.execute(f"SELECT id, schema, caption, countries, topics, datasets, properties, search_text "
                    f"FROM entities WHERE id = {self.ph}", (entity_id,))
        row = cur.fetchone()
        if row is None:
            return None
        r = dict(row) if not self.dsn else dict(zip(
            ["id", "schema", "caption", "countries", "topics", "datasets", "properties", "search_text"], row))
        return Entity(r["id"], r["schema"], r["caption"], json.loads(r["countries"]),
                      json.loads(r["topics"]), json.loads(r["datasets"]),
                      json.loads(r["properties"]), r["search_text"])

    def neighbours(self, entity_id: str) -> list[tuple[str, str, str]]:
        """Return (neighbour_id, relation_schema, direction) for one entity."""
        p = self.ph
        cur = self.conn.cursor()
        cur.execute(f"SELECT target_id, schema FROM relations WHERE source_id = {p}", (entity_id,))
        out = [(r[0], r[1], "out") for r in cur.fetchall()]
        cur.execute(f"SELECT source_id, schema FROM relations WHERE target_id = {p}", (entity_id,))
        out += [(r[0], r[1], "in") for r in cur.fetchall()]
        return out

    def counts(self) -> dict[str, int]:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM entities")
        e = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM relations")
        r = cur.fetchone()[0]
        return {"entities": e, "relations": r}
