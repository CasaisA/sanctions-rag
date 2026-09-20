-- Postgres schema. Mirrors the SQLite DDL in store.py.
CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    schema      TEXT NOT NULL,
    caption     TEXT NOT NULL,
    countries   JSONB NOT NULL DEFAULT '[]',
    topics      JSONB NOT NULL DEFAULT '[]',
    datasets    JSONB NOT NULL DEFAULT '[]',
    properties  JSONB NOT NULL DEFAULT '{}',
    search_text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS entities_schema_idx  ON entities (schema);
CREATE INDEX IF NOT EXISTS entities_topics_idx  ON entities USING GIN (topics);
-- Postgres full-text as a baseline to compare our BM25 against.
CREATE INDEX IF NOT EXISTS entities_fts_idx
    ON entities USING GIN (to_tsvector('simple', search_text));

CREATE TABLE IF NOT EXISTS relations (
    id         TEXT PRIMARY KEY,
    schema     TEXT NOT NULL,
    source_id  TEXT NOT NULL,
    target_id  TEXT NOT NULL,
    role       TEXT,
    properties JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS relations_source_idx ON relations (source_id);
CREATE INDEX IF NOT EXISTS relations_target_idx ON relations (target_id);
