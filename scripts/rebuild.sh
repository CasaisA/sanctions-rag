#!/usr/bin/env bash
# End to end: data -> store -> labelled queries -> evaluation table.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
[ -s data/raw/us_ofac_cons.jsonl ] || ./scripts/fetch_data.sh
$PY -m sanctions_rag.ingest data/raw/*.jsonl --db data/sanctions.db
$PY -m sanctions_rag.build_eval --db data/sanctions.db
$PY -m sanctions_rag.evaluate --db data/sanctions.db --out data/eval/results.json
