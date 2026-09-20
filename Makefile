.PHONY: install data index eval gate test lint serve docker up clean
install: ; pip install -e ".[api,dev]"
data:    ; ./scripts/fetch_data.sh
index:   ; python -m sanctions_rag.ingest data/raw/*.jsonl --db data/sanctions.db
eval:    ; python -m sanctions_rag.build_eval && python -m sanctions_rag.evaluate
gate:    ; python -m sanctions_rag.gate
test:    ; pytest -q
lint:    ; ruff check src tests
serve:   ; uvicorn sanctions_rag.api:app --reload
docker:  ; docker build -t sanctions-rag:local .
up:      ; docker compose up --build
clean:   ; rm -rf data/sanctions.db .pytest_cache .ruff_cache **/__pycache__
