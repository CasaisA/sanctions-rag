# Sanctions research assistant

Hybrid retrieval, graph expansion and an agentic investigation loop over open sanctions
data. Built to answer questions an analyst actually asks — *is this name listed, under
what alias, and who sits behind the company* — and to **measure** whether each retrieval
strategy earns its place.

Data: [OpenSanctions](https://www.opensanctions.org/) (OFAC SDN slice, OFAC consolidated
non-SDN, UK HMT investment bans), loaded as FollowTheMoney entities and relations.
Corpus as evaluated: **19,588 entities and 1,463 relations**, of which 1,446 are ownership
edges.

## Quickstart

```bash
pip install -e ".[api,dev]"
./scripts/rebuild.sh                 # fetch -> ingest -> build eval set -> evaluate

sanctions-rag search "Gazprombank" -k 5
sanctions-rag ask "Which Russian banks are sanctioned?"
sanctions-rag investigate "Who owns GPB INTERNATIONAL SA?"
sanctions-rag link "Gazprombank Joint Stock Company" "GPB INTERNATIONAL SA"

uvicorn sanctions_rag.api:app --reload      # or: docker compose up
```

The core runs on numpy and scipy alone, no index server and no model download. FastAPI,
Postgres, a neural encoder and an LLM are all optional extras that slot into the same
interfaces.

## Retrieval evaluation

300 labelled queries derived from the corpus, four families that break different things.
Relevance is single-entity: the query is generated from one record, that record is the
only correct answer.

| system | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 | latency/query |
|---|---|---|---|---|---|---|
| `bm25` | 0.750 | 0.833 | 0.850 | 0.782 | 0.798 | 1 ms |
| `vector` | 0.440 | 0.627 | 0.667 | 0.515 | 0.552 | 10 ms |
| `hybrid-rrf` | 0.640 | 0.757 | 0.803 | 0.692 | 0.718 | 10 ms |
| `hybrid+rerank` | 0.763 | 0.867 | 0.907 | 0.808 | 0.832 | 16 ms |


Per-family recall@5 is where the interesting part is:

|---|---|---|---|---|
| alias (known-as names) | 0.97 | 0.69 | 0.77 | 0.97 |
| attribute (name + DOB / reg. no.) | 1.00 | 0.85 | 0.92 | 1.00 |
| partial (surname + country) | 0.71 | 0.21 | 0.57 | 0.65 |
| noisy (dropped + swapped chars) | 0.65 | 0.75 | 0.76 | 0.84 |

corpus=19588 queries=300 encoder=tfidf-svd(word+char3) reranker=feature-reranker build: bm25 0.5s vectors 11.9s
**What this says, including the inconvenient part.** BM25 is a very strong baseline for
entity retrieval and dense vectors alone are much worse — 0.627 against 0.833 recall@5.
Naive RRF fusion *degrades* BM25, because averaging in a weak run pulls good hits down.
Dense retrieval earns its keep in exactly one family: noisy strings, where it beats BM25
0.75 to 0.65, which is what you would hope for from character-level semantics. The
combination only wins once a reranker re-sorts the fused pool using name-level evidence:
recall@10 goes 0.850 → 0.907 and MRR 0.782 → 0.808, at 16 ms per query.

The honest conclusion for a production system: ship BM25 first, add dense retrieval for
transliteration and typo robustness, and always rerank. "We use vector search" would have
been the wrong answer here.

Reproduce with `python -m sanctions_rag.evaluate`; results land in `data/eval/results.json`.

### How the labelled set is built

`build_eval.py` generates queries from entities, seeded and reproducible:

| family | query | tests |
|---|---|---|
| `alias` | an alias the entity is also listed under | name variation across lists |
| `partial` | surname + country code | low-specificity queries |
| `attribute` | name + birth date or registration number | mixed free text and structured fields |
| `noisy` | name with one character dropped and two swapped | transliteration and typos |

This is weaker than editorial relevance judgements: it cannot capture "these three
entities are all plausible answers", and it inherits any bias in the source captions. It
is reproducible at zero cost and it is enough to *rank strategies against each other*,
which is what it is used for.

## Architecture

```
OpenSanctions JSONL
   └─ ingest.py ──> Store (SQLite by default, Postgres via DSN)
                      ├─ entities   (flattened search_text, topics, datasets)
                      └─ relations  (Ownership, Directorship, Family, Associate…)

query ─┬─ BM25 (pure python, Okapi)        ─┐
       └─ VectorIndex (TF-IDF+SVD or        ├─ RRF fusion ─ reranker ─ graph expansion ─> hits
          sentence-transformers)           ─┘
                                                                    │
                                          rag.py (grounded answer, every claim cited)
                                                                    │
                                          agent.py  plan → search → critique → replan
```

**Agent loop.** `investigate()` decomposes the question into sub-queries, retrieves for
each, expands one hop along ownership edges, drafts a cited answer, and then runs a critic
that checks every sub-query has supporting evidence. If not, it issues follow-up queries
and goes round again, bounded by `max_iterations`. Every step lands in a trace, so a bad
answer can be read rather than guessed at:

```
plan      Russian | russian banks sanctioned owns them        hits=2
search    Russian                                             hits=6
search    russian banks sanctioned owns them                  hits=6
critique  all sub-questions have supporting evidence          hits=12
stopped: all sub-questions have supporting evidence after 1 iteration(s)
```

**Graph-aware retrieval.** Ownership, directorship and family edges are first-class.
`GraphExpander.expand()` pulls in neighbours of a hit with a decay factor so they support
an answer without displacing direct matches; `path()` answers "how are these two linked"
with a bounded BFS.

## Design decisions

- **BM25 in plain Python, not Elasticsearch.** 20k entities index in 0.5 s. A search
  cluster would be infrastructure to maintain in exchange for nothing measurable here.
- **RRF instead of score interpolation.** BM25 scores and cosine similarities are not
  comparable, and RRF needs no per-corpus tuning.
- **Brute-force cosine, not an ANN index.** 19,588 × 256 floats is 20 MB and a full scan
  is ~10 ms. FAISS would be a dependency bought with no latency to spend it on.
- **A feature reranker as the default.** For entity retrieval, name coverage and
  trigram overlap beat a general-purpose cross-encoder, and an analyst can be told why a
  result ranked where it did. The cross-encoder is one env var away.
- **Everything swappable.** Encoder, reranker, generator and store are interfaces with a
  dependency-free default, so the repo runs offline and still demonstrates the real thing.

## Interfaces

| | |
|---|---|
| CLI | `search`, `ask`, `investigate`, `link` |
| HTTP | `GET /search`, `POST /ask`, `POST /investigate`, `GET /link`, `GET /entity/{id}`, `GET /health` |
| Container | `Dockerfile` (non-root, healthcheck), `docker-compose.yml` (API + Postgres) |
| Kubernetes | `k8s/deployment.yaml` — 2 replicas, startup/readiness/liveness probes, DSN from a secret |
| CI | GitHub Actions: lint, tests, **retrieval quality gate**, container build + smoke test |
| Observability | JSON logs with request ids and per-stage timings, `/metrics` in Prometheus format |

## Optional extras

```bash
SANCTIONS_RAG_ENCODER=sentence-transformers   # dense neural embeddings
SANCTIONS_RAG_RERANKER=cross-encoder          # ms-marco cross-encoder
ANTHROPIC_API_KEY=...                         # LLM writes the prose over the same evidence
SANCTIONS_RAG_DSN=postgresql://...            # Postgres instead of SQLite
```

## Evaluation as a build step

Retrieval quality is treated like a test. `eval_baseline.json` is committed; CI rebuilds
the index from a fresh data slice, re-runs the 300 queries and fails the build if any
system drops more than 0.02 below its baseline, or if `hybrid+rerank` stops being the
best system by recall@10. A ranking change therefore has to update the baseline in the
pull request, where a human can see the trade.

```bash
make eval gate        # locally: rebuild metrics, then enforce thresholds
```

## Tests

```bash
make test       # 18 tests: normalisation, BM25, RRF, store/graph, planner, critic,
                #           quality gate, metrics rendering
```

## What this is not

Not a production system. There is no incremental indexing, no auth, no rate limiting, no
entity resolution across lists beyond what OpenSanctions already did, and the labelled set
is machine-generated rather than annotated by an analyst. It is a demonstrator: the
retrieval, the evaluation and the agent loop are real and measured; the operational
hardening is deliberately out of scope.
