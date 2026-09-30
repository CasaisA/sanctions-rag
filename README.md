# Sanctions research assistant

Hybrid retrieval, graph expansion and an agentic investigation loop over open sanctions
data. Built to answer questions an analyst actually asks — *is this name listed, under
what alias, and who sits behind the company* — and to **measure** whether each retrieval
strategy earns its place.

Data: [OpenSanctions](https://www.opensanctions.org/) (OFAC SDN slice, OFAC consolidated
non-SDN, UK HMT investment bans), loaded as FollowTheMoney entities and relations.
Corpus as evaluated: **19,588 entities and 1,463 relations**, of which 1,446 are ownership
edges.

Data licence: OpenSanctions data is published under
[CC BY-NC 4.0](https://www.opensanctions.org/licensing/) (non-commercial use, with
attribution). This repository does not ship the raw data; `scripts/fetch_data.sh` downloads
it from OpenSanctions. The evaluation queries in `data/eval/` are derived from it and carry
the same licence.

## Quickstart

```bash
pip install -e ".[api,dev]"
./scripts/rebuild.sh                 # fetch -> ingest -> build eval set -> evaluate

sanctions-rag search "Gazprombank" -k 5
sanctions-rag ask "Which Russian banks are sanctioned?"
sanctions-rag investigate "Who owns GPB INTERNATIONAL SA?"
sanctions-rag investigate "Who owns GPB INTERNATIONAL SA?" --engine claude   # needs [agent]
sanctions-rag link "Gazprombank Joint Stock Company" "GPB INTERNATIONAL SA"

uvicorn sanctions_rag.api:app --reload      # or: docker compose up
```

The core runs on numpy, scipy and rapidfuzz, no index server and no model download. FastAPI,
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

**With a brute-force name matcher and learned reranker weights.** `namematch.py` compares
the query against every one of the ~50k distinct listed names and aliases, with no index:
names are normalised (accents folded, Cyrillic transliterated, legal forms such as
LLC / OOO / JSC dropped) and scored with Jaro-Winkler, token-sort and token-set similarity,
about 50 ms per query with rapidfuzz. It runs as a third retriever in the fusion, and its
best-name score replaces caption trigram overlap as the reranker's name feature.

The reranker's weights were hand-set and judged on the same 300 queries, so the numbers
above are optimistic. `train_rerank.py` now fits them on 600 training queries generated
from entities that never appear in the test set (listwise softmax, L-BFGS), and the
pipeline uses those weights (`data/eval/reranker_weights.json`). Out-of-sample:

| system | recall@1 | recall@5 | recall@10 | mrr@10 | noisy recall@5 |
|---|---|---|---|---|---|
| `hybrid+rerank` (hand weights, in-sample) | 0.763 | 0.867 | 0.907 | 0.808 | 0.84 |
| `names` alone | 0.743 | 0.760 | 0.777 | 0.751 | **0.89** |
| `hybrid+names+rerank` (learned weights) | **0.820** | **0.880** | **0.913** | **0.849** | **0.89** |

The matcher alone is the best system for typos and aliases and useless for "surname +
country" (0.15), which is BM25's job; combined, the stack is the best on every overall
metric and is the pipeline default and the CI gate's primary system. The learned weights
put name similarity and query coverage first (0.42, 0.41), keep some retriever prior
(0.14) and drop the "sanctioned" topic bonus to zero, since nearly every indexed entity
carries it. It also separates cases embeddings blur: for the typo "Gazpronbank" the
matcher scores Gazprombank 92.5 against 72.6 for Gazprom Neft Lubricants, where bge-small
puts the typo at cosine 0.75 from Gazprombank and 0.73 from Gazprom Neft. The full
pipeline still ranks Gazprombank second on that query, because Lubricants is surfaced by
two retrievers and the prior rewards it. `sanctions-rag match NAME` shows the matched name
and each measure.

**Claude as identity adjudicator.** Retrieval finds what is *similar*; screening needs
what is *the same*. `adjudicate.py` shows Claude (Opus 5.5, low effort, via the Agent SDK)
the query and the top 10 candidates with aliases, dates, identifiers and countries, and asks
which one is the same entity, or none, with a confidence and a reason. The pick moves to rank
1; nothing else changes. Same 300 test queries; the right entity is in the top 10 for
274 of them, which caps what reordering can reach:

| | recall@1 | mrr@10 | Claude calls |
|---|---|---|---|
| retrieval only (`hybrid+names+rerank`) | 0.820 | 0.849 | 0 |
| + Claude when top-2 reranker gap < 0.05 | 0.840 | 0.863 | 32 |
| + Claude when gap < 0.1 | 0.863 | 0.878 | 59 |
| + Claude on every query | 0.873 | 0.883 | 300 |

Asked on every query it fixed 16 rankings and broke none. Its confidence is informative:
high-confidence picks were right 229/229, medium 23/24, low 7/12, and every wrong pick
was low confidence on a query that is genuinely ambiguous ("ZAO ru", "SANCHEZ co": a legal
form or a surname plus a country), where it said so in its reason. Typo queries go from
0.80 to 0.95 recall@1. Gating at a 0.1 gap keeps most of the gain with a fifth of the
calls. A production policy follows directly: accept high-confidence decisions, route low
ones to an analyst. Each call takes 5-7 s on the Agent SDK, most of it process start-up
and thinking; the full run took under 5 minutes at 6 in parallel. Results in
`data/eval/results-adjudicator.json`; `python -m sanctions_rag.adjudicate` re-scores from
the cached decisions without calling Claude again.

**Encoder comparison, including an open-weights Voyage model.** Same 300 queries, vector
retrieval on its own, then inside the full stack. `voyage-4-nano` (0.3B parameters,
Apache 2.0, the free local member of the family Anthropic's docs recommend for embeddings)
runs through `SANCTIONS_RAG_ENCODER=sentence-transformers
SANCTIONS_RAG_EMBED_MODEL=voyageai/voyage-4-nano SANCTIONS_RAG_TRUST_REMOTE_CODE=1
SANCTIONS_RAG_EMBED_DIM=1024`; it needs `transformers` 4.x.

| encoder | size | vector recall@1 | vector recall@5 | noisy recall@5 | `hybrid+names+rerank` recall@1 / mrr@10 | corpus encode |
|---|---|---|---|---|---|---|
| LSA (TF-IDF + SVD) | none | 0.440 | 0.627 | 0.75 | 0.820 / 0.849 | 12 s |
| `BAAI/bge-small-en-v1.5` | 33M | 0.597 | 0.727 | 0.60 | not run | 14 min CPU |
| `voyageai/voyage-4-nano` | 300M | 0.710 | 0.803 | **0.88** | 0.830 / 0.857 | 12 min Apple GPU |
| `BAAI/bge-m3` | 568M | **0.773** | **0.847** | **0.88** | **0.837 / 0.863** | 5 min Apple GPU |

The two multilingual models are the first vector retrievers here that compete with BM25
on their own; bge-m3 beats it at rank 1 (0.773 vs 0.750) and needs no remote code. Unlike
bge-small both are robust to typos (0.88 on noisy queries) and to script: they put
"Газпромбанк" and "Gazpronbank" on Gazprombank, well clear of Gazprom Neft. Run bge-m3 with
`SANCTIONS_RAG_ENCODER=sentence-transformers SANCTIONS_RAG_EMBED_MODEL=BAAI/bge-m3`. In the full
stack the gain is small, because the name matcher already covers typos and aliases, and the
reranker weights were fitted with LSA vectors in the pool, not retrained for it.

**With a small neural encoder.** Swapping the LSA vectors for `BAAI/bge-small-en-v1.5`
(`SANCTIONS_RAG_ENCODER=fastembed`, 67 MB, ONNX, no torch) on the same 300 queries:

| system | recall@1 | recall@5 | recall@10 | mrr@10 | vs LSA encoder |
|---|---|---|---|---|---|
| `vector` | 0.597 | 0.727 | 0.773 | 0.648 | recall@5 +0.10 |
| `hybrid-rrf` | 0.720 | 0.817 | 0.853 | 0.761 | fusion no longer hurts BM25 |
| `hybrid+rerank` | 0.777 | 0.867 | 0.897 | 0.817 | recall@10 −0.01, MRR +0.01 |

The neural model is the better *standalone* retriever, mostly on aliases (0.88 vs 0.69
recall@5), and with it naive fusion stops dragging BM25 down. It is worse on noisy strings
(0.60 vs 0.75): character trigrams see a swapped letter, a word-piece model mostly does not.
After reranking, the two encoders end up within noise of each other. Encoding the corpus
takes about 14 minutes on a laptop CPU once, then vectors load from `data/cache`. Results in
`data/eval/results-fastembed.json`.

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

**Claude-driven agent (`--engine claude`).** The loop above fixes the plan in code. The
Claude engine (`claude_agent.py`, built on the Claude Agent SDK) hands those decisions to
the model: it gets four read-only tools over the same stack (`search_entities`,
`get_entity`, `get_neighbours`, `find_path`) and decides what to search, which hits are the
same entity, which ownership edges to follow and when to stop. Two guarantees stay in code:
the tool set is closed (no shell, files or web, and no user settings are loaded), and every
id the report cites is checked against the ids the tools actually returned, so a fabricated
citation shows up in the trace as `unsupported` instead of passing silently.

```
search_entities query=GPB INTERNATIONAL SA                          hits=8
get_entity      entity_id=NK-UpAr7xoNrR6um5PCyigtRX                 hits=1
get_neighbours  entity_id=NK-UpAr7xoNrR6um5PCyigtRX                 hits=1
get_entity      entity_id=NK-f7u9JR4f7piS8PngjTkUSa                 hits=1
search_entities query=Gazprombank                                   hits=8
search_entities query=Bank GPB JSC Gazprombank Aktsionernoe ...     hits=8
verify          8 citations retrieved, 0 unsupported                hits=8
```

The SDK drives the local Claude Code binary and uses its credentials (`ANTHROPIC_API_KEY`,
or an existing Claude Code login for personal use). Model via `--model` or
`SANCTIONS_RAG_AGENT_MODEL`.

**Obsidian export.** `sanctions-rag export-obsidian` writes the connected part of the graph
(1,927 entities, 1,456 relations) as an Obsidian vault in `vault/`: one note per entity,
relations as `[[wikilinks]]` phrased in both directions ("owns" / "owned by"), aliases in
frontmatter so any listed name resolves, and tags that colour the graph view (sanctioned,
politically exposed, person, vessel). `--seed "Gazprombank" --hops 2` exports one
neighbourhood; `--all` includes unconnected entities.

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
  name similarity beat a general-purpose cross-encoder, and an analyst can be told why a
  result ranked where it did. Its five weights are fitted on training queries disjoint
  from the test set. The cross-encoder is one env var away.
- **Everything swappable.** Encoder, reranker, generator and store are interfaces with a
  dependency-free default, so the repo runs offline and still demonstrates the real thing.

## Interfaces

| | |
|---|---|
| CLI | `search`, `ask`, `investigate [--engine rules\|claude]`, `link`, `export-obsidian` |
| HTTP | `GET /search`, `POST /ask`, `POST /investigate` (`"engine": "claude"`), `GET /link`, `GET /entity/{id}`, `GET /health` |
| Container | `Dockerfile` (non-root, healthcheck), `docker-compose.yml` (API + Postgres) |
| Kubernetes | `k8s/deployment.yaml` — 2 replicas, startup/readiness/liveness probes, DSN from a secret |
| CI | GitHub Actions: lint, tests, **retrieval quality gate**, container build + smoke test |
| Observability | JSON logs with request ids and per-stage timings, `/metrics` in Prometheus format |

## Optional extras

```bash
SANCTIONS_RAG_ENCODER=fastembed               # small neural encoder on ONNX, no torch (.[embed])
SANCTIONS_RAG_EMBED_MODEL=BAAI/bge-small-en-v1.5   # any fastembed model; vectors cached in data/cache
SANCTIONS_RAG_ENCODER=sentence-transformers   # dense neural embeddings via torch
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
make test       # 19 tests: normalisation, BM25, RRF, store/graph, planner, critic,
                #           quality gate, metrics rendering
```

## What this is not

Not a production system. There is no incremental indexing, no auth, no rate limiting, no
entity resolution across lists beyond what OpenSanctions already did, and the labelled set
is machine-generated rather than annotated by an analyst. It is a demonstrator: the
retrieval, the evaluation and the agent loop are real and measured; the operational
hardening is deliberately out of scope.
