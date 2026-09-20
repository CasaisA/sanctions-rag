"""Compare retrieval strategies on the labelled set: recall@k, MRR@10, nDCG@10."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from .bm25 import BM25
from .embed import get_encoder
from .hybrid import HybridRetriever
from .rerank import get_reranker
from .store import Store
from .vector import VectorIndex


def _metrics(ranked: list[str], relevant: set[str], ks=(1, 5, 10, 20)) -> dict:
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = 1.0 if relevant & set(ranked[:k]) else 0.0
    rr = 0.0
    for i, doc in enumerate(ranked[:10], start=1):
        if doc in relevant:
            rr = 1.0 / i
            break
    out["mrr@10"] = rr
    dcg = sum(1.0 / math.log2(i + 1) for i, doc in enumerate(ranked[:10], start=1) if doc in relevant)
    out["ndcg@10"] = dcg / 1.0  # single relevant document, so ideal DCG is 1
    return out


def run(db: str, queries_path: str, out_path: str | None = None, pool: int = 100) -> dict:
    store = Store(db)
    docs = [(e.id, e.search_text) for e in store.iter_entities()]
    print(f"corpus: {len(docs)} entities")

    t0 = time.time()
    bm25 = BM25().index(docs)
    t_bm25 = time.time() - t0

    t0 = time.time()
    encoder = get_encoder()
    vectors = VectorIndex(encoder).index(docs)
    t_vec = time.time() - t0

    hybrid = HybridRetriever(bm25, vectors)
    reranker = get_reranker()

    queries = [json.loads(l) for l in Path(queries_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    systems = {
        "bm25": lambda q: [d for d, _ in bm25.search(q, k=20)],
        "vector": lambda q: [d for d, _ in vectors.search(q, k=20)],
        "hybrid-rrf": lambda q: [d for d, _ in hybrid.search(q, k=20, pool=pool)],
        "hybrid+rerank": lambda q: [d for d, _ in reranker.rerank(q, hybrid.search(q, k=pool, pool=pool), store, k=20)],
    }

    results: dict = {"corpus": len(docs), "queries": len(queries),
                     "encoder": getattr(encoder, "info", None) and encoder.info.name,
                     "reranker": getattr(reranker, "name", "?"),
                     "build_seconds": {"bm25": round(t_bm25, 1), "vector": round(t_vec, 1)},
                     "overall": {}, "by_family": {}, "latency_ms": {}}

    for sys_name, fn in systems.items():
        agg: dict[str, float] = {}
        per_family: dict[str, dict[str, float]] = {}
        counts: dict[str, int] = {}
        t0 = time.time()
        for q in queries:
            ranked = fn(q["query"])
            m = _metrics(ranked, set(q["relevant"]))
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
            fam = q["family"]
            counts[fam] = counts.get(fam, 0) + 1
            pf = per_family.setdefault(fam, {})
            for k, v in m.items():
                pf[k] = pf.get(k, 0.0) + v
        elapsed = time.time() - t0
        n = len(queries)
        results["overall"][sys_name] = {k: round(v / n, 3) for k, v in agg.items()}
        results["latency_ms"][sys_name] = round(1000 * elapsed / n, 1)
        for fam, pf in per_family.items():
            results["by_family"].setdefault(fam, {})[sys_name] = {
                k: round(v / counts[fam], 3) for k, v in pf.items()}

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def as_table(res: dict) -> str:
    cols = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    lines = ["| system | " + " | ".join(cols) + " | latency/query |",
             "|---|" + "---|" * (len(cols) + 1)]
    for sysname, m in res["overall"].items():
        lines.append("| " + sysname + " | " + " | ".join(f"{m[c]:.3f}" for c in cols)
                     + f" | {res['latency_ms'][sysname]:.0f} ms |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sanctions.db")
    ap.add_argument("--queries", default="data/eval/queries.jsonl")
    ap.add_argument("--out", default="data/eval/results.json")
    args = ap.parse_args()
    res = run(args.db, args.queries, args.out)
    print(as_table(res))
    print("\nby family (recall@5):")
    for fam, systems in sorted(res["by_family"].items()):
        row = "  ".join(f"{s}={m['recall@5']:.2f}" for s, m in systems.items())
        print(f"  {fam:10s} {row}")


if __name__ == "__main__":
    main()
