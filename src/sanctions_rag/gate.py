"""Quality gate: fail the build when retrieval regresses.

Treats evaluation output like a test result. The baseline is committed, so a pull
request that changes ranking has to either keep the numbers or update the baseline
deliberately, which makes the regression visible in review.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# How much a metric may drop before the build fails. Retrieval numbers move a little
# with the upstream data slice, so the gate is a band, not an equality check.
TOLERANCE = 0.02


def check(results: dict, baseline: dict, tolerance: float = TOLERANCE) -> list[str]:
    failures: list[str] = []
    for system, metrics in baseline.get("overall", {}).items():
        got = results.get("overall", {}).get(system)
        if got is None:
            failures.append(f"{system}: missing from results")
            continue
        for metric, expected in metrics.items():
            actual = got.get(metric)
            if actual is None:
                failures.append(f"{system}.{metric}: missing")
            elif actual < expected - tolerance:
                failures.append(f"{system}.{metric}: {actual:.3f} < {expected:.3f} - {tolerance}")
    best = results.get("overall", {}).get("hybrid+rerank", {})
    if best.get("recall@10", 0) < max(r.get("recall@10", 0) for r in results.get("overall", {}).values()):
        failures.append("hybrid+rerank is no longer the best system by recall@10")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/eval/results.json")
    ap.add_argument("--baseline", default="eval_baseline.json")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    baseline = json.loads(Path(args.baseline).read_text())
    failures = check(results, baseline, args.tolerance)

    for system, metrics in results.get("overall", {}).items():
        print(f"  {system:16s} recall@10={metrics.get('recall@10', 0):.3f}  mrr@10={metrics.get('mrr@10', 0):.3f}")
    if failures:
        print("\nquality gate FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nquality gate passed")


if __name__ == "__main__":
    main()
