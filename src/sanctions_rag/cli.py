"""Command line entry point: search, ask, investigate, link."""
from __future__ import annotations

import argparse
import json

from .agent import investigate, link
from .pipeline import Pipeline
from .rag import answer as rag_answer


def main() -> None:
    ap = argparse.ArgumentParser(prog="sanctions-rag")
    ap.add_argument("--db", default="data/sanctions.db")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=10)
    s.add_argument("--expand", action="store_true")
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("-k", type=int, default=8)
    i = sub.add_parser("investigate"); i.add_argument("question"); i.add_argument("-n", type=int, default=3)
    l = sub.add_parser("link"); l.add_argument("a"); l.add_argument("b"); l.add_argument("--hops", type=int, default=3)

    args = ap.parse_args()
    p = Pipeline.build(args.db)

    if args.cmd == "search":
        for doc_id, score in p.search(args.query, k=args.k, expand=args.expand):
            e = p.store.get(doc_id)
            tags = ",".join(e.topics[:2]) if e and e.topics else "-"
            print(f"{score:7.4f}  {e.schema if e else '?':13s} {e.caption if e else doc_id:55s} {tags}")
    elif args.cmd == "ask":
        ans = rag_answer(args.question, p.search(args.question, k=args.k, expand=True), p.store, limit=args.k)
        print(ans.text)
    elif args.cmd == "investigate":
        rep = investigate(p, args.question, max_iterations=args.n)
        print(rep.text)
        print("\n--- trace")
        for st in rep.trace:
            print(f"  {st.kind:9s} {st.detail[:80]:82s} hits={st.hits}")
        print(f"  stopped: {rep.stopped_because} after {rep.iterations} iteration(s)")
    elif args.cmd == "link":
        print(json.dumps(link(p, args.a, args.b, max_hops=args.hops), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
