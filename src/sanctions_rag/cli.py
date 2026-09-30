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
    m = sub.add_parser("match", help="brute-force name match with per-measure scores")
    m.add_argument("name"); m.add_argument("-k", type=int, default=10)
    m.add_argument("--min-score", type=float, default=0.0)
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("-k", type=int, default=8)
    i = sub.add_parser("investigate"); i.add_argument("question"); i.add_argument("-n", type=int, default=3)
    i.add_argument("--engine", choices=["rules", "claude"], default="rules",
                   help="rules: fixed plan/critique loop; claude: Claude plans and calls the tools")
    i.add_argument("--model", default=None, help="Claude model for --engine claude")
    o = sub.add_parser("export-obsidian", help="write the entity graph as an Obsidian vault")
    o.add_argument("--out", default="vault")
    o.add_argument("--all", action="store_true", help="include entities with no relations")
    o.add_argument("--seed", help="only the neighbourhood of this entity (name or id)")
    o.add_argument("--hops", type=int, default=2)
    l = sub.add_parser("link"); l.add_argument("a"); l.add_argument("b"); l.add_argument("--hops", type=int, default=3)

    args = ap.parse_args()
    if args.cmd == "export-obsidian":
        from .obsidian import export
        from .store import Store

        store = Store(args.db)
        seed = args.seed
        if seed and store.get(seed) is None:  # a name, not an id: resolve with the search stack
            hits = Pipeline.build(args.db).search(seed, k=1)
            seed = hits[0][0] if hits else None
            if seed is None:
                raise SystemExit(f"no entity matches {args.seed!r}")
        stats = export(store, args.out, mode="all" if args.all else "connected", seed=seed, hops=args.hops)
        print(f"wrote {stats['notes']} notes, {stats['relations']} relations -> {args.out}/ "
              "(open the folder as a vault in Obsidian)")
        return
    if args.cmd == "match":
        from .namematch import NameMatcher
        from .store import Store

        store = Store(args.db)
        for mt in NameMatcher().index(store.iter_entities()).match(args.name, k=args.k, min_score=args.min_score):
            e = store.get(mt.entity_id)
            ms = "  ".join(f"{k}={v:.0f}" for k, v in mt.measures.items())
            print(f"{mt.score:5.1f}  {e.caption[:44]:45s} via {mt.matched_name[:34]!r:37s} {ms}")
        return
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
        if args.engine == "claude":
            from .claude_agent import investigate as claude_investigate

            rep = claude_investigate(p, args.question, model=args.model)
        else:
            rep = investigate(p, args.question, max_iterations=args.n)
        print(rep.text)
        print("\n--- trace")
        for st in rep.trace:
            print(f"  {st.kind:15s} {st.detail[:76]:78s} hits={st.hits}")
        unit = "turn(s)" if args.engine == "claude" else "iteration(s)"
        print(f"  stopped: {rep.stopped_because} after {rep.iterations} {unit}")
    elif args.cmd == "link":
        print(json.dumps(link(p, args.a, args.b, max_hops=args.hops), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
