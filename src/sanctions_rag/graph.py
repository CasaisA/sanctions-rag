"""Graph-aware retrieval: expand hits along ownership and control edges.

A sanctions analyst rarely wants only the matching entity; they want who owns it and
what it owns. Expansion is one hop by default, scored down by a decay factor so that
neighbours support the answer without displacing direct matches.
"""
from __future__ import annotations

from collections import deque


class GraphExpander:
    def __init__(self, store, decay: float = 0.45,
                 follow: tuple[str, ...] = ("Ownership", "Directorship", "Family", "Associate")) -> None:
        self.store, self.decay, self.follow = store, decay, set(follow)

    def expand(self, hits: list[tuple[str, float]], hops: int = 1, max_extra: int = 20):
        """Return hits plus their neighbourhood, keeping the original ranking on top."""
        seen = {doc_id for doc_id, _ in hits}
        out = list(hits)
        frontier = deque((doc_id, score, 0) for doc_id, score in hits)
        while frontier and len(out) - len(hits) < max_extra:
            node, score, depth = frontier.popleft()
            if depth >= hops:
                continue
            for nb, schema, _direction in self.store.neighbours(node):
                if schema not in self.follow or nb in seen:
                    continue
                if self.store.get(nb) is None:
                    continue
                seen.add(nb)
                out.append((nb, score * self.decay))
                frontier.append((nb, score * self.decay, depth + 1))
        return out

    def path(self, src: str, dst: str, max_hops: int = 3) -> list[str] | None:
        """Shortest relation path between two entities, for the 'how are they linked' question."""
        if src == dst:
            return [src]
        prev: dict[str, str] = {src: ""}
        q = deque([(src, 0)])
        while q:
            node, d = q.popleft()
            if d >= max_hops:
                continue
            for nb, _schema, _dir in self.store.neighbours(node):
                if nb in prev:
                    continue
                prev[nb] = node
                if nb == dst:
                    path, cur = [dst], node
                    while cur:
                        path.append(cur)
                        cur = prev[cur]
                    return list(reversed(path))
                q.append((nb, d + 1))
        return None
