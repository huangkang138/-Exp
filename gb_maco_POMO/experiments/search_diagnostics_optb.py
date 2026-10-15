"""Read-only replay of Opt-B for attribution; verified against the original result."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .compare_learnable_bias import canonical, equal_plateau


@dataclass
class Trace:
    routes: dict[tuple[int, ...], list[int]]
    checks: int
    attempts: int
    seed_keys: list[tuple[int, ...]]
    root_by_key: dict[tuple[int, ...], int]
    parent_by_key: dict[tuple[int, ...], tuple[int, ...] | None]


def replay(starts: list[list[int]], distance: list[list[int]], budget: int = 50_000) -> Trace:
    """Mirror Opt-B BFS and record the first seed/parent that discovers each route."""
    if not starts or budget < 1:
        raise ValueError("positive budget and nonempty starts required")
    length = sum(distance[starts[0][i]][starts[0][(i + 1) % len(starts[0])]]
                 for i in range(len(starts[0])))
    routes: dict[tuple[int, ...], list[int]] = {}
    root_by_key: dict[tuple[int, ...], int] = {}
    parent_by_key: dict[tuple[int, ...], tuple[int, ...] | None] = {}
    queue: deque[list[int]] = deque()
    seed_keys = []
    for index, route in enumerate(starts):
        if sum(distance[route[i]][route[(i + 1) % len(route)]]
               for i in range(len(route))) != length:
            raise ValueError("Opt-B starts must have identical length")
        key = canonical(route)
        if key not in routes:
            routes[key] = route.copy()
            root_by_key[key] = index
            parent_by_key[key] = None
            queue.append(route)
            seed_keys.append(key)
    checks = attempts = 0
    while queue and checks < budget:
        path = queue.popleft()
        parent = canonical(path)
        attempts += 1
        n = len(path)
        for i in range(n - 2):
            a, b = path[i], path[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                if checks == budget:
                    trace = Trace(routes, checks, attempts, seed_keys, root_by_key, parent_by_key)
                    _verify(starts, distance, budget, trace)
                    return trace
                c, d = path[j], path[(j + 1) % n]
                delta = distance[a][c] + distance[b][d] - distance[a][b] - distance[c][d]
                checks += 1
                if delta == 0:
                    neighbor = path[:i + 1] + list(reversed(path[i + 1:j + 1])) + path[j + 1:]
                    key = canonical(neighbor)
                    if key not in routes:
                        routes[key] = neighbor.copy()
                        root_by_key[key] = root_by_key[parent]
                        parent_by_key[key] = parent
                        queue.append(neighbor)
    trace = Trace(routes, checks, attempts, seed_keys, root_by_key, parent_by_key)
    _verify(starts, distance, budget, trace)
    return trace


def _verify(starts: list[list[int]], distance: list[list[int]], budget: int, trace: Trace) -> None:
    official = equal_plateau(starts, distance, budget)
    if (list(trace.routes) != list(official.routes) or trace.checks != official.checks
            or trace.attempts != official.attempts):
        raise RuntimeError("diagnostic Opt-B replay differs from the original search")


def discovered_components(trace: Trace, distance: list[list[int]]) -> list[int]:
    """Components among discovered best tours only; makes no claim about unseen tours."""
    parent = {key: key for key in trace.routes}

    def find(key: tuple[int, ...]) -> tuple[int, ...]:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for key, path in trace.routes.items():
        n = len(path)
        for i in range(n - 2):
            a, b = path[i], path[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c, d = path[j], path[(j + 1) % n]
                if distance[a][c] + distance[b][d] != distance[a][b] + distance[c][d]:
                    continue
                neighbor = canonical(path[:i + 1] + list(reversed(path[i + 1:j + 1])) + path[j + 1:])
                if neighbor in parent:
                    x, y = find(key), find(neighbor)
                    parent[x] = y
    counts: dict[tuple[int, ...], int] = {}
    for key in parent:
        root = find(key)
        counts[root] = counts.get(root, 0) + 1
    return sorted(counts.values(), reverse=True)
