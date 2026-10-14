"""Enumerate equal-length 2-opt neighbors of already converged best tours.

This is the exact Opt-B rule used in the 2026-10-14 A/B/C experiment. A
distance check is one candidate edge-pair delta evaluation; breadth-first
search and canonical cycle keys prevent revisits and loops.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


def canonical(route: list[int]) -> tuple[int, ...]:
    first = route.index(min(route))
    forward = route[first:] + route[:first]
    backward = [forward[0], *reversed(forward[1:])]
    return min(tuple(forward), tuple(backward))


def tour_length(route: list[int], distance: list[list[int]]) -> int:
    return sum(distance[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))


@dataclass
class SearchResult:
    routes: dict[tuple[int, ...], list[int]]
    best_length: int
    checks: int = 0
    discovered: int = 0
    attempts: int = 0


def _record(result: SearchResult, route: list[int], length: int) -> bool:
    key = canonical(route)
    if length < result.best_length:
        result.best_length = length
        result.routes.clear()
    if length == result.best_length and key not in result.routes:
        result.routes[key] = route.copy()
        result.discovered += 1
        return True
    return False


def equal_plateau(
    starting_routes: list[list[int]], distance: list[list[int]], budget: int,
) -> SearchResult:
    """Breadth-first search over zero-delta 2-opt neighbors of best tours."""
    if not starting_routes or budget < 1:
        raise ValueError("starting routes and positive budget required")
    optimum_so_far = tour_length(starting_routes[0], distance)
    result = SearchResult({}, optimum_so_far)
    queue: deque[list[int]] = deque()
    for route in starting_routes:
        if tour_length(route, distance) != optimum_so_far:
            raise ValueError("plateau starts must have the same length")
        if _record(result, route, optimum_so_far):
            queue.append(route)
    initial_keys = set(result.routes)
    while queue and result.checks < budget:
        path = queue.popleft()
        result.attempts += 1
        n = len(path)
        for i in range(n - 2):
            a, b = path[i], path[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                if result.checks == budget:
                    result.discovered = len(result.routes.keys() - initial_keys)
                    return result
                c, d = path[j], path[(j + 1) % n]
                delta = distance[a][c] + distance[b][d] - distance[a][b] - distance[c][d]
                result.checks += 1
                if delta == 0:
                    neighbor = path[: i + 1] + list(reversed(path[i + 1 : j + 1])) + path[j + 1 :]
                    if _record(result, neighbor, result.best_length):
                        queue.append(neighbor)
    result.discovered = len(result.routes.keys() - initial_keys)
    return result
