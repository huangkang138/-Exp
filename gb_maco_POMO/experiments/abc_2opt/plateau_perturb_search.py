"""Bounded searches around tours already converged by ordinary 2-opt.

One distance check means evaluating one 2-opt edge-pair delta. The caller owns
the shared per-instance ceiling; a partial perturbation descent is discarded.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random


def tour_length(route: list[int], distance: list[list[int]]) -> int:
    return sum(distance[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))


def canonical(route: list[int]) -> tuple[int, ...]:
    first = route.index(min(route))
    forward = route[first:] + route[:first]
    backward = [forward[0], *reversed(forward[1:])]
    return min(tuple(forward), tuple(backward))


@dataclass
class SearchResult:
    routes: dict[tuple[int, ...], list[int]]
    best_length: int
    checks: int = 0
    discovered: int = 0
    attempts: int = 0
    completed_descents: int = 0


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


def double_bridge(route: list[int], rng: Random) -> list[int]:
    """Change four cycle connections while preserving segment orientations."""
    n = len(route)
    if n < 8:
        raise ValueError("double bridge needs at least eight cities")
    a, b, c, d = sorted(rng.sample(range(1, n), 4))
    return route[:a] + route[c:d] + route[b:c] + route[a:b] + route[d:]


def _bounded_two_opt(
    route: list[int], distance: list[list[int]], allowance: int,
) -> tuple[list[int] | None, int, int]:
    """Return a fully converged tour, or None if the pair budget runs out."""
    path = route.copy()
    n = len(path)
    length = tour_length(path, distance)
    checks = 0
    while True:
        best_delta = 0
        best_pair = None
        for i in range(n - 2):
            a, b = path[i], path[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                if checks == allowance:
                    return None, length, checks
                c, d = path[j], path[(j + 1) % n]
                delta = distance[a][c] + distance[b][d] - distance[a][b] - distance[c][d]
                checks += 1
                if delta < best_delta:
                    best_delta, best_pair = delta, (i, j)
        if best_pair is None:
            if tour_length(path, distance) != length:
                raise RuntimeError("inconsistent 2-opt length")
            return path, length, checks
        i, j = best_pair
        path[i + 1 : j + 1] = reversed(path[i + 1 : j + 1])
        length += best_delta


def perturb_and_descend(
    starting_routes: list[list[int]], distance: list[list[int]], budget: int,
    seed: int,
) -> SearchResult:
    """Explore new basins from best tours using seeded double-bridge kicks."""
    if not starting_routes or budget < 1:
        raise ValueError("starting routes and positive budget required")
    best = tour_length(starting_routes[0], distance)
    result = SearchResult({}, best)
    for route in starting_routes:
        if tour_length(route, distance) != best:
            raise ValueError("perturbation starts must have the same length")
        _record(result, route, best)
    initial_keys = set(result.routes)
    rng = Random(seed)
    # Every completed trial consumes at least one full 2-opt scan. Repeated
    # trials may revisit a basin; their work still counts against the ceiling.
    while result.checks < budget:
        pool = list(result.routes.values())
        source = pool[rng.randrange(len(pool))]
        candidate = double_bridge(source, rng)
        result.attempts += 1
        refined, length, used = _bounded_two_opt(candidate, distance, budget - result.checks)
        result.checks += used
        if refined is None:
            break
        result.completed_descents += 1
        _record(result, refined, length)
    result.discovered = len(result.routes.keys() - initial_keys)
    return result
