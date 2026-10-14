"""Deterministic 2-opt refinement using the benchmark's integer distance matrix."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TwoOptResult:
    route: list[int]
    length: int
    moves: int
    pair_checks: int


def two_opt(route: list[int], distance: list[list[int]]) -> TwoOptResult:
    """Apply best-improvement edge swaps until no strictly shorter tour remains.

    The start city stays fixed. Every accepted swap is checked against the same
    closed-tour distance definition used by POMO, so integer rounding is identical.
    """
    city_count = len(route)
    if city_count < 2 or len(set(route)) != city_count:
        raise ValueError("route must contain distinct cities")
    if len(distance) != city_count or any(len(row) != city_count for row in distance):
        raise ValueError("distance matrix shape does not match route")
    path = route.copy()
    length = sum(distance[path[i]][path[(i + 1) % city_count]] for i in range(city_count))
    moves = 0
    pair_checks = 0
    while True:
        best_delta = 0
        best_pair: tuple[int, int] | None = None
        for i in range(city_count - 2):
            a, b = path[i], path[i + 1]
            for j in range(i + 2, city_count):
                if i == 0 and j == city_count - 1:
                    continue  # Reversing the whole cycle only changes its direction.
                c, d = path[j], path[(j + 1) % city_count]
                delta = distance[a][c] + distance[b][d] - distance[a][b] - distance[c][d]
                pair_checks += 1
                if delta < best_delta:
                    best_delta, best_pair = delta, (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        path[i + 1 : j + 1] = reversed(path[i + 1 : j + 1])
        length += best_delta
        moves += 1
    verified = sum(distance[path[i]][path[(i + 1) % city_count]] for i in range(city_count))
    if verified != length or set(path) != set(route):
        raise RuntimeError("2-opt produced an inconsistent tour")
    return TwoOptResult(path, length, moves, pair_checks)
