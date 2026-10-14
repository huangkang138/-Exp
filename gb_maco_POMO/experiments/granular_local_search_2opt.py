"""Post-inference 2-opt restricted to swaps that touch a cross-ball edge.

This isolates the contribution of a granular-ball edge filter. It is not the
stateful GB-MACO local search, which also uses archive key edges and FES.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GranularTwoOptResult:
    route: list[int]
    length: int
    moves: int
    pair_checks: int
    filtered_pairs: int


def granular_two_opt(
    route: list[int], distance: list[list[int]], city_to_ball: list[int],
) -> GranularTwoOptResult:
    """Apply best-improvement swaps while checking only cross-ball cut pairs.

    A pair is eligible when either of its two removed edges crosses a granular
    ball boundary. All other search and distance rules match ordinary 2-opt.
    """
    city_count = len(route)
    if city_count < 2 or set(route) != set(range(city_count)):
        raise ValueError("route must be a permutation of city indices")
    if len(distance) != city_count or any(len(row) != city_count for row in distance):
        raise ValueError("distance matrix shape does not match route")
    if len(city_to_ball) != city_count:
        raise ValueError("city-to-ball mapping does not match route")

    path = route.copy()
    length = sum(distance[path[i]][path[(i + 1) % city_count]] for i in range(city_count))
    moves = pair_checks = filtered_pairs = 0
    while True:
        best_delta = 0
        best_pair: tuple[int, int] | None = None
        for i in range(city_count - 2):
            a, b = path[i], path[i + 1]
            for j in range(i + 2, city_count):
                if i == 0 and j == city_count - 1:
                    continue
                c, d = path[j], path[(j + 1) % city_count]
                if city_to_ball[a] == city_to_ball[b] and city_to_ball[c] == city_to_ball[d]:
                    filtered_pairs += 1
                    continue
                pair_checks += 1
                delta = distance[a][c] + distance[b][d] - distance[a][b] - distance[c][d]
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
        raise RuntimeError("granular 2-opt produced an inconsistent tour")
    return GranularTwoOptResult(path, length, moves, pair_checks, filtered_pairs)
