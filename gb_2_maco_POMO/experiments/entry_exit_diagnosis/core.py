"""Closed-tour geometry and mutually exclusive edge categories for offline diagnosis."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median


def edges(route):
    return {tuple(sorted((route[i], route[(i + 1) % len(route)]))) for i in range(len(route))}


def canonical(route):
    first = route.index(min(route))
    path = route[first:] + route[:first]
    return min(tuple(path), tuple([path[0], *reversed(path[1:])]))


def spatial_labels(cities):
    """Reuse the project's 3 x median-MST-edge gap rule; return city labels too."""
    n = len(cities)
    distances = [[math.hypot(a.x - b.x, a.y - b.y) for b in cities] for a in cities]
    connected = {0}
    nearest = [(distances[0][j], 0, j) for j in range(n)]
    mst = []
    while len(connected) < n:
        length, left, right = min((nearest[j] for j in range(n) if j not in connected),
                                  key=lambda item: (item[0], item[1], item[2]))
        connected.add(right)
        mst.append((length, left, right))
        for j in range(n):
            if j not in connected and (distances[right][j], right, j) < nearest[j]:
                nearest[j] = (distances[right][j], right, j)
    threshold = 3 * median(length for length, _, _ in mst)
    parent = list(range(n))

    def find(city):
        while parent[city] != city:
            parent[city] = parent[parent[city]]
            city = parent[city]
        return city

    for length, a, b in mst:
        if length <= threshold:
            parent[find(a)] = find(b)
    components = defaultdict(list)
    for city in range(n):
        components[find(city)].append(city)
    groups = sorted((group for group in components.values() if len(group) >= 3), key=min)
    labels = [-1] * n
    for label, group in enumerate(groups):
        for city in group:
            labels[city] = label
    return labels, groups, threshold


def segments(route, labels):
    """Treat each circular contiguous visit as a separate segment, including revisits."""
    n = len(route)
    starts = [i for i in range(n) if labels[route[i - 1]] != labels[route[i]]]
    if not starts:
        return [{"group": labels[route[0]], "entry": route[0], "exit": route[-1],
                 "cities": list(route), "previous_group": None, "next_group": None}]
    first = starts[0]
    rotated = route[first:] + route[:first]
    result = []
    current = [rotated[0]]
    for city in rotated[1:]:
        if labels[city] == labels[current[-1]]:
            current.append(city)
        else:
            result.append(current)
            current = [city]
    result.append(current)
    return [{"group": labels[part[0]], "entry": part[0], "exit": part[-1],
             "cities": part, "previous_group": labels[result[i - 1][-1]],
             "next_group": labels[result[(i + 1) % len(result)][0]]}
            for i, part in enumerate(result)]


def cyclic_signature(values):
    if len(values) == 1:
        return tuple(values)
    return min(tuple(values[i:] + values[:i]) for i in range(len(values)))


def order_signature(route, labels):
    groups = [part["group"] for part in segments(route, labels)]
    return min(cyclic_signature(groups), cyclic_signature(list(reversed(groups))))


def edge_classes(route, balls):
    """A crosses a ball, B touches a cross-ball endpoint, C is deeper inside."""
    n = len(route)
    interface_cities = set()
    for i in range(n):
        a, b = route[i], route[(i + 1) % n]
        if balls[a] != balls[b]:
            interface_cities.update((a, b))
    result = {}
    for i in range(n):
        a, b = route[i], route[(i + 1) % n]
        kind = "A" if balls[a] != balls[b] else ("B" if a in interface_cities or b in interface_cities else "C")
        result[tuple(sorted((a, b)))] = kind
    return result


def compare(failed, optimal, balls):
    left, right = edge_classes(failed, balls), edge_classes(optimal, balls)
    removed, added = set(left) - set(right), set(right) - set(left)
    counts = {f"remove_{kind}": sum(left[edge] == kind for edge in removed) for kind in "ABC"}
    counts.update({f"add_{kind}": sum(right[edge] == kind for edge in added) for kind in "ABC"})
    total = len(removed) + len(added)
    counts.update({"wrong_edges": total, "edge_distance": len(removed),
                   "interface_error_ratio": sum(counts[f"{side}_{kind}"] for side in ("remove", "add")
                                                for kind in "AB") / total if total else 0,
                   "normal_interface_ratio": sum(kind in "AB" for kind in left.values()) / len(left)})
    return counts, removed, added, left, right


def segment_comparison(failed, optimal, balls):
    fs = segments(failed, balls)
    if order_signature(failed, balls) != order_signature(optimal, balls):
        return "ball_order_differs", fs, segments(optimal, balls)
    # Orient the optimum so the circular group order matches the failed route.
    expected = [part["group"] for part in fs]
    for oriented in (optimal, list(reversed(optimal))):
        n = len(oriented)
        for offset in range(n):
            trial = oriented[offset:] + oriented[:offset]
            os = segments(trial, balls)
            if [part["group"] for part in os] == expected:
                counts = defaultdict(int)
                for a, b in zip(fs, os):
                    if a["entry"] == b["entry"] and a["exit"] == b["exit"]:
                        kind = "same_interface_internal_diff" if a["cities"] != b["cities"] else "same_segment"
                    elif a["entry"] == b["entry"]:
                        kind = "same_entry_different_exit"
                    elif a["exit"] == b["exit"]:
                        kind = "different_entry_same_exit"
                    else:
                        kind = "both_interface_cities_differ"
                    counts[kind] += 1
                return dict(counts), fs, os
    return "ball_order_differs", fs, segments(optimal, balls)
