"""Replay a sampled prefix, replace one legal decision, then resume sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass
from time import perf_counter

import torch

from gb_2_maco_POMO.attention import build_city_features


@dataclass
class CaseRoutes:
    groups: dict[str, list[list[int]]]
    reference_traces: list[dict]
    fork_records: list[dict]
    timing: dict[str, float]


def _choice(logits, sample: bool):
    return torch.distributions.Categorical(logits=logits).sample() if sample else logits.argmax(dim=-1)


def _trace_step(ball_logits, city_logits, chosen_ball, chosen_city, mapping, current):
    """Record the actual pre-decision probabilities; no truth labels enter inference."""
    balls = torch.isfinite(ball_logits)
    cities = torch.isfinite(city_logits) & (mapping == chosen_ball)
    return {
        "current_city": int(current), "current_ball": int(mapping[current]),
        "ball_probabilities": {str(i): float(p) for i, p in enumerate(
            torch.softmax(ball_logits, -1).tolist()) if bool(balls[i])},
        "chosen_ball": int(chosen_ball),
        "city_probabilities": {str(i): float(p) for i, p in enumerate(
            torch.softmax(city_logits.masked_fill(~cities, -torch.inf), -1).tolist())
            if bool(cities[i])},
        "chosen_city": int(chosen_city),
        "cross_ball": bool(mapping[current] != mapping[chosen_city]),
    }


def _candidates(trace: list[dict], kind: str):
    """Prefer raw cross-ball links and the next two choices within the entered ball."""
    priority = set()
    for index, step in enumerate(trace):
        if step["cross_ball"]:
            priority.add(index)
            entered = step["chosen_ball"]
            remaining = 2
            for later in range(index + 1, len(trace)):
                if trace[later]["chosen_ball"] != entered:
                    break
                priority.add(later)
                remaining -= 1
                if not remaining:
                    break
    field = "city_probabilities" if kind == "city" else "ball_probabilities"
    eligible = [i for i, step in enumerate(trace) if len(step[field]) > 1]
    if kind == "ball":
        cross = [i for i in eligible if trace[i]["cross_ball"]]
        preferred = cross or [i for i in eligible if i in priority]
    else:
        preferred = [i for i in eligible if i in priority]
    return preferred or eligible


def _fork(policy, encoded, mapping, reference, trace, position, kind, seed):
    n = len(reference)
    prefix = reference[:position + 1]  # trace index 0 chooses route position 1
    first = torch.tensor([[reference[0]]], device=encoded.device)
    current = torch.tensor([[prefix[-1]]], device=encoded.device)
    available = torch.ones((1, 1, n), dtype=torch.bool, device=encoded.device)
    available[0, 0, prefix] = False
    ball_logits, city_logits = policy.action_logits(encoded, first, current, available, mapping)
    expected = trace[position]
    if expected["current_city"] != prefix[-1]:
        raise RuntimeError("reference trace does not match its route prefix")
    route = prefix.copy()
    # Fork RNG is isolated from the S0 stream and deterministic for this slot.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if kind == "city":
            chosen_ball = expected["chosen_ball"]
            mask = available & (mapping[:, None, :] == chosen_ball)
            mask[0, 0, expected["chosen_city"]] = False
            alternative = city_logits.masked_fill(~mask, -torch.inf)
            if not torch.isfinite(alternative).any():
                raise RuntimeError("city fork has no legal alternative")
            chosen_city = int(_choice(alternative, True).item())
        else:
            alternative = ball_logits.clone()
            alternative[0, 0, expected["chosen_ball"]] = -torch.inf
            if not torch.isfinite(alternative).any():
                raise RuntimeError("ball fork has no legal alternative")
            chosen_ball = int(_choice(alternative, True).item())
            city_mask = available & (mapping[:, None, :] == chosen_ball)
            chosen_city = int(_choice(city_logits.masked_fill(~city_mask, -torch.inf), True).item())
        if chosen_city == expected["chosen_city"] or kind == "city" and chosen_ball != expected["chosen_ball"]:
            raise RuntimeError("fork did not change the requested decision")
        route.append(chosen_city)
        available[0, 0, chosen_city] = False
        current[0, 0] = chosen_city
        while len(route) < n:
            city, _, _ = policy.choose_step(encoded, first, current, available, mapping, sample=True)
            selected = int(city.item())
            if not bool(available[0, 0, selected]):
                raise RuntimeError("fork continuation selected a visited city")
            route.append(selected)
            available[0, 0, selected] = False
            current[0, 0] = selected
    if sorted(route) != list(range(n)) or route[:position + 1] != prefix:
        raise RuntimeError("fork produced an illegal route or changed the prefix")
    return route, chosen_ball, chosen_city


@torch.inference_mode()
def generate_case(policy, prepared, seed: int) -> CaseRoutes:
    """Generate S0 exactly as historical repeat-major POMO, then fork four references."""
    device = next(policy.parameters()).device
    if device.type != "cpu":
        raise ValueError("historical S0 comparison requires CPU")
    n = len(prepared.cities)
    started = perf_counter()
    torch.manual_seed(seed)
    features = build_city_features(prepared.cities, prepared.granular_balls,
                                   prepared.city_to_ball).to(device)
    _, encoded = policy.encode(features)
    mapping = torch.tensor(prepared.city_to_ball, dtype=torch.long, device=device)[None, :]
    first = torch.arange(n, dtype=torch.long, device=device)[None, :]
    rounds = []
    common_seconds = 0.0
    traces = [[{"start": start, "reference_round": repeat, "steps": []}
               for start in range(n)] for repeat in range(4)]
    for repeat in range(9):
        route = torch.empty((1, n, n), dtype=torch.long, device=device)
        route[:, :, 0] = first
        current = first
        available = torch.ones((1, n, n), dtype=torch.bool, device=device)
        available.scatter_(2, first[..., None], False)
        for position in range(1, n):
            ball_logits, city_logits = policy.action_logits(encoded, first, current,
                                                            available, mapping)
            ball = _choice(ball_logits, repeat > 0)
            city_mask = available & (mapping[:, None, :] == ball[..., None])
            conditional = city_logits.masked_fill(~city_mask, -torch.inf)
            city = _choice(conditional, repeat > 0)
            if 1 <= repeat <= 4:
                for start in range(n):
                    traces[repeat - 1][start]["steps"].append(_trace_step(
                        ball_logits[0, start], city_logits[0, start], int(ball[0, start]),
                        int(city[0, start]), mapping[0], int(current[0, start])))
            route[:, :, position] = city
            available.scatter_(2, city[..., None], False)
            current = city
        actual = route[0].tolist()
        if any(sorted(item) != list(range(n)) for item in actual):
            raise RuntimeError("S0 produced an illegal tour")
        rounds.append(actual)
        if repeat == 4:
            common_seconds = perf_counter() - started
    groups = {"S0": [r for round_routes in rounds for r in round_routes]}
    s0_seconds = perf_counter() - started
    records = []
    forked = {kind: [[] for _ in range(4)] for kind in ("city", "ball")}
    fork_slot_seconds = {kind: [] for kind in ("city", "ball")}
    for kind in ("city", "ball"):
        for slot in range(4):
            slot_started = perf_counter()
            for start in range(n):
                selected = None
                attempts = []
                for offset in range(4):
                    ref = (slot + offset) % 4
                    trace = traces[ref][start]["steps"]
                    choices = _candidates(trace, kind)
                    if choices:
                        selection_seed = seed * 10_000_000 + start * 100_000 + slot * 1000 + ref * 10 + (kind == "ball")
                        position = random.Random(selection_seed).choice(choices)
                        fork_seed = selection_seed + 5_000_000
                        reference = rounds[ref + 1][start]
                        route, new_ball, new_city = _fork(
                            policy, encoded, mapping, reference, trace, position, kind, fork_seed)
                        selected = route
                        before = {frozenset((reference[i], reference[(i + 1) % n])) for i in range(n)}
                        after = {frozenset((route[i], route[(i + 1) % n])) for i in range(n)}
                        records.append({"kind": kind, "start": start, "slot": slot,
                            "reference_round": ref + 1, "step": position + 1,
                            "current_city": trace[position]["current_city"],
                            "old_ball": trace[position]["chosen_ball"], "new_ball": new_ball,
                            "old_city": trace[position]["chosen_city"], "new_city": new_city,
                            "cross_ball": trace[position]["cross_ball"],
                            "selection_seed": selection_seed, "fork_seed": fork_seed,
                            "status": "forked", "reason": "", "edges_changed": len(after - before),
                            "reference_route": reference, "route": route})
                        break
                    attempts.append(ref + 1)
                if selected is None:
                    # Preserve the route count without inventing an illegal action.
                    fallback_seed = seed * 10_000_000 + start * 100_000 + slot * 1000 + (kind == "ball") + 9_000_000
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(fallback_seed)
                        initial = torch.tensor([[start]], device=device)
                        current = initial.clone()
                        available = torch.ones((1, 1, n), dtype=torch.bool, device=device)
                        available[0, 0, start] = False
                        selected = [start]
                        while len(selected) < n:
                            city, _, _ = policy.choose_step(encoded, initial, current, available, mapping, sample=True)
                            value = int(city.item())
                            selected.append(value)
                            available[0, 0, value] = False
                            current[0, 0] = value
                    records.append({"kind": kind, "start": start, "slot": slot,
                        "reference_round": "", "step": "", "current_city": "",
                        "old_ball": "", "new_ball": "", "old_city": "", "new_city": "",
                        "cross_ball": "", "selection_seed": "", "fork_seed": fallback_seed,
                        "status": "fallback", "reason": f"no legal alternative in references {attempts}",
                        "edges_changed": "", "reference_route": [], "route": selected})
                if sorted(selected) != list(range(n)):
                    raise RuntimeError("fallback produced an illegal route")
                forked[kind][slot].append(selected)
            fork_slot_seconds[kind].append(perf_counter() - slot_started)
    common = [*rounds[:5]]
    groups["S2"] = [r for block in [*common, *forked["city"]] for r in block]
    groups["S3"] = [r for block in [*common, *forked["ball"]] for r in block]
    groups["S4"] = [r for block in [*common, *forked["city"][:2], *forked["ball"][:2]] for r in block]
    if any(len(value) != 9 * n for value in groups.values()):
        raise RuntimeError("each group must contain exactly nine routes per start")
    for ref in range(4):
        for start in range(n):
            traces[ref][start]["route"] = rounds[ref + 1][start]
            traces[ref][start]["mapping"] = prepared.city_to_ball
            traces[ref][start]["visited_before_each_step"] = [
                rounds[ref + 1][start][:i] for i in range(1, n)]
    return CaseRoutes(groups, [entry for group in traces for entry in group], records,
                      {"s0_seconds": s0_seconds, "common_seconds": common_seconds,
                       "city_slot_seconds": fork_slot_seconds["city"],
                       "ball_slot_seconds": fork_slot_seconds["ball"],
                       "fork_seconds": sum(map(sum, fork_slot_seconds.values()))})
