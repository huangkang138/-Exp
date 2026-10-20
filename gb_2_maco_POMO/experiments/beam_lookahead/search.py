"""Deterministic B1/B2 search over the unchanged A0 two-stage policy.

One decode step means one route state passed through ``action_logits``.
The counter includes local beams and greedy completions, even when the latter
are used only to score a prefix. It excludes the shared encoder pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from gb_2_maco_POMO.attention import build_city_features
from gb_2_maco_POMO.experiments.shared.opt_b_search import tour_length

from .settings import Settings


@dataclass(frozen=True)
class Prefix:
    cities: tuple[int, ...]
    logp: float
    estimate: int | None = None
    structure: frozenset[tuple[int, int, int]] = frozenset()


@dataclass
class SearchResult:
    routes: list[list[int]]
    decode_steps: int
    inference_seconds: float
    truncated: bool
    lookahead_rounds: int


class BudgetSpent(RuntimeError):
    pass


class Decoder:
    def __init__(self, policy, prepared, budget: int | None):
        self.policy = policy
        self.device = next(policy.parameters()).device
        self.n = len(prepared.cities)
        self.mapping = tuple(prepared.city_to_ball)
        self.distance = prepared.distance_matrix
        features = build_city_features(prepared.cities, prepared.granular_balls,
                                       prepared.city_to_ball).to(self.device)
        _, self.encoded = policy.encode(features)
        self.mapping_tensor = torch.tensor([self.mapping], device=self.device)
        self.budget = budget
        self.steps = 0
        self.greedy_choice: dict[tuple[int, ...], int] = {}

    def actions(self, prefix: Prefix) -> list[tuple[int, int, float]]:
        if self.budget is not None and self.steps >= self.budget:
            raise BudgetSpent
        self.steps += 1
        used = set(prefix.cities)
        available = torch.tensor([[[city not in used for city in range(self.n)]]],
                                 dtype=torch.bool, device=self.device)
        first = torch.tensor([[prefix.cities[0]]], device=self.device)
        current = torch.tensor([[prefix.cities[-1]]], device=self.device)
        ball_logits, city_logits = self.policy.action_logits(
            self.encoded, first, current, available, self.mapping_tensor)
        greedy_ball = int(ball_logits[0, 0].argmax().item())
        greedy_mask = available[0, 0] & (self.mapping_tensor[0] == greedy_ball)
        self.greedy_choice[prefix.cities] = int(city_logits[0, 0].masked_fill(
            ~greedy_mask, -torch.inf).argmax().item())
        ball_logp = torch.log_softmax(ball_logits[0, 0], dim=-1)
        results = []
        # Conditional city normalization is over all legal cities in this ball.
        # Top-k filtering is applied afterward; it never changes A0 log-probability.
        for ball in range(ball_logp.numel()):
            if not torch.isfinite(ball_logp[ball]):
                continue
            legal = available[0, 0] & (self.mapping_tensor[0] == ball)
            conditional = city_logits[0, 0].masked_fill(~legal, -torch.inf)
            city_logp = torch.log_softmax(conditional, dim=-1)
            for city in torch.nonzero(legal).flatten().tolist():
                results.append((ball, city, float((ball_logp[ball] + city_logp[city]).item())))
        return sorted(results, key=lambda item: (-item[2], item[1]))

    def greedy(self, prefix: Prefix) -> Prefix:
        state = prefix
        while len(state.cities) < self.n:
            actions = self.actions(state)
            city = self.greedy_choice[state.cities]
            logp = next(item[2] for item in actions if item[1] == city)
            state = Prefix(state.cities + (city,), state.logp + logp)
        return state

    def length(self, route: tuple[int, ...]) -> int:
        return int(tour_length(route, self.distance))

    def edges(self, route: tuple[int, ...]) -> frozenset[tuple[int, int, int]]:
        result = set()
        for left, right in zip(route, route[1:] + route[:1]):
            kind = int(self.mapping[left] != self.mapping[right])
            result.add((kind, min(left, right), max(left, right)))
        return frozenset(result)


def _select_b2(candidates: list[Prefix], width: int, slack: float) -> list[Prefix]:
    """3 probability + 3 completed distance + 2 structural slots for width 8."""
    unique = list({part.cities: part for part in candidates}.values())
    chosen: list[Prefix] = []

    def add(items, limit):
        for part in items:
            if part not in chosen:
                chosen.append(part)
                if len(chosen) >= limit:
                    break

    prob_count, distance_count = (3, 3) if width == 8 else (width * 3 // 8, width * 3 // 8)
    add(sorted(unique, key=lambda p: (-p.logp, p.cities)), prob_count)
    add(sorted(unique, key=lambda p: (p.estimate, p.cities)), min(width, prob_count + distance_count))
    best = min(p.estimate for p in unique)
    quality = [p for p in unique if p.estimate <= best * (1 + slack) and p not in chosen]
    while quality and len(chosen) < width:
        item = max(quality, key=lambda p: (
            min((len(p.structure ^ other.structure) for other in chosen), default=len(p.structure)),
            -p.estimate, p.cities))
        chosen.append(item)
        quality.remove(item)
    add(sorted(unique, key=lambda p: (p.estimate, p.cities)), width)
    return chosen


def _lookahead(decoder: Decoder, prefix: Prefix, first_city: int, first_logp: float,
               config: Settings, quota: int | None) -> list[Prefix]:
    """Try legal within-ball and cross-ball continuations until first cross.

    A local branch may exit before consuming its first ball, and later global
    search may re-enter it. Each surviving fragment gets an A0 greedy suffix.
    """
    entered_ball = decoder.mapping[first_city]
    beam = [Prefix(prefix.cities + (first_city,), prefix.logp + first_logp)]
    finished: list[Prefix] = []
    start_steps = decoder.steps
    for _ in range(config.lookahead_depth - 1):
        expanded = []
        for part in beam:
            # In compute-budget mode reserve a full greedy completion. A
            # partial local search still evaluates the same legal A0 policy.
            if quota is not None and decoder.steps - start_steps >= quota - decoder.n:
                expanded.append(part)
                continue
            if len(part.cities) == decoder.n:
                finished.append(part)
                continue
            for _, city, logp in decoder.actions(part):
                new = Prefix(part.cities + (city,), part.logp + logp)
                if decoder.mapping[city] != entered_ball:
                    finished.append(new)  # first connection into the next ball
                else:
                    expanded.append(new)
        if not expanded:
            beam = []
            break
        beam = sorted(expanded, key=lambda p: (-p.logp, p.cities))[:config.local_beam_width]
    finished.extend(beam)
    scored = []
    if quota is not None:
        finished = sorted(set(finished), key=lambda p: (-p.logp, p.cities))[:1]
    for part in finished:
        complete = decoder.greedy(part)
        scored.append((decoder.length(complete.cities), complete.cities, part))
    choices = []
    for length, route, fragment in sorted(scored, key=lambda item: (item[0], item[1])):
        if fragment.cities not in {item.cities for item in choices}:
            choices.append(Prefix(fragment.cities, fragment.logp, length, decoder.edges(route)))
        if len(choices) == (1 if quota is not None else 3):
            break
    return choices


@torch.inference_mode()
def generate(policy, prepared, mode: str, config: Settings, *, start: int,
             budget: int | None = None, width: int | None = None) -> SearchResult:
    if mode not in ("B1", "B2"):
        raise ValueError(mode)
    started = perf_counter()
    decoder = Decoder(policy, prepared, budget)
    beam_width = width or config.beam_width
    initial = Prefix((start,), 0.0)
    greedy = decoder.greedy(initial).cities
    beam = [initial]
    final_pool: list[Prefix] = []
    rejected: list[Prefix] = []
    truncated = False
    lookahead_rounds = 0
    for _ in range(1, decoder.n):
        if all(len(part.cities) == decoder.n for part in beam):
            break
        candidates = []
        # Full lookahead runs in equal-output mode. Under a hard decode cap,
        # reserve ordinary beam continuation and final route completions;
        # spend surplus on complete top-2/top-3 lookahead rounds only.
        remaining = max(decoder.n - len(part.cities) for part in beam)
        reserve = beam_width * remaining + decoder.n
        expected_candidates = len(beam) * config.ball_top_k * config.entry_top_k
        surplus = (None if budget is None else budget - decoder.steps - reserve)
        use_lookahead = mode == "B2" and (surplus is None or surplus >= expected_candidates * decoder.n)
        quota = None if surplus is None else max(decoder.n, surplus // max(expected_candidates, 1))
        try:
            for part in beam:
                if len(part.cities) == decoder.n:
                    candidates.append(part)
                    continue
                actions = decoder.actions(part)
                if mode == "B2":
                    selected_balls = []
                    filtered = []
                    for ball, city, logp in actions:
                        if ball not in selected_balls:
                            if len(selected_balls) >= config.ball_top_k:
                                continue
                            selected_balls.append(ball)
                        if sum(item[0] == ball for item in filtered) < config.entry_top_k:
                            filtered.append((ball, city, logp))
                    actions = filtered
                for _, city, logp in actions:
                    next_prefix = Prefix(part.cities + (city,), part.logp + logp)
                    if use_lookahead:
                        candidates.extend(_lookahead(decoder, part, city, logp, config, quota))
                    else:
                        candidates.append(next_prefix)
        except BudgetSpent:
            truncated = True
            break
        if not candidates:
            break
        final_pool = candidates
        if use_lookahead:
            lookahead_rounds += 1
        beam = (sorted(candidates, key=lambda p: (-p.logp, p.cities))[:beam_width]
                if not use_lookahead else _select_b2(candidates, beam_width, config.quality_slack))
        rejected.extend(part for part in candidates if part not in beam)
    routes = [list(greedy)]
    # Complete only as many beam prefixes as the remaining decode budget allows.
    extras = (sorted(final_pool, key=lambda p: (-p.logp, p.cities)) if
              not final_pool or final_pool[0].estimate is None else
              sorted(final_pool, key=lambda p: (p.estimate, -p.logp, p.cities)))
    # The final beam can contain the greedy route. Rejected prefixes provide
    # another legal candidate so both groups can still submit nine tours.
    alternatives = extras + sorted(rejected, key=lambda p: (-len(p.cities), -p.logp, p.cities))
    ordered = beam + [part for part in alternatives if part not in beam]
    for part in ordered:
        if len(routes) >= config.routes_per_start:
            break
        try:
            complete = decoder.greedy(part).cities
        except BudgetSpent:
            truncated = True
            break
        if complete != greedy and list(complete) not in routes:
            routes.append(list(complete))
    if any(sorted(route) != list(range(decoder.n)) or route[0] != start for route in routes):
        raise RuntimeError("search produced an illegal route")
    return SearchResult(routes, decoder.steps, perf_counter() - started,
                        truncated, lookahead_rounds)
