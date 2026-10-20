"""Small, validated inference-only experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("config.toml")


@dataclass(frozen=True)
class Settings:
    beam_width: int
    ball_top_k: int
    entry_top_k: int
    local_beam_width: int
    lookahead_depth: int
    quality_slack: float
    routes_per_start: int
    budget_multipliers: tuple[int, ...]
    validation_device: str
    formal_device: str
    seed: int
    optb_checks_per_case: int


def load(path: Path = DEFAULT_CONFIG) -> Settings:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    search, experiment = data["search"], data["experiment"]
    config = Settings(**{key: search[key] for key in (
        "beam_width", "ball_top_k", "entry_top_k", "local_beam_width",
        "lookahead_depth", "quality_slack", "routes_per_start")},
        budget_multipliers=tuple(search["budget_multipliers"]),
        **experiment)
    if any(getattr(config, name) <= 0 for name in (
        "beam_width", "ball_top_k", "entry_top_k", "local_beam_width",
        "lookahead_depth", "routes_per_start", "optb_checks_per_case")):
        raise ValueError("search widths, depth, routes and Opt-B checks must be positive")
    if config.routes_per_start != config.beam_width + 1:
        raise ValueError("routes_per_start must equal greedy route + beam_width")
    if config.quality_slack < 0 or not config.budget_multipliers or any(
        value <= 0 for value in config.budget_multipliers):
        raise ValueError("invalid quality slack or decode budgets")
    if config.validation_device not in ("cpu", "cuda") or config.formal_device not in ("cpu", "cuda"):
        raise ValueError("validation_device and formal_device must be cpu or cuda")
    return config
