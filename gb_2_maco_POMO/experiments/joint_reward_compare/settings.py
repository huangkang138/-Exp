"""Separate TOML files for identical J1/J2 training and the J2 reward."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from gb_2_maco_POMO.experiments.joint_ball_city.settings import Settings, load

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RewardSettings:
    path: Path
    two_opt_max_pair_checks_per_route: int
    quality_ratio: float
    raw_advantage_weight: float
    two_opt_advantage_weight: float
    diversity_advantage_weight: float
    cross_ball_novelty_weight: float
    within_ball_novelty_weight: float


def load_pair(config_dir: Path = HERE) -> tuple[Settings, Settings, RewardSettings]:
    j1 = load(config_dir / "config_j1.toml")
    j2 = load(config_dir / "config_j2.toml")
    for field in Settings.__dataclass_fields__:
        if field not in ("config_path", "output_dir") and getattr(j1, field) != getattr(j2, field):
            raise ValueError(f"J1/J2 protocol differs at {field}")
    if j1.output_dir == j2.output_dir or j1.validation_interval != 2:
        raise ValueError("J1/J2 require separate outputs and validation interval 2")
    path = (config_dir / "reward.toml").resolve()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    fields = set(RewardSettings.__dataclass_fields__) - {"path"}
    if set(raw) != fields:
        raise ValueError(f"reward keys differ: missing={fields-set(raw)}, extra={set(raw)-fields}")
    reward = RewardSettings(path, **raw)
    if (reward.two_opt_max_pair_checks_per_route < 1
            or reward.quality_ratio != 1.005
            or (reward.raw_advantage_weight, reward.two_opt_advantage_weight,
                reward.diversity_advantage_weight) != (0.5, 0.5, 0.1)
            or (reward.cross_ball_novelty_weight,
                reward.within_ball_novelty_weight) != (0.5, 0.5)):
        raise ValueError("J2 reward differs from the requested fixed coefficients")
    return j1, j2, reward
