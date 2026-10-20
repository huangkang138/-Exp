"""Only the joint experiment's TOML controls its run."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from gb_2_maco_POMO.experiments.region_supervised_pomo.config import INITIAL

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("config.toml")


@dataclass(frozen=True)
class Settings:
    config_path: Path
    data_root: Path
    initial_checkpoint: Path
    output_dir: Path
    epochs: int
    validation_interval: int
    batch_size: int
    micro_batch_size: int
    learning_rate: float
    seed: int
    device: str
    validation_samples_per_start: int
    validation_optb_checks: int
    multi_quality_gap_tolerance_pct: float
    multi_allowed_certified_hit_loss: int


def load(path: Path = DEFAULT_CONFIG) -> Settings:
    path = path.resolve()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    required = set(Settings.__dataclass_fields__) - {"config_path"}
    if set(raw) != required:
        raise ValueError(f"config keys differ: missing={required - set(raw)}, extra={set(raw) - required}")
    data_root = (PROJECT / raw.pop("data_root")).resolve()
    initial = (PROJECT / raw.pop("initial_checkpoint")).resolve()
    output = (PROJECT / raw.pop("output_dir")).resolve()
    config = Settings(path, data_root, initial, output, **raw)
    if config.validation_interval not in (1, 2):
        raise ValueError("validation_interval must be 1 or 2")
    if (data_root != (PROJECT / "benachmark_Train_MATSP_Multiopt_500").resolve()
            or initial != INITIAL.resolve() or config.epochs != 10
            or config.seed != 42
            or config.device != "cuda" or config.batch_size != 64
            or config.micro_batch_size != 8 or config.learning_rate != 1e-4
            or config.validation_samples_per_start != 8
            or config.validation_optb_checks != 50000
            or config.multi_quality_gap_tolerance_pct != 0.5
            or config.multi_allowed_certified_hit_loss != 1):
        raise ValueError("joint experiment differs from the specified fixed protocol")
    return config
