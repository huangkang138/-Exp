"""Validated experiment settings; paths remain relative to the project root."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

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
    auxiliary_lambda: float
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
    initial_checkpoint = (PROJECT / raw.pop("initial_checkpoint")).resolve()
    output_dir = (PROJECT / raw.pop("output_dir")).resolve()
    config = Settings(path, data_root, initial_checkpoint, output_dir, **raw)
    if (config.epochs != 10 or config.validation_interval != 2 or config.seed != 42
            or config.auxiliary_lambda != 0.3 or config.device not in ("cpu", "cuda")
            or config.batch_size <= 0 or config.micro_batch_size <= 0
            or config.batch_size < config.micro_batch_size
            or config.learning_rate <= 0 or config.validation_samples_per_start != 8
            or config.validation_optb_checks != 50000
            or config.multi_quality_gap_tolerance_pct < 0
            or config.multi_allowed_certified_hit_loss < 0):
        raise ValueError("dual-reward protocol differs from the approved experiment")
    if data_root != (PROJECT / "benachmark_Train_MATSP_Multiopt_500").resolve():
        raise ValueError("dual-reward training dataset changed")
    from gb_2_maco_POMO.experiments.region_supervised_pomo.config import INITIAL
    if initial_checkpoint != INITIAL.resolve():
        raise ValueError("A0 initialization checkpoint changed")
    return config
