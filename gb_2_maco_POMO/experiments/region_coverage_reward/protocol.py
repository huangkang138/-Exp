"""Write source and input hashes for the independent experiment directory."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import torch

from gb_2_maco_POMO.experiments.hierarchical.original25 import CHECKPOINT as SOFT

from .train import DATA_ROOT, INITIAL, PROJECT


ROOT = PROJECT / "gb_2_maco_POMO/results/region_coverage_reward"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    module = Path(__file__).resolve().parent
    files = ("references.py", "reward.py", "validation.py", "train.py",
             "evaluate25.py", "run_all.py", "run_tests.py", "aggregate.py",
             "calibrate.py", "augment_validation.py")
    protocol = {
        "project": str(PROJECT), "dataset": str(DATA_ROOT),
        "dataset_manifest_sha256": sha(DATA_ROOT / "manifest.csv"),
        "initial_checkpoint": str(INITIAL), "initial_checkpoint_sha256": sha(INITIAL),
        "fixed_soft_checkpoint": str(SOFT), "fixed_soft_checkpoint_sha256": sha(SOFT),
        "source_sha256": {name: sha(module / name) for name in files},
        "groups": {"G0": "no_bias_raw_length", "Q": "no_bias_two_opt_quality",
                   "G1": "no_bias_quality_plus_observed_region",
                   "G2": "learned_bias_raw_length", "G3": "learned_bias_quality_plus_observed_region"},
        "training_seeds": [42, 43, 44], "inference_seeds": [42, 43, 44, 45, 46],
        "training_instances": 500, "validation_instances": 100,
        "certified_training_instances": 282, "certified_validation_instances": 56,
        "training_epochs_max": 5, "early_stop_patience": 2,
        "batch_size": 8, "optimizer": "Adam", "learning_rate": 0.0001,
        "training_candidate_2opt_max": 16, "training_optb_entrances_max": 8,
        "training_optb_pair_checks_max": 8000,
        "validation_samples_per_start": 8, "validation_optb_pair_checks": 50000,
        "test_all_starts": True, "test_argmax_rounds": 1,
        "test_samples_per_start": 8, "test_optb_pair_checks": 50000,
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "training_test_separation": "MSTSP1-25 truth is loaded only by final test wrapper after local search",
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "experiment_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    print(ROOT / "experiment_protocol.json")


if __name__ == "__main__":
    main()
