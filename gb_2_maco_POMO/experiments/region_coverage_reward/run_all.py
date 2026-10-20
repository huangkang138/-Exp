"""Resume-safe orchestration; pre-run all five seed-42 groups before full runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train import GROUPS, PROJECT, run


ROOT = PROJECT / "gb_2_maco_POMO/results/region_coverage_reward"


def progress(path: Path):
    result = path / "train_result.json"
    if result.is_file():
        state = json.loads(result.read_text(encoding="utf-8"))
        return int(state["completed_epochs"]), bool(state["stopped_early"])
    return 0, False


def execute(phase):
    ROOT.mkdir(parents=True, exist_ok=True)
    jobs = ([(group, 42, 1) for group in GROUPS] if phase == "prerun"
            else [(group, seed, 5) for seed in (42, 43, 44) for group in GROUPS])
    for group, seed, limit in jobs:
        output = ROOT / group / f"train_seed_{seed}"
        finished, stopped = progress(output)
        if finished >= limit or (phase == "full" and stopped):
            print(f"skip {group} seed={seed}: completed {finished}", flush=True)
            continue
        if phase == "full" and seed == 42 and not (output / "checkpoints/last.pt").is_file():
            raise RuntimeError("seed-42 pre-run must finish for all groups before full continuation")
        if output.exists() and not (output / "checkpoints/last.pt").is_file():
            raise RuntimeError(f"incomplete output requires diagnosis: {output}")
        baseline = ROOT / "G0" / f"train_seed_{seed}"
        if group != "G0" and not (baseline / "checkpoints/epoch_000.pt").is_file():
            raise RuntimeError(f"G0 baseline missing for seed {seed}")
        print(f"start {group} seed={seed} through epoch {limit}, resume={output.exists()}", flush=True)
        run(group, seed, output, stop_after=limit, resume=output.exists(),
            baseline_dir=baseline if group != "G0" else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prerun", "full"), required=True)
    args = parser.parse_args()
    execute(args.phase)


if __name__ == "__main__":
    main()
