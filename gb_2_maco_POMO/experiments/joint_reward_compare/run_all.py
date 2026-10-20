"""One command for J1/J2 preflight, training, formal testing and report."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path

from gb_2_maco_POMO.experiments.dual_reward.train import save_json
from gb_2_maco_POMO.experiments.joint_ball_city import evaluate as formal_test
from gb_2_maco_POMO.experiments.joint_ball_city import train as j1_train

from . import diagnostics, preflight, report, train_j2
from .settings import HERE, load_pair


def status(root: Path, stage: str, state: str, **details):
    payload = {"updated_at": datetime.now().astimezone().isoformat(),
               "stage": stage, "state": state, **details}
    (root / "run_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"[{stage}] {state}", flush=True)


def run(config_dir: Path = HERE):
    j1, j2, reward = load_pair(config_dir.resolve())
    root = j1.output_dir.parent
    if root != j2.output_dir.parent or j1.output_dir.exists() or j2.output_dir.exists():
        raise FileExistsError("J1/J2 output directories must be new siblings")
    root.mkdir(parents=True, exist_ok=True)
    try:
        status(root, "preflight", "running")
        save_json(root / "preflight.json", preflight.run(j1, j2, reward))
        status(root, "preflight", "completed")
        status(root, "J1_training", "running")
        j1_train.run(j1.output_dir, j1)
        status(root, "J1_training", "completed")
        status(root, "J2_training", "running")
        train_j2.run(j2.output_dir, j2, reward)
        status(root, "J2_training", "completed")
        for label, config in (("J1", j1), ("J2", j2)):
            status(root, f"{label}_formal_test", "running")
            formal_test.run(config.output_dir, config)
            status(root, f"{label}_formal_test", "completed")
        status(root, "report", "running")
        diagnostics.run(root, j1.output_dir, j2.output_dir)
        report.build(root, j1.output_dir, j2.output_dir)
        status(root, "all", "completed", report=str(root / "analysis.md"))
    except Exception as error:
        status(root, "all", "failed", error=str(error), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=HERE)
    args = parser.parse_args()
    run(args.config_dir)


if __name__ == "__main__":
    main()
