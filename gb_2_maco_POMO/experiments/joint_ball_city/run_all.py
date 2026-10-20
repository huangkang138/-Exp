"""One command: CUDA preflight, 10 epochs, dual selection, formal test and report."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path

from . import cost, evaluate, report, train
from .settings import DEFAULT_CONFIG, Settings, load


def _status(output: Path, stage: str, state: str, **details) -> None:
    payload = {"updated_at": datetime.now().astimezone().isoformat(),
               "stage": stage, "state": state, **details}
    (output / "run_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{stage}] {state}", flush=True)


def run(config: Settings, output: Path | None = None) -> None:
    output = (output or config.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"new output directory required: {output}")
    output.mkdir(parents=True)
    try:
        _status(output, "preflight_and_training", "running")
        train.run(output, config)
        _status(output, "preflight_and_training", "completed")
        _status(output, "formal_test", "running")
        evaluate.run(output, config)
        _status(output, "formal_test", "completed")
        _status(output, "inference_cost", "running")
        cost.run(output, config)
        _status(output, "inference_cost", "completed")
        report.create(output)
        _status(output, "all", "completed", report=str(output / "analysis.md"))
    except Exception as exc:
        _status(output, "all", "failed", error=str(exc), traceback=traceback.format_exc())
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run(load(args.config), args.output_dir)


if __name__ == "__main__":
    main()
