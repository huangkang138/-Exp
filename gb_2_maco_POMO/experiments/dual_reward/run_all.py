"""One user-run command: preflight, TOML-controlled A/B training, test, report."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path

from . import evaluate, report, train
from .settings import DEFAULT_CONFIG, Settings, load as load_settings


def _status(output: Path, stage: str, state: str, **details):
    payload = {"updated_at": datetime.now().astimezone().isoformat(),
               "stage": stage, "state": state, **details}
    (output / "run_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{stage}] {state}", flush=True)


def run(config: Settings, output: Path | None = None):
    output = output or config.output_dir
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"new output directory required; historical result preserved: {output}")
    output.mkdir(parents=True)
    # train.run owns a fresh output directory. Keep this parent for status.
    work = output / "experiment"
    try:
        _status(output, "preflight_and_training", "running")
        train.run(work, config)
        _status(output, "preflight_and_training", "completed")
        _status(output, "formal_test", "running")
        evaluate.run(work, config)
        _status(output, "formal_test", "completed")
        report.create(work)
        _status(output, "all", "completed", report=str(work / "analysis.md"))
    except Exception as exc:
        _status(output, "all", "failed", error=str(exc), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="optional new output path; defaults to config.toml")
    args = parser.parse_args()
    run(load_settings(args.config), args.output_dir)


if __name__ == "__main__":
    main()
