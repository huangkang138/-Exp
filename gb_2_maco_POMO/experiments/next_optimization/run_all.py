"""One command for the two experiments, with existing completed work reused."""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path

from . import dataset_diagnosis, generate_complex_dataset, solution_union, train_complex


ROOT = Path(__file__).resolve().parents[3] / "gb_2_maco_POMO/results/next_optimization"
STATUS = ROOT / "run_status.json"


def _record(stage: str, state: str, **details) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().astimezone().isoformat(),
               "stage": stage, "state": state, **details}
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"[{stage}] {state}", flush=True)


def _existing_or_run(stage: str, output: Path, required: tuple[str, ...], action) -> None:
    if output.exists():
        missing = [name for name in required if not (output / name).is_file()]
        if missing:
            raise RuntimeError(f"{stage}: existing output is incomplete: {missing}; "
                               "preserved for inspection")
        _record(stage, "reused", output=str(output))
        return
    _record(stage, "running", output=str(output))
    action(output)
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"{stage}: finished without required outputs: {missing}")
    _record(stage, "completed", output=str(output))


def run() -> None:
    union = ROOT / "solution_union"
    diagnosis = ROOT / "dataset_diagnosis"
    dataset = ROOT / "complex_training/dataset"
    training = ROOT / "complex_training/run_seed_42"

    try:
        _existing_or_run("solution_union", union,
                         ("summary.csv", "per_instance.csv", "analysis.md",
                          "merged_routes/MSTSP1.json", "merged_routes/MSTSP25.json"),
                         solution_union.run)
        _existing_or_run("dataset_diagnosis", diagnosis,
                         ("distribution_summary.csv", "structural_features.csv",
                          "analysis.md"), dataset_diagnosis.run)

        if dataset.exists():
            if not (dataset / "READY").is_file() or (dataset / "INCOMPLETE").exists():
                raise RuntimeError("existing complex dataset is incomplete; preserved for inspection")
            _record("complex_dataset", "reused", output=str(dataset))
        else:
            _record("complex_dataset", "running", output=str(dataset))
            try:
                generate_complex_dataset.run(dataset)
            except RuntimeError as exc:
                if "original data already covers" in str(exc):
                    _record("complex_dataset", "stopped_sufficient_original_data",
                            reason=str(exc))
                    return
                raise
            if not (dataset / "READY").is_file():
                raise RuntimeError("complex dataset finished without READY marker")
            _record("complex_dataset", "completed", output=str(dataset))

        if training.exists():
            if (training / "stopped.json").is_file():
                _record("complex_training", "stopped_cpu_estimate",
                        output=str(training))
                return
            result_path = training / "train_result.json"
            if not result_path.is_file():
                raise RuntimeError("existing training output is incomplete; preserved for inspection")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("selected_epoch") == 1 and not result.get("formal_test_executed"):
                raise RuntimeError("selected epoch 1 but formal test is incomplete; "
                                   "preserved for inspection")
            if result.get("formal_test_executed") and not (training / "test/summary.json").is_file():
                raise RuntimeError("training says test completed, but test summary is missing")
            _record("complex_training", "reused", output=str(training),
                    selected_epoch=result.get("selected_epoch"),
                    formal_test_executed=result.get("formal_test_executed"))
            return

        _record("complex_training", "running", output=str(training))
        train_complex.run(dataset, training, "cpu")
        if (training / "stopped.json").is_file():
            _record("complex_training", "stopped_cpu_estimate", output=str(training))
            return
        result_path = training / "train_result.json"
        if not result_path.is_file():
            raise RuntimeError("training finished without train_result.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("formal_test_executed") and not (training / "test/summary.json").is_file():
            raise RuntimeError("formal test summary missing")
        _record("all", "completed", output=str(training),
                selected_epoch=result.get("selected_epoch"),
                formal_test_executed=result.get("formal_test_executed"))
    except Exception as exc:
        _record("all", "failed", error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    run()
