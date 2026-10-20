"""Prepare, calibrate, then run the inference-only B1/B2 comparison.

Commands (from project root):
  python -m gb_2_maco_POMO.experiments.beam_lookahead.run smoke
  python -m gb_2_maco_POMO.experiments.beam_lookahead.run validate
  python -m gb_2_maco_POMO.experiments.beam_lookahead.run formal

Formal evaluation can use a saved validation lock or an explicitly supplied
fixed budget. The latter is recorded as an uncalibrated test protocol.
No command trains or modifies the A0 checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from gb_maco_POMO.experiments.compare import BENCHMARKS, load_ground_truth
from gb_2_maco_POMO.experiments.hierarchical import original25
from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import load_checkpoint
from gb_2_maco_POMO.experiments.region_supervised_pomo.config import INITIAL, INITIAL_SHA256
from gb_2_maco_POMO.experiments.region_supervised_pomo.data import _project_components
from gb_2_maco_POMO.experiments.shared.data import load_manifest
from gb_2_maco_POMO.runner import prepare_tsp

from .evaluate import evaluate
from .search import generate
from .settings import DEFAULT_CONFIG, Settings, load


PROJECT = Path(__file__).resolve().parents[3]
OUTPUT = PROJECT / "gb_2_maco_POMO/results/beam_lookahead"
VALIDATION_ROOT = PROJECT / "benachmark_Train_MATSP_Multiopt_500"
A0_TEST = PROJECT / "gb_2_maco_POMO/results/dual_reward/run_seed_42_multiopt_500_10epoch/experiment/test"


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staged.replace(path)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def prepare(config: Settings, phase: str = "smoke"):
    digest = hashlib.sha256(INITIAL.read_bytes()).hexdigest()
    if digest != INITIAL_SHA256:
        raise RuntimeError("A0 checkpoint hash changed")
    if phase not in ("smoke", "validate", "formal"):
        raise ValueError(f"unknown phase: {phase}")
    device = config.formal_device if phase == "formal" else config.validation_device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{phase}: CUDA requested but unavailable")
    policy, _ = load_checkpoint(INITIAL)
    policy.to(device).eval()
    print(f"{phase} inference device: {device}", flush=True)
    return policy, digest


def search_case(policy, prepared, mode: str, config: Settings,
                budget_multiplier: int | None) -> tuple[list[list[int]], list[dict]]:
    n = len(prepared.cities)
    routes, trace = [], []
    width = config.beam_width if budget_multiplier is None or mode == "B2" else max(
        config.beam_width, budget_multiplier - 1)
    for start in range(n):
        result = generate(policy, prepared, mode, config, start=start,
                          budget=None if budget_multiplier is None else budget_multiplier * n,
                          width=width)
        offset = len(routes)
        routes.extend(result.routes)
        trace.append({"start": start, "first_route_index": offset,
            "route_count": len(result.routes), "decode_steps": result.decode_steps,
            "lookahead_rounds": result.lookahead_rounds,
            "decode_budget": "" if budget_multiplier is None else budget_multiplier * n,
            "truncated": result.truncated, "inference_seconds": result.inference_seconds})
    return routes, trace


def validation_cases() -> list:
    _, validation = load_manifest({"root": str(VALIDATION_ROOT), "train_dir": "train",
        "validation_dir": "validation"})
    # Deterministic spread across both N and data distribution; no test case.
    selected = {}
    for item in sorted(validation, key=lambda item: (item.city_count, item.distribution, str(item.path))):
        selected.setdefault((item.city_count, item.distribution), item)
    candidates = list(selected.values())
    if len(candidates) < 3:
        raise RuntimeError("independent validation lacks three size/distribution cohorts")
    return [candidates[0], candidates[len(candidates) // 2], candidates[-1]]


def smoke(policy, config: Settings) -> None:
    prepared = prepare_tsp(validation_cases()[0].path)
    for mode in ("B1", "B2"):
        result = generate(policy, prepared, mode, config, start=0)
        print(f"{mode}: {len(result.routes)} routes; {result.decode_steps} decode steps; "
              f"{result.inference_seconds:.2f}s", flush=True)


def validate(policy, config: Settings, output: Path, checkpoint_sha: str,
             config_sha: str) -> None:
    lock_path = output / "validation_lock.json"
    if lock_path.exists():
        raise FileExistsError(lock_path)
    rows = []
    cases = validation_cases()
    for item in cases:
        prepared = prepare_tsp(item.path)
        for multiplier in config.budget_multipliers:
            for mode in ("B1", "B2"):
                routes, trace = search_case(policy, prepared, mode, config, multiplier)
                rows.append({"instance": item.path.name, "distribution": item.distribution,
                    "city_count": item.city_count, "mode": mode, "multiplier": multiplier,
                    "routes": len(routes), "decode_steps": sum(r["decode_steps"] for r in trace),
                    "truncated_starts": sum(r["truncated"] for r in trace),
                    "lookahead_starts": sum(r["lookahead_rounds"] > 0 for r in trace),
                    "mean_routes_per_start": len(routes) / item.city_count,
                    "inference_seconds": sum(r["inference_seconds"] for r in trace)})
                print(f"validation {item.path.name} {mode} {multiplier}N complete", flush=True)
    save_csv(output / "validation.csv", rows)
    eligible = [multiplier for multiplier in config.budget_multipliers if all(
        row["mean_routes_per_start"] >= 2 and
        (row["mode"] != "B2" or row["lookahead_starts"] == row["city_count"])
        for row in rows if row["multiplier"] == multiplier)]
    if not eligible:
        raise RuntimeError("neither candidate budget lets both groups produce useful extra routes")
    selected = min(eligible)
    save_json(lock_path, {"budget_multiplier": selected, "validation_cases": [
        str(item.path) for item in cases], "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha,
        "selection_rule": "smallest candidate budget with at least two routes per start and B2 lookahead on every start"})
    print(f"locked compute budget: {selected}N decode steps/start", flush=True)


def _baseline() -> tuple[dict, list[dict]]:
    summary = json.loads((A0_TEST / "summary.json").read_text(encoding="utf-8"))["A0"]
    if (summary["checkpoint_sha256"] != INITIAL_SHA256 or summary["distance_hits"] != 25
            or summary["opt_b_hits"] != 669 or summary["regions_hit"] != 105
            or abs(summary["mean_opt_b_fbeta"] - 0.9011984576780031) > 1e-10):
        raise RuntimeError("A0 formal baseline does not match the approved reference")
    with (A0_TEST / "per_instance.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["mode"] == "A0"]
    if len(rows) != 25:
        raise RuntimeError("A0 per-instance baseline incomplete")
    return summary, rows


def formal(policy, config: Settings, output: Path, checkpoint_sha: str,
           config_sha: str, *, budget_multiplier: int | None,
           conditions: tuple[str, ...]) -> None:
    lock_path = output / "validation_lock.json"
    if budget_multiplier is None:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if (lock["checkpoint_sha256"] != checkpoint_sha or
                lock["config_sha256"] != config_sha):
            raise RuntimeError("validation lock no longer matches checkpoint or config")
        budget_multiplier = int(lock["budget_multiplier"])
        budget_source = "independent_validation"
    else:
        if budget_multiplier not in config.budget_multipliers:
            raise ValueError("manual budget must be one of config.budget_multipliers")
        budget_source = "manual_fixed_budget_without_completed_validation"
    baseline, baseline_rows = _baseline()
    formal_dir = output / "formal"
    protocol = {"config": asdict(config), "config_sha256": config_sha,
        "checkpoint": str(INITIAL), "checkpoint_sha256": checkpoint_sha,
        "budget_multiplier": budget_multiplier, "budget_source": budget_source,
        "conditions": list(conditions),
        "routes_per_start_max": config.routes_per_start,
        "optb_checks_per_case": config.optb_checks_per_case,
        "decode_step_definition": "one route-state action_logits call; includes local beam and rollout",
        "baseline_source": str(A0_TEST),
        "search_sha256": hashlib.sha256(Path(__file__).with_name("search.py").read_bytes()).hexdigest(),
        "evaluate_sha256": hashlib.sha256(Path(__file__).with_name("evaluate.py").read_bytes()).hexdigest()}
    protocol = json.loads(json.dumps(protocol))
    protocol_path = formal_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("existing formal result has a different protocol; use another output directory")
    else:
        if formal_dir.exists() and any(formal_dir.iterdir()):
            raise RuntimeError("formal result directory exists without protocol.json")
        formal_dir.mkdir(parents=True, exist_ok=True)
        save_json(protocol_path, protocol)
    paper = original25._csv(original25.PAPER)
    rows = []
    for condition in conditions:
        multiplier = None if condition == "equal_output" else budget_multiplier
        for index, (stem, n, _) in enumerate(BENCHMARKS, 1):
            name = f"MSTSP{index}"
            prepared = prepare_tsp(original25.DATA / f"{stem}.tsp")
            if len(prepared.cities) != n:
                raise RuntimeError(f"{name}: unexpected city count")
            for mode in ("B1", "B2"):
                label = f"{condition}_{mode}"
                case_dir = formal_dir / label / name
                metrics_path = case_dir / "metrics.json"
                if metrics_path.exists():
                    for filename in ("raw_routes.json", "route_lineage.json",
                                     "optb_provenance.json", "search_trace.json"):
                        if not (case_dir / filename).is_file():
                            raise RuntimeError(f"{name} {label}: incomplete saved case")
                    row = json.loads(metrics_path.read_text(encoding="utf-8"))
                    rows.append(row)
                    print(f"{label} {name}: resumed", flush=True)
                    continue
                routes, trace = search_case(policy, prepared, mode, config, multiplier)
                if any(record["route_count"] > config.routes_per_start for record in trace):
                    raise RuntimeError("output route budget exceeded")
                result, lineage, provenance = evaluate(prepared, routes,
                    original25.DATA / f"{stem}.solution", config.optb_checks_per_case)
                truth = load_ground_truth(original25.DATA / f"{stem}.solution", n)
                tours = tuple(sorted(truth))
                regions = [set(tours[j] for j in part) for part in
                           _project_components(tours, prepared.distance_matrix)]
                returned = set(map(tuple, provenance["b_best_routes"])) & truth
                save_json(case_dir / "raw_routes.json", routes)
                save_json(case_dir / "route_lineage.json", lineage)
                save_json(case_dir / "optb_provenance.json", provenance)
                save_json(case_dir / "search_trace.json", trace)
                row = {"instance": name, "condition": condition, "mode": mode,
                    "city_count": n, "seed": config.seed + index,
                    "checkpoint_sha256": checkpoint_sha,
                    "paper_fbeta": paper[name]["paper_fbeta"],
                    "regions_hit": sum(bool(returned & region) for region in regions),
                    "known_regions": len(regions),
                    "decode_steps": sum(record["decode_steps"] for record in trace),
                    "truncated_starts": sum(record["truncated"] for record in trace),
                    "lookahead_starts": sum(record["lookahead_rounds"] > 0 for record in trace),
                    "inference_seconds": sum(record["inference_seconds"] for record in trace),
                    **result}
                save_json(metrics_path, row)
                rows.append(row)
                print(f"{label} {name}: distance={row['b_best']} "
                      f"tours={row['b_hits']} regions={row['regions_hit']}", flush=True)
    save_csv(formal_dir / "per_instance.csv", rows)
    paired = []
    by_key = {(r["instance"], r["condition"], r["mode"]): r for r in rows}
    keys = ("raw_best", "raw_gap_pct", "a_best", "a_gap_pct", "a_hits",
            "a_fbeta", "b_best", "b_gap_pct", "b_hits", "b_precision",
            "b_recall", "b_fbeta", "regions_hit", "decode_steps", "a_checks",
            "b_checks", "inference_seconds")
    base_by_case = {r["instance"]: r for r in baseline_rows}
    for condition in conditions:
        for index in range(1, 26):
            name = f"MSTSP{index}"
            record = {"instance": name, "condition": condition,
                "reference_length": base_by_case[name]["reference_length"],
                "paper_fbeta": base_by_case[name]["paper_fbeta"]}
            for mode in ("B1", "B2"):
                record.update({f"{mode}_{key}": by_key[name, condition, mode][key] for key in keys})
            paired.append(record)
    save_csv(formal_dir / "paired_per_instance.csv", paired)
    summaries = {"A0": baseline}
    for condition in conditions:
        for mode in ("B1", "B2"):
            part = [r for r in rows if r["condition"] == condition and r["mode"] == mode]
            summaries[f"{condition}_{mode}"] = {
                "distance_hits": sum(r["b_distance_hit"] for r in part),
                "two_opt_hits": sum(r["a_hits"] for r in part),
                "opt_b_hits": sum(r["b_hits"] for r in part),
                "regions_hit": sum(r["regions_hit"] for r in part),
                "mean_opt_b_fbeta": sum(r["b_fbeta"] for r in part) / 25,
                "decode_steps": sum(r["decode_steps"] for r in part),
                "inference_seconds": sum(r["inference_seconds"] for r in part),
                "two_opt_checks": sum(r["a_checks"] for r in part),
                "optb_checks": sum(r["b_checks"] for r in part)}
    save_json(formal_dir / "summary.json", summaries)
    from .report import write_report
    write_report(formal_dir, summaries, paired)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "validate", "formal"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--budget-multiplier", type=int,
                        help="formal only: use a fixed budget without validation lock")
    parser.add_argument("--conditions", choices=("equal_decode_budget", "equal_output", "both"),
                        default="equal_decode_budget", help="formal comparison(s) to run")
    args = parser.parse_args()
    config = load(args.config)
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    output = args.output_dir.resolve()
    policy, checkpoint_sha = prepare(config, args.phase)
    if args.phase == "smoke":
        smoke(policy, config)
    elif args.phase == "validate":
        output.mkdir(parents=True, exist_ok=True)
        validate(policy, config, output, checkpoint_sha, config_sha)
    else:
        selected = (("equal_output", "equal_decode_budget") if args.conditions == "both"
                    else (args.conditions,))
        formal(policy, config, output, checkpoint_sha, config_sha,
               budget_multiplier=args.budget_multiplier, conditions=selected)


if __name__ == "__main__":
    main()
