"""Training-only offline signal and cost check; formal budgets remain fixed."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from gb_2_maco_POMO.experiments.learnable_ball_bias.rollout import rollout_batch
from gb_2_maco_POMO.experiments.shared.data import load_manifest

from .references import build_references
from .reward import compute_advantages
from .train import DATA_ROOT, initialized_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train, validation = load_manifest({"root": str(DATA_ROOT), "train_dir": "train",
                                       "validation_dir": "validation"})
    references = build_references(DATA_ROOT, train, validation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
    model, _ = initialized_model("G3", 42, device)
    model.train()
    rows = []
    for n in sorted({item.city_count for item in train}):
        for certified in (True, False):
            selected = [item for item in train if item.city_count == n
                        and references[str(item.path)]["certified"] == certified][:3]
            for item in selected:
                outcome = rollout_batch(model, [item], device, sample=True)
                for name, candidates, entrances, budget in (("fixed", 16, 8, 8000),
                                                             ("probe", 32, 16, 16000)):
                    started = time.perf_counter()
                    _, diag, _ = compute_advantages(
                        outcome, [item], references, "G3", {"two_opt": {}, "opt_b": {}},
                        selected_limit=candidates, entrance_limit=entrances,
                        optb_budget=budget)
                    rows.append({"instance": item.path.name, "city_count": n,
                                 "certified": certified, "budget": name,
                                 "selected": diag.selected, "eligible": diag.eligible,
                                 "entrances": diag.entrances, "observed_regions": diag.regions,
                                 "nonzero_advantage_routes": diag.nonzero_coverage_advantages,
                                 "two_opt_checks": diag.two_opt_checks,
                                 "optb_checks": diag.optb_checks,
                                 "reward_seconds": time.perf_counter() - started})
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"source": "training split only", "seed": 42,
                                  "sample_rule": "first 3 certified and first 3 uncertified per city count",
                                  "probe_is_not_formal_training_budget": True, "rows": rows},
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in ("fixed", "probe"):
        subset = [r for r in rows if r["budget"] == name]
        signal = sum(r["nonzero_advantage_routes"] > 0 for r in subset)
        print(name, "signal_cases", signal, "/", len(subset),
              "large_signal_cases", sum(r["nonzero_advantage_routes"] > 0
                                         for r in subset if r["city_count"] >= 50),
              "large_cases", sum(r["city_count"] >= 50 for r in subset), flush=True)


if __name__ == "__main__":
    main()
