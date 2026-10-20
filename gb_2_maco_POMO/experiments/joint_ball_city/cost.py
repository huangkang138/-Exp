"""Same-device, same-seed inference and CUDA memory accounting."""

from __future__ import annotations

from time import perf_counter

import torch

from gb_2_maco_POMO.experiments.dual_reward.train import save_csv, save_json
from gb_2_maco_POMO.experiments.hierarchical.original25 import BENCHMARKS, DATA
from gb_2_maco_POMO.experiments.learnable_ball.learnable_inference import (
    generate_learned_routes, load_checkpoint as load_original,
)
from gb_2_maco_POMO.runner import prepare_tsp

from .inference import generate_routes, load_checkpoint


def run(root, config) -> dict:
    prepared = [prepare_tsp(DATA / f"{stem}.tsp") for stem, _, _ in BENCHMARKS]
    sources = {"A0": config.initial_checkpoint,
               "Joint_shortest": root / "checkpoints/best_shortest.pt",
               "Joint_diversity": root / "checkpoints/best_diversity.pt"}
    rows, summary = [], {}
    for label, path in sources.items():
        model, _ = (load_original(path) if label == "A0" else load_checkpoint(path))
        generator = generate_learned_routes if label == "A0" else generate_routes
        model.cuda().eval()
        parameters = sum(p.numel() for p in model.parameters())
        extra = sum(p.numel() for p in model.joint_head.parameters()) if label != "A0" else 0
        torch.cuda.reset_peak_memory_stats()
        total = 0.0
        for index, case in enumerate(prepared, 1):
            torch.cuda.synchronize()
            started = perf_counter()
            routes, _ = generator(model, case, list(range(len(case.cities))),
                                  config.validation_samples_per_start, config.seed + index)
            torch.cuda.synchronize()
            seconds = perf_counter() - started
            if len(routes) != len(case.cities) * (1 + config.validation_samples_per_start):
                raise RuntimeError(f"{label} MSTSP{index}: inference route count differs")
            total += seconds
            rows.append({"model": label, "instance": f"MSTSP{index}",
                         "route_count": len(routes), "inference_seconds": seconds})
        summary[label] = {"parameters": parameters, "new_joint_parameters": extra,
                          "full_25_inference_seconds": total,
                          "mean_case_inference_seconds": total / 25,
                          "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        del model
        torch.cuda.empty_cache()
    baseline = summary["A0"]
    for label in ("Joint_shortest", "Joint_diversity"):
        summary[label]["time_ratio_to_A0"] = (summary[label]["full_25_inference_seconds"] /
                                               baseline["full_25_inference_seconds"])
        summary[label]["extra_parameters_vs_A0"] = (summary[label]["parameters"] -
                                                      baseline["parameters"])
        summary[label]["extra_peak_bytes_vs_A0"] = (summary[label]["cuda_peak_allocated_bytes"] -
                                                     baseline["cuda_peak_allocated_bytes"])
    save_csv(root / "inference_cost_per_instance.csv", rows)
    save_json(root / "inference_cost.json", summary)
    return summary
