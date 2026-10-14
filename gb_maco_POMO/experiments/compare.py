"""用训练好的 POMO 在原 benchmark 上测试，并与已有 GB-MACO 复现结果比较。

运行：python -m gb_maco_POMO.experiments.compare
默认读取最近完成的训练 run 的 best.pt；也可用 --checkpoint 明确指定。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    # 允许在 PyCharm 中直接点击运行本文件，仍按项目包导入现有模块。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "gb_maco_POMO.experiments"

from gb_maco.runner import BENCHMARKS
from ..attention import POMOPolicy
from ..pomo import evaluate
from ..runner import prepare_tsp
from .train_loop import select_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POMO_ROOT = PROJECT_ROOT / "gb_maco_POMO"
BENCHMARK_ROOT = PROJECT_ROOT / "benchmark_MSTSP"
BASELINE_CSV = PROJECT_ROOT / "reproduction" / "results" / "summary" / "instance_summary.csv"
PAPER_CSV = PROJECT_ROOT / "reproduction" / "config" / "paper_results.csv"
# 原复现脚本还不是 Python package；把它的目录加入搜索路径，直接复用同一套 Fβ/DI 定义。
REPRODUCTION_SCRIPTS = PROJECT_ROOT / "reproduction" / "scripts"
sys.path.insert(0, str(REPRODUCTION_SCRIPTS))
from calculate_metrics import canonical_cycle, diversity_indicator, fbeta, load_ground_truth

FIELDS = (
    "instance", "file", "type", "city_count", "optimal_length", "gb_maco_length",
    "pomo_length", "gb_maco_gap_pct", "pomo_gap_pct", "pomo_improvement_pct",
    "paper_fbeta", "gb_maco_fbeta", "pomo_fbeta", "gb_maco_di", "pomo_di",
    "truth_solution_count", "pomo_best_solution_count", "baseline_runs",
    "gb_maco_seconds", "pomo_seconds",
)


def find_checkpoint() -> Path:
    """找到最近一个已完成训练的 run；不把仍在训练的 latest.pt 当成最终模型。"""
    completed = sorted(
        (p for p in (POMO_ROOT / "results").glob("run_*/final_result.json")
        if any((root / p.parent.name / "best.pt").is_file()
               for root in (POMO_ROOT / "results" / "checkpoints", POMO_ROOT / "checkpoints"))),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not completed:
        raise FileNotFoundError("找不到已完成训练的 best.pt，请用 --checkpoint 指定")
    for root in (POMO_ROOT / "results" / "checkpoints", POMO_ROOT / "checkpoints"):
        checkpoint = root / completed[0].parent.name / "best.pt"
        if checkpoint.is_file():
            return checkpoint
    raise AssertionError("已完成模型的 checkpoint 消失")


def load_baseline(path: Path) -> dict[str, dict[str, str]]:
    """读取既有复现实验汇总，校验每个 MSTSP 编号只出现一次。"""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "instance", "city_count", "route_cost_best", "runs",
            "Fbeta_mean", "DI_mean", "elapsed_mean",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"GB-MACO 汇总缺少列：{sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    by_instance = {row["instance"]: row for row in rows}
    if len(by_instance) != len(rows):
        raise ValueError("GB-MACO 汇总中的 instance 有重复")
    return by_instance


def load_paper(path: Path) -> dict[str, float]:
    """只读取论文 Table 2 中明确记录的 Fβ；paper_di 为空，不构造论文 DI。"""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"instance", "paper_fbeta"}.issubset(reader.fieldnames or []):
            raise ValueError("论文结果 CSV 缺少 instance 或 paper_fbeta 列")
        rows = list(reader)
    values = {row["instance"]: float(row["paper_fbeta"]) for row in rows}
    if len(values) != len(rows) or not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()):
        raise ValueError("论文 Fβ 存在重复实例或无效值")
    return values


def read_optimum(path: Path) -> int:
    """.solution 每行第一个数字是该闭合路线的长度；取其中最短值。"""
    lengths = [int(line.split()[0]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lengths or min(lengths) <= 0:
        raise ValueError(f".solution 没有有效的最优路线长度：{path}")
    return min(lengths)


def load_policy(checkpoint: Path, device: torch.device) -> tuple[POMOPolicy, dict[str, object]]:
    """只恢复网络权重供推理；优化器和随机状态是续训用的。"""
    # 这里读取的是本项目自己生成的 checkpoint，weights_only 限制反序列化对象。
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = state["config"]["model"]
    policy = POMOPolicy(
        embedding_dim=int(model["embedding_dim"]),
        head_num=int(model["num_heads"]),
        encoder_layer_num=int(model["encoder_layers"]),
        candidate_mode=str(model.get("candidate_mode", "hard")),
        same_ball_bias=float(model.get("same_ball_bias", 2.0)),
        adjacent_ball_bias=float(model.get("adjacent_ball_bias", 1.0)),
        bias_mode=str(model.get("bias_mode", "fixed")),
    ).to(device)
    policy.load_state_dict(state["model_state_dict"], strict=True)
    policy.eval()
    return policy, state


def evaluate_cases(
    policy: POMOPolicy, device: torch.device, baseline: dict[str, dict[str, str]],
    paper: dict[str, float],
    benchmark_root: Path, selected: set[str] | None,
) -> list[dict[str, object]]:
    """每例生成 N 条路线；与原算法一样，只把本次最短的去重路线当成返回解集。"""
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for index, (stem, count, _budget) in enumerate(BENCHMARKS, start=1):
            name = f"MSTSP{index}"
            if selected is not None and name not in selected:
                continue
            source = benchmark_root / f"{stem}.tsp"
            original = baseline.get(name)
            if original is None or name not in paper or not source.is_file():
                raise FileNotFoundError(f"缺少同一算例的论文 Fβ、GB-MACO 复现结果或 TSP：{name}")
            if int(original["city_count"]) != count:
                raise ValueError(f"{name}：原算法汇总的城市数不匹配")
            solution_path = benchmark_root / f"{stem}.solution"
            optimal = read_optimum(solution_path)
            truth = load_ground_truth(solution_path, count)
            original_length = float(original["route_cost_best"])
            original_fbeta = float(original["Fbeta_mean"])
            original_di = float(original["DI_mean"])
            if not math.isfinite(original_length) or original_length < optimal:
                raise ValueError(f"{name}：原算法路线长度无效或小于 .solution 最优值")
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (original_fbeta, original_di)):
                raise ValueError(f"{name}：原算法 Fβ/DI 汇总值无效")

            # 原复现实验只写当前最优长度的路线；POMO 也只保留 N 条路线中
            # 长度等于本次最短值的路线，旋转或逆向相同的闭环只计一次。
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            prepared = prepare_tsp(source)
            outcome = evaluate(policy, prepared)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            lengths = outcome.tour_lengths.cpu().tolist()
            predicted = float(min(lengths))
            if not math.isfinite(predicted) or predicted < optimal:
                raise ValueError(f"{name}：POMO 路线长度无效或小于 .solution 最优值")
            returned = {
                canonical_cycle(route)
                for route, length in zip(outcome.routes, lengths)
                if length == predicted
            }
            pomo_fbeta = fbeta(returned, truth)
            pomo_di = diversity_indicator(returned, truth)
            if pomo_fbeta is None or pomo_di is None:
                raise ValueError(f"{name}：.solution 没有可用的最优路线集合")
            rows.append({
                "instance": name, "file": source.name,
                "type": "Simple" if index <= 6 else "Geometry" if index <= 12 else "Composite",
                "city_count": count, "optimal_length": optimal,
                "gb_maco_length": original_length, "pomo_length": predicted,
                "gb_maco_gap_pct": 100 * (original_length / optimal - 1),
                "pomo_gap_pct": 100 * (predicted / optimal - 1),
                "pomo_improvement_pct": 100 * (original_length - predicted) / original_length,
                "paper_fbeta": paper[name],
                "gb_maco_fbeta": original_fbeta, "pomo_fbeta": pomo_fbeta,
                "gb_maco_di": original_di, "pomo_di": pomo_di,
                "truth_solution_count": len(truth), "pomo_best_solution_count": len(returned),
                "baseline_runs": int(original["runs"]),
                "gb_maco_seconds": float(original["elapsed_mean"]), "pomo_seconds": elapsed,
            })
            print(
                f"{name}: cost {original_length:g}/{predicted:g}, "
                f"Fβ paper/POMO {paper[name]:.3f}/{pomo_fbeta:.3f}, "
                f"cost GB-MACO/POMO {original_length:g}/{predicted:g}",
                flush=True,
            )
    return rows


def save_results(
    rows: list[dict[str, object]], output: Path, checkpoint: Path, state: dict[str, object],
    baseline_csv: Path, paper_csv: Path, benchmark_root: Path, device: torch.device,
) -> None:
    """保存逐例原始数值、汇总和图片；保留输入文件路径供日后复核。"""
    from .compare_plots import draw_comparison

    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_instance.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "run_id": state["run_id"], "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_epoch": state["epoch"], "best_epoch": state["best_epoch"],
        "benchmark_dir": str(benchmark_root.resolve()),
        "paper_fbeta_source": str(paper_csv.resolve()),
        "gb_maco_reproduction_source": str(baseline_csv.resolve()),
        "device": str(device), "decision": "N starts, greedy argmax, shortest closed tour",
        "instances": len(rows),
        "pomo_shorter": sum(float(r["pomo_length"]) < float(r["gb_maco_length"]) for r in rows),
        "equal": sum(float(r["pomo_length"]) == float(r["gb_maco_length"]) for r in rows),
        "gb_maco_shorter": sum(float(r["pomo_length"]) > float(r["gb_maco_length"]) for r in rows),
        "mean_gb_maco_gap_pct": sum(float(r["gb_maco_gap_pct"]) for r in rows) / len(rows),
        "mean_pomo_gap_pct": sum(float(r["pomo_gap_pct"]) for r in rows) / len(rows),
        "mean_paper_fbeta": sum(float(r["paper_fbeta"]) for r in rows) / len(rows),
        "mean_gb_maco_fbeta": sum(float(r["gb_maco_fbeta"]) for r in rows) / len(rows),
        "mean_pomo_fbeta": sum(float(r["pomo_fbeta"]) for r in rows) / len(rows),
        "mean_gb_maco_di": sum(float(r["gb_maco_di"]) for r in rows) / len(rows),
        "mean_pomo_di": sum(float(r["pomo_di"]) for r in rows) / len(rows),
        "note": "主图比较论文 Table 2 的 Fβ 与 POMO；论文 DI 原始数值缺失，DI 图仅画 POMO。",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    draw_comparison(rows, output / "figures")
    print(f"对比结果：{output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="训练好的 POMO 与原 GB-MACO 同算例对比")
    parser.add_argument("--checkpoint", type=Path, help="默认最近完成的 run 的 best.pt")
    parser.add_argument("--baseline-csv", type=Path, default=BASELINE_CSV)
    parser.add_argument("--paper-csv", type=Path, default=PAPER_CSV)
    parser.add_argument("--benchmark-dir", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, help="默认对应 run 的 results/run_xxx/comparison")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--instances", nargs="+", help="可选：仅测 MSTSP1 等指定算例")
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint or find_checkpoint()
    device = select_device(args.device)
    policy, state = load_policy(checkpoint, device)
    selected = set(args.instances) if args.instances else None
    all_names = {f"MSTSP{i}" for i in range(1, len(BENCHMARKS) + 1)}
    if selected is not None and not selected.issubset(all_names):
        parser.error(f"未知算例：{sorted(selected - all_names)}")
    baseline = load_baseline(args.baseline_csv)
    paper = load_paper(args.paper_csv)
    print(f"模型：{checkpoint}；epoch={state['epoch']}；设备={device}", flush=True)
    rows = evaluate_cases(policy, device, baseline, paper, args.benchmark_dir, selected)
    output = args.output_dir or POMO_ROOT / "results" / str(state["run_id"]) / "comparison"
    save_results(rows, output, checkpoint, state, args.baseline_csv, args.paper_csv, args.benchmark_dir, device)


if __name__ == "__main__":
    main()
