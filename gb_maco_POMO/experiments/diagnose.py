"""只读诊断训练后 POMO 在 25 个 MSTSP 算例上的路线与候选集结果。

运行：python -m gb_maco_POMO.experiments.diagnose
使用已有 best.pt 和 evaluate()；不训练、不修改选路或粒球算法。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    # 直接点击运行本文件时，也按项目包导入现有评估函数。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "gb_maco_POMO.experiments"

from gb_maco.runner import BENCHMARKS
from ..pomo import evaluate
from ..runner import prepare_tsp
from .compare import (
    BENCHMARK_ROOT, POMO_ROOT, canonical_cycle, fbeta, find_checkpoint,
    load_ground_truth, load_policy, read_optimum,
)
from .train_loop import select_device


FIELDS = (
    "实例", "最优真实距离", "POMO best distance", "Gap %", "POMO 生成路线数",
    "去重后的路线数", "真实最优路线命中数", "Precision", "Recall", "Fβ",
    "Fallback 次数", "真实最优路线总数", "全部生成路线去重数",
)


def diagnose(checkpoint: Path, benchmark_dir: Path, device: torch.device) -> tuple[list[dict[str, object]], str]:
    """逐例复用正式推理，返回原始数值和加载的训练 run 编号。"""
    policy, state = load_policy(checkpoint, device)
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for number, (stem, city_count, _budget) in enumerate(BENCHMARKS, start=1):
            path = benchmark_dir / f"{stem}.tsp"
            solution_path = benchmark_dir / f"{stem}.solution"
            prepared = prepare_tsp(path)
            if len(prepared.cities) != city_count:
                raise ValueError(f"{path.name} 的城市数量与原 benchmark 定义不一致")
            result = evaluate(policy, prepared)
            lengths = result.tour_lengths.cpu().tolist()
            best = min(lengths)
            optimum = read_optimum(solution_path)
            truth = load_ground_truth(solution_path, city_count)

            # POMO 的 N 个起点各生成一条路线。Fβ 延续现有 compare.py 的口径：
            # 只把本次最短长度的路线当作返回解集；旋转/逆向等价的路线只算一次。
            all_unique = {canonical_cycle(route) for route in result.routes}
            returned = {
                canonical_cycle(route)
                for route, length in zip(result.routes, lengths)
                if length == best
            }
            hits = len(returned & truth)
            precision = hits / len(returned)
            recall = hits / len(truth)
            fbeta_value = fbeta(returned, truth)
            assert fbeta_value is not None

            # candidate_decisions 覆盖 N 条路线各自的 N-1 次选城；一次决策最多记一次 fallback。
            fallback_count = sum(decision.fallback_used for decision in result.candidate_decisions)
            rows.append({
                "实例": f"MSTSP{number}", "最优真实距离": optimum,
                "POMO best distance": int(best), "Gap %": 100 * (best - optimum) / optimum,
                "POMO 生成路线数": len(result.routes), "去重后的路线数": len(returned),
                "真实最优路线命中数": hits, "Precision": precision,
                "Recall": recall, "Fβ": fbeta_value, "Fallback 次数": fallback_count,
                "真实最优路线总数": len(truth), "全部生成路线去重数": len(all_unique),
            })
            print(
                f"MSTSP{number}: optimal={optimum}, best={int(best)}, "
                f"gap={rows[-1]['Gap %']:.2f}%, hits={hits}, "
                f"Fβ={fbeta_value:.3f}, fallback={fallback_count}", flush=True,
            )
    return rows, str(state["run_id"])


def save(rows: list[dict[str, object]], output_dir: Path, checkpoint: Path) -> None:
    """CSV 保留计算精度；Markdown 方便直接阅读全部 25 例。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "diagnostics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# POMO 路线诊断（25 个 MSTSP 算例）", "",
        f"模型：`{checkpoint.resolve()}`", "",
        "去重后的路线数指**达到本次最短长度**且按起点旋转、反向等价去重后的数量；"
        "Precision、Recall、Fβ 都使用这一集合。最后一列另列出全部 N 条路线的去重数量。", "",
        "Fallback 次数统计该算例的全部 N 条 POMO 路线、全部选城步骤；"
        "不是单条最佳路线的次数。", "",
        "| " + " | ".join(FIELDS) + " |",
        "| " + " | ".join("---" for _ in FIELDS) + " |",
    ]
    for row in rows:
        cells = [f"{row[key]:.4f}" if key in ("Gap %", "Precision", "Recall", "Fβ") else str(row[key]) for key in FIELDS]
        lines.append("| " + " | ".join(cells) + " |")
    (output_dir / "diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="只读诊断训练后 POMO 的 25 个 benchmark 算例")
    parser.add_argument("--checkpoint", type=Path, help="默认最近完成的 run 的 best.pt")
    parser.add_argument("--benchmark-dir", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, help="默认对应 run 的 comparison 目录")
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint or find_checkpoint()
    rows, run_id = diagnose(checkpoint, args.benchmark_dir, select_device(args.device))
    output_dir = args.output_dir or POMO_ROOT / "results" / run_id / "comparison"
    save(rows, output_dir, checkpoint)
    print(f"诊断文件：{output_dir / 'diagnostics.csv'}", flush=True)


if __name__ == "__main__":
    main()
