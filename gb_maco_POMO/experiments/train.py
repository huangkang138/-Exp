"""仅用现有 simple1_9.tsp 连续训练 POMO，并保存实验指标与曲线。

运行：python -m gb_maco_POMO.experiments.train
快速验证：python -m gb_maco_POMO.experiments.train --epochs 5 --eval-interval 1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from ..attention import POMOPolicy
from ..pomo import evaluate, train_step
from ..runner import prepare_tsp
from .metrics import plot_training_curve, summarize_candidates


DATASET = Path(__file__).resolve().parents[2] / "benchmark_MSTSP" / "simple1_9.tsp"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
HISTORY_FIELDS = (
    "epoch", "train_mean_length", "train_best_length", "eval_best_length",
    "mean_reward", "loss", "fallback_count", "fallback_rate",
    "average_candidate_size",
)
FALLBACK_FIELDS = (
    "epoch", "pomo_index", "step", "current_city", "candidate_size",
    "fallback_used", "remaining_city_count",
)


def run_experiment(
    *, epochs: int = 2000, eval_interval: int = 50, seed: int = 42,
    learning_rate: float = 1e-4, results_root: Path = RESULTS_ROOT,
) -> dict[str, int | float | str | dict[str, int]]:
    """训练同一个小算例，按 epoch 保存真实指标、逐步候选记录和 PNG。

    Policy 和 Adam 在循环外只创建一次，因此各 epoch 接着上一轮参数训练。
    初始评估在训练前；周期评估与最终评估均调用现有 evaluate()。
    """
    if epochs < 1 or eval_interval < 1 or learning_rate <= 0:
        raise ValueError("epochs、eval_interval 和 learning_rate 必须为正数")
    torch.manual_seed(seed)
    # 9 城市算例的计算很小；单线程避免 CPU 线程调度占用大部分时间。
    torch.set_num_threads(1)
    prepared = prepare_tsp(DATASET)
    policy = POMOPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    output_dir = Path(results_root) / DATASET.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_eval_length = float(evaluate(policy, prepared).tour_lengths.min().item())
    eval_lengths: list[float] = []
    history: list[dict[str, float | int | None]] = []
    total_decisions = 0
    total_fallback_count = 0
    total_candidate_size = 0
    fallback_by_remaining: dict[str, int] = {}

    # 两个 CSV 持续写入：一个 epoch 一行训练摘要；每次选城一行候选统计。
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as history_file, \
         (output_dir / "fallback_stats.csv").open("w", newline="", encoding="utf-8") as fallback_file:
        history_writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
        fallback_writer = csv.DictWriter(fallback_file, fieldnames=FALLBACK_FIELDS)
        history_writer.writeheader()
        fallback_writer.writeheader()

        for epoch in range(1, epochs + 1):
            step_result = train_step(policy, prepared, optimizer)
            rollout = step_result.rollout
            candidate_summary = summarize_candidates(rollout.candidate_decisions)
            total_decisions += candidate_summary.total_decisions
            total_fallback_count += candidate_summary.fallback_count
            total_candidate_size += sum(
                decision.candidate_size for decision in rollout.candidate_decisions
            )
            for decision in rollout.candidate_decisions:
                fallback_writer.writerow({
                    "epoch": epoch,
                    "pomo_index": decision.pomo_index,
                    "step": decision.step,
                    "current_city": decision.current_city,
                    "candidate_size": decision.candidate_size,
                    "fallback_used": decision.fallback_used,
                    "remaining_city_count": decision.remaining_city_count,
                })
                if decision.fallback_used:
                    remaining = str(decision.remaining_city_count)
                    fallback_by_remaining[remaining] = fallback_by_remaining.get(remaining, 0) + 1

            # 未评估的 epoch 留空；不伪造两个评估点之间的结果。
            eval_best_length: float | None = None
            if epoch % eval_interval == 0:
                eval_best_length = float(evaluate(policy, prepared).tour_lengths.min().item())
                eval_lengths.append(eval_best_length)
            row: dict[str, float | int | None] = {
                "epoch": epoch,
                "train_mean_length": float(rollout.tour_lengths.mean().item()),
                "train_best_length": float(rollout.tour_lengths.min().item()),
                "eval_best_length": eval_best_length,
                "mean_reward": float(rollout.rewards.mean().item()),
                "loss": step_result.loss,
                "fallback_count": candidate_summary.fallback_count,
                "fallback_rate": candidate_summary.fallback_rate,
                "average_candidate_size": candidate_summary.average_candidate_size,
            }
            history_writer.writerow(row)
            history.append(row)
            if epoch % eval_interval == 0 or epoch == epochs:
                print(
                    f"Epoch {epoch}/{epochs} | train mean={row['train_mean_length']:.2f} "
                    f"| train best={row['train_best_length']:.2f} "
                    f"| eval best={eval_best_length if eval_best_length is not None else '-'} "
                    f"| loss={step_result.loss:.4f} "
                    f"| fallback={candidate_summary.fallback_rate:.3f} "
                    f"| avg candidates={candidate_summary.average_candidate_size:.2f}",
                    flush=True,
                )

    # 即使 epochs 不是 eval_interval 的整数倍，最终值仍对应训练后的 Policy。
    final_eval_length = float(evaluate(policy, prepared).tour_lengths.min().item())
    eval_lengths.append(final_eval_length)
    result: dict[str, int | float | str | dict[str, int]] = {
        "dataset": DATASET.name,
        "seed": seed,
        "epochs": epochs,
        "eval_interval": eval_interval,
        "initial_eval_length": initial_eval_length,
        "final_eval_length": final_eval_length,
        "best_eval_length": min(eval_lengths),
        "final_train_mean_length": float(history[-1]["train_mean_length"]),
        "final_train_best_length": float(history[-1]["train_best_length"]),
        "total_decisions": total_decisions,
        "total_fallback_count": total_fallback_count,
        "overall_fallback_rate": total_fallback_count / total_decisions,
        "average_candidate_size": total_candidate_size / total_decisions,
        "fallback_by_remaining_city_count": fallback_by_remaining,
    }
    (output_dir / "final_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_training_curve(history, output_dir / "training_curve.png")
    print("Training finished.")
    print(
        f"Initial eval length: {initial_eval_length:.2f} | "
        f"Final eval length: {final_eval_length:.2f} | "
        f"Best eval length: {min(eval_lengths):.2f}"
    )
    print(
        f"Total fallback: {total_fallback_count} | "
        f"Fallback rate: {total_fallback_count / total_decisions:.4f} | "
        f"Average candidate size: {total_candidate_size / total_decisions:.2f}"
    )
    print(f"Results saved to: {output_dir}")
    return result


def main(argv: list[str] | None = None) -> None:
    """解析实验参数；无参数时按约定运行 2000 epoch、每 50 epoch 评估。"""
    parser = argparse.ArgumentParser(description="训练 simple1_9 的 Attention + POMO")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args(argv)
    run_experiment(
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        seed=args.seed,
        learning_rate=args.learning_rate,
        results_root=args.results_root,
    )


if __name__ == "__main__":
    main()
