"""汇总候选城市统计，并把真实训练记录绘成路线长度曲线。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..pomo import CandidateDecision


@dataclass(frozen=True)
class CandidateSummary:
    """一次 rollout 或整场实验的候选城市统计。"""

    total_decisions: int
    fallback_count: int
    fallback_rate: float
    average_candidate_size: float


def summarize_candidates(decisions: Sequence[CandidateDecision]) -> CandidateSummary:
    """按实际选城次数统计 fallback 占比及 Action Mask 前的平均候选数。"""
    total = len(decisions)
    if total == 0:
        raise ValueError("没有候选城市决策，无法统计")
    fallback_count = sum(decision.fallback_used for decision in decisions)
    candidate_total = sum(decision.candidate_size for decision in decisions)
    return CandidateSummary(
        total_decisions=total,
        fallback_count=fallback_count,
        fallback_rate=fallback_count / total,
        average_candidate_size=candidate_total / total,
    )


def plot_training_curve(history: list[dict[str, float | int | None]], path: Path) -> None:
    """绘制每轮训练长度和按 eval_interval 采样的确定性评估长度。"""
    import matplotlib

    matplotlib.use("Agg")  # 保存 PNG，不依赖 Windows 图形窗口。
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    evaluation_rows = [row for row in history if row["eval_best_length"] is not None]
    figure, axes = plt.subplots(figsize=(9, 5))
    axes.plot(epochs, [row["train_mean_length"] for row in history], label="Train mean length")
    axes.plot(epochs, [row["train_best_length"] for row in history], label="Train best length")
    axes.plot(
        [row["epoch"] for row in evaluation_rows],
        [row["eval_best_length"] for row in evaluation_rows],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Eval best length",
    )
    axes.set(title="POMO Training on simple1_9", xlabel="Epoch", ylabel="Tour Length")
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
