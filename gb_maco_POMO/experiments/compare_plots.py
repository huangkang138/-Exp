"""复用 reproduction 中文图的字体、配色、实例分区与坐标布局绘制 POMO 对比图。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# reproduction/scripts 不是 package；导入它已有的中文绘图辅助函数，保持图形风格一致。
SCRIPTS = Path(__file__).resolve().parents[2] / "reproduction" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from plot_results_zh import (
    LINE_COLOR, PAPER_COLOR, configure_chinese_font, finish, ordered,
    plot_bar, shade_types,
)


FIGURE_NAMES = (
    "fbeta_pomo_zh.png", "fbeta_paper_vs_pomo_zh.png", "fbeta_difference_zh.png",
    "di_pomo_zh.png", "runtime_by_instance_zh.png", "paper_error_overview_zh.png",
)


def _single_metric(
    data: pd.DataFrame, column: str, destination: Path,
    *, ylabel: str, title: str, explanation: str,
) -> None:
    """对应旧图的 Fβ、DI 单算法折线：每例一个 POMO 指标值。"""
    data = ordered(data)
    figure, ax = plt.subplots(figsize=(12, 5.5))
    type_handles = shade_types(ax, data["type"].tolist())
    line = ax.errorbar(
        range(len(data)), data[column], marker="o", capsize=3,
        linewidth=1.8, color=LINE_COLOR, label="训练后 POMO", zorder=3,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(handles=[line, *type_handles], loc="best")
    finish(
        ax, data["instance"].tolist(), destination,
        y_label=ylabel, title=title, explanation=explanation,
    )


def _paired_metric(
    data: pd.DataFrame, baseline: str, pomo: str, destination: Path,
    *, ylabel: str, title: str, explanation: str,
) -> None:
    """对应旧图 fbeta_paper_vs_reproduced_zh：同一实例两条折线。"""
    data = ordered(data)
    figure, ax = plt.subplots(figsize=(12, 5.5))
    type_handles = shade_types(ax, data["type"].tolist())
    pomo_line, = ax.plot(range(len(data)), data[pomo], marker="o", color=LINE_COLOR, label="训练后 POMO")
    paper_line, = ax.plot(range(len(data)), data[baseline], marker="s", color=PAPER_COLOR, label="论文 Table 2")
    ax.set_ylim(0, 1.05)
    ax.legend(handles=[paper_line, pomo_line, *type_handles], loc="best")
    finish(ax, data["instance"].tolist(), destination, y_label=ylabel, title=title, explanation=explanation)


def draw_comparison(rows: list[dict[str, object]], figures: Path) -> None:
    """按 reproduction 的六类图，主对比使用论文 Fβ；论文 DI 缺失时只画 POMO。"""
    configure_chinese_font()
    figures.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)

    _single_metric(
        data, "pomo_fbeta", figures / FIGURE_NAMES[0],
        ylabel="Fβ", title="训练后 POMO 的 Fβ",
        explanation="越接近 1 越好：衡量真实最优路线找得准、找得全",
    )
    _paired_metric(
        data, "paper_fbeta", "pomo_fbeta", figures / FIGURE_NAMES[1],
        ylabel="Fβ", title="论文 Table 2 与训练后 POMO 的 Fβ 对比",
        explanation="论文 Fβ 为原文报告值；POMO Fβ 用相同指标公式从路线集合计算",
    )

    data["delta_fbeta"] = data["pomo_fbeta"] - data["paper_fbeta"]
    difference = ordered(data)
    delta_colors = ["#2E7D32" if value >= 0 else "#C0392B" for value in difference["delta_fbeta"]]
    plot_bar(
        difference, "delta_fbeta", figures / FIGURE_NAMES[2],
        y_label="Fβ 差值", title="POMO − 论文 Table 2 的 Fβ 差值",
        explanation="0 表示相同；负值表示 POMO 低于论文报告值",
        colors=delta_colors, zero_line=True,
    )

    _single_metric(
        data, "pomo_di", figures / FIGURE_NAMES[3],
        ylabel="多样性指标 DI", title="训练后 POMO 的 DI",
        explanation="论文 DI 无原始数值；这里仅展示 POMO 的计算结果",
    )

    plot_bar(
        data, "pomo_seconds", figures / FIGURE_NAMES[4],
        y_label="运行时间（秒）", title="POMO 各测试实例推理耗时",
        explanation="包含粒球预处理；GPU 推理，不与原算法的历史耗时直接比较",
    )

    data["absolute_difference"] = data["delta_fbeta"].abs()
    plot_bar(
        data, "absolute_difference", figures / FIGURE_NAMES[5],
        y_label="Fβ 绝对差值", title="POMO 与论文 Fβ 差异总览",
        explanation="越接近 0，表示 POMO 与论文在该实例的 Fβ 越接近",
    )
