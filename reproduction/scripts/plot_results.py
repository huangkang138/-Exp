"""根据汇总数据自动生成论文/PPT 可用的静态图片。

脚本读取 ``instance_summary.csv`` 和 ``paper_results.csv``，生成 Fβ、DI、
论文对比、差异及运行时间图。数据缺失时只跳过对应图片，不影响其他图。
使用无界面的 Agg 绘图后端，因此不需要打开图形窗口。
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from common import REPRODUCTION_ROOT, instance_sort_key, load_config, result_root


TYPE_COLORS = {"Simple": "#DCEAF7", "Geometry": "#E4F2E4", "Composite": "#F8E6D4"}
LINE_COLOR = "#245B8A"


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    """按 MSTSP 的数字编号排序 DataFrame，避免 MSTSP10 排在 MSTSP2 前。"""
    return frame.sort_values("instance", key=lambda column: column.map(instance_sort_key)).reset_index(drop=True)


def shade_types(ax: plt.Axes, types: list[str]) -> list[Patch]:
    """按 Simple、Geometry、Composite 给图表添加背景分区并返回图例。"""
    for index, kind in enumerate(types):
        ax.axvspan(index - 0.5, index + 0.5, color=TYPE_COLORS.get(kind, "#EEEEEE"), alpha=0.45, zorder=0)
    present = list(dict.fromkeys(types))
    return [Patch(facecolor=TYPE_COLORS[kind], alpha=0.45, label=kind) for kind in present]


def finish(ax: plt.Axes, labels: list[str], destination: Path, *, y_label: str, title: str) -> None:
    """统一设置标题、坐标轴、标签和网格，并以 240 dpi 保存 PNG。"""
    ax.set_xticks(range(len(labels)), labels, rotation=50, ha="right")
    ax.set_xlabel("Instance")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"Generated: {destination}")


def plot_reproduced(
    frame: pd.DataFrame,
    value_column: str,
    std_column: str,
    destination: Path,
    *,
    y_label: str,
    title: str,
    ylim: tuple[float, float] | None = None,
) -> bool:
    """绘制某个复现指标的均值曲线；有标准差时同时显示误差棒。"""
    data = ordered(frame.dropna(subset=[value_column]).copy())
    if data.empty:
        print(f"Skipped {destination.name}: no {value_column} data")
        return False
    figure, ax = plt.subplots(figsize=(12, 5.5))
    positions = list(range(len(data)))
    type_handles = shade_types(ax, data["type"].tolist())
    std_values = pd.to_numeric(data[std_column], errors="coerce")
    errors = std_values.to_numpy() if std_values.notna().any() else None
    line = ax.errorbar(
        positions,
        data[value_column],
        yerr=errors,
        marker="o",
        capsize=3,
        linewidth=1.8,
        color=LINE_COLOR,
        label="Reproduced mean",
        zorder=3,
    )
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(handles=[line, *type_handles], loc="best")
    finish(ax, data["instance"].tolist(), destination, y_label=y_label, title=title)
    return True


def plot_comparison(
    merged: pd.DataFrame,
    reproduced: str,
    paper: str,
    destination: Path,
    *,
    y_label: str,
    title: str,
) -> bool:
    """只对同时拥有论文值和复现值的实例绘制双线对比图。"""
    data = ordered(merged.dropna(subset=[reproduced, paper]).copy())
    if data.empty:
        print(f"Skipped {destination.name}: paper comparison data unavailable")
        return False
    figure, ax = plt.subplots(figsize=(12, 5.5))
    positions = list(range(len(data)))
    type_handles = shade_types(ax, data["type"].tolist())
    reproduced_line, = ax.plot(positions, data[reproduced], marker="o", label="Reproduced", color=LINE_COLOR)
    paper_line, = ax.plot(positions, data[paper], marker="s", label="Paper GB-MACO", color="#B84A3A")
    ax.legend(handles=[paper_line, reproduced_line, *type_handles], loc="best")
    finish(ax, data["instance"].tolist(), destination, y_label=y_label, title=title)
    return True


def plot_bars(data: pd.DataFrame, value: str, destination: Path, *, y_label: str, title: str) -> bool:
    """绘制柱状图；额外绘制数据点，使数值为 0 时仍然可见。"""
    data = ordered(data.dropna(subset=[value]).copy())
    if data.empty:
        print(f"Skipped {destination.name}: no {value} data")
        return False
    figure, ax = plt.subplots(figsize=(12, 5.5))
    positions = list(range(len(data)))
    type_handles = shade_types(ax, data["type"].tolist())
    ax.bar(positions, data[value], color=LINE_COLOR, width=0.68, zorder=2, label=title)
    ax.scatter(positions, data[value], color=LINE_COLOR, s=24, zorder=3)
    ax.legend(handles=type_handles, loc="best")
    finish(ax, data["instance"].tolist(), destination, y_label=y_label, title=title)
    return True


def main() -> int:
    """读取当前可用数据并依次尝试生成全部七类图片。"""
    config = load_config()
    summary_path = result_root(config) / "summary" / "instance_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Run calculate_metrics.py first: {summary_path}")
    summary = pd.read_csv(summary_path)
    for column in ["Fbeta_mean", "Fbeta_std", "DI_mean", "DI_std", "elapsed_mean"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    paper = pd.read_csv(REPRODUCTION_ROOT / "config" / "paper_results.csv")
    paper["paper_fbeta"] = pd.to_numeric(paper["paper_fbeta"], errors="coerce")
    paper["paper_di"] = pd.to_numeric(paper["paper_di"], errors="coerce")
    merged = summary.merge(paper[["instance", "paper_fbeta", "paper_di"]], on="instance", how="left")

    figures = result_root(config) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_reproduced(summary, "Fbeta_mean", "Fbeta_std", figures / "fbeta_reproduced.png", y_label="F-beta", title="GB-MACO Reproduced F-beta", ylim=(0, 1.05))
    plot_comparison(merged, "Fbeta_mean", "paper_fbeta", figures / "fbeta_paper_vs_reproduced.png", y_label="F-beta", title="Paper vs Reproduced F-beta")

    # 差异只使用论文值和复现值同时存在的实例，缺失值不会被当成 0。
    difference = merged.dropna(subset=["Fbeta_mean", "paper_fbeta"]).copy()
    difference["delta_fbeta"] = difference["Fbeta_mean"] - difference["paper_fbeta"]
    if not difference.empty:
        difference = ordered(difference)
        figure, ax = plt.subplots(figsize=(12, 5.5))
        positions = list(range(len(difference)))
        colors = ["#2E7D32" if value >= 0 else "#C0392B" for value in difference["delta_fbeta"]]
        ax.bar(positions, difference["delta_fbeta"], color=colors, zorder=2)
        ax.scatter(positions, difference["delta_fbeta"], color=colors, s=28, zorder=3)
        ax.axhline(0, color="black", linewidth=1)
        finish(ax, difference["instance"].tolist(), figures / "fbeta_difference.png", y_label="Delta F-beta", title="Reproduced minus Paper F-beta")
    else:
        print("Skipped fbeta_difference.png: no paired F-beta data")

    plot_reproduced(summary, "DI_mean", "DI_std", figures / "di_reproduced.png", y_label="DI", title="GB-MACO Reproduced Diversity Indicator", ylim=(0, 1.05))
    plot_comparison(merged, "DI_mean", "paper_di", figures / "di_paper_vs_reproduced.png", y_label="DI", title="Paper vs Reproduced DI")
    plot_bars(summary, "elapsed_mean", figures / "runtime_by_instance.png", y_label="Mean runtime (seconds)", title="Runtime by Instance")

    if paper["paper_fbeta"].notna().all():
        overview = difference.copy()
        overview["absolute_error"] = overview["delta_fbeta"].abs()
        plot_bars(overview, "absolute_error", figures / "reproduction_error_overview.png", y_label="Absolute F-beta error", title="F-beta Reproduction Error Overview")
    else:
        print("Skipped reproduction_error_overview.png: paper F-beta is incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
