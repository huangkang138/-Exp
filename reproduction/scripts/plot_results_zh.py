"""根据现有汇总 CSV 生成便于阅读的中文结果图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
import pandas as pd

from common import REPRODUCTION_ROOT, instance_sort_key, load_config, result_root


TYPE_COLORS = {"Simple": "#DCEAF7", "Geometry": "#E4F2E4", "Composite": "#F8E6D4"}
TYPE_NAMES = {"Simple": "简单型", "Geometry": "几何型", "Composite": "复合型"}
LINE_COLOR = "#245B8A"
PAPER_COLOR = "#B84A3A"


def configure_chinese_font() -> FontProperties:
    """优先使用 Windows 自带微软雅黑，避免中文保存成方框。"""
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    font_path = next((path for path in candidates if path.is_file()), None)
    if font_path is None:
        raise FileNotFoundError("未找到微软雅黑或黑体，无法可靠生成中文图片")
    font = FontProperties(fname=str(font_path))
    plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return font


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("instance", key=lambda col: col.map(instance_sort_key)).reset_index(drop=True)


def shade_types(ax: plt.Axes, types: list[str]) -> list[Patch]:
    for index, kind in enumerate(types):
        ax.axvspan(index - 0.5, index + 0.5, color=TYPE_COLORS[kind], alpha=0.45, zorder=0)
    return [
        Patch(facecolor=TYPE_COLORS[kind], alpha=0.45, label=TYPE_NAMES[kind])
        for kind in dict.fromkeys(types)
    ]


def finish(
    ax: plt.Axes,
    labels: list[str],
    destination: Path,
    *,
    y_label: str,
    title: str,
    explanation: str,
) -> None:
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_xlabel("测试实例")
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=15, pad=24)
    ax.text(0.5, 1.01, explanation, transform=ax.transAxes, ha="center", va="bottom", color="#555555")
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"已生成：{destination}")


def plot_metric(
    frame: pd.DataFrame,
    value_column: str,
    std_column: str,
    destination: Path,
    *,
    y_label: str,
    title: str,
    explanation: str,
) -> None:
    data = ordered(frame.dropna(subset=[value_column]).copy())
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
        label="本次复现结果",
        zorder=3,
    )
    ax.set_ylim(max(0.0, float(data[value_column].min()) - 0.01), 1.002)
    ax.legend(handles=[line, *type_handles], loc="best")
    finish(ax, data["instance"].tolist(), destination, y_label=y_label, title=title, explanation=explanation)


def plot_comparison(merged: pd.DataFrame, destination: Path) -> None:
    data = ordered(merged.dropna(subset=["Fbeta_mean", "paper_fbeta"]).copy())
    figure, ax = plt.subplots(figsize=(12, 5.5))
    positions = list(range(len(data)))
    type_handles = shade_types(ax, data["type"].tolist())
    reproduced, = ax.plot(positions, data["Fbeta_mean"], marker="o", color=LINE_COLOR, label="本次复现")
    paper, = ax.plot(positions, data["paper_fbeta"], marker="s", color=PAPER_COLOR, label="论文结果")
    ax.legend(handles=[paper, reproduced, *type_handles], loc="best")
    finish(
        ax,
        data["instance"].tolist(),
        destination,
        y_label="Fβ",
        title="论文结果与本次复现 Fβ 对比",
        explanation="两条线越重合，说明复现结果与论文越一致",
    )


def plot_bar(
    data: pd.DataFrame,
    value: str,
    destination: Path,
    *,
    y_label: str,
    title: str,
    explanation: str,
    colors: list[str] | str = LINE_COLOR,
) -> None:
    data = ordered(data.dropna(subset=[value]).copy())
    figure, ax = plt.subplots(figsize=(12, 5.5))
    positions = list(range(len(data)))
    type_handles = shade_types(ax, data["type"].tolist())
    ax.bar(positions, data[value], color=colors, width=0.68, zorder=2)
    ax.scatter(positions, data[value], color=colors, s=24, zorder=3)
    ax.legend(handles=type_handles, loc="best")
    finish(ax, data["instance"].tolist(), destination, y_label=y_label, title=title, explanation=explanation)


def main() -> int:
    configure_chinese_font()
    config = load_config()
    summary = pd.read_csv(result_root(config) / "summary" / "instance_summary.csv")
    for column in ["Fbeta_mean", "Fbeta_std", "DI_mean", "DI_std", "elapsed_mean"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    paper = pd.read_csv(REPRODUCTION_ROOT / "config" / "paper_results.csv")
    paper["paper_fbeta"] = pd.to_numeric(paper["paper_fbeta"], errors="coerce")
    merged = summary.merge(paper[["instance", "paper_fbeta"]], on="instance", how="left")
    figures = result_root(config) / "figures"

    plot_metric(
        summary, "Fbeta_mean", "Fbeta_std", figures / "fbeta_reproduced_zh.png",
        y_label="Fβ", title="GB-MACO 本次复现的 Fβ",
        explanation="越接近 1 越好：同时反映找到真实最优路线的完整性和准确性",
    )
    plot_comparison(merged, figures / "fbeta_paper_vs_reproduced_zh.png")

    difference = ordered(merged.dropna(subset=["Fbeta_mean", "paper_fbeta"]).copy())
    difference["delta_fbeta"] = difference["Fbeta_mean"] - difference["paper_fbeta"]
    delta_colors = ["#2E7D32" if value >= 0 else "#C0392B" for value in difference["delta_fbeta"]]
    plot_bar(
        difference, "delta_fbeta", figures / "fbeta_difference_zh.png",
        y_label="Fβ 差值", title="本次复现 Fβ − 论文 Fβ",
        explanation="0 表示完全一致；负值表示本次结果低于论文",
        colors=delta_colors,
    )

    plot_metric(
        summary, "DI_mean", "DI_std", figures / "di_reproduced_zh.png",
        y_label="多样性指标 DI", title="GB-MACO 本次复现的多样性指标",
        explanation="越接近 1，说明算法返回的路线对真实最优路线覆盖得越完整",
    )
    plot_bar(
        summary, "elapsed_mean", figures / "runtime_by_instance_zh.png",
        y_label="平均运行时间（秒）", title="各测试实例运行耗时",
        explanation="柱子越高，说明该实例计算耗时越长；当前每个实例仅运行 1 次",
    )

    difference["absolute_error"] = difference["delta_fbeta"].abs()
    plot_bar(
        difference, "absolute_error", figures / "reproduction_error_overview_zh.png",
        y_label="Fβ 绝对误差", title="Fβ 复现误差总览",
        explanation="越接近 0 越好；表示本次结果与论文结果的偏离程度",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
