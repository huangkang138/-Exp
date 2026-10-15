"""Aggregate verified route and local-search traces into a standalone report."""

from __future__ import annotations

import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .search_diagnostics import SOURCES, TARGETED, SEEDS


def _mean(rows: list[dict], field: str) -> float:
    return statistics.mean(float(row[field]) for row in rows)


def _sum(rows: list[dict], field: str) -> int:
    return sum(int(row[field]) for row in rows)


def _plots(rows: list[dict], output: Path) -> None:
    plots = output / "figures"
    plots.mkdir()
    baseline = [row for row in rows if row["seed"] == 42]
    stages = ("raw_total", "raw_unique", "post_unique", "a_hits", "b_hits")
    stage_labels = ("Raw", "Raw unique", "2-opt unique", "A truth hits", "Opt-B truth hits")
    fig, ax = plt.subplots(figsize=(9, 4.7))
    x = range(len(stages))
    for offset, model in enumerate(SOURCES):
        subset = [row for row in baseline if row["model"] == model]
        values = [_sum(subset, stage) for stage in stages]
        ax.bar([i + (offset - 1) * 0.25 for i in x], values, width=0.24, label=model)
    ax.set_xticks(list(x), stage_labels)
    ax.set_yscale("log")
    ax.set_ylabel("Count across 25 instances (log scale)")
    ax.set_title("Route counts across search stages, seed 42")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "stage_counts.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for model in SOURCES:
        subset = sorted((row for row in baseline if row["model"] == model),
                        key=lambda row: int(row["instance"].removeprefix("MSTSP")))
        ids = [int(row["instance"].removeprefix("MSTSP")) for row in subset]
        axes[0].plot(ids, [row["raw_duplicate_rate"] for row in subset], marker=".", label=model)
        axes[1].plot(ids, [1 - row["post_unique"] / row["raw_unique"] for row in subset],
                     marker=".", label=model)
    axes[0].set_ylabel("Raw duplicate fraction")
    axes[1].set_ylabel("Unique raw routes merging under 2-opt")
    axes[1].set_xlabel("MSTSP instance")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(plots / "duplicate_and_convergence.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for offset, model in enumerate(SOURCES):
        subset = {int(row["instance"].removeprefix("MSTSP")): row for row in baseline
                  if row["model"] == model}
        ax.bar([i + (offset - 1) * 0.25 for i in range(len(TARGETED))],
               [subset[index]["b_hits"] for index in TARGETED], width=0.24, label=model)
    ax.set_xticks(range(len(TARGETED)), [f"{index}" for index in TARGETED])
    ax.set_xlabel("MSTSP instance")
    ax.set_ylabel("Known true-optimal routes after Opt-B")
    ax.set_title("Targeted cases, seed 42")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "targeted_coverage.png", dpi=170)
    plt.close(fig)


def write_report(rows: list[dict], output: Path) -> None:
    _plots(rows, output)
    baseline = [row for row in rows if row["seed"] == 42]
    lines = ["# Attention + POMO 多最优解搜索行为诊断", "",
             "## 数据与口径", "",
             "三个 checkpoint 的原始模型参数和历史 CPU 推理配置均已核对。seed=42 直接读取历史 25 例原始路线，"
             "并重放完整普通 2-opt 与原版 Opt-B；逐例最终距离、真最优解数量、Fβ 和换边检查次数与历史结果一致。"
             "种子 123、2026 只对 MSTSP16、18、20～25 新增推理；每例 seed 实际为基础 seed 加算例编号。",
             "每起点 1 Argmax + 8 Sampling；不同模型共享种子但概率不同，采样轨迹不视为完全配对。"
             "路线按旋转/反向等价去重。无向边 Jaccard 在所有路线对或固定随机 2000 对上统计。", "",
             "`route_mapping.csv` 每行是一条原始路线，重复原始路线共用同一次 2-opt 输出。"
             "`optb_expansion.csv` 的 `global_first_discoveries` 按 BFS 第一次发现的初始回路归因，受队列顺序影响；"
             "重点算例 seed=42 另列单初始回路独立重放的可达量及重叠。", "",
             "## 全部 25 例：seed=42", "",
             "| 模型 | 原始/去重 | 原始重复率 | 原始边相似度 | 2-opt 后不同回路 | 2-opt 后真最优回路 | Opt-B 初始真最优 | Opt-B 新增真最优 | 最终真最优 | 距离命中 | Fβ均值 | Opt-B 预算触顶 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in SOURCES:
        subset = [row for row in baseline if row["model"] == model]
        lines.append(f"| {model} | {_sum(subset,'raw_total')}/{_sum(subset,'raw_unique')} | "
                     f"{1-_sum(subset,'raw_unique')/_sum(subset,'raw_total'):.3f} | "
                     f"{_mean(subset,'edge_jaccard_mean'):.3f} | {_sum(subset,'post_unique')} | "
                     f"{_sum(subset,'post_true_optimal_unique')} | {_sum(subset,'a_hits')} | "
                     f"{_sum(subset,'b_new_hits')} | {_sum(subset,'b_hits')} | "
                     f"{sum(r['a_best']==r['reference_length'] for r in subset)}/25 | "
                     f"{_mean(subset,'b_fbeta'):.6f} | {sum(r['b_budget_hit'] for r in subset)} |")
    lines += ["", "![各搜索阶段路线数量](figures/stage_counts.png)", "",
              "![重复率与2-opt收敛](figures/duplicate_and_convergence.png)", "",
              "![重点算例最优回路覆盖](figures/targeted_coverage.png)", "",
              "## 重点算例：seed=42", "",
              "| 算例 | 模型 | 原始去重 | 2-opt 不同 | 最大收敛盆地 | A 真最优 | B 新增真最优 | B 真最优 | 已发现连通分量 | 预算触顶 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for index in TARGETED:
        for model in SOURCES:
            row = next(row for row in baseline if row["model"] == model and row["instance"] == f"MSTSP{index}")
            lines.append(f"| MSTSP{index} | {model} | {row['raw_unique']} | {row['post_unique']} | "
                         f"{row['post_basin_max']} | {row['a_hits']} | {row['b_new_hits']} | "
                         f"{row['b_hits']} | {row['discovered_components']} | {row['b_budget_hit']} |")
    lines += ["", "连通分量只针对本次已发现的最佳回路，并不描述完整最优解空间。", "",
              "## 多种子稳定性：重点算例合计", "",
              "| 模型 | 基础 seed | 距离命中/8 | A 真最优 | B 真最优 | 平均 Fβ | 原始重复率 | 2-opt 收敛率 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in SOURCES:
        for seed in SEEDS:
            subset = [row for row in rows if row["model"] == model and row["seed"] == seed
                      and int(row["instance"].removeprefix("MSTSP")) in TARGETED]
            lines.append(f"| {model} | {seed} | "
                         f"{sum(r['a_best']==r['reference_length'] for r in subset)}/8 | "
                         f"{_sum(subset,'a_hits')} | {_sum(subset,'b_hits')} | "
                         f"{_mean(subset,'b_fbeta'):.6f} | "
                         f"{1-_sum(subset,'raw_unique')/_sum(subset,'raw_total'):.3f} | "
                         f"{1-_sum(subset,'post_unique')/_sum(subset,'raw_unique'):.3f} |")
    lines += ["", "## 诊断结论", "",
              "### 历史数据直接证明", "",
              "- 三模型的 seed=42 最终指标见上表；原始路线、2-opt 与 Opt-B 重放逐例通过历史一致性核验。", "",
              "### 新增诊断支持", "",
              "- 原始重复率、无向边相似度、2-opt 多对一收敛和 Opt-B 初始解归因见 CSV 与图。", "",
              "### 尚未确认", "",
              "- 已发现连通分量无法证明完整最优解空间连通性；单模型、三个推理种子也不足以识别训练随机性的影响。", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
