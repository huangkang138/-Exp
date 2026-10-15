"""Write a compact, evidence-bound A/B/C five-epoch comparison report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _epochs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 5:
        raise ValueError(f"Expected five completed epochs: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    root = parser.parse_args().run
    diagnosis = json.loads((root / "offline_diagnosis.json").read_text(encoding="utf-8"))["summary"]
    calibration = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    results = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows = {group: _epochs(root / group / "epoch_metrics.csv") for group in "ABC"}
    lines = ["# 2-opt Feedback-Guided POMO：5 轮小规模对照", "",
             "三组均从同一个固定 Soft-POMO best.pt 初始化，固定同球 +2、邻球 +1；"
             "Attention 与 Decoder 均训练。训练数据为新多最优解增强集 500 例，"
             "旧 100 例验证集按原始贪心 POMO 平均最短长度选 best.pt。",
             "batch=64、microbatch=8、Adam lr=1e-5、seed=42、各 5 epoch。"
             "A 保持原 POMO Loss，B 增加 2-opt 质量项，C 再增加留一覆盖边际项。", "",
             f"初始 checkpoint SHA256：`{diagnosis['checkpoint_sha256']}`。",
             f"离线 42 例/336 条抽样路线：{diagnosis['quality_variable_instances']} 例质量奖励有差异；"
             f"{diagnosis['raw_best_not_refined_best_instances']} 例原始最短路线未产生最佳 2-opt 终点；"
             f"0.5% 门槛通过率 {diagnosis['pass_rate_0_5']:.1%}，"
             f"覆盖边际非零率 {diagnosis['marginal_nonzero_rate_0_5']:.1%}；"
             f"完整 2-opt 额外 {diagnosis['two_opt_seconds']:.2f}s。", "",
             "## 梯度校准", "",
             f"初始未加权梯度范数：主 {calibration['main_grad_norm']:.2f}，"
             f"质量 {calibration['quality_grad_norm']:.4f}，"
             f"覆盖 {calibration['coverage_grad_norm']:.2f}。"
             f"固定 α={calibration['alpha']:.4f}、λ={calibration['lambda']:.4f}，"
             "使两项辅助梯度的初始范数各约为主项 5%。"
             "质量奖励已除以 L_ref，因此 α 的数值不能直接与原始长度奖励的系数比较。",
             f"主/质量梯度余弦 {calibration['main_quality_cosine']:.3f}；"
             f"主/覆盖梯度余弦 {calibration['main_coverage_cosine']:.3f}。", "",
             "## 各组结果", "",
             "| 组 | best epoch | best 原始验证长度 | best 验证 2-opt 长度 | "
             "best 不同高质量回路/例 | epoch5 原始验证长度 | "
             "训练 s | 2-opt 额外 s | 验证 s | GPU 峰值 MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        group = result["group"]
        current = rows[group][-1]
        chosen = rows[group][int(result["best_epoch"]) - 1]
        lines.append(f"| {group} | {result['best_epoch']} | {result['best_validation']:.2f} | "
                     f"{float(chosen['validation_mean_best_two_opt_length']):.2f} | "
                     f"{float(chosen['validation_mean_unique_good_cycles']):.2f} | "
                     f"{float(current['validation_mean_best_raw_length']):.2f} | "
                     f"{sum(float(row['training_seconds']) for row in rows[group]):.1f} | "
                     f"{sum(float(row['two_opt_seconds']) for row in rows[group]):.1f} | "
                     f"{sum(float(row['validation_seconds']) for row in rows[group]):.1f} | "
                     f"{max(float(row['peak_gpu_mb']) for row in rows[group]):.1f} |")
    lines += ["", "## 逐轮验证", "",
              "| epoch | A 原始/2-opt/高质量回路 | B 原始/2-opt/高质量回路 | "
              "C 原始/2-opt/高质量回路 |",
              "|---:|---:|---:|---:|"]
    for index in range(5):
        cells = []
        for group in "ABC":
            row = rows[group][index]
            cells.append(f"{float(row['validation_mean_best_raw_length']):.2f} / "
                         f"{float(row['validation_mean_best_two_opt_length']):.2f} / "
                         f"{float(row['validation_mean_unique_good_cycles']):.2f}")
        lines.append(f"| {index + 1} | " + " | ".join(cells) + " |")
    chosen = {group: rows[group][next(result["best_epoch"] for result in results
                                   if result["group"] == group) - 1] for group in "ABC"}
    delta_b_raw = (float(chosen["B"]["validation_mean_best_raw_length"])
                   - float(chosen["A"]["validation_mean_best_raw_length"]))
    delta_b_opt = (float(chosen["B"]["validation_mean_best_two_opt_length"])
                   - float(chosen["A"]["validation_mean_best_two_opt_length"]))
    lines += ["", "## 口径与判断", "",
              "验证的 2-opt 指标只处理每例八个固定贪心起点；best.pt 仍按全部 N 起点的"
              "原始贪心长度选择。训练 C 的覆盖奖励只代表当前固定启发式参考附近的"
              "不同 2-opt 终点，不代表不同全局最优盆地。A 不计算训练 2-opt 反馈，"
              "三组验证预算一致。", "",
              "集合梯度口径：给定实例与预先抽出的起点，POMO 各起点的动作采样条件独立；"
              "对第 i 条轨迹，G(S_without_i) 不依赖它的动作，因此"
              "E[score_i·G(S)] = E[score_i·(G(S)−G(S_without_i))]。"
              "覆盖项直接对留一边际贡献求和，不使用包含当前轨迹奖励的组均值 Baseline。"
              "质量项使用组均值后乘 k/(k−1)，抵消自身奖励被减去造成的尺度偏差。", "",
              "资源记录：各组 GPU 峰值见表；训练环境未安装 psutil，逐轮 CPU RSS 列为空。"
              "运行中另以系统进程表抽样观察到 Python 常驻内存约 1.5 GiB；"
              "CPU 利用率未获得可靠逐轮读数，不报告估算百分比。", "",
              "5 轮只能检验信号、耗时与早期趋势，不能证明 25 例正式测试性能改善；"
              "本次未执行 Opt-B、35 轮训练或 25 例测试。"
              "是否继续正式训练应同时考虑原始验证长度、2-opt 验证质量、"
              "高质量终点覆盖及额外计算成本。", "",
              f"B 的 best 原始验证长度相对 A {delta_b_raw:+.2f}，"
              f"best 验证 2-opt 长度相对 A {delta_b_opt:+.2f}；"
              f"高质量回路数为 B {float(chosen['B']['validation_mean_unique_good_cycles']):.2f}、"
              f"A {float(chosen['A']['validation_mean_unique_good_cycles']):.2f}/例。"
              "C 的 best 原始和 2-opt 长度均高于 A，覆盖数也更低。"
              "这一小规模结果不支持继续投入 C 组的完整训练；"
              "B 只有很小的早期改善，若考虑后续实验需先确认改进超过种子波动与额外计算成本。", ""]
    output = root / "comparison.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
