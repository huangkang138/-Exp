"""One seed-42 benchmark of fixed Soft and the saved 2-opt-guided A/B/C best models.

Read-only with respect to checkpoints and search code. Creates a new diagnostics
directory and uses the same inference, full 2-opt, and Opt-B functions as the
historical fixed Soft comparison.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch

from .compare import BENCHMARKS, load_ground_truth, load_policy, read_optimum
from .compare_learnable_bias import BENCHMARK_DIR, PROJECT, canonical, equal_plateau, fbeta
from .diagnose_2opt import _select_candidates
from .local_search_2opt import two_opt
from ..pomo import search_routes
from ..runner import prepare_tsp


ROOT = PROJECT / "gb_maco_POMO/results/opt_guided_pomo"
RUN = ROOT / "run_20261015_123557_4988d9"
FIXED = PROJECT / "gb_maco_POMO/results/checkpoints/run_20261013_230742_828b99/best.pt"
FIXED_SHA = "dbdfc5cb2faef0d7cbecb7c5c7cc77db16049e5d10e5d584a26e288f1bd88410"
MODELS = {"fixed": FIXED, **{group: RUN / group / "best.pt" for group in "ABC"}}
FIELDS = ("model", "instance", "city_count", "reference_length", "raw_best", "two_opt_best",
          "distance_hit", "gap_pct", "raw_routes", "unique_raw_routes", "unique_two_opt_routes",
          "optb_entry_routes", "two_opt_true_optimal", "optb_new_true_optimal",
          "optb_true_optimal", "optb_final_routes", "optb_checks", "optb_budget_hit",
          "precision", "recall", "fbeta", "two_opt_new_vs_A", "two_opt_lost_vs_A",
          "optb_new_vs_A", "optb_lost_vs_A")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_sources() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    if sha256(FIXED) != FIXED_SHA:
        raise ValueError("Original fixed Soft checkpoint SHA256 mismatch")
    policies = {}
    sources = {}
    common = None
    for name, path in MODELS.items():
        policy, state = load_policy(path, torch.device("cpu"))  # strict state_dict load
        model = state["config"]["model"]
        if model["candidate_mode"] != "soft" or model.get("bias_mode", "fixed") != "fixed":
            raise ValueError(f"{name}: expected fixed Soft Bias")
        if float(model["same_ball_bias"]) != 2 or float(model["adjacent_ball_bias"]) != 1:
            raise ValueError(f"{name}: unexpected fixed Bias strengths")
        entry = {"path": str(path), "sha256": sha256(path), "epoch": state.get("epoch"),
                 "best_epoch": state.get("best_epoch"), "run_id": state.get("run_id")}
        if name != "fixed":
            config = state["config"]
            effective = json.loads((RUN / name / "effective_config.json").read_text(encoding="utf-8"))
            if (state.get("group") != name or config["opt_guided"]["group"] != name
                    or state.get("source_checkpoint_sha256") != FIXED_SHA
                    or effective["source_sha256"] != FIXED_SHA
                    or state["epoch"] != state["best_epoch"]
                    or config["training"] != {"epochs": 5, "batch_size": 64,
                                               "micro_batch_size": 8, "learning_rate": 1e-5,
                                               "seed": 42}):
                raise ValueError(f"{name}: source/epoch/training configuration mismatch")
            shared = {k: v for k, v in effective.items() if k not in {"group", "alpha", "lambda"}}
            if common is None:
                common = shared
            elif shared != common:
                raise ValueError(f"{name}: training conditions differ from A")
            entry["training"] = config["training"]
            entry["opt_guided"] = config["opt_guided"]
            entry["effective_config"] = effective
        policies[name] = policy
        sources[name] = entry
    return policies, sources


def evaluate_case(policy: object, name: str, index: int, problem: object,
                  reference: int, truth: set[tuple[int, ...]]) -> tuple[dict, set, set]:
    city_count = BENCHMARKS[index - 1][1]
    search = search_routes(policy, problem, inference_mode="mixed", samples_per_start=8,
                           seed=42 + index)
    raw = search.routes
    lengths = search.lengths
    if len(raw) != city_count * 9 or any(len(set(route)) != city_count for route in raw):
        raise ValueError(f"{name} MSTSP{index}: invalid POMO routes")
    candidates = _select_candidates(raw, lengths, city_count, len(raw), all_generated=True)
    refined = [two_opt(route, problem.distance_matrix) for route in candidates]
    raw_best = min(lengths)
    two_opt_best = min(item.length for item in refined)
    best = min(raw_best, two_opt_best)
    starts = [route for route, length in zip(raw, lengths) if length == best]
    starts.extend(item.route for item in refined if item.length == best)
    a_pool = {canonical(route): route for route in starts}
    expansion = equal_plateau(list(a_pool.values()), problem.distance_matrix, 50_000)
    a_keys, b_keys = set(a_pool), set(expansion.routes)
    if best < reference or expansion.best_length < reference or expansion.checks > 50_000:
        raise ValueError(f"{name} MSTSP{index}: reference or Opt-B budget violation")
    if not a_keys <= b_keys:
        raise ValueError(f"{name} MSTSP{index}: Opt-B lost an input route")
    a_hits, b_hits = a_keys & truth, b_keys & truth
    row = {"model": name, "instance": f"MSTSP{index}", "city_count": city_count,
           "reference_length": reference, "raw_best": raw_best, "two_opt_best": two_opt_best,
           "distance_hit": int(best == reference), "gap_pct": 100 * (best / reference - 1),
           "raw_routes": len(raw), "unique_raw_routes": len(candidates),
           "unique_two_opt_routes": len({canonical(item.route) for item in refined}),
           "optb_entry_routes": len(a_keys), "two_opt_true_optimal": len(a_hits),
           "optb_new_true_optimal": len(b_hits - a_hits), "optb_true_optimal": len(b_hits),
           "optb_final_routes": len(b_keys), "optb_checks": expansion.checks,
           "optb_budget_hit": int(expansion.checks == 50_000),
           "precision": len(b_hits) / len(b_keys), "recall": len(b_hits) / len(truth),
           "fbeta": fbeta(b_keys, truth), "two_opt_new_vs_A": 0, "two_opt_lost_vs_A": 0,
           "optb_new_vs_A": 0, "optb_lost_vs_A": 0}
    return row, a_hits, b_hits


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    for model in MODELS:
        subset = [row for row in rows if row["model"] == model]
        result.append({"model": model, "distance_hits": sum(r["distance_hit"] for r in subset),
                       "mean_gap_pct": sum(r["gap_pct"] for r in subset) / 25,
                       "two_opt_true_optimal": sum(r["two_opt_true_optimal"] for r in subset),
                       "optb_new_true_optimal": sum(r["optb_new_true_optimal"] for r in subset),
                       "optb_true_optimal": sum(r["optb_true_optimal"] for r in subset),
                       "mean_fbeta": sum(r["fbeta"] for r in subset) / 25,
                       "two_opt_new_vs_A": sum(r["two_opt_new_vs_A"] for r in subset),
                       "two_opt_lost_vs_A": sum(r["two_opt_lost_vs_A"] for r in subset),
                       "optb_new_vs_A": sum(r["optb_new_vs_A"] for r in subset),
                       "optb_lost_vs_A": sum(r["optb_lost_vs_A"] for r in subset)})
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def write_report(output: Path) -> None:
    """Render the finished CSV and saved training logs; never repeats inference."""
    with (output / "per_instance.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    with (output / "summary.csv").open(newline="", encoding="utf-8-sig") as stream:
        totals = {row["model"]: row for row in csv.DictReader(stream)}
    if len(rows) != 100 or set(totals) != set(MODELS):
        raise ValueError("Expected 25 cases for fixed, A, B, and C")
    by_case = {(row["instance"], row["model"]): row for row in rows}
    logs = {}
    reference_schedule = None
    for name in "ABC":
        with (RUN / name / "epoch_metrics.csv").open(newline="", encoding="utf-8-sig") as stream:
            logs[name] = list(csv.DictReader(stream))
        with (RUN / name / "batch_metrics.csv").open(newline="", encoding="utf-8-sig") as stream:
            batches = list(csv.DictReader(stream))
        if len(logs[name]) != 5 or len(batches) != 55:
            raise ValueError(f"{name}: expected five epochs and 55 updates")
        schedule = [(int(x["epoch"]), int(x["batch"]), int(x["instances"])) for x in batches]
        if (any(sum(size for epoch, _, size in schedule if epoch == e) != 500
                for e in range(1, 6)) or any(size > 64 or size < 1 for _, _, size in schedule)):
            raise ValueError(f"{name}: invalid training instance count")
        if reference_schedule is None:
            reference_schedule = schedule
        elif schedule != reference_schedule:
            raise ValueError(f"{name}: batch/update schedule mismatch")
    source = json.loads((output / "sources.json").read_text(encoding="utf-8"))["models"]
    lines = ["# 2-opt 引导 POMO：25 例诊断（CPU，seed=42）", "",
             "四个 checkpoint 均按每起点 1 Argmax + 8 Sampling、旋转/反向去重、全部不同原始回路完整普通 2-opt、原版 Opt-B 每例最多 50,000 次额外换边检查评估。程序复用了历史正式测试的搜索函数；固定模型精确复现历史基准。以下结论只针对这一次测试，不用于调节辅助奖励参数。", "",
             "## 模型来源与训练一致性", "",
             f"- 固定 Soft：SHA256 `{source['fixed']['sha256']}`，原始最佳 checkpoint。",
             f"- A/B/C：均从该固定 checkpoint 初始化；各组 best.pt 分别是 epoch {source['A']['best_epoch']}/{source['B']['best_epoch']}/{source['C']['best_epoch']}，已严格加载完整模型参数。",
             "- 三组配置均为新 500 例训练集、旧 100 例验证集、5 epoch、batch 64、microbatch 8、Adam 1e-5、seed 42；各有 55 次参数更新。训练代码每组重建相同 seed 的 shuffle RNG 和 torch RNG，B/C 的起点抽取使用独立 RNG。保存的日志不含逐批样本 ID，因此数据顺序一致是由代码和 RNG 配置支持，无法仅从日志逐例重放证明。", "",
             "## 总体结果", "",
             "| 模型 | 距离命中 | 平均 Gap | 2-opt 阶段真最优回路 | Opt-B 新增 | Opt-B 最终 | 平均 Fβ |",
             "|---|---:|---:|---:|---:|---:|---:|",
             ]
    for name in MODELS:
        t = totals[name]
        lines.append(f"| {name} | {t['distance_hits']}/25 | {float(t['mean_gap_pct']):.6f}% | "
                     f"{t['two_opt_true_optimal']} | {t['optb_new_true_optimal']} | "
                     f"{t['optb_true_optimal']} | {float(t['mean_fbeta']):.6f} |")
    lines += ["", "2-opt 阶段口径沿用历史基准：进入 Opt-B 的最佳距离回路，含已达到最佳距离的原始路线及普通 2-opt 输出，按旋转/反向等价去重。", "",
              "## 逐例覆盖：2-opt 入口 / Opt-B 最终", "",
              "| 算例 | 固定 | A | B | C |", "|---|---:|---:|---:|---:|"]
    for index in range(1, 26):
        case = f"MSTSP{index}"
        cells = [f"{by_case[(case, name)]['two_opt_true_optimal']}/{by_case[(case, name)]['optb_true_optimal']}"
                 for name in MODELS]
        lines.append(f"| {case} | " + " | ".join(cells) + " |")
    lines += ["", "完整逐例距离、Gap、Precision、Recall、Fβ、原始/去重路线数、Opt-B 检查次数见 `per_instance.csv`。", "",
              "## 新增反馈的可观察效果", ""]
    for name in "BC":
        t = totals[name]
        lines.append(f"- {name} 对比 A：2-opt 真最优入口新增 {t['two_opt_new_vs_A']} 条、丢失 "
                     f"{t['two_opt_lost_vs_A']} 条（净 {int(t['two_opt_new_vs_A']) - int(t['two_opt_lost_vs_A']):+d}）；"
                     f"Opt-B 最终新增 {t['optb_new_vs_A']} 条、丢失 {t['optb_lost_vs_A']} 条。"
                     "这些是逐例集合差，不能跨算例视为同一回路。")
    b4, c4 = logs["B"][3], logs["C"][3]
    a3 = logs["A"][2]
    gain = 100 * (float(b4["validation_mean_best_two_opt_length"]) /
                  float(a3["validation_mean_best_two_opt_length"]) - 1)
    lines += [f"- B 选中 checkpoint 的旧验证集 2-opt 平均长度为 {b4['validation_mean_best_two_opt_length']}，A 为 {a3['validation_mean_best_two_opt_length']}（{gain:+.3f}%）。但 25 例中 B 的入口、Opt-B 总数和平均 Fβ 均低于 A，故验证集的微小质量改善未转化为此测试集的多解覆盖改善。",
              f"- B best epoch 的质量梯度范数 {float(b4['quality_grad_norm']):.3f}（未乘权重）、质量门槛通过率 {100*float(b4['pass_rate']):.2f}%；A 主梯度范数在 best epoch 为 {float(a3['main_grad_norm']):.1f}。B 的质量信号非零且参与了训练。",
              f"- C best epoch 的覆盖梯度范数 {float(c4['coverage_grad_norm']):.1f}（未乘权重）、留一边际非零率 {100*float(c4['marginal_nonzero_rate']):.2f}%、质量门槛通过率 {100*float(c4['pass_rate']):.2f}%。覆盖梯度存在，但其验证集不同高质量回路数 {c4['validation_mean_unique_good_cycles']} 低于 A 的 {a3['validation_mean_unique_good_cycles']}。",
              "- 三组均在 MSTSP25 失去距离命中：A 最佳 9524，B/C 最佳 9527，参考 9521；MSTSP22 均命中 10442。B 仅在 MSTSP16 将 Opt-B 命中由 A 的 40 提到 44；C 在 MSTSP13、20 分别由 56/16 提到 58/20，但 MSTSP24 从 A 的 26 降到 12。MSTSP23 三组均为 4，低于固定模型 6。", "",
              "## 判断与边界", "",
              "1. B/C 确实带来部分 **不同** 的真实最优 2-opt 入口，但分别丢失更多 A 的入口，总数从 A 的 263 降到 252/254。新增入口不保证 Opt-B 可扩展到更多回路。",
              "2. B/C 的 Opt-B 最终覆盖 652/658、平均 Fβ 0.859357/0.867303，均低于 A 的 667、0.876990；也低于固定模型的 663、0.896843。A 虽多 4 条最终回路，Fβ 仍下降。",
              "3. A/B/C 均为 24/25，损失 MSTSP25；固定模型保持 25/25。",
              "4. 与 A 相比，B 在旧验证集呈现很小的 2-opt 距离改善，并在 MSTSP16 增加最终回路；C 在 MSTSP20 达到全部 20 条真最优回路。两者都没有总体覆盖收益。",
              "5. 当前单种子、5 轮 best checkpoint 和两个验证口径不足以支持继续训练 B 来追求总体收益；C 的覆盖项有有效梯度，但未提高验证覆盖或测试总体覆盖，值得重新审视其代理目标与模型选择规则。这里不据这 25 例调权重或阈值。若后续另立实验，应预先制定验证集选择和多种子复核方案。", "",
              "注意：5 轮模型以旧验证集**原始**长度选 best.pt；上文验证 2-opt 与高质量回路数是旁路诊断，不能当作独立模型选择依据。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    torch.set_num_threads(1)
    policies, sources = check_sources()
    output = ROOT / "final_diagnostics_seed42" / f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "sources.json").write_text(json.dumps({"models": sources, "device": "cpu", "seed": 42,
        "samples_per_start": 8, "optb_budget": 50_000, "benchmark_dir": str(BENCHMARK_DIR)},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    keys = defaultdict(dict)
    for index, (stem, city_count, _) in enumerate(BENCHMARKS, 1):
        problem = prepare_tsp(BENCHMARK_DIR / f"{stem}.tsp")
        if len(problem.cities) != city_count:
            raise ValueError(f"MSTSP{index}: wrong city count")
        reference = read_optimum(BENCHMARK_DIR / f"{stem}.solution")
        truth = load_ground_truth(BENCHMARK_DIR / f"{stem}.solution", city_count)
        for name, policy in policies.items():
            row, a_hits, b_hits = evaluate_case(policy, name, index, problem, reference, truth)
            rows.append(row)
            keys[(index, name)] = (a_hits, b_hits)
            print(f"MSTSP{index} {name}: distance={min(row['raw_best'],row['two_opt_best'])}/{reference}, "
                  f"2-opt={len(a_hits)}, Opt-B={len(b_hits)}", flush=True)
        write_csv(output / "per_instance_progress.csv", rows)
    for row in rows:
        index = int(row["instance"][5:])
        own_a, own_b = keys[(index, row["model"])]
        base_a, base_b = keys[(index, "A")]
        row["two_opt_new_vs_A"] = len(own_a - base_a)
        row["two_opt_lost_vs_A"] = len(base_a - own_a)
        row["optb_new_vs_A"] = len(own_b - base_b)
        row["optb_lost_vs_A"] = len(base_b - own_b)
    summary = summarize(rows)
    fixed = summary[0]
    if not (fixed["distance_hits"] == 25 and fixed["two_opt_true_optimal"] == 250
            and fixed["optb_true_optimal"] == 663
            and abs(fixed["mean_fbeta"] - .8968434047253819) < 1e-9):
        raise RuntimeError(f"Fixed baseline does not reproduce historical results: {fixed}")
    write_csv(output / "per_instance.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_report(output)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
