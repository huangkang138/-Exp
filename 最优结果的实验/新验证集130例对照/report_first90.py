"""Build a colored, paired report from the first completed cases of a saved run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT = Path(r"D:\Project\球粒计算\GB-MACO_python")
DATA = PROJECT / "benachmark_STSP_Complex_Multiopt_Test"
HISTORY = PROJECT / "gb_maco_POMO/results/learnable_bias/comparison/run_20261014_233707"
PAPER = PROJECT / "reproduction/config/paper_results.csv"
BLUE, GREEN, YELLOW = "#38BDF8", "#34D399", "#FBBF24"


def cell(value: object, color: str) -> str:
    return f'<span style="color:{color}">{value}</span>'


def canonical(route: list[int]) -> tuple[int, ...]:
    first = route.index(min(route))
    forward = route[first:] + route[:first]
    backward = [forward[0], *reversed(forward[1:])]
    return min(tuple(forward), tuple(backward))


def table(headers: list[tuple[str, str | None]], rows: list[list[object]]) -> list[str]:
    result = ['<table border="1" cellspacing="0" cellpadding="6">',
              '<thead><tr>' + ''.join(
                  f'<th style="color:{color}">{title}</th>' if color else f'<th>{title}</th>'
                  for title, color in headers
              ) + '</tr></thead>', '<tbody>']
    for row in rows:
        result.append('<tr>' + ''.join(f'<td>{item}</td>' for item in row) + '</tr>')
    result += ['</tbody>', '</table>']
    return result


def read_cases(run_dir: Path, count: int):
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    names = protocol["cases"][:count]
    if len(names) != count or count < 1:
        raise ValueError("Requested count is outside the saved run")
    with (DATA / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = {row["name"]: row for row in csv.DictReader(stream)}
    models = {}
    for model in ("fixed_soft", "learnable_bias"):
        model_info = json.loads((run_dir / model / "model.json").read_text(encoding="utf-8"))
        if model_info["checkpoint_sha256"] != protocol["checkpoints"][model]:
            raise ValueError(f"{model}: checkpoint differs from saved protocol")
        models[model] = []
        for name in names:
            path = run_dir / model / "cases" / f"{name}.json"
            if not path.is_file():
                raise FileNotFoundError(f"{model} has not completed {name}: {path}")
            row = json.loads(path.read_text(encoding="utf-8"))
            source = manifest[name]
            if (row["name"] != name or row["file"] != source["file"] or
                    row["n"] != int(source["n"]) or
                    row["reference_length"] != int(source["exact_optimal_length"])):
                raise ValueError(f"{model}: saved result mismatches manifest for {name}")
            models[model].append(row)
    for fixed, learned in zip(models["fixed_soft"], models["learnable_bias"]):
        for key in ("name", "file", "n", "reference_length", "starts",
                    "samples_per_start", "raw_routes"):
            if fixed[key] != learned[key]:
                raise ValueError(f"Models have different inputs or generation budgets: {fixed['name']} {key}")
    return protocol, names, models, manifest


def proof_witness_coverage(run_dir: Path, names: list[str], manifest: dict) -> dict:
    """Coverage of the saved certified subset, never full optimal-set recall."""
    result = {"fixed_soft": [], "learnable_bias": []}
    for name in names:
        source = manifest[name]
        proof = json.loads((DATA / source["proof_file"]).read_text(encoding="utf-8"))
        witnesses = {canonical(route) for route in proof["optimal_tours"]}
        expected = int(source["certified_optimal_witness_count"])
        if proof["instance"] != name or len(witnesses) != expected:
            raise ValueError(f"Invalid certified witness set: {name}")
        for model in result:
            path = run_dir / model / "best_routes" / f"{name}.json"
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["name"] != name:
                raise ValueError(f"Wrong saved best routes: {path}")
            returned = {canonical(route) for route in saved["after_optb"]["routes"]}
            hits = len(returned & witnesses)
            result[model].append({"name": name, "hit": hits, "total": expected,
                                  "coverage": hits / expected})
    return result


def summary(rows: list[dict]) -> dict:
    count = len(rows)
    return {
        "cases": count,
        "raw_hit": sum(r["raw_best"] == r["reference_length"] for r in rows),
        "a_hit": sum(r["a_best"] == r["reference_length"] for r in rows),
        "b_hit": sum(r["b_best"] == r["reference_length"] for r in rows),
        "raw_gap": sum(r["raw_gap_pct"] for r in rows) / count,
        "a_gap": sum(r["a_gap_pct"] for r in rows) / count,
        "b_gap": sum(r["b_gap_pct"] for r in rows) / count,
        "a_routes": sum(r["a_proven_optimal_routes"] for r in rows),
        "b_routes": sum(r["b_proven_optimal_routes"] for r in rows),
        "inference_seconds": sum(r["inference_seconds"] for r in rows),
        "two_opt_seconds": sum(r["two_opt_seconds"] for r in rows),
        "optb_seconds": sum(r["optb_seconds"] for r in rows),
        "total_seconds": sum(r["total_seconds"] for r in rows),
    }


def write_report(run_dir: Path, output_dir: Path, count: int):
    protocol, names, models, manifest = read_cases(run_dir, count)
    fixed, learned = models["fixed_soft"], models["learnable_bias"]
    sf, sl = summary(fixed), summary(learned)
    witnesses = proof_witness_coverage(run_dir, names, manifest)
    wf, wl = witnesses["fixed_soft"], witnesses["learnable_bias"]
    for stats, values in ((sf, wf), (sl, wl)):
        stats["saved_witness_hits"] = sum(row["hit"] for row in values)
        stats["saved_witness_total"] = sum(row["total"] for row in values)
        stats["saved_witness_coverage_micro"] = stats["saved_witness_hits"] / stats["saved_witness_total"]
        stats["saved_witness_coverage_macro"] = sum(row["coverage"] for row in values) / count
    wins = {
        "fixed_shorter": sum(a["b_best"] < b["b_best"] for a, b in zip(fixed, learned)),
        "same_distance": sum(a["b_best"] == b["b_best"] for a, b in zip(fixed, learned)),
        "learned_shorter": sum(b["b_best"] < a["b_best"] for a, b in zip(fixed, learned)),
        "fixed_more_routes": sum(a["b_proven_optimal_routes"] > b["b_proven_optimal_routes"]
                                 for a, b in zip(fixed, learned)),
        "same_routes": sum(a["b_proven_optimal_routes"] == b["b_proven_optimal_routes"]
                           for a, b in zip(fixed, learned)),
        "learned_more_routes": sum(b["b_proven_optimal_routes"] > a["b_proven_optimal_routes"]
                                   for a, b in zip(fixed, learned)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paired = []
    for a, b, proof_a, proof_b in zip(fixed, learned, wf, wl):
        paired.append({
            "name": a["name"], "n": a["n"], "tier": a["tier"], "shape": a["shape"],
            "reference_length": a["reference_length"],
            "fixed_raw_best": a["raw_best"], "learned_raw_best": b["raw_best"],
            "fixed_2opt_best": a["a_best"], "learned_2opt_best": b["a_best"],
            "fixed_optb_best": a["b_best"], "learned_optb_best": b["b_best"],
            "fixed_optb_gap_pct": a["b_gap_pct"], "learned_optb_gap_pct": b["b_gap_pct"],
            "fixed_2opt_optimal_routes": a["a_proven_optimal_routes"],
            "learned_2opt_optimal_routes": b["a_proven_optimal_routes"],
            "fixed_optb_optimal_routes": a["b_proven_optimal_routes"],
            "learned_optb_optimal_routes": b["b_proven_optimal_routes"],
            "saved_witness_count": proof_a["total"],
            "fixed_saved_witness_hits": proof_a["hit"],
            "learned_saved_witness_hits": proof_b["hit"],
            "fixed_saved_witness_coverage": proof_a["coverage"],
            "learned_saved_witness_coverage": proof_b["coverage"],
            "fixed_2opt_candidates": a["two_opt_candidates"],
            "learned_2opt_candidates": b["two_opt_candidates"],
            "fixed_total_seconds": a["total_seconds"],
            "learned_total_seconds": b["total_seconds"],
        })
    with (output_dir / "paired_first90.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    with PAPER.open(newline="", encoding="utf-8-sig") as stream:
        paper_rows = list(csv.DictReader(stream))
    paper_mean = sum(float(r["paper_fbeta"]) for r in paper_rows) / len(paper_rows)
    old = json.loads((HISTORY / "summary.json").read_text(encoding="utf-8"))
    if old["cases"] != len(paper_rows) or old["fixed_baseline_sha256"] == "":
        raise ValueError("Historical paper comparison is incomplete")
    summary_data = {
        "scope": "first 90 new benchmark cases only",
        "run_dir": str(run_dir), "protocol": protocol,
        "fixed_soft": sf, "learnable_bias": sl, "paired_wins": wins,
        "historical_25_only": {
            "fixed_fbeta": old["fixed_mean_fbeta"],
            "learnable_fbeta": old["learnable_mean_fbeta"],
            "paper_fbeta": paper_mean,
        },
    }
    (output_dir / "summary_first90.json").write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 新测试集前 90 例：固定 Soft-POMO 与可学习 Bias-POMO", "",
        f"同一测试集清单的前 **{count}** 例；两个模型均已逐例完成。"
        f"城市数范围 {fixed[0]['n']}～{fixed[-1]['n']}。",
        f"POMO 推理设备 {protocol['device']}，种子 {protocol['seed']}；"
        f"每例最多 {protocol['max_starts']} 个均匀起点，每起点 1 Argmax + "
        f"{protocol['samples_per_start']} Sampling；去重后最多 "
        f"{protocol['max_2opt_candidates']} 条原始路线做完整普通 2-opt，"
        f"Opt-B 最多 {protocol['optb_extra_pair_checks']:,} 次额外换边检查。",
        "两个模型的输入算例、起点和预算上限相同；去重后的实际 2-opt 条数可能不同。"
        "本次预算与下面历史 25 例的全起点、全候选预算不同。", "",
        f'{cell("蓝色：数据集已证最短距离及历史论文指标", BLUE)}；'
        f'{cell("绿色：固定 Soft-POMO", GREEN)}；'
        f'{cell("黄色：可学习 Bias-POMO", YELLOW)}。', "",
        f"**Opt-B 后最短距离命中率：固定 {sf['b_hit']}/{count}（{sf['b_hit']/count:.1%}），"
        f"可学习 {sl['b_hit']}/{count}（{sl['b_hit']/count:.1%}）；"
        f"逐例最佳距离：固定更短 {wins['fixed_shorter']}、相同 {wins['same_distance']}、"
        f"可学习更短 {wins['learned_shorter']}。**", "",
    ]
    metrics = [
        ("原始路线最短距离命中率", f"{sf['raw_hit']}/{count} ({sf['raw_hit']/count:.1%})",
         f"{sl['raw_hit']}/{count} ({sl['raw_hit']/count:.1%})"),
        ("2-opt 后最短距离命中率", f"{sf['a_hit']}/{count} ({sf['a_hit']/count:.1%})",
         f"{sl['a_hit']}/{count} ({sl['a_hit']/count:.1%})"),
        ("Opt-B 后最短距离命中率", f"{sf['b_hit']}/{count} ({sf['b_hit']/count:.1%})",
         f"{sl['b_hit']}/{count} ({sl['b_hit']/count:.1%})"),
        ("已保存证明回路覆盖率（按回路汇总）",
         f"{sf['saved_witness_hits']}/{sf['saved_witness_total']} "
         f"({sf['saved_witness_coverage_micro']:.2%})",
         f"{sl['saved_witness_hits']}/{sl['saved_witness_total']} "
         f"({sl['saved_witness_coverage_micro']:.2%})"),
        ("已保存证明回路覆盖率（逐例等权）",
         f"{sf['saved_witness_coverage_macro']:.2%}",
         f"{sl['saved_witness_coverage_macro']:.2%}"),
        ("完整最优回路召回率（旧报告口径）", "无法计算", "无法计算"),
        ("原始路线平均 Gap", f"{sf['raw_gap']:.4f}%", f"{sl['raw_gap']:.4f}%"),
        ("2-opt 后平均 Gap", f"{sf['a_gap']:.4f}%", f"{sl['a_gap']:.4f}%"),
        ("Opt-B 后平均 Gap", f"{sf['b_gap']:.4f}%", f"{sl['b_gap']:.4f}%"),
        ("2-opt 后真最优回路（逐例求和）", sf["a_routes"], sl["a_routes"]),
        ("Opt-B 后真最优回路（逐例求和）", sf["b_routes"], sl["b_routes"]),
        ("POMO 推理耗时", f"{sf['inference_seconds']:.1f}s", f"{sl['inference_seconds']:.1f}s"),
        ("2-opt 耗时", f"{sf['two_opt_seconds']:.1f}s", f"{sl['two_opt_seconds']:.1f}s"),
        ("Opt-B 额外耗时", f"{sf['optb_seconds']:.1f}s", f"{sl['optb_seconds']:.1f}s"),
        ("逐例总耗时求和", f"{sf['total_seconds']:.1f}s", f"{sl['total_seconds']:.1f}s"),
    ]
    lines += ["## 总体结果", ""] + table(
        [("指标", None), ("固定 Soft", GREEN), ("可学习 Bias", YELLOW)],
        [[label, cell(a, GREEN), cell(b, YELLOW)] for label, a, b in metrics]
    )
    lines += ["", "最佳距离逐例胜负：固定更短 "
              f"{wins['fixed_shorter']}、相同 {wins['same_distance']}、可学习更短 "
              f"{wins['learned_shorter']}。",
              "Opt-B 后真最优回路数逐例胜负：固定更多 "
              f"{wins['fixed_more_routes']}、相同 {wins['same_routes']}、可学习更多 "
              f"{wins['learned_more_routes']}。", "",
              "## 逐例距离与 Gap", "",
              "Gap =（模型最佳路线长度 ÷ 数据集已证明的全局最短距离 − 1）×100%。", ""]
    distance_rows = []
    for a, b in zip(fixed, learned):
        distance_rows.append([
            a["name"], a["n"], cell(a["reference_length"], BLUE),
            cell(a["raw_best"], GREEN), cell(b["raw_best"], YELLOW),
            cell(a["a_best"], GREEN), cell(b["a_best"], YELLOW),
            cell(a["b_best"], GREEN), cell(b["b_best"], YELLOW),
            cell(f"{a['b_gap_pct']:.4f}%", GREEN), cell(f"{b['b_gap_pct']:.4f}%", YELLOW),
        ])
    lines += table([
        ("算例", None), ("城市", None), ("已证最短", BLUE),
        ("固定混合", GREEN), ("可学混合", YELLOW),
        ("固定 2-opt", GREEN), ("可学 2-opt", YELLOW),
        ("固定 Opt-B", GREEN), ("可学 Opt-B", YELLOW),
        ("固定 Gap", GREEN), ("可学 Gap", YELLOW),
    ], distance_rows)
    lines += ["", "## 逐例最优回路与耗时", "",
              "最优回路按旋转、反向等价去重；只要合法路线长度等于已证最短距离，"
              "就计为本次找到的真最优回路。", ""]
    route_rows = []
    for a, b in zip(fixed, learned):
        route_rows.append([
            a["name"], cell(a["a_proven_optimal_routes"], GREEN),
            cell(b["a_proven_optimal_routes"], YELLOW),
            cell(a["b_proven_optimal_routes"], GREEN),
            cell(b["b_proven_optimal_routes"], YELLOW),
            cell(a["two_opt_candidates"], GREEN), cell(b["two_opt_candidates"], YELLOW),
            cell(f"{a['total_seconds']:.2f}", GREEN),
            cell(f"{b['total_seconds']:.2f}", YELLOW),
        ])
    lines += table([
        ("算例", None), ("固定 2-opt 最优回路", GREEN), ("可学 2-opt 最优回路", YELLOW),
        ("固定 Opt-B 最优回路", GREEN), ("可学 Opt-B 最优回路", YELLOW),
        ("固定 2-opt 条数", GREEN), ("可学 2-opt 条数", YELLOW),
        ("固定总耗时 s", GREEN), ("可学总耗时 s", YELLOW),
    ], route_rows)
    lines += ["", "## 已保存证明回路覆盖率", "",
              "覆盖率 = 本次 Opt-B 回路中命中的已保存证明回路数 ÷ 该例已保存证明回路数。"
              "这些证明回路只是全体最优回路的**子集**；此覆盖率不是完整召回率，"
              "也不能据此计算本次 Fβ。", ""]
    witness_rows = []
    for a, b in zip(wf, wl):
        witness_rows.append([
            a["name"], cell(a["total"], BLUE),
            cell(a["hit"], GREEN), cell(b["hit"], YELLOW),
            cell(f"{a['coverage']:.2%}", GREEN),
            cell(f"{b['coverage']:.2%}", YELLOW),
        ])
    lines += table([
        ("算例", None), ("已保存证明回路", BLUE),
        ("固定命中", GREEN), ("可学习命中", YELLOW),
        ("固定证明集覆盖率", GREEN), ("可学习证明集覆盖率", YELLOW),
    ], witness_rows)
    lines += ["", "## 论文 Fβ：仅原 25 例历史对照", "",
              "论文 Table 2 的 Fβ 对应原 25 例 MSTSP；**这 90 例是另一套新算例，"
              "不能计算它们相对论文的 Fβ 差值**。下面只列历史 25 例结果供参照，"
              "其推理设备与搜索预算也不同于本次 90 例。", ""]
    lines += table([
        ("历史原 25 例", None), ("固定 Soft", GREEN), ("可学习 Bias", YELLOW),
        ("论文 Table 2", BLUE),
    ], [["平均 Fβ", cell(f"{old['fixed_mean_fbeta']:.4f}", GREEN),
         cell(f"{old['learnable_mean_fbeta']:.4f}", YELLOW),
         cell(f"{paper_mean:.4f}", BLUE)]])
    lines += ["", "本新数据集没有穷举全部最优回路，无法给出完整召回率和 Fβ；"
              "已证最短距离可用于准确判断本次找到的回路是否全局最优。",
              "原始逐例记录保留在运行目录的 `fixed_soft/cases/` 与 `learnable_bias/cases/`。", ""]
    report = output_dir / "comparison_前90例_固定-可学习-论文说明.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {count}-case report: {report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=90)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    write_report(args.run_dir.resolve(),
                 (args.output_dir or (args.run_dir / f"comparison_first{args.count}")).resolve(),
                 args.count)


if __name__ == "__main__":
    main()
