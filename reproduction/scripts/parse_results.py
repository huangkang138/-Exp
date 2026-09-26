"""把每次运行的原始文件解析为统一 CSV。

脚本读取 ``results/raw/MSTSP*/run_*`` 下的元数据、日志和
``.alg_solution``，提取路线、路线长度、解数量、耗时和实际参数，最终
生成 ``results/parsed/all_runs.csv``。原始输出没有的指标先留空。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from common import instance_sort_key, load_config, result_root


FIELDS = [
    "instance", "type", "city_count", "run", "seed", "route", "route_cost",
    "number_of_solutions", "Fbeta", "DI", "elapsed_seconds", "MaxFes", "NP",
    "antNum", "alpha", "beta", "rho", "exit_code", "output_file",
    "stdout_file", "stderr_file",
]


def parse_solution(path: Path, city_count: int) -> tuple[list[list[int]], float | None]:
    """读取一个 .alg_solution 或 .solution 文件。

    每行第一列是路线代价，后续列是城市编号。返回全部路线，以及文件中
    的最小路线代价；文件不存在或为空时返回空路线和 None。
    """
    routes: list[list[int]] = []
    costs: list[float] = []
    if not path.is_file():
        return routes, None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < city_count + 1:
            raise ValueError(f"{path}:{line_number}: expected cost plus {city_count} cities")
        costs.append(float(parts[0]))
        routes.append([int(value) for value in parts[1 : city_count + 1]])
    return routes, min(costs) if costs else None


def main() -> int:
    """扫描全部 raw 运行目录，生成一行代表一次 run 的 all_runs.csv。"""
    config = load_config()
    raw_root = result_root(config) / "raw"
    rows: list[dict[str, object]] = []
    metadata_files = sorted(
        raw_root.glob("MSTSP*/run_*/metadata.json"),
        key=lambda path: (instance_sort_key(path.parents[1].name), path.parent.name),
    )
    for metadata_path in metadata_files:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_dir = metadata_path.parent
        # 元数据中只认原程序真实登记过的 .alg_solution 文件。
        output_candidates = [
            run_dir / relative
            for relative in metadata.get("output_files", [])
            if relative.endswith(".alg_solution")
        ]
        output_path = output_candidates[0] if output_candidates else Path()
        routes, route_cost = parse_solution(output_path, int(metadata["city_count"])) if output_candidates else ([], None)
        stdout = (run_dir / "stdout.txt").read_text(encoding="utf-8") if (run_dir / "stdout.txt").exists() else ""
        match = re.search(r"best_solutions=(\d+)", stdout)
        reported_count = int(match.group(1)) if match else None
        row = {
            "instance": metadata["instance"],
            "type": metadata["instance_type"],
            "city_count": metadata["city_count"],
            "run": metadata["run"],
            "seed": metadata.get("seed"),
            "route": json.dumps(routes[0], separators=(",", ":")) if routes else "",
            "route_cost": route_cost,
            "number_of_solutions": len(routes) if routes else reported_count,
            # Fβ 和 DI 需要真实最优解集合，由 calculate_metrics.py 后续填写。
            "Fbeta": "",
            "DI": "",
            "elapsed_seconds": metadata["elapsed_seconds"],
            "MaxFes": metadata["MaxFes"],
            "NP": metadata.get("NP"),
            "antNum": metadata["antNum"],
            "alpha": metadata["alpha"],
            "beta": metadata["beta"],
            "rho": metadata["rho"],
            "exit_code": metadata.get("exit_code"),
            "output_file": str(output_path) if output_candidates else "",
            "stdout_file": str(run_dir / "stdout.txt"),
            "stderr_file": str(run_dir / "stderr.txt"),
        }
        rows.append(row)

    destination = result_root(config) / "parsed" / "all_runs.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Parsed {len(rows)} run(s): {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
