"""按原 main.cpp 的算例顺序、运行次数和预算执行实验。"""

import argparse
from pathlib import Path

from .aco import GBMACO, SearchConfig
from .granular_ball import Point2D, gbc
from .tsp import city_to_ball_mapping, greed_route, init_distance_matrix, path_len, read_cities


# 原 C++：main.cpp:994-1005 的 MaxFES / Func_name，以及 1026 的 CITYS_NUM。
# Head.h:40-43 设定 FUNC_BEGIN=0、FUNC_END=24、RUNS=30。
BENCHMARKS: tuple[tuple[str, int, int], ...] = (
    ("simple1_9", 9, 60_000),
    ("simple2_10", 10, 60_000),
    ("simple3_10", 10, 60_000),
    ("simple4_11", 11, 60_000),
    ("simple5_12", 12, 60_000),
    ("simple6_12", 12, 60_000),
    ("geometry1_10", 10, 60_000),
    ("geometry2_12", 12, 60_000),
    ("geometry3_10", 10, 60_000),
    ("geometry4_10", 10, 60_000),
    ("geometry5_10", 10, 60_000),
    ("geometry6_15", 15, 60_000),
    ("composite1_28", 28, 1_200_000),
    ("composite2_34", 34, 1_200_000),
    ("composite3_22", 22, 1_200_000),
    ("composite4_33", 33, 1_200_000),
    ("composite5_35", 35, 1_200_000),
    ("composite6_39", 39, 1_200_000),
    ("composite7_42", 42, 1_200_000),
    ("composite8_45", 45, 1_200_000),
    ("composite9_48", 48, 1_200_000),
    ("composite10_55", 55, 1_200_000),
    ("composite11_59", 59, 1_200_000),
    ("composite12_60", 60, 1_200_000),
    ("composite13_66", 66, 1_200_000),
)

# 原 C++：main.cpp:1010-1011 拼接 alpha、beta、rho 和 antNum。
SOURCE_OUTPUT_DIR = "Results_1_1.5 _0.3 150"


def main() -> None:
    """默认执行原 main() 的 25 个算例、每例 30 次；可选单例检查。
    原 C++：main.cpp:990 main；整合 main.cpp:294 read_file、916 RESULTS_OUT。
    """
    parser = argparse.ArgumentParser(description="运行 Python 版 GB-MACO")
    parser.add_argument("input", nargs="?", type=Path, help="可选：只运行一个原源码算例")
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark_MSTSP"))
    parser.add_argument("--runs", type=int, default=30, help="每个算例运行次数；原源码为 30")
    parser.add_argument("--max-fes", type=int, default=None, help="可选：检查时覆盖原预算")
    parser.add_argument("--seed", type=int, default=None, help="可选：Python 随机种子")
    parser.add_argument("--output-dir", type=Path, default=Path(SOURCE_OUTPUT_DIR))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs 必须至少为 1")
    if args.max_fes is not None and args.max_fes < 1:
        parser.error("--max-fes 必须至少为 1")

    if args.input is None:
        cases = [(name, count, fes, args.data_dir / f"{name}.tsp") for name, count, fes in BENCHMARKS]
    else:
        source_case = next((case for case in BENCHMARKS if case[0] == args.input.stem), None)
        if source_case is None:
            parser.error("单例检查只支持原 main.cpp 列出的 25 个算例")
        name, count, fes = source_case
        cases = [(name, count, fes, args.input)]

    # 原 C++：main.cpp:1013 创建输出目录；数据文件始终只读。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, expected_count, source_fes, input_path in cases:
        cities = read_cities(input_path)
        if len(cities) != expected_count:
            raise ValueError(f"{name}：原源码期望 {expected_count} 个城市，实际 {len(cities)} 个")
        matrix = init_distance_matrix(cities)
        balls = gbc([Point2D(city.x, city.y, city.number) for city in cities])
        mapping = city_to_ball_mapping(cities, balls)
        initial_route = greed_route(cities, matrix, 0, mapping, balls)
        budget = args.max_fes if args.max_fes is not None else source_fes

        for run in range(args.runs):
            config = SearchConfig(max_fes=budget, seed=None if args.seed is None else args.seed + run)
            search = GBMACO(cities, matrix, mapping, balls, initial_route, config)
            routes = search.solve()
            best_length = routes[0].length
            result_path = args.output_dir / f"{name}_RUN_{run}.alg_solution"
            best_count = 0
            with result_path.open("w", encoding="utf-8") as result:
                for route in routes:
                    if route.length != best_length:
                        break
                    # 原 C++：main.cpp:927-938，每条路径重新计算长度，城市编号后保留制表符。
                    result.write(f"{path_len(route, matrix)}\t" + "".join(f"{city}\t" for city in route.city_path) + "\n")
                    best_count += 1
            # 原 C++：main.cpp:1184 分别输出最优路线数和候选队列总数。
            print(f"{name} run {run + 1}: best={best_length}, best_solutions={best_count}, "
                  f"candidate_routes={len(routes)}, output={result_path}")


if __name__ == "__main__":
    main()
