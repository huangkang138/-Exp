"""命令行入口：读取原始 .tsp 坐标，运行 GB-MACO 并保存路径。"""

import argparse
from pathlib import Path

from .aco import GBMACO, SearchConfig
from .granular_ball import Point2D, gbc
from .tsp import city_to_ball_mapping, greed_route, init_distance_matrix, read_cities


def main() -> None:
    """接收算例路径与搜索预算，逐次运行完整算法。
    原 C++：main.cpp:990 main；整合 main.cpp:294 read_file、916 RESULTS_OUT。
    """
    parser = argparse.ArgumentParser(description="运行 Python 版 GB-MACO")
    parser.add_argument("input", type=Path, help="原项目中的纯坐标 .tsp 文件")
    parser.add_argument("--runs", type=int, default=1, help="独立运行次数，默认 1")
    parser.add_argument("--max-fes", type=int, default=60_000, help="每次运行的评价预算")
    parser.add_argument("--seed", type=int, default=None, help="Python 随机种子")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="结果目录")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs 必须至少为 1")

    # 原始数据只读取；数据集不会被复制或修改。
    cities = read_cities(args.input)
    matrix = init_distance_matrix(cities)
    balls = gbc([Point2D(city.x, city.y, city.number) for city in cities])
    mapping = city_to_ball_mapping(cities, balls)
    initial_route = greed_route(cities, matrix, 0, mapping, balls)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for run in range(args.runs):
        config = SearchConfig(max_fes=args.max_fes, seed=None if args.seed is None else args.seed + run)
        search = GBMACO(cities, matrix, mapping, balls, initial_route, config)
        routes = search.solve()
        best_length = routes[0].length
        result_path = args.output_dir / f"{args.input.stem}_RUN_{run}.alg_solution"
        with result_path.open("w", encoding="utf-8") as result:
            for route in routes:
                if route.length != best_length:
                    break
                result.write(f"{route.length}\t" + "\t".join(map(str, route.city_path)) + "\n")
        print(f"{args.input.stem} run {run + 1}: best={best_length}, solutions={len(routes)}, output={result_path}")


if __name__ == "__main__":
    main()
