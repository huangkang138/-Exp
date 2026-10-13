"""按原 main.cpp 的算例顺序、运行次数和预算执行实验。

默认配置 = 学长两项改进 + 结构感知粒球 v3：
    --gbc v3          粒球换成 granular_ball_v3（结构感知版）
    --candidate soft_neighbor  候选城市改为"软加权"，不再硬屏蔽
    --keep-tolerance 1.05      保留不比最优差 5% 以内的高质量路线
    --final-polish             搜索结束后对候选队列再做一轮不计预算的 2-opt

加 --legacy 可一键回到"原论文复现"的完全原始配置（v1 粒球 + 不做候选限制
+ 只留并列最优 + 不做收尾精修），复现论文数字时用这个。
"""

import argparse
from pathlib import Path

from .aco import GBMACO, SearchConfig
from .candidate import ADJACENCY_RULES, CANDIDATE_MODES, build_candidate_masks, mask_summary
from .granular_ball import Point2D, gbc
from .granular_ball_v3 import gbc_v3
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

# 两套粒球实现，命令行直接选
GB_CALLABLES = {"v1": gbc, "v3": gbc_v3}


def build_config(args, budget: int, seed: int | None) -> SearchConfig:
    """把命令行参数翻译成一次搜索的 SearchConfig。"""
    if args.legacy:
        return SearchConfig(max_fes=budget, seed=seed)
    return SearchConfig(
        max_fes=budget,
        seed=seed,
        candidate_mode=args.candidate,
        candidate_rule=args.candidate_rule,
        neighbor_margin=args.neighbor_margin,
        neighbor_k=args.neighbor_k,
        outside_scale=args.outside_scale,
        anneal_fes=args.anneal_fes,
        keep_tolerance=args.keep_tolerance,
        queue_limit=args.queue_limit,
        final_polish=args.final_polish,
        polish_relax=args.polish_relax,
        polish_rotations=args.polish_rotations,
        polish_rounds=args.polish_rounds,
    )


def describe(args) -> str:
    """一行说清这次跑的是什么配置，方便贴进实验记录。"""
    if args.legacy:
        return "配置：legacy（原论文复现：v1 粒球 + 无候选限制 + 只留并列最优）"
    return (
        f"配置：粒球={args.gbc} 候选={args.candidate}(规则={args.candidate_rule}"
        f", margin={args.neighbor_margin}, k={args.neighbor_k}"
        f", outside={args.outside_scale}, anneal={args.anneal_fes}) "
        f"保留阈值={args.keep_tolerance} 存档上限={args.queue_limit} "
        f"收尾精修={args.final_polish}(换起点={args.polish_rotations}"
        f", 轮数={args.polish_rounds}, 关关键边保护={args.polish_relax})"
    )


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
    # —— 学长两项改进 + 粒球版本的开关 ——
    parser.add_argument("--gbc", choices=sorted(GB_CALLABLES), default="v3",
                        help="粒球实现：v1=原版，v3=结构感知版（默认）")
    parser.add_argument("--candidate", choices=CANDIDATE_MODES, default="soft_neighbor",
                        help="候选城市口径；off 表示不做限制（原论文行为）")
    parser.add_argument("--candidate-rule", choices=ADJACENCY_RULES, default="knn",
                        help="'临近粒球'怎么判：knn=球心 k 近邻（默认），mst=球心最小生成树，gap=圆盘间隙")
    parser.add_argument("--neighbor-k", type=int, default=1,
                        help="knn 口径下每个球连最近的几个球，1 最紧、越大越松")
    parser.add_argument("--neighbor-margin", type=float, default=1.0,
                        help="gap 口径下的间隙系数")
    parser.add_argument("--outside-scale", type=float, default=0.05,
                        help="soft 模式下候选集外城市的权重折扣（0=硬屏蔽）")
    parser.add_argument("--anneal-fes", type=float, default=0.5,
                        help=">0 时折扣在前这段 FES 比例内由 0 线性放开")
    parser.add_argument("--keep-tolerance", type=float, default=1.25,
                        help="存档门槛：保留 length <= 当前最优 * 该值 的路线；1.0 = 只留并列最优")
    parser.add_argument("--queue-limit", type=int, default=600,
                        help="高质量路线存档上限，0 = 用默认值 300")
    parser.add_argument("--polish-rotations", type=int, default=8,
                        help="收尾精修时每条路线换几个起点各扫一遍 2-opt；1 = 不换起点")
    parser.add_argument("--polish-rounds", type=int, default=2,
                        help="收尾精修最多扫几遍（没有再改进就提前停）")
    parser.add_argument("--final-polish", dest="final_polish", action="store_true", default=True,
                        help="搜索结束后对高质量存档再做一轮不计预算的 2-opt（默认开）")
    parser.add_argument("--no-final-polish", dest="final_polish", action="store_false",
                        help="关闭收尾精修")
    parser.add_argument("--no-polish-relax", dest="polish_relax", action="store_false",
                        default=True,
                        help="收尾精修时保留「关键边保护」（默认关掉保护，允许更多两刀交换）")
    parser.add_argument("--legacy", action="store_true",
                        help="一键回到原论文复现配置，忽略上面所有改进开关")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs 必须至少为 1")
    if args.max_fes is not None and args.max_fes < 1:
        parser.error("--max-fes 必须至少为 1")
    if args.keep_tolerance < 1.0:
        parser.error("--keep-tolerance 不能小于 1.0")
    if not 0.0 <= args.outside_scale <= 1.0:
        parser.error("--outside-scale 必须落在 0 到 1 之间")
    if args.neighbor_k < 1:
        parser.error("--neighbor-k 必须至少为 1")
    if args.polish_rotations < 1 or args.polish_rounds < 1:
        parser.error("--polish-rotations / --polish-rounds 必须至少为 1")

    if args.input is None:
        cases = [(name, count, fes, args.data_dir / f"{name}.tsp") for name, count, fes in BENCHMARKS]
    else:
        source_case = next((case for case in BENCHMARKS if case[0] == args.input.stem), None)
        if source_case is None:
            parser.error("单例检查只支持原 main.cpp 列出的 25 个算例")
        name, count, fes = source_case
        cases = [(name, count, fes, args.input)]

    granular_ball_fn = GB_CALLABLES[args.gbc]
    print(describe(args), flush=True)

    # 原 C++：main.cpp:1013 创建输出目录；数据文件始终只读。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, expected_count, source_fes, input_path in cases:
        cities = read_cities(input_path)
        if len(cities) != expected_count:
            raise ValueError(f"{name}：原源码期望 {expected_count} 个城市，实际 {len(cities)} 个")
        matrix = init_distance_matrix(cities)
        points = [Point2D(city.x, city.y, city.number) for city in cities]
        balls = granular_ball_fn(points)
        mapping = city_to_ball_mapping(cities, balls)
        initial_route = greed_route(cities, matrix, 0, mapping, balls)
        budget = args.max_fes if args.max_fes is not None else source_fes

        masks = None if args.legacy else build_candidate_masks(
            mapping, balls, args.candidate, args.neighbor_margin, len(cities),
            args.candidate_rule, args.neighbor_k,
        )
        stats = mask_summary(masks, len(cities))

        for run in range(args.runs):
            seed = None if args.seed is None else args.seed + run
            config = build_config(args, budget, seed)
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
                  f"candidate_routes={len(routes)}, balls={len(balls)}, "
                  f"cand_avg={search.candidate_stats['avg_size']:.1f}/{len(cities)}, "
                  f"output={result_path}", flush=True)


if __name__ == "__main__":
    main()
