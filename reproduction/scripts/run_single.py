"""单实例实验入口。

示例：``python reproduction/scripts/run_single.py --instance MSTSP1 --runs 1``。
脚本按每次运行分目录保存完整日志、参数、耗时和原算法输出，适合先做小规模
验证，也可以对同一个实例连续运行指定次数。
"""

from __future__ import annotations

import argparse
import json

from common import execute_run


def main() -> int:
    """解析命令行参数，依次完成指定实例的若干次独立运行。"""
    parser = argparse.ArgumentParser(description="运行一个 GB-MACO 基准实例")
    parser.add_argument("--instance", required=True, help="实例编号：MSTSP1 到 MSTSP25")
    parser.add_argument("--runs", type=int, default=1, help="运行次数，默认 1")
    parser.add_argument("--seed-start", type=int, default=None, help="首个随机种子；后续 run 依次加 1")
    parser.add_argument("--profile", default="source_original", help="实验参数档位，默认使用源码实际参数")
    parser.add_argument("--timeout", type=int, default=None, help="每次运行的超时秒数")
    parser.add_argument("--force", action="store_true", help="覆盖已经存在的对应 run 目录文件")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    failures = 0
    for run_number in range(1, args.runs + 1):
        seed = None if args.seed_start is None else args.seed_start + run_number - 1
        try:
            metadata = execute_run(
                args.instance,
                run_number,
                profile=args.profile,
                seed=seed,
                timeout_seconds=args.timeout,
                force=args.force,
            )
            print(json.dumps(metadata, ensure_ascii=False))
            failures += metadata["exit_code"] != 0
        except Exception as error:  # 单次失败只计数，不阻止后续已请求的 run。
            failures += 1
            print(f"run {run_number} failed: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
