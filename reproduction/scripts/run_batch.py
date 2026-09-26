"""批量实验入口。

支持选择部分实例或全部 25 个实例，按顺序串行运行，并通过已有元数据
实现断点续跑。某一次失败只写入汇总，不会导致整个批次提前终止。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from common import (
    execute_run,
    instance_sort_key,
    is_successful_run,
    load_config,
    normalize_instance,
    result_root,
    run_directory,
)


def main() -> int:
    """解析批量参数，串行执行所有任务并生成 batch_summary.json。"""
    parser = argparse.ArgumentParser(description="顺序执行可断点续跑的 GB-MACO 批量实验")
    parser.add_argument("--instances", nargs="+", required=True, help="填写 all 或一个/多个 MSTSP 实例名")
    parser.add_argument("--runs", type=int, default=None, help="每个实例运行次数；默认读取配置文件")
    parser.add_argument("--seed-start", type=int, default=None, help="整个批次的起始随机种子")
    parser.add_argument("--profile", default="source_original", help="实验参数档位")
    parser.add_argument("--timeout", type=int, default=None, help="每次运行的超时秒数")
    parser.add_argument("--force", action="store_true", help="强制重新执行已有成功结果")
    args = parser.parse_args()

    config = load_config()
    runs = args.runs or int(config["default_runs"])
    if runs < 1:
        parser.error("--runs must be at least 1")
    if len(args.instances) == 1 and args.instances[0].lower() == "all":
        instances = sorted(config["instances"], key=instance_sort_key)
    else:
        instances = [normalize_instance(name, config) for name in args.instances]

    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "instances": instances,
        "requested_runs_per_instance": runs,
        "seed_start": args.seed_start,
        "completed": [],
        "skipped": [],
        "failed": [],
    }
    # 全局运行序号在整个批次中连续递增，确保显式种子不会跨实例重复。
    sequence = 0
    for instance in instances:
        for run_number in range(1, runs + 1):
            sequence += 1
            run_dir = run_directory(config, instance, run_number)
            label = f"{instance}/run_{run_number:03d}"
            if not args.force and is_successful_run(run_dir):
                summary["skipped"].append(label)
                print(f"SKIP {label}")
                continue
            seed = None if args.seed_start is None else args.seed_start + sequence - 1
            try:
                metadata = execute_run(
                    instance,
                    run_number,
                    profile=args.profile,
                    seed=seed,
                    timeout_seconds=args.timeout,
                    force=args.force or run_dir.exists(),
                )
                if metadata["exit_code"] == 0:
                    summary["completed"].append(label)
                    print(f"OK   {label}")
                else:
                    summary["failed"].append({"run": label, "exit_code": metadata["exit_code"]})
                    print(f"FAIL {label}: exit={metadata['exit_code']}")
            except Exception as error:  # 单次异常只记录，继续执行剩余任务。
                summary["failed"].append({"run": label, "error": str(error)})
                print(f"FAIL {label}: {error}")

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary_path = result_root(config) / "raw" / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Batch summary: {summary_path}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
