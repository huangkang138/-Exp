"""用两份历史代码和 best.pt 评估任意 .tsp 验证目录，不训练模型。

从本文件所在目录运行：
python evaluate_two_backups.py --data-dir D:/path/to/validation

每个模型在独立进程中导入各自备份的源码，避免同名 gb_maco_POMO 包串用。
口径与训练时验证一致：每例从全部 N 个城市出发，各生成一条 Argmax 路线，
取其中最短长度；不执行 Sampling、2-opt 或 Opt-B。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4


HERE = Path(__file__).resolve().parent
BACKUPS = {
    "fixed_soft": HERE / "固定Soft-POMO_代码与模型",
    "learnable_bias": HERE / "旧数据可学习Bias-POMO_代码与模型",
}
FIELDS = ("model", "file", "city_count", "best_length")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def worker(root: Path, model_name: str, data_dir: Path, output: Path,
           device_name: str) -> None:
    # 此进程只导入一个备份包；两份历史代码同名，不能在一个进程中混用。
    sys.path.insert(0, str(root))
    import torch

    from gb_maco_POMO.attention import POMOPolicy
    from gb_maco_POMO.experiments.batch_rollout import rollout_batch
    from gb_maco_POMO.experiments.data import DatasetInstance, batches_by_city_count
    from gb_maco_POMO.tsp import read_cities

    manifest = json.loads((root / "BACKUP_MANIFEST.json").read_text(encoding="utf-8"))
    checkpoint = root / "checkpoints/best.pt"
    if file_sha256(checkpoint) != manifest["checkpoint_sha256"]["best.pt"]:
        raise ValueError(f"{model_name}: checkpoint SHA256 不匹配")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]["model"]
    kwargs = {
        "embedding_dim": int(config["embedding_dim"]),
        "head_num": int(config["num_heads"]),
        "encoder_layer_num": int(config["encoder_layers"]),
        "candidate_mode": str(config["candidate_mode"]),
        "same_ball_bias": float(config["same_ball_bias"]),
        "adjacent_ball_bias": float(config["adjacent_ball_bias"]),
    }
    if "bias_mode" in config:
        kwargs["bias_mode"] = str(config["bias_mode"])
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无法使用 CUDA")
    device = torch.device(device_name)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    policy = POMOPolicy(**kwargs).to(device)
    policy.load_state_dict(state["model_state_dict"], strict=True)
    policy.eval()

    paths = sorted(data_dir.rglob("*.tsp"))
    if not paths:
        raise FileNotFoundError(f"没有找到 .tsp 文件：{data_dir}")
    instances = [DatasetInstance(path, len(read_cities(path)), path.parent.name)
                 for path in paths]
    rows = []
    with torch.no_grad():
        for batch in batches_by_city_count(instances, batch_size=64):
            # POMO 同时保留 N 条路线；大算例减小实例批量以控制显存。
            city_count = batch[0].city_count
            micro_size = 1 if city_count >= 256 else (4 if city_count >= 128 else 8)
            for offset in range(0, len(batch), micro_size):
                micro = batch[offset:offset + micro_size]
                lengths = rollout_batch(policy, micro, device,
                                        sample_actions=False).lengths
                for instance, best in zip(micro, lengths.min(dim=1).values.tolist()):
                    rows.append({"model": model_name,
                                 "file": instance.path.relative_to(data_dir).as_posix(),
                                 "city_count": instance.city_count,
                                 "best_length": int(best)})
                with output.open("w", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(sorted(rows, key=lambda row: row["file"]))
                print(f"{model_name}: {len(rows)}/{len(paths)} 例，刚完成 N={city_count}",
                      flush=True)
    rows.sort(key=lambda row: row["file"])
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{model_name}: {len(rows)} 例，平均最短路线 {sum(r['best_length'] for r in rows)/len(rows):.2f}",
          flush=True)


def parent(data_dir: Path, output_dir: Path | None, device_name: str) -> None:
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)
    data_dir = data_dir.resolve()
    output = output_dir or (HERE / "新验证集评估" /
                            f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, root in BACKUPS.items():
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
                        "--root", str(root), "--model", name, "--data-dir", str(data_dir),
                        "--output-dir", str(output / f"{name}.csv"), "--device", device_name],
                       cwd=root, check=True)
    by_model = {}
    for name in BACKUPS:
        with (output / f"{name}.csv").open(newline="", encoding="utf-8-sig") as stream:
            by_model[name] = {row["file"]: row for row in csv.DictReader(stream)}
    if set(by_model["fixed_soft"]) != set(by_model["learnable_bias"]):
        raise ValueError("两个模型评估的文件不一致")
    paired = []
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for file in sorted(by_model["fixed_soft"]):
        fixed = by_model["fixed_soft"][file]
        learned = by_model["learnable_bias"][file]
        if fixed["city_count"] != learned["city_count"]:
            raise ValueError(f"城市数不一致：{file}")
        n = int(fixed["city_count"])
        a, b = int(fixed["best_length"]), int(learned["best_length"])
        grouped[n].append((a, b))
        paired.append({"file": file, "city_count": n, "fixed_soft": a,
                       "learnable_bias": b, "learnable_minus_fixed": b - a})
    with (output / "paired.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=paired[0])
        writer.writeheader()
        writer.writerows(paired)
    summary = []
    for name, values in [(str(n), grouped[n]) for n in sorted(grouped)] + [
            ("all", [pair for group in grouped.values() for pair in group])]:
        summary.append({"city_count": name, "instances": len(values),
                        "fixed_mean": sum(a for a, _ in values) / len(values),
                        "learnable_mean": sum(b for _, b in values) / len(values),
                        "learnable_better": sum(b < a for a, b in values),
                        "equal": sum(b == a for a, b in values),
                        "fixed_better": sum(a < b for a, b in values)})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    (output / "protocol.json").write_text(json.dumps({
        "data_dir": str(data_dir), "models": {name: str(root / "checkpoints/best.pt")
                                                for name, root in BACKUPS.items()},
        "device": device_name, "seed": 42,
        "metric": "minimum raw Argmax route length over all N POMO starts",
        "postprocessing": "none",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"结果：{output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="用两份 best.pt 对比新 .tsp 验证集")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="独立的新输出目录；默认自动创建")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.root not in BACKUPS.values() or args.model not in BACKUPS:
            parser.error("非法内部调用")
        if args.root != BACKUPS[args.model]:
            parser.error("模型与备份目录不匹配")
        worker(args.root, args.model, args.data_dir, args.output_dir, args.device)
    else:
        parent(args.data_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
