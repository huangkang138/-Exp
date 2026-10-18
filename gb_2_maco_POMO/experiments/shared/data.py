"""读取真实 .tsp 清单，并把相同城市数的实例放进同一训练批次。"""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetInstance:
    path: Path
    city_count: int
    distribution: str


def load_manifest(data: dict[str, object]) -> tuple[list[DatasetInstance], list[DatasetInstance]]:
    """校验全部文件与 SHA256，训练集和验证集按 manifest 的 split 分开。"""
    root = Path(str(data["root"]))
    train_root = Path(str(data.get("train_root", root)))
    roots = {"train": train_root, "validation": root}
    directories = {
        "train": (train_root / str(data["train_dir"])).resolve(),
        "validation": (root / str(data["validation_dir"])).resolve(),
    }
    partitions: dict[str, list[DatasetInstance]] = {"train": [], "validation": []}
    missing: list[str] = []
    seen: set[Path] = set()
    for selected_split, selected_root in roots.items():
        manifest = selected_root / "manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"找不到数据清单：{manifest}")
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"split", "distribution", "city_count", "relative_path", "sha256"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"manifest.csv 缺少列：{sorted(required - set(reader.fieldnames or []))}")
            for line_number, row in enumerate(reader, start=2):
                split = row["split"]
                if split not in partitions:
                    raise ValueError(f"清单第 {line_number} 行的 split 无效：{split}")
                if split != selected_split:
                    continue
                path = (selected_root / row["relative_path"]).resolve()
                if not path.is_relative_to(directories[split]) or path.suffix.lower() != ".tsp":
                    raise ValueError(f"清单第 {line_number} 行的路径不属于 {split}：{path}")
                if path in seen:
                    raise ValueError(f"清单中重复列出文件：{path}")
                seen.add(path)
                if not path.is_file():
                    missing.append(row["relative_path"])
                    continue
                if hashlib.sha256(path.read_bytes()).hexdigest().lower() != row["sha256"].lower():
                    raise ValueError(f"文件 SHA256 与清单不一致：{path}")
                try:
                    city_count = int(row["city_count"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"清单第 {line_number} 行的 city_count 无效") from exc
                if city_count < 2:
                    raise ValueError(f"清单第 {line_number} 行至少需要两个城市")
                partitions[split].append(DatasetInstance(path, city_count, row["distribution"]))
    if missing:
        raise FileNotFoundError(f"清单中的 {len(missing)} 个 .tsp 不存在，例如：{missing[:3]}")
    if not partitions["train"] or not partitions["validation"]:
        raise ValueError("manifest.csv 必须同时有非空 train 和 validation")
    return partitions["train"], partitions["validation"]


def batches_by_city_count(
    instances: list[DatasetInstance], batch_size: int, rng: random.Random | None = None,
) -> list[list[DatasetInstance]]:
    """同批 N 必须相同，因为 Encoder/Decoder 的城市维需要一致。"""
    groups: dict[int, list[DatasetInstance]] = defaultdict(list)
    for instance in instances:
        groups[instance.city_count].append(instance)
    batches: list[list[DatasetInstance]] = []
    for city_count in sorted(groups):
        group = groups[city_count]
        if rng is not None:
            rng.shuffle(group)
        batches.extend(group[i:i + batch_size] for i in range(0, len(group), batch_size))
    if rng is not None:
        rng.shuffle(batches)
    return batches
