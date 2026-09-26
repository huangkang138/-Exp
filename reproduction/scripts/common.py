"""复现实验工具的公共模块。

本文件集中处理配置读取、项目路径定位、运行目录判断，以及调用原
``gb_maco.runner`` 的单次实验。其他脚本复用这里的逻辑，避免各自
拼接路径或启动命令造成实验参数不一致。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPRODUCTION_ROOT / "config" / "experiment_config.json"


def load_config() -> dict[str, Any]:
    """读取实验配置文件，并返回解析后的字典。"""
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def project_root(config: dict[str, Any] | None = None) -> Path:
    """根据配置计算 GB-MACO_python 项目根目录的绝对路径。"""
    config = config or load_config()
    return (REPRODUCTION_ROOT / config["project_root"]).resolve()


def result_root(config: dict[str, Any] | None = None) -> Path:
    """返回所有复现实验结果统一保存目录的绝对路径。"""
    config = config or load_config()
    return (project_root(config) / config["output_root"]).resolve()


def instance_sort_key(name: str) -> int:
    """提取 MSTSP 编号，用于按 MSTSP1、MSTSP2……而非字符串顺序排序。"""
    match = re.fullmatch(r"MSTSP(\d+)", name.upper())
    if not match:
        raise ValueError(f"Invalid instance name: {name}")
    return int(match.group(1))


def normalize_instance(name: str, config: dict[str, Any]) -> str:
    """统一实例名大小写，并检查该实例是否存在于配置中。"""
    normalized = name.upper()
    if normalized not in config["instances"]:
        raise ValueError(f"Unknown instance: {name}")
    return normalized


def run_directory(config: dict[str, Any], instance: str, run_number: int) -> Path:
    """生成某个实例某一次运行的独立结果目录。"""
    return result_root(config) / "raw" / instance / f"run_{run_number:03d}"


def is_successful_run(run_dir: Path) -> bool:
    """检查一次运行是否已有退出码为 0 且输出文件完整的结果。

    批量实验用这个判断实现断点续跑：只有真正成功且文件仍存在的 run
    才会跳过，失败或不完整的 run 会重新执行。
    """
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    output_files = metadata.get("output_files", [])
    return metadata.get("exit_code") == 0 and bool(output_files) and all(
        (run_dir / path).is_file() for path in output_files
    )


def _metadata_parameters(config: dict[str, Any], instance_data: dict[str, Any]) -> dict[str, Any]:
    """整理需要写入 metadata.json 的实际算法参数。"""
    source = config["source_parameters"]
    return {
        "MaxFes": instance_data["max_fes"],
        "NP": source["NP"],
        "antNum": source["ant_num"],
        "alpha": source["alpha"],
        "beta": source["beta"],
        "rho": source["rho"],
        "pheromone_update_percentage": source["pheromone_update_percentage"],
        "tau0": source["initial_pheromone"],
        "max_pheromone_matrices": source["max_pheromone_matrices"],
    }


def execute_run(
    instance: str,
    run_number: int,
    *,
    profile: str = "source_original",
    seed: int | None = None,
    timeout_seconds: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """执行并保存一次相互隔离的实验。

    输入实例名、run 编号、参数档位、随机种子和超时时间；函数会调用原
    Python 入口，完整保存 stdout、stderr、原算法输出及 metadata.json，
    最后返回本次运行的元数据。这里不修改任何核心算法参数或实现。
    """
    config = load_config()
    instance = normalize_instance(instance, config)
    profile_data = config["profiles"].get(profile)
    if profile_data is None:
        raise ValueError(f"Unknown profile: {profile}")
    if profile_data.get("status") != "runnable":
        raise ValueError(
            f"Profile {profile!r} is {profile_data.get('status')}; "
            "paper NP has not been mapped to Python ant_count"
        )

    root = project_root(config)
    instance_data = config["instances"][instance]
    dataset = root / config["benchmark_dir"] / instance_data["file"]
    if not dataset.is_file():
        raise FileNotFoundError(dataset)

    run_dir = run_directory(config, instance, run_number)
    if run_dir.exists() and not force:
        raise FileExistsError(f"Run directory already exists: {run_dir}; use --force to replace its files")
    run_dir.mkdir(parents=True, exist_ok=True)
    program_output = run_dir / "program_output"
    program_output.mkdir(parents=True, exist_ok=True)

    timeout = timeout_seconds or int(config["timeout_seconds"])
    command = [
        sys.executable,
        "-m",
        config["runner_module"],
        str(dataset),
        "--runs",
        "1",
        "--max-fes",
        str(instance_data["max_fes"]),
        "--output-dir",
        str(program_output),
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])

    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    timed_out = False
    # 每次 run 都使用独立子进程，保证退出码、耗时和日志可以单独追踪。
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code: int | None = completed.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout or ""
        stderr = (error.stderr or "") + f"\nTimed out after {timeout} seconds.\n"
        exit_code = None
    elapsed = perf_counter() - started

    (run_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    # 记录原程序真正生成的文件，后续解析器只读取这些有证据的输出。
    outputs = sorted(
        str(path.relative_to(run_dir)).replace("\\", "/")
        for path in program_output.rglob("*")
        if path.is_file()
    )
    metadata = {
        "schema_version": 1,
        "instance": instance,
        "instance_type": instance_data["type"],
        "city_count": instance_data["city_count"],
        "run": run_number,
        "dataset_file": str(dataset),
        "timestamp": started_at.isoformat(),
        "profile": profile,
        "seed": seed,
        "seed_source": "explicit" if seed is not None else "runner_time_based_unobservable",
        **_metadata_parameters(config, instance_data),
        "executable": sys.executable,
        "command": command,
        "working_directory": str(root),
        "elapsed_seconds": elapsed,
        "timeout_seconds": timeout,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "output_files": outputs,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
