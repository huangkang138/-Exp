"""读取训练 TOML，并按配置文件 → 环境变量 → 命令行的顺序覆盖。"""

from __future__ import annotations

import os
import tomllib
from math import isfinite
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).with_name("train_config.toml")


def load_config(path: Path, overrides: dict[str, object]) -> dict[str, dict[str, object]]:
    """把路径转为绝对路径；返回的字典就是保存到实验目录的实际配置。"""
    config_path = path.resolve()
    with config_path.open("rb") as stream:
        config: dict[str, dict[str, object]] = tomllib.load(stream)
    for section in ("training", "model", "validation", "data", "checkpoint", "runtime", "output"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"配置文件缺少 [{section}]：{config_path}")

    # GB_POMO_DATA_DIR 只覆盖数据根目录；CUDA_VISIBLE_DEVICES 由 PyTorch 自行读取。
    if os.getenv("GB_POMO_DATA_DIR"):
        config["data"]["root"] = str(Path(os.environ["GB_POMO_DATA_DIR"]).expanduser().resolve())
    for option, value in overrides.items():
        if value is None:
            continue
        section, key = option.split(".", 1)
        # 命令行里的相对路径按运行时当前目录理解；TOML 里的相对路径按配置文件目录理解。
        if option in ("data.root", "output.results_root", "output.checkpoints_root", "checkpoint.resume") and value:
            value = str(Path(str(value)).expanduser().resolve())
        config[section][key] = value

    for section, key in (("data", "root"), ("output", "results_root"),
                         ("output", "checkpoints_root")):
        value = Path(str(config[section][key])).expanduser()
        config[section][key] = str((config_path.parent / value).resolve())
    resume = str(config["checkpoint"]["resume"])
    if resume:
        value = Path(resume).expanduser()
        config["checkpoint"]["resume"] = str((config_path.parent / value).resolve())

    positive_ints = (
        ("training", "epochs"), ("training", "batch_size"),
        ("training", "micro_batch_size"), ("validation", "interval"),
        ("validation", "batch_size"), ("checkpoint", "save_interval"),
        ("model", "embedding_dim"), ("model", "encoder_layers"), ("model", "num_heads"),
    )
    for section, key in positive_ints:
        value = config[section][key]
        if type(value) is not int or value < 1:
            raise ValueError(f"[{section}] {key} 必须是正整数")
    if type(config["training"]["seed"]) is not int:
        raise ValueError("[training] seed 必须是整数")
    rate = config["training"]["learning_rate"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not isfinite(rate) or rate <= 0:
        raise ValueError("[training] learning_rate 必须为正数")
    if config["training"]["micro_batch_size"] > config["training"]["batch_size"]:
        raise ValueError("micro_batch_size 不能大于 batch_size")
    if config["model"]["embedding_dim"] % config["model"]["num_heads"]:
        raise ValueError("embedding_dim 必须能被 num_heads 整除")
    model = config["model"]
    model.setdefault("candidate_mode", "hard")
    model.setdefault("same_ball_bias", 2.0)
    model.setdefault("adjacent_ball_bias", 1.0)
    if model["candidate_mode"] not in ("hard", "soft", "none"):
        raise ValueError("candidate_mode 必须是 hard、soft 或 none")
    same, adjacent = model["same_ball_bias"], model["adjacent_ball_bias"]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
           for v in (same, adjacent)) or not 0 <= adjacent <= same <= 10:
        raise ValueError("偏好必须满足 0 <= adjacent <= same <= 10")
    if config["runtime"]["device"] not in ("auto", "cuda", "cpu"):
        raise ValueError("[runtime] device 只能是 auto、cuda 或 cpu")
    return config
