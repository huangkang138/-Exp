"""User-started, resumable preflight -> training -> test -> report pipeline."""

from __future__ import annotations

import argparse

import torch

from .common import load_data
from .config import INFERENCE_SEEDS, TRAIN_SEEDS
from .preflight import inspect
from .report import main as write_report
from .test25 import run_test
from .train import run_group, save_json
from .config import RESULTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "preflight", "train", "test", "report"),
                        default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    if args.phase in ("all", "preflight"):
        stats = inspect(device)
        RESULTS.mkdir(parents=True, exist_ok=True)
        save_json(RESULTS / f"preflight_{device.type}.json", stats)
        print("epoch-0, mask and gradient preflight passed", flush=True)
    if args.phase in ("all", "train"):
        train, validation = load_data()
        for seed in TRAIN_SEEDS:
            for group in ("A", "B"):
                run_group(group, seed, train, validation, device)
    if args.phase in ("all", "test"):
        for seed in TRAIN_SEEDS:
            for inference in INFERENCE_SEEDS:
                for group in ("A", "B"):
                    run_test(group, seed, inference)
    if args.phase in ("all", "report"):
        write_report()


if __name__ == "__main__":
    main()
