"""Frozen paths and budgets for the region-supervision experiment."""

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
PACKAGE = PROJECT / "datasets/MSTSP_Region_Supervision_v1"
ORIGINAL = PROJECT / "benachmark_Train_MATSP_Multiopt_500"
INITIAL = PROJECT / "gb_2_maco_POMO/results/learnable_ball_20261015/checkpoints/best.pt"
INITIAL_SHA256 = "efffddda5fe28f678dd835b7d238ffe4e89d740f40045dd761db1e4a46bfd4bc"
RESULTS = PROJECT / "gb_2_maco_POMO/results/region_supervised_pomo"
GROUPS = ("A", "B", "C")
TRAIN_SEEDS = (42, 43, 44)
INFERENCE_SEEDS = (42, 43, 44, 45, 46)
EPOCHS = 3
BATCH_SIZE = 64
MICRO_BATCH_SIZE = 8
LEARNING_RATE = 1e-4
LAMBDA = 100.0
SAMPLES_PER_START = 8
OPTB_CHECKS = 50_000
