"""Frozen experiment inputs and budgets."""

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "benachmark_Train_MATSP_Multiopt_500"
RESULTS = PROJECT / "gb_2_maco_POMO/results/dynamic_ball_attention"
INITIAL = PROJECT / "gb_2_maco_POMO/results/learnable_ball_20261015/checkpoints/best.pt"
INITIAL_SHA256 = "efffddda5fe28f678dd835b7d238ffe4e89d740f40045dd761db1e4a46bfd4bc"
TRAIN_MANIFEST_SHA256 = "982f6970838629d33ddf49583780d9f7ef866bfca761243b40e21fc9471ad82f"
VALIDATION_MANIFEST_SHA256 = "f85d9675f2c9237d312bc93503251785db068e21ca4bae5cc8c5c6d48a161c99"
TRAIN_SEEDS = (42, 43, 44)
INFERENCE_SEEDS = (42, 43, 44, 45, 46)
EPOCHS = 3
BATCH_SIZE = 64
MICRO_BATCH_SIZE = 8
LEARNING_RATE = 1e-4
OPT_B_CHECKS = 50_000
