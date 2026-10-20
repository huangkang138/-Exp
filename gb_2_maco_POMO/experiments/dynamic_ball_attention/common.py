"""Checkpoint and data checks shared by preflight, training and testing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from gb_2_maco_POMO.experiments.learnable_ball.learnable_policy import LearnableBallPolicy
from gb_2_maco_POMO.experiments.shared.data import batches_by_city_count, load_manifest

from .config import (BATCH_SIZE, DATA, INITIAL, INITIAL_SHA256,
                     TRAIN_MANIFEST_SHA256, VALIDATION_MANIFEST_SHA256)
from .model import DynamicBallPolicy


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data():
    if (sha256(DATA / "train_manifest.csv") != TRAIN_MANIFEST_SHA256 or
            sha256(DATA / "validation_manifest.csv") != VALIDATION_MANIFEST_SHA256):
        raise RuntimeError("Multiopt 500 split manifest changed")
    train, validation = load_manifest({"root": str(DATA), "train_dir": "train",
                                       "validation_dir": "validation"})
    if (len(train), len(validation), len(batches_by_city_count(train, BATCH_SIZE))) != (500, 100, 11):
        raise RuntimeError("expected 500/100 split and 11 updates per epoch")
    if {x.path for x in train} & {x.path for x in validation}:
        raise RuntimeError("train/validation paths overlap")
    return train, validation


def initial_model(group: str, device: torch.device):
    if group not in ("A", "B") or sha256(INITIAL) != INITIAL_SHA256:
        raise RuntimeError("unknown group or initial checkpoint changed")
    state = torch.load(INITIAL, map_location="cpu", weights_only=True)
    architecture = state["architecture"]
    arguments = (int(architecture["embedding_dim"]), int(architecture["num_heads"]),
                 int(architecture["encoder_layers"]))
    model = (LearnableBallPolicy if group == "A" else DynamicBallPolicy)(*arguments)
    incompatible = model.load_state_dict(state["model_state_dict"], strict=(group == "A"))
    if group == "B" and (set(incompatible.missing_keys) != {"ball_attention_alpha"}
                         or incompatible.unexpected_keys):
        raise RuntimeError(f"dynamic checkpoint incompatible: {incompatible}")
    if group == "B" and model.ball_attention_alpha.item() != 0:
        raise RuntimeError("dynamic residual must start at zero")
    return model.to(device), arguments


def load_selected(checkpoint: Path, group: str, device: torch.device):
    if checkpoint.resolve() == INITIAL.resolve():
        return initial_model(group, device)[0]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("group") != group or state.get("initial_sha256") != INITIAL_SHA256:
        raise RuntimeError("checkpoint experiment identity mismatch")
    args = state["architecture"]
    model = (LearnableBallPolicy if group == "A" else DynamicBallPolicy)(
        int(args["embedding_dim"]), int(args["num_heads"]), int(args["encoder_layers"]))
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.to(device).eval()
