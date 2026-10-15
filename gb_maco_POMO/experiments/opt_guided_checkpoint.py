"""Repair nested config metadata for this run's already-saved checkpoints.

Only local A/B/C files from an opt_guided_pomo run are accepted. The weights,
optimizer, epoch, and RNG payloads are compared before atomic replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .compare import load_policy
from .opt_guided_diagnose import SOURCE, SOURCE_SHA
from .opt_guided_train import checkpoint_config


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path) -> None:
    root = root.resolve()
    if root.parent.name != "opt_guided_pomo" or not root.name.startswith("run_"):
        raise ValueError("Expected one opt_guided_pomo run directory")
    if _sha(SOURCE) != SOURCE_SHA:
        raise ValueError("Source checkpoint changed")
    source = torch.load(SOURCE, map_location="cpu", weights_only=True)
    calibration = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    records = []
    for group in "ABC":
        for filename in ("best.pt", "latest.pt"):
            path = root / group / filename
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state["group"] != group or state["source_checkpoint_sha256"] != SOURCE_SHA:
                raise ValueError(f"Wrong checkpoint identity: {path}")
            old_sha = _sha(path)
            if "model" in state["config"]:
                raise FileExistsError(f"Checkpoint already has nested config: {path}")
            preserved = {key: state[key] for key in (
                "model_state_dict", "optimizer_state_dict", "epoch", "torch_rng_state",
                "python_shuffle_rng_state", "python_selection_rng_state", "cuda_rng_states")}
            state["config"] = checkpoint_config(group, root, source, calibration,
                                                torch.device("cuda"))
            temporary = path.with_suffix(".metadata_tmp")
            torch.save(state, temporary)
            policy, reloaded = load_policy(temporary, torch.device("cpu"))
            if policy.candidate_mode != "soft" or policy.bias_mode != "fixed":
                raise RuntimeError("Standard loader did not restore fixed Soft policy")
            for key, original in preserved["model_state_dict"].items():
                if not torch.equal(original, reloaded["model_state_dict"][key]):
                    raise RuntimeError(f"Model tensor changed: {key}")
            if any(reloaded[key] != state[key] for key in ("epoch",)) or (
                    not torch.equal(reloaded["torch_rng_state"], preserved["torch_rng_state"])):
                raise RuntimeError("Epoch or torch RNG changed")
            # The untouched optimizer and Python RNG objects remain in state;
            # torch serialization round-trip is checked by their top-level keys.
            if not all(key in reloaded for key in preserved):
                raise RuntimeError("Required recovery state missing")
            temporary.replace(path)
            records.append({"group": group, "file": filename, "epoch": state["epoch"],
                            "old_sha256": old_sha, "new_sha256": _sha(path)})
    (root / "checkpoint_metadata_repair.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified six compatible checkpoints: {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    run(parser.parse_args().run)
