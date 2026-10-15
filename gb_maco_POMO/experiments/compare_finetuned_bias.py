"""Evaluate the completed Bias Head fine-tune against the saved fixed checkpoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch

from .compare_learnable_bias import run as run_comparison
from .finetune_bias_head import DEFAULT_CONFIG, EXPECTED_SOURCE_SHA256


ROOT = DEFAULT_CONFIG.parent.parent / "results/bias_head_finetune"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, help="fine-tune best.pt; defaults to latest completed run")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.checkpoint is None:
        runs = sorted((ROOT / "training").glob("run_*/final_result.json"), reverse=True)
        if not runs:
            raise FileNotFoundError("no completed Bias Head fine-tune run found")
        result = json.loads(runs[0].read_text(encoding="utf-8"))
        checkpoint = Path(result["checkpoints_dir"]) / "best.pt"
    else:
        checkpoint = args.checkpoint.resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("source_checkpoint_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("checkpoint did not originate from the specified fixed Soft-POMO model")
    output = args.output_dir or ROOT / "comparison" / f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    run_comparison(checkpoint, output)


if __name__ == "__main__":
    main()
