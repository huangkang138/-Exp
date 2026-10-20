"""Training-only quality references, frozen once per epoch."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from gb_2_maco_POMO.experiments.learnable_ball_bias.validation import heuristic_reference
from gb_2_maco_POMO.runner import prepare_tsp


def build_references(data_root: Path, train_items, validation_items):
    train_paths = {item.path.resolve() for item in train_items}
    val_paths = {item.path.resolve() for item in validation_items}
    if len(train_paths) != 500 or len(val_paths) != 100 or train_paths & val_paths:
        raise RuntimeError("expected disjoint 500/100 split")
    references = {}
    with (data_root / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "train":
                continue
            path = (data_root / row["relative_path"]).resolve()
            if path not in train_paths:
                raise RuntimeError(f"manifest/path mismatch: {path}")
            proof = row.get("proof_file", "")
            certified = row.get("multiopt_certified") == "1" and bool(row.get("proven_optimal_length"))
            if certified:
                proof_path = data_root / proof
                if not proof or not proof_path.is_file():
                    raise RuntimeError(f"missing certified training proof: {path}")
                certificate = json.loads(proof_path.read_text(encoding="utf-8"))["certificate"]
                length = int(row["proven_optimal_length"])
                proof_length = certificate.get("exact_optimal_length", certificate.get("proven_optimal_length"))
                if proof_length is None or int(proof_length) != length:
                    raise RuntimeError(f"training proof length mismatch: {path}")
            else:
                length = heuristic_reference(prepare_tsp(path).distance_matrix)
            references[str(path)] = {"length": length, "certified": certified,
                                     "source": str(data_root / proof) if certified else "frozen-nearest-neighbor-2opt"}
    if len(references) != 500:
        raise RuntimeError("training references missing")
    return references


def reference_digest(references):
    payload = json.dumps(references, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
