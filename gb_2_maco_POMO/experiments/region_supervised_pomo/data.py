"""Merge the original split with certified labels without duplicating cases."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from gb_2_maco_POMO.experiments.shared.data import DatasetInstance, load_manifest
from gb_2_maco_POMO.experiments.shared.opt_b_search import canonical, equal_plateau, tour_length
from gb_2_maco_POMO.runner import prepare_tsp

from .config import ORIGINAL, PACKAGE


@dataclass(frozen=True)
class LabeledCase:
    instance: DatasetInstance
    label_id: str
    tours: tuple[tuple[int, ...], ...]
    regions: tuple[tuple[int, ...], ...]
    optimal_length: int
    complete: bool
    source_type: str


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_components(tours: tuple[tuple[int, ...], ...], distance):
    """Use the real Opt-B search for exactly one expansion per labeled tour."""
    n = len(tours[0])
    pair_checks = (n - 2) * (n - 1) // 2 - 1
    index = {key: i for i, key in enumerate(tours)}
    edges = [set() for _ in tours]
    for i, route in enumerate(tours):
        result = equal_plateau([list(route)], distance, pair_checks)
        if result.checks != pair_checks:
            raise RuntimeError(f"Opt-B neighbor budget changed: {result.checks} != {pair_checks}")
        for neighbor in result.routes:
            j = index.get(neighbor)
            if j is not None:
                edges[i].add(j)
                edges[j].add(i)
    pending = set(range(len(tours)))
    components = []
    while pending:
        component = set()
        stack = [min(pending)]
        pending.remove(stack[0])
        while stack:
            i = stack.pop()
            component.add(i)
            for j in edges[i] & pending:
                pending.remove(j)
                stack.append(j)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def load_experiment_data(*, audit_path: Path | None = None):
    if not (PACKAGE / "verify_labels.py").is_file():
        raise FileNotFoundError(f"verified extracted label package missing: {PACKAGE}")
    original_train, original_validation = load_manifest({
        "root": str(ORIGINAL), "train_dir": "train", "validation_dir": "validation"})
    partitions = {"train": list(original_train), "validation": list(original_validation)}
    if (len(original_train), len(original_validation)) != (500, 100):
        raise RuntimeError("original 500/100 split changed")
    original_hashes = {split: {sha256(item.path): item for item in items}
                       for split, items in partitions.items()}
    if any(len(hashes) != len(partitions[split]) for split, hashes in original_hashes.items()):
        raise RuntimeError("duplicate coordinates in original split")
    if set(original_hashes["train"]) & set(original_hashes["validation"]):
        raise RuntimeError("original train/validation coordinate overlap")
    seen_hashes = set(original_hashes["train"]) | set(original_hashes["validation"])
    labels: dict[str, dict[Path, LabeledCase]] = {"train": {}, "validation": {}}
    audit = []
    counts = defaultdict(int)
    with (PACKAGE / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 438:
        raise RuntimeError("label manifest must have 438 cases")
    for row in rows:
        split = row["split"]
        if split not in partitions:
            raise RuntimeError(f"invalid label split: {split}")
        coordinate = (PACKAGE / row["tsp_path"]).resolve()
        label = json.loads((PACKAGE / row["label_path"]).read_text(encoding="utf-8"))
        digest = sha256(coordinate)
        if digest != label["coordinates_sha256"]:
            raise RuntimeError(f"coordinate hash mismatch: {row['id']}")
        source = row["source_type"]
        if source == "original_certified_annotated":
            item = original_hashes[split].get(digest)
            if item is None or coordinate.read_bytes() != item.path.read_bytes():
                raise RuntimeError(f"original label split/content mismatch: {row['id']}")
        elif source == "new_exact_imbalanced":
            if digest in seen_hashes:
                raise RuntimeError(f"new case duplicates original/other case: {row['id']}")
            seen_hashes.add(digest)
            item = DatasetInstance(coordinate, int(row["city_count"]), row["distribution"])
            partitions[split].append(item)
        else:
            raise RuntimeError(f"unexpected label source: {source}")
        if item.path in labels[split]:
            raise RuntimeError(f"duplicate label: {item.path}")
        tours = tuple(canonical(t["tour"]) for t in label["tours"])
        if len(tours) != len(set(tours)) or len(tours) != int(row["labeled_optimal_tours"]):
            raise RuntimeError(f"duplicate/incorrect tours: {row['id']}")
        problem = prepare_tsp(item.path)
        optimum = int(row["global_optimal_distance"])
        if any(tour_length(list(t), problem.distance_matrix) != optimum for t in tours):
            raise RuntimeError(f"tour length differs from certified optimum: {row['id']}")
        regions = _project_components(tours, problem.distance_matrix)
        provided = tuple(sorted(tuple(sorted(r["known_tour_indices"])) for r in label["regions"]))
        audit.append({"id": row["id"], "split": split, "city_count": item.city_count,
                      "source_type": source, "package_regions": len(provided),
                      "project_regions": len(regions), "same_partition": regions == provided,
                      "complete": label["certification"]["all_optimal_tours_enumerated"]})
        labels[split][item.path] = LabeledCase(
            item, row["id"], tours, regions, optimum,
            bool(label["certification"]["all_optimal_tours_enumerated"]), source)
        counts[(split, source)] += 1
    expected = {("train", "original_certified_annotated"): 282,
                ("validation", "original_certified_annotated"): 56,
                ("train", "new_exact_imbalanced"): 80,
                ("validation", "new_exact_imbalanced"): 20}
    if dict(counts) != expected or tuple(len(partitions[s]) for s in ("train", "validation")) != (580, 120):
        raise RuntimeError(f"merged dataset size/source mismatch: {dict(counts)}")
    if len({sha256(i.path) for i in partitions["train"] + partitions["validation"]}) != 700:
        raise RuntimeError("merged train/validation coordinate overlap")
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
            writer.writeheader()
            writer.writerows(audit)
        (audit_path.parent / "region_rule_report.json").write_text(json.dumps({
            "rule": "gb_2_maco_POMO.experiments.shared.opt_b_search.equal_plateau; one complete neighbor expansion per labeled route",
            "cases": len(audit), "matching_partitions": sum(x["same_partition"] for x in audit),
            "changed_ids": [x["id"] for x in audit if not x["same_partition"]],
            "saved_label_version": "project_optb_labeled_induced_components_v1",
            "partial_label_caveat": "Components of known tours may be connected through unknown optimal tours."
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        derived = {case.label_id: {"split": split, "tour_order": [list(t) for t in case.tours],
                                   "known_region_tour_indices": [list(c) for c in case.regions],
                                   "complete": case.complete}
                   for split in labels for case in labels[split].values()}
        (audit_path.parent / "project_region_labels_v1.json").write_text(
            json.dumps(derived, ensure_ascii=False) + "\n", encoding="utf-8")
    return partitions, labels, audit
