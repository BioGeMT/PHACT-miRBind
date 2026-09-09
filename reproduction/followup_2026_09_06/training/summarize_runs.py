#!/usr/bin/env python3
"""Verify retained predictions and report all completed matched-seed runs."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workspace import historical_path

import csv
import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from run_condition import ROOT, EXPECTED_ROWS, PHACT_CACHE, sha256
from phact_mirbind.cache.manifest import find_manifest

PAIRS = [("guide_real", "sequence"), ("target_real", "sequence"),
         ("target_real", "conservation_both"), ("guide_real", "guide_shuffled"),
         ("target_real", "target_shuffled")]


def write_tsv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    results = {}
    for seed in protocol["seeds"]:
        for condition in protocol["conditions"]:
            path = ROOT / "runs" / f"seed_{seed}" / condition / "result.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            if result["smoke"] or result["condition"] != condition or result["seed"] != seed:
                raise ValueError(f"Run identity mismatch: {path}")
            checkpoint_path = historical_path(result["checkpoint"])
            if checkpoint_path.parent.resolve() != path.parent.resolve():
                raise ValueError(f"Checkpoint is outside its run: {path}")
            if sha256(checkpoint_path) != result["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            summary = checkpoint["summary"]
            for field, expected in (("condition", condition), ("seed", seed),
                                    ("initial_state_sha256", result["initial_state_sha256"])):
                if summary[field] != expected:
                    raise ValueError(f"Checkpoint {field} mismatch: {path}")
            if checkpoint["epoch"] != result["best_epoch"] or not np.isclose(
                    checkpoint["val_auprc"], result["validation_ap"], rtol=0, atol=1e-12):
                raise ValueError(f"Checkpoint selection mismatch: {path}")
            results[(seed, condition)] = result
    metrics = []
    for split in ("test", "leftout"):
        manifest_path = find_manifest(PHACT_CACHE / split)
        manifest = json.loads(manifest_path.read_text())
        labels = np.concatenate([torch.load(manifest_path.parent / row["file"],
                                             map_location="cpu", weights_only=True)["labels"].numpy()
                                 for row in manifest["shards"]])
        expected_ids = np.array([f"{split}_{i + 1}" for i in range(EXPECTED_ROWS[split])])
        for (seed, condition), result in results.items():
            path = ROOT / "runs" / f"seed_{seed}" / condition / f"{split}.npz"
            if sha256(path) != result[split]["prediction_sha256"]:
                raise ValueError(f"Prediction checksum mismatch: {path}")
            with np.load(path) as archive:
                if not np.array_equal(archive["ids"], expected_ids) or not np.array_equal(archive["labels"], labels):
                    raise ValueError(f"Evaluation row mismatch: {path}")
                predictions = archive["predictions"]
                if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
                    raise ValueError(f"Invalid probabilities: {path}")
                ap = float(average_precision_score(labels, archive["predictions"]))
            if not np.isclose(ap, result[split]["ap"], rtol=0, atol=1e-12):
                raise ValueError(f"AP mismatch: {path}")
            metrics.append(dict(seed=seed, condition=condition, split=split, ap=ap,
                                best_epoch=result["best_epoch"], validation_ap=result["validation_ap"]))
    deltas = []
    for seed in protocol["seeds"]:
        for first, second in PAIRS:
            if (seed, first) not in results or (seed, second) not in results:
                continue
            if second.endswith("shuffled") and results[(seed, first)]["initial_state_sha256"] != results[(seed, second)]["initial_state_sha256"]:
                raise ValueError(f"Unmatched initialization: {seed} {first} {second}")
            for split in ("test", "leftout"):
                delta = results[(seed, first)][split]["ap"] - results[(seed, second)][split]["ap"]
                deltas.append(dict(seed=seed, comparison=f"{first} minus {second}", split=split, ap_delta=delta))
    write_tsv(ROOT / "seed_metrics.tsv", metrics, ["seed", "condition", "split", "ap", "best_epoch", "validation_ap"])
    write_tsv(ROOT / "paired_seed_deltas.tsv", deltas, ["seed", "comparison", "split", "ap_delta"])
    lines = [f"Completed {len(results)} of {len(protocol['seeds']) * len(protocol['conditions'])} planned runs.", "",
             "AP is calculated separately for each fitted model. These are not ensemble scores.", "",
             "| Condition | Split | Completed seeds | Mean AP | Seed SD | Range |", "|---|---|---:|---:|---:|---:|"]
    for condition in protocol["conditions"]:
        for split in ("test", "leftout"):
            values = [row["ap"] for row in metrics if row["condition"] == condition and row["split"] == split]
            if values:
                sd = f"{np.std(values, ddof=1):.6f}" if len(values) > 1 else "—"
                lines.append(f"| {condition} | {split} | {len(values)} | {np.mean(values):.6f} | {sd} | {min(values):.6f}–{max(values):.6f} |")
    lines += ["", "Paired differences are in paired_seed_deltas.tsv. Three training seeds give a limited description of training variability; no confidence interval across seeds is inferred.", "",
              "The evaluation collections were inspected during earlier development. These repeats do not create an untouched external test."]
    (ROOT / "TRAINING_RESULTS.md").write_text("\n".join(lines) + "\n")
    return len(results)


if __name__ == "__main__":
    print(f"Verified {summarize()} completed runs")
