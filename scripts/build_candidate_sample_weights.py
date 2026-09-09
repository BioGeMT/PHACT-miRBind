#!/usr/bin/env python3
"""Build paired training weights for reviewed Manakov candidate rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_candidate_ids(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        ids = {
            row["manakov_row_id"]
            for row in csv.DictReader(handle, delimiter="\t")
        }
    if "" in ids:
        raise ValueError("Candidate rows must have Manakov row IDs")
    return ids


def build_weights(
    train_rows: Path,
    candidate_ids: set[str],
    candidate_weight: float,
    additional_rows: int,
) -> tuple[np.ndarray, list[int]]:
    weights: list[float] = []
    matched_ids: set[str] = set()
    matched_indices: list[int] = []
    with train_rows.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            row_id = row["manakov_row_id"]
            is_candidate = row_id in candidate_ids
            weights.append(candidate_weight if is_candidate else 1.0)
            if is_candidate:
                matched_ids.add(row_id)
                matched_indices.append(index)
    missing = candidate_ids - matched_ids
    if missing:
        raise ValueError(f"Candidate IDs absent from training rows: {sorted(missing)}")
    weights.extend([1.0] * additional_rows)
    return np.asarray(weights, dtype=np.float32), matched_indices


def main() -> None:
    args = parse_args()
    candidate_ids = read_candidate_ids(args.candidates)
    weights, matched_indices = build_weights(
        args.train_rows,
        candidate_ids,
        args.candidate_weight,
        args.additional_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, weights=weights)
    summary = {
        "train_rows": str(args.train_rows),
        "candidates": str(args.candidates),
        "candidate_weight": args.candidate_weight,
        "candidate_rows": len(candidate_ids),
        "candidate_zero_based_indices": matched_indices,
        "additional_rows_with_weight_one": args.additional_rows,
        "total_weights": len(weights),
        "weight_sum": float(weights.sum()),
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-weight", type=float, required=True)
    parser.add_argument("--additional-rows", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.candidate_weight <= 1.0:
        parser.error("--candidate-weight must be in [0, 1]")
    if args.additional_rows < 0:
        parser.error("--additional-rows must be non-negative")
    return args


if __name__ == "__main__":
    main()
