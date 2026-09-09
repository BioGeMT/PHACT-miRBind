#!/usr/bin/env python3
"""Audit high-confidence in-sample disagreements without auto-deleting them."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


OUTPUT_COLUMNS = (
    "manakov_row_id",
    "label",
    "prediction",
    "error_confidence",
    "gene",
    "noncodingRNA",
    "noncodingRNA_name",
    "noncodingRNA_fam",
    "feature",
)


def main() -> None:
    args = parse_args()
    arrays = np.load(args.predictions)
    labels = arrays["labels"].astype(np.int8)
    predictions = arrays["predictions"].astype(np.float64)
    high_confidence = ((labels == 0) & (predictions >= args.threshold)) | (
        (labels == 1) & (predictions <= 1.0 - args.threshold)
    )
    known_conflicts = read_row_ids(args.known_conflicts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / "high_confidence_disagreements.tsv"
    selected_rows = 0
    known_overlap = 0
    with args.rows.open(newline="") as source, evidence_path.open(
        "w", newline=""
    ) as destination:
        reader = csv.DictReader(source, delimiter="\t")
        writer = csv.DictWriter(
            destination,
            fieldnames=OUTPUT_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        row_count = 0
        for index, row in enumerate(reader):
            row_count += 1
            if int(row["label"]) != int(labels[index]):
                raise ValueError(f"Label/cache mismatch at input row {index + 1}")
            if not high_confidence[index]:
                continue
            row_id = int(row["manakov_row_id"])
            known_overlap += int(row_id in known_conflicts)
            prediction = float(predictions[index])
            writer.writerow(
                {
                    **{column: row[column] for column in OUTPUT_COLUMNS if column in row},
                    "prediction": f"{prediction:.8f}",
                    "error_confidence": f"{max(prediction, 1.0 - prediction):.8f}",
                }
            )
            selected_rows += 1
    if row_count != len(labels):
        raise ValueError(f"TSV rows {row_count} do not match predictions {len(labels)}")

    report = {
        "rows": row_count,
        "threshold": args.threshold,
        "high_confidence_disagreements": selected_rows,
        "negative_predicted_positive": int(
            ((labels == 0) & (predictions >= args.threshold)).sum()
        ),
        "positive_predicted_negative": int(
            ((labels == 1) & (predictions <= 1.0 - args.threshold)).sum()
        ),
        "known_cross_study_conflict_rows": len(known_conflicts),
        "known_conflicts_among_disagreements": known_overlap,
        "interpretation": (
            "Model disagreement is diagnostic only and is not sufficient evidence "
            "to delete a training row."
        ),
        "evidence": str(evidence_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--known-conflicts", type=Path)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0.5 < args.threshold <= 1.0:
        parser.error("--threshold must be in (0.5, 1.0]")
    return args


def read_row_ids(path: Path | None) -> set[int]:
    if path is None:
        return set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {
            int(row["manakov_row_id"])
            for row in reader
            if row.get("recommend_exclude") == "1"
        }


if __name__ == "__main__":
    main()
