#!/usr/bin/env python3
"""Compare paired candidate-row ablations with a validation-selected threshold."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default="baseline,downweight025,remove",
        help="Comma-separated suffixes of candidate_ablation_seed42_<variant> directories.",
    )
    parser.add_argument("--candidate-tsv", type=Path, required=True)
    parser.add_argument(
        "--candidate-split", choices=("val", "test", "leftout"), default="test"
    )
    parser.add_argument(
        "--baseline-directory",
        type=Path,
        help="Optional baseline prediction directory when it is outside --root.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def confusion_counts(
    labels: np.ndarray, predictions: np.ndarray, threshold: float
) -> dict[str, int | float]:
    predicted_positive = predictions >= threshold
    positive = labels == 1
    tp = int(np.count_nonzero(predicted_positive & positive))
    fp = int(np.count_nonzero(predicted_positive & ~positive))
    fn = int(np.count_nonzero(~predicted_positive & positive))
    tn = int(np.count_nonzero(~predicted_positive & ~positive))
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "errors": fp + fn,
        "error_rate": float((fp + fn) / labels.size),
    }


def validation_error_threshold(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Return an exact score threshold minimizing FP + FN.

    Ties are resolved in favor of the threshold closest to 0.5. The rule is fit
    only on validation predictions and is then applied unchanged to other splits.
    """
    order = np.argsort(-predictions, kind="stable")
    sorted_scores = predictions[order]
    sorted_labels = labels[order].astype(np.int64, copy=False)
    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(1 - sorted_labels)
    group_ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), labels.size - 1]

    positives = int(sorted_labels.sum())
    errors = positives - cumulative_tp[group_ends] + cumulative_fp[group_ends]
    thresholds = sorted_scores[group_ends].astype(np.float64, copy=False)

    all_negative_threshold = float(
        np.nextafter(
            sorted_scores[0],
            np.asarray(np.inf, dtype=sorted_scores.dtype),
            dtype=sorted_scores.dtype,
        )
    )
    errors = np.r_[positives, errors]
    thresholds = np.r_[all_negative_threshold, thresholds]

    minimum = errors.min()
    tied = np.flatnonzero(errors == minimum)
    best = tied[np.argmin(np.abs(thresholds[tied] - 0.5))]
    return float(thresholds[best])


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        labels = data["labels"].astype(np.int8, copy=False)
        predictions = data["predictions"].astype(np.float64, copy=False)
    return labels, predictions


def read_candidate_indices(path: Path) -> tuple[np.ndarray, list[str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    ids = [row["test_row_id"] for row in rows]
    # Analysis row IDs are one-based physical TSV row numbers.
    indices = np.asarray([int(row_id) - 1 for row_id in ids], dtype=np.int64)
    return indices, ids


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    if "baseline" not in variants:
        raise ValueError("The variants must include baseline")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    metrics: list[dict[str, object]] = []
    validation_thresholds: dict[str, float] = {}

    for variant in variants:
        directory = (
            args.baseline_directory
            if variant == "baseline" and args.baseline_directory is not None
            else args.root / f"candidate_ablation_seed42_{variant}"
        )
        loaded[variant] = {}
        for split in ("val", "test", "leftout"):
            loaded[variant][split] = load_predictions(directory / f"{split}_predictions.npz")
        val_labels, val_predictions = loaded[variant]["val"]
        validation_thresholds[variant] = validation_error_threshold(val_labels, val_predictions)

        for split in ("val", "test", "leftout"):
            labels, predictions = loaded[variant][split]
            for threshold_source, threshold in (
                ("fixed_0.5", 0.5),
                ("validation_min_errors", validation_thresholds[variant]),
            ):
                counts = confusion_counts(labels, predictions, threshold)
                metrics.append(
                    {
                        "variant": variant,
                        "split": split,
                        "threshold_source": threshold_source,
                        "auprc": float(average_precision_score(labels, predictions)),
                        **counts,
                    }
                )

    candidate_indices, candidate_ids = read_candidate_indices(args.candidate_tsv)
    candidate_count = len(candidate_indices)
    baseline_labels, baseline_predictions = loaded["baseline"][args.candidate_split]
    if candidate_indices.max(initial=-1) >= baseline_labels.size:
        raise IndexError("Candidate test_row_id exceeds exported test predictions")
    if not np.all(baseline_labels[candidate_indices] == 1):
        raise ValueError("Expected every candidate-linked test row to be positive")

    candidate_rows: list[dict[str, object]] = []
    for variant in variants:
        _, predictions = loaded[variant][args.candidate_split]
        for row_id, index in zip(candidate_ids, candidate_indices, strict=True):
            candidate_rows.append(
                {
                    "test_row_id": row_id,
                    "variant": variant,
                    "prediction": float(predictions[index]),
                    "delta_from_baseline": float(
                        predictions[index] - baseline_predictions[index]
                    ),
                    "predicted_positive_at_validation_threshold": int(
                        predictions[index] >= validation_thresholds[variant]
                    ),
                }
            )

    write_tsv(args.output_dir / "metrics.tsv", metrics)
    write_tsv(args.output_dir / "candidate_score_changes.tsv", candidate_rows)

    metric_lookup = {
        (row["variant"], row["split"], row["threshold_source"]): row for row in metrics
    }
    lines = [
        "# Candidate-row ablation comparison",
        "",
        "Thresholds minimize FP + FN on validation and are applied unchanged to test and left-out.",
        "",
        "| variant | validation threshold | val AUPRC | test AUPRC | left-out AUPRC | test FP | test FN | left-out FP | left-out FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        val = metric_lookup[(variant, "val", "validation_min_errors")]
        test = metric_lookup[(variant, "test", "validation_min_errors")]
        leftout = metric_lookup[(variant, "leftout", "validation_min_errors")]
        lines.append(
            f"| {variant} | {validation_thresholds[variant]:.9f} | "
            f"{val['auprc']:.9f} | {test['auprc']:.9f} | {leftout['auprc']:.9f} | "
            f"{test['fp']} | {test['fn']} | {leftout['fp']} | {leftout['fn']} |"
        )

    lines.extend(["", f"## Linked positive {args.candidate_split} rows", ""])
    for variant in variants:
        _, predictions = loaded[variant][args.candidate_split]
        deltas = predictions[candidate_indices] - baseline_predictions[candidate_indices]
        rescued = int(
            np.count_nonzero(
                predictions[candidate_indices] >= validation_thresholds[variant]
            )
        )
        lines.append(
            f"- {variant}: mean score change {deltas.mean():+.9f}; "
            f"scores increased for {np.count_nonzero(deltas > 0)}/{candidate_count}; "
            f"predicted positive for {rescued}/{candidate_count} at its validation threshold."
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
