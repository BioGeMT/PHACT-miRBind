#!/usr/bin/env python3
"""Consolidate mature-miRNA PHACT-P1 signal and predictive evidence.

This analysis is deliberately evaluation-only: it consumes retained score
tables, caches, checkpoints' exported predictions, and benchmark labels.  It
does not train or modify a model.
"""

from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workspace import WORKSPACE, DATASETS, REPO

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


DEFAULT_SCRATCH = WORKSPACE
DEFAULT_HOME_REPO = REPO
DEFAULT_OLD_RELEASE = WORKSPACE / "archive/legacy_leaderboard_artifacts"
DEFAULT_DATASETS = DATASETS

MODEL_LABELS = {
    "sequence_only": "Sequence-only CNN",
    "old_mirna": "Old miRNA CountNodes_3 CNN",
    "p1_mirna": "P1 miRNA CNN",
    "old_target": "Target PHACT CNN",
    "old_both": "Old miRNA + target PHACT CNN",
    "p1_both": "P1 miRNA + target CNN",
    "p1_conservation": "P1 miRNA + target + conservation CNN",
    "p1_multimodal": "P1 multimodal fusion",
    "p1_multicandidate": "P1 multi-candidate fusion",
}

PLOT_MODEL_ORDER = [
    "sequence_only",
    "old_mirna",
    "p1_mirna",
    "old_target",
    "p1_both",
    "p1_conservation",
    "p1_multimodal",
    "p1_multicandidate",
]

PAIR_DEFINITIONS = [
    ("p1_mirna_vs_sequence", "p1_mirna", "sequence_only"),
    ("old_mirna_vs_sequence", "old_mirna", "sequence_only"),
    ("p1_mirna_vs_old_mirna", "p1_mirna", "old_mirna"),
    ("target_vs_sequence", "old_target", "sequence_only"),
    ("p1_both_vs_target", "p1_both", "old_target"),
    ("p1_both_vs_old_both", "p1_both", "old_both"),
]

SUBGROUPS = ["all", "p1_available", "p1_missing", "p1_flat", "p1_variable"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch-root", type=Path, default=DEFAULT_SCRATCH,
        help="PHACT scratch workspace root.",
    )
    parser.add_argument(
        "--home-repo", type=Path, default=DEFAULT_HOME_REPO,
        help="Original PHACT-miRBind checkout.",
    )
    parser.add_argument(
        "--old-release", type=Path, default=DEFAULT_OLD_RELEASE,
        help="Retained old-score leaderboard artifacts.",
    )
    parser.add_argument(
        "--datasets", type=Path, default=DEFAULT_DATASETS,
        help="Directory containing Manakov test and left-out TSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SCRATCH / "analyses/final" / "p1_evidence",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    parser.add_argument("--bootstrap-batch-size", type=int, default=5)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_dataset(path: Path) -> dict[str, np.ndarray]:
    labels: list[int] = []
    sequences: list[str] = []
    names: list[str] = []
    mature_ids: list[str] = []
    with require_file(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"label", "noncodingRNA", "noncodingRNA_name", "mirgenedb_mature_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            labels.append(int(row["label"]))
            sequences.append(row["noncodingRNA"].upper().replace("U", "T"))
            names.append(row["noncodingRNA_name"])
            mature_ids.append(row["mirgenedb_mature_id"])
    return {
        "labels": np.asarray(labels, dtype=np.int8),
        "sequences": np.asarray(sequences),
        "names": np.asarray(names),
        "mature_ids": np.asarray(mature_ids),
    }


def load_prediction_csv(path: Path, split: str, expected_rows: int) -> np.ndarray:
    predictions = np.full(expected_rows, np.nan, dtype=np.float64)
    seen = np.zeros(expected_rows, dtype=bool)
    with require_file(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "prediction"]:
            raise ValueError(f"Unexpected prediction schema in {path}: {reader.fieldnames}")
        prefix = f"{split}_"
        for row in reader:
            row_id = row["id"]
            if not row_id.startswith(prefix):
                raise ValueError(f"Unexpected row ID {row_id!r} in {path}")
            index = int(row_id[len(prefix):]) - 1
            if index < 0 or index >= expected_rows or seen[index]:
                raise ValueError(f"Invalid or duplicate row ID {row_id!r} in {path}")
            predictions[index] = float(row["prediction"])
            seen[index] = True
    if not seen.all() or not np.isfinite(predictions).all():
        raise ValueError(f"Missing or non-finite predictions in {path}")
    if np.any((predictions < 0) | (predictions > 1)):
        raise ValueError(f"Predictions outside [0, 1] in {path}")
    return predictions


def load_sequence_predictions(path: Path, expected_labels: np.ndarray) -> np.ndarray:
    with np.load(require_file(path)) as archive:
        if set(archive.files) != {"labels", "predictions"}:
            raise ValueError(f"Unexpected NPZ keys in {path}: {archive.files}")
        labels = archive["labels"].astype(np.int8, copy=False)
        predictions = archive["predictions"].astype(np.float64, copy=False)
    if not np.array_equal(labels, expected_labels):
        raise ValueError(f"Labels in {path} do not match the benchmark dataset")
    if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
        raise ValueError(f"Invalid predictions in {path}")
    return predictions


def model_prediction_paths(
    scratch_root: Path, old_release: Path, split: str,
) -> dict[str, Path]:
    old_predictions = old_release / "predictions"
    p1_predictions = scratch_root / "releases/mirbench_v7" / "predictions"
    split_inputs = "error_analysis_inputs" if split == "test" else "split_error_analysis_inputs"
    return {
        "sequence_only": scratch_root / "runs/experiments" / split_inputs / f"seq_only_{split}_predictions.npz",
        "old_mirna": old_predictions / "phact_countnodes3_mirna" / f"{split}.csv",
        "old_target": old_predictions / "phact_countnodes3_target" / f"{split}.csv",
        "old_both": old_predictions / "phact_countnodes3_both" / f"{split}.csv",
        "p1_mirna": p1_predictions / "phact_p1_mirna_cnn" / f"{split}.csv",
        "p1_both": p1_predictions / "phact_p1_mirna_target_cnn" / f"{split}.csv",
        "p1_conservation": p1_predictions / "phact_p1_conservation_cnn" / f"{split}.csv",
        "p1_multimodal": p1_predictions / "phact_p1_agentomics_selected_fusion" / f"{split}.csv",
        "p1_multicandidate": p1_predictions / "phact_p1_agentomics_multicandidate_fusion" / f"{split}.csv",
    }


def load_all_predictions(
    scratch_root: Path,
    old_release: Path,
    split: str,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    paths = model_prediction_paths(scratch_root, old_release, split)
    result: dict[str, np.ndarray] = {}
    for model_id, path in paths.items():
        if path.suffix == ".npz":
            result[model_id] = load_sequence_predictions(path, labels)
        else:
            result[model_id] = load_prediction_csv(path, split, len(labels))
    return result


BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def classify_cached_p1_profiles(
    cache_dir: Path,
    sequences: np.ndarray,
    labels: np.ndarray,
    flat_tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    categories: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    offset = 0
    shard_paths = sorted(cache_dir.glob("*_shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No cache shards in {cache_dir}")
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        scores = shard["mirna_phact"].float().numpy()
        missing = shard["mirna_phact_missing"].squeeze(-1).bool().numpy()
        shard_labels = shard["labels"].numpy().astype(np.int8, copy=False)
        rows, length, channels = scores.shape
        if channels != 4 or length != 28:
            raise ValueError(f"Unexpected P1 tensor shape in {shard_path}: {scores.shape}")
        if not np.array_equal(shard_labels, labels[offset:offset + rows]):
            raise ValueError(f"Cache labels are misaligned in {shard_path}")

        base_indices = np.full((rows, length), -1, dtype=np.int8)
        for local_index, sequence in enumerate(sequences[offset:offset + rows]):
            encoded = [BASE_TO_INDEX.get(base, -1) for base in sequence[:length]]
            base_indices[local_index, :len(encoded)] = encoded
        valid = (~missing) & (base_indices >= 0)
        if np.any((~missing) & (base_indices < 0)):
            raise ValueError(f"A scored position lacks an A/C/G/T reference base in {shard_path}")
        safe_indices = np.where(base_indices >= 0, base_indices, 0)
        reference = np.take_along_axis(scores, safe_indices[..., None], axis=2).squeeze(2)
        contrast = reference - (scores.sum(axis=2) - reference) / 3.0
        minima = np.where(valid, contrast, np.inf).min(axis=1)
        maxima = np.where(valid, contrast, -np.inf).max(axis=1)
        has_profile = valid.any(axis=1)
        shard_ranges = maxima - minima
        shard_ranges[~has_profile] = np.nan
        shard_categories = np.full(rows, "variable", dtype="U8")
        shard_categories[~has_profile] = "missing"
        shard_categories[has_profile & (shard_ranges <= flat_tolerance)] = "flat"
        categories.append(shard_categories)
        ranges.append(shard_ranges)
        offset += rows
    if offset != len(labels):
        raise ValueError(f"Cache rows ({offset}) do not match dataset rows ({len(labels)})")
    return np.concatenate(categories), np.concatenate(ranges)


def read_arm_ranges(path: Path, model_name: str) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    with require_file(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["phact_model"] != model_name:
                continue
            scores = np.asarray([float(row[f"score_{base}"]) for base in "ACGT"])
            actual = row["actual_nt"].upper().replace("U", "T")
            if actual not in BASE_TO_INDEX:
                raise ValueError(f"Unsupported actual nucleotide {actual!r} in {path}")
            reference = scores[BASE_TO_INDEX[actual]]
            contrast = reference - (scores.sum() - reference) / 3.0
            values[(row["pre_mirna"], row["arm"])].append(float(contrast))
    if not values:
        raise ValueError(f"No rows for model {model_name!r} in {path}")
    return {key: max(group) - min(group) for key, group in values.items()}


def compare_arm_signal(
    old_table: Path,
    new_table: Path,
    tolerance: float = 1e-12,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    old_ranges = read_arm_ranges(old_table, "1")
    new_ranges = read_arm_ranges(new_table, "param_1")
    keys = sorted(old_ranges.keys() & new_ranges.keys())
    if keys != sorted(old_ranges) or keys != sorted(new_ranges):
        raise ValueError("Old and new parameter-1 arm keys are not identical")
    rows: list[dict[str, object]] = []
    for precursor, arm in keys:
        old_range = old_ranges[(precursor, arm)]
        new_range = new_ranges[(precursor, arm)]
        old_flat = old_range <= tolerance
        new_flat = new_range <= tolerance
        if old_flat and new_flat:
            outcome = "both_flat"
        elif new_range > old_range + tolerance:
            outcome = "stronger"
        elif new_range < old_range - tolerance:
            outcome = "weaker"
        else:
            outcome = "unchanged_variable"
        rows.append({
            "pre_mirna": precursor,
            "arm": arm,
            "old_range": old_range,
            "consensus_p1_range": new_range,
            "range_delta": new_range - old_range,
            "old_flat": old_flat,
            "consensus_p1_flat": new_flat,
            "outcome": outcome,
        })
    outcomes = count_values(row["outcome"] for row in rows)
    old_values = np.asarray([float(row["old_range"]) for row in rows])
    new_values = np.asarray([float(row["consensus_p1_range"]) for row in rows])
    summary = {
        "matched_arms": len(rows),
        "old_signal_arms": int((old_values > tolerance).sum()),
        "old_flat_arms": int((old_values <= tolerance).sum()),
        "consensus_p1_signal_arms": int((new_values > tolerance).sum()),
        "consensus_p1_flat_arms": int((new_values <= tolerance).sum()),
        "stronger_arms": outcomes.get("stronger", 0),
        "weaker_arms": outcomes.get("weaker", 0),
        "both_flat_arms": outcomes.get("both_flat", 0),
        "unchanged_variable_arms": outcomes.get("unchanged_variable", 0),
        "old_median_range": float(np.median(old_values)),
        "consensus_p1_median_range": float(np.median(new_values)),
        "median_range_percent_change": float(
            100.0 * (np.median(new_values) / np.median(old_values) - 1.0)
        ),
    }
    return rows, summary


def count_values(values: Iterable[object]) -> dict[object, int]:
    result: dict[object, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def subgroup_masks(categories: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(categories), dtype=bool),
        "p1_available": categories != "missing",
        "p1_missing": categories == "missing",
        "p1_flat": categories == "flat",
        "p1_variable": categories == "variable",
    }


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions have different lengths")
    positive = int(labels.sum())
    result: dict[str, float | int] = {
        "n": len(labels),
        "positive": positive,
        "positive_rate": positive / len(labels),
        "auprc": float("nan"),
        "auroc": float("nan"),
        "brier": float(brier_score_loss(labels, predictions)),
        "log_loss": float(log_loss(labels, predictions, labels=[0, 1])),
    }
    if 0 < positive < len(labels):
        result["auprc"] = float(average_precision_score(labels, predictions))
        result["auroc"] = float(roc_auc_score(labels, predictions))
    return result


def calculate_subgroup_metrics(
    split: str,
    labels: np.ndarray,
    categories: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    masks = subgroup_masks(categories)
    for subgroup in SUBGROUPS:
        mask = masks[subgroup]
        for model_id, model_predictions in predictions.items():
            metrics = calculate_metrics(labels[mask], model_predictions[mask])
            rows.append({
                "split": split,
                "subgroup": subgroup,
                "model_id": model_id,
                "model": MODEL_LABELS[model_id],
                **metrics,
            })
    return rows


@dataclass
class WeightedAveragePrecision:
    labels: np.ndarray
    predictions: np.ndarray

    def __post_init__(self) -> None:
        self.order = np.argsort(-self.predictions, kind="mergesort")
        self.sorted_labels = self.labels[self.order].astype(np.float64, copy=False)
        sorted_predictions = self.predictions[self.order]
        self.ends = np.flatnonzero(np.r_[sorted_predictions[1:] != sorted_predictions[:-1], True])
        self.starts = np.r_[0, self.ends[:-1] + 1]
        unweighted = self.calculate(np.ones((1, len(self.labels)), dtype=np.float64))[0]
        expected = average_precision_score(self.labels, self.predictions)
        if not math.isclose(unweighted, expected, rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"Weighted AP implementation mismatch: {unweighted} != {expected}")

    def calculate(self, row_weights: np.ndarray) -> np.ndarray:
        weights = row_weights[:, self.order]
        positive_weights = weights * self.sorted_labels
        cumulative_weight = np.cumsum(weights, axis=1)
        cumulative_positive = np.cumsum(positive_weights, axis=1)
        group_positive = np.add.reduceat(positive_weights, self.starts, axis=1)
        denominators = cumulative_weight[:, self.ends]
        precision = np.divide(
            cumulative_positive[:, self.ends],
            denominators,
            out=np.zeros_like(denominators),
            where=denominators > 0,
        )
        total_positive = cumulative_positive[:, -1]
        numerator = np.sum(precision * group_positive, axis=1)
        return np.divide(
            numerator,
            total_positive,
            out=np.full_like(total_positive, np.nan),
            where=total_positive > 0,
        )


def cluster_bootstrap_average_precision(
    labels: np.ndarray,
    sequences: np.ndarray,
    predictions: dict[str, np.ndarray],
    replicates: int,
    seed: int,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], int]:
    _, groups = np.unique(sequences, return_inverse=True)
    group_count = int(groups.max()) + 1
    rng = np.random.default_rng(seed)
    calculators = {
        model_id: WeightedAveragePrecision(labels, model_predictions)
        for model_id, model_predictions in predictions.items()
    }
    distributions = {
        model_id: np.empty(replicates, dtype=np.float64) for model_id in predictions
    }
    probability = np.full(group_count, 1.0 / group_count)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        sampled_group_counts = rng.multinomial(group_count, probability, size=stop - start)
        row_weights = sampled_group_counts[:, groups].astype(np.float64, copy=False)
        for model_id, calculator in calculators.items():
            distributions[model_id][start:stop] = calculator.calculate(row_weights)
    for model_id, values in distributions.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite bootstrap AUPRC values for {model_id}")
    return distributions, group_count


def summarize_paired_bootstrap(
    split: str,
    overall_metrics: dict[str, float],
    distributions: dict[str, np.ndarray],
    group_count: int,
    subgroup: str = "all",
    pair_definitions: list[tuple[str, str, str]] = PAIR_DEFINITIONS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for comparison, model_a, model_b in pair_definitions:
        delta_distribution = distributions[model_a] - distributions[model_b]
        low, high = np.quantile(delta_distribution, [0.025, 0.975])
        rows.append({
            "split": split,
            "subgroup": subgroup,
            "comparison": comparison,
            "model_a": MODEL_LABELS[model_a],
            "model_b": MODEL_LABELS[model_b],
            "auprc_a": overall_metrics[model_a],
            "auprc_b": overall_metrics[model_b],
            "auprc_delta_a_minus_b": overall_metrics[model_a] - overall_metrics[model_b],
            "cluster_bootstrap_ci_low": float(low),
            "cluster_bootstrap_ci_high": float(high),
            "bootstrap_probability_delta_gt_zero": float(np.mean(delta_distribution > 0)),
            "bootstrap_clusters": group_count,
            "bootstrap_unit": "exact_mature_miRNA_sequence",
            "bootstrap_replicates": len(delta_distribution),
        })
    return rows


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def metric_lookup(
    rows: list[dict[str, object]], split: str, subgroup: str, model_id: str, metric: str,
) -> float:
    matches = [
        row for row in rows
        if row["split"] == split and row["subgroup"] == subgroup and row["model_id"] == model_id
    ]
    if len(matches) != 1:
        raise ValueError((split, subgroup, model_id, metric, len(matches)))
    return float(matches[0][metric])


def bootstrap_lookup(
    rows: list[dict[str, object]], split: str, comparison: str, subgroup: str = "all",
) -> dict[str, object]:
    matches = [
        row for row in rows
        if row["split"] == split
        and row["subgroup"] == subgroup
        and row["comparison"] == comparison
    ]
    if len(matches) != 1:
        raise ValueError((split, subgroup, comparison, len(matches)))
    return matches[0]


def make_overview_plot(
    output_path: Path,
    arm_summary: dict[str, object],
    coverage_summary: dict[str, dict[str, object]],
    metric_rows: list[dict[str, object]],
    bootstrap_rows: list[dict[str, object]],
) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
    })
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    fig.suptitle("What consensus-tree PHACT-P1 contributes to miRNA–target prediction", fontsize=19, fontweight="bold")

    # A: arm-level positional signal.
    ax = axes[0, 0]
    signal = [arm_summary["old_signal_arms"], arm_summary["consensus_p1_signal_arms"]]
    flat = [arm_summary["old_flat_arms"], arm_summary["consensus_p1_flat_arms"]]
    labels = ["Old parameter 1", "Consensus-tree P1"]
    x = np.arange(2)
    ax.bar(x, signal, color="#2a9d8f", label="Position-variable")
    ax.bar(x, flat, bottom=signal, color="#aab4c3", label="Flat")
    for index in range(2):
        ax.text(index, signal[index] / 2, f"{signal[index]} variable", ha="center", va="center", color="white", fontweight="bold")
        ax.text(index, signal[index] + flat[index] / 2, f"{flat[index]} flat", ha="center", va="center", fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Candidate mature arms")
    ax.set_title("A  P1 signal is stronger, but flat arms are not rescued", loc="left")
    ax.text(
        0.5, 0.03,
        f"Median range: {arm_summary['old_median_range']:.3f} → {arm_summary['consensus_p1_median_range']:.3f} "
        f"({arm_summary['median_range_percent_change']:+.1f}%)",
        transform=ax.transAxes, ha="center", va="bottom",
    )

    # B: benchmark-row P1 coverage.
    ax = axes[0, 1]
    splits = ["test", "leftout"]
    colors = {"variable": "#2a9d8f", "flat": "#f2a93b", "missing": "#aab4c3"}
    bottoms = np.zeros(2)
    for category in ["variable", "flat", "missing"]:
        percentages = [100.0 * coverage_summary[split][category] / coverage_summary[split]["rows"] for split in splits]
        ax.bar(np.arange(2), percentages, bottom=bottoms, color=colors[category], label=category.capitalize())
        for index, percentage in enumerate(percentages):
            if percentage >= 6:
                ax.text(index, bottoms[index] + percentage / 2, f"{percentage:.1f}%", ha="center", va="center", fontweight="bold")
        bottoms += percentages
    ax.set_xticks(np.arange(2), ["Test", "Left-out"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Rows (%)")
    ax.set_title("B  P1 availability differs sharply between splits", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper center")

    # C: overall model comparison.
    ax = axes[1, 0]
    y = np.arange(len(PLOT_MODEL_ORDER))
    test_scores = [metric_lookup(metric_rows, "test", "all", model, "auprc") for model in PLOT_MODEL_ORDER]
    leftout_scores = [metric_lookup(metric_rows, "leftout", "all", model, "auprc") for model in PLOT_MODEL_ORDER]
    for index, (test_score, leftout_score) in enumerate(zip(test_scores, leftout_scores)):
        ax.plot([test_score, leftout_score], [index, index], color="#bac2cf", linewidth=2, zorder=1)
    ax.scatter(test_scores, y, color="#e45756", marker="s", s=45, label="Test", zorder=2)
    ax.scatter(leftout_scores, y, color="#4c78a8", marker="o", s=45, label="Left-out", zorder=2)
    ax.set_yticks(y, [MODEL_LABELS[model] for model in PLOT_MODEL_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("AUPRC")
    ax.set_title("C  Richer models improve test, not left-out", loc="left")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    # D: P1 miRNA contribution within coverage strata.
    ax = axes[1, 1]
    subgroups = ["all", "p1_missing", "p1_flat", "p1_variable"]
    display = ["All rows", "P1 missing", "P1 flat", "P1 variable"]
    x = np.arange(len(subgroups))
    for split, offset, color, marker, label in [
        ("test", -0.10, "#e45756", "s", "Test"),
        ("leftout", 0.10, "#4c78a8", "o", "Left-out"),
    ]:
        rows = [
            bootstrap_lookup(bootstrap_rows, split, "p1_mirna_vs_sequence", subgroup)
            for subgroup in subgroups
        ]
        deltas = np.asarray([float(row["auprc_delta_a_minus_b"]) for row in rows])
        low = np.asarray([float(row["cluster_bootstrap_ci_low"]) for row in rows])
        high = np.asarray([float(row["cluster_bootstrap_ci_high"]) for row in rows])
        ax.errorbar(
            x + offset,
            deltas,
            yerr=np.vstack([deltas - low, high - deltas]),
            fmt=marker,
            color=color,
            capsize=3,
            markersize=6,
            linewidth=1.5,
            label=label,
        )
    ax.axhline(0, color="#202733", linewidth=1)
    ax.set_xticks(x, display, rotation=15, ha="right")
    ax.set_ylabel("AUPRC difference: P1 miRNA − sequence-only")
    ax.set_title("D  Mature P1 adds no consistent predictive gain", loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    fig.text(
        0.5, 0.012,
        "P1 signal range = max − min of PHACT(reference nucleotide) − mean PHACT(alternatives) across the mature arm. "
        "Bootstrap resamples exact mature-miRNA sequence groups.",
        ha="center", fontsize=9, color="#566176",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_report(
    path: Path,
    arm_summary: dict[str, object],
    coverage_summary: dict[str, dict[str, object]],
    metric_rows: list[dict[str, object]],
    bootstrap_rows: list[dict[str, object]],
) -> None:
    sequence_test = metric_lookup(metric_rows, "test", "all", "sequence_only", "auprc")
    p1_test = metric_lookup(metric_rows, "test", "all", "p1_mirna", "auprc")
    sequence_leftout = metric_lookup(metric_rows, "leftout", "all", "sequence_only", "auprc")
    p1_leftout = metric_lookup(metric_rows, "leftout", "all", "p1_mirna", "auprc")
    lines = [
        "# PHACT-P1 evidence audit",
        "",
        "This audit uses retained predictions and score/cache artifacts only. No model was retrained.",
        "",
        "## Main conclusion",
        "",
        "Consensus-tree parameter 1 increases the magnitude of mature-position variation for most already-variable arms, "
        "but it does not rescue arms that were flat under old parameter 1. Adding mature-miRNA P1 channels to the matched "
        "sequence CNN produces effectively unchanged benchmark AUPRC.",
        "",
        "## Mature-arm signal",
        "",
        f"- Matched candidate arms: {arm_summary['matched_arms']}",
        f"- Old parameter 1: {arm_summary['old_signal_arms']} variable, {arm_summary['old_flat_arms']} flat",
        f"- Consensus-tree P1: {arm_summary['consensus_p1_signal_arms']} variable, {arm_summary['consensus_p1_flat_arms']} flat",
        f"- Range became stronger in {arm_summary['stronger_arms']} arms and weaker in {arm_summary['weaker_arms']} arms",
        f"- Median mature-region range: {arm_summary['old_median_range']:.6f} → {arm_summary['consensus_p1_median_range']:.6f} "
        f"({arm_summary['median_range_percent_change']:+.1f}%)",
        "",
        "The flat set is unchanged. Therefore the defensible score claim is stronger contrast in most non-flat arms, "
        "not recovery of signal in previously flat arms.",
        "",
        "## Coverage in the evaluated rows",
        "",
    ]
    for split in ["test", "leftout"]:
        summary = coverage_summary[split]
        lines.append(
            f"- {split.capitalize()}: {summary['variable']:,} variable ({100*summary['variable']/summary['rows']:.1f}%), "
            f"{summary['flat']:,} flat ({100*summary['flat']/summary['rows']:.1f}%), "
            f"{summary['missing']:,} missing ({100*summary['missing']/summary['rows']:.1f}%)"
        )
    lines.extend([
        "",
        "The much larger missing-profile fraction in left-out is an important explanation to consider when interpreting "
        "different test and left-out behavior.",
        "",
        "## Clean sequence-only comparison",
        "",
        "| Split | Sequence-only | P1 miRNA CNN | Difference | miRNA-cluster 95% CI |",
        "|---|---:|---:|---:|---:|",
    ])
    for split, sequence_score, p1_score in [
        ("Test", sequence_test, p1_test),
        ("Left-out", sequence_leftout, p1_leftout),
    ]:
        row = bootstrap_lookup(bootstrap_rows, split.lower().replace("-", ""), "p1_mirna_vs_sequence")
        lines.append(
            f"| {split} | {sequence_score:.6f} | {p1_score:.6f} | {p1_score-sequence_score:+.6f} | "
            f"[{row['cluster_bootstrap_ci_low']:+.6f}, {row['cluster_bootstrap_ci_high']:+.6f}] |"
        )
    lines.extend([
        "",
        "The confidence intervals include zero on both splits, so these retained predictions do not support a claim that "
        "mature-miRNA P1 improves prediction over sequence alone.",
        "",
        "### P1 contribution by input category",
        "",
        "| Split | P1 category | AUPRC difference | miRNA-cluster 95% CI | Sequence clusters |",
        "|---|---|---:|---:|---:|",
    ])
    for split in ["test", "leftout"]:
        for subgroup, label in [
            ("p1_missing", "Missing"),
            ("p1_flat", "Flat"),
            ("p1_variable", "Variable"),
        ]:
            row = bootstrap_lookup(
                bootstrap_rows, split, "p1_mirna_vs_sequence", subgroup,
            )
            lines.append(
                f"| {split.capitalize()} | {label} | {row['auprc_delta_a_minus_b']:+.6f} | "
                f"[{row['cluster_bootstrap_ci_low']:+.6f}, {row['cluster_bootstrap_ci_high']:+.6f}] | "
                f"{row['bootstrap_clusters']} |"
            )
    lines.extend([
        "",
        "Any favorable point estimate is concentrated in rows with genuinely variable P1 profiles; flat or missing profiles "
        "do not show a consistent benefit. The subgroup confidence intervals should be used rather than the point estimates "
        "alone. In particular, the left-out flat subgroup contains only five exact mature-miRNA sequences and is not broad "
        "enough for a general conclusion.",
        "",
        "## Interpretation of the other models",
        "",
        "| Split | Target PHACT − sequence-only AUPRC | miRNA-cluster 95% CI |",
        "|---|---:|---:|",
    ])
    for split in ["test", "leftout"]:
        row = bootstrap_lookup(bootstrap_rows, split, "target_vs_sequence")
        lines.append(
            f"| {split.capitalize()} | {row['auprc_delta_a_minus_b']:+.6f} | "
            f"[{row['cluster_bootstrap_ci_low']:+.6f}, {row['cluster_bootstrap_ci_high']:+.6f}] |"
        )
    lines.extend([
        "",
        "Target-side PHACT produces the visible test-set improvement. Conservation and multimodal branches increase test "
        "AUPRC further, but the target-only effect reverses on left-out and none of the richer models surpasses the simple "
        "sequence/P1-miRNA CNN there. Their performance cannot "
        "be attributed specifically to mature-miRNA P1 because they add target scores, conservation, metadata, alternative "
        "sequence encodings, and/or RiNALMo representations simultaneously.",
        "",
        "## Remaining decision",
        "",
        "No additional training is required for the practical question, ‘Does adding P1 outperform the retained no-PHACT "
        "CNN?’ A constant-channel control would only be needed for the narrower mechanistic question of whether P1 values "
        "help independently of the extra input-layer parameters.",
        "",
        "## Output files",
        "",
        "- `p1_evidence_overview.png` / `.pdf`: consolidated figure",
        "- `p1_arm_signal_comparison.tsv`: per-arm old-versus-consensus P1 ranges",
        "- `p1_row_coverage.tsv`: row counts by P1 input category",
        "- `p1_subgroup_metrics.tsv`: metrics for every retained model and P1 subgroup",
        "- `p1_paired_cluster_bootstrap.tsv`: paired AUPRC differences and confidence intervals",
        "- `p1_evidence_summary.json`: machine-readable summary and provenance",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    old_arm_table = args.scratch_root / "data/scores/main_repo" / "phact_mirna_arm_position_qntnorm_transformed_all_models.tsv"
    new_arm_table = args.scratch_root / "runs/param1-training" / "data" / "phact_mirna_arm_position_qntnorm_transformed_param_1.tsv"
    arm_rows, arm_summary = compare_arm_signal(old_arm_table, new_arm_table)
    write_tsv(args.output_dir / "p1_arm_signal_comparison.tsv", arm_rows)

    metric_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    coverage_summary: dict[str, dict[str, object]] = {}
    split_provenance: dict[str, object] = {}

    for split_index, split in enumerate(["test", "leftout"]):
        dataset_path = args.datasets / f"AGO2_eCLIP_Manakov2022_{split}.tsv"
        dataset = read_dataset(dataset_path)
        labels = dataset["labels"]
        predictions = load_all_predictions(args.scratch_root, args.old_release, split, labels)
        cache_dir = args.scratch_root / "runs/param1-training" / "cache" / "param_1_target_score" / split
        categories, profile_ranges = classify_cached_p1_profiles(
            cache_dir, dataset["sequences"], labels,
        )
        category_counts = count_values(categories.tolist())
        summary = {
            "rows": len(labels),
            "positive": int(labels.sum()),
            "variable": category_counts.get("variable", 0),
            "flat": category_counts.get("flat", 0),
            "missing": category_counts.get("missing", 0),
            "median_range_among_available": float(np.nanmedian(profile_ranges)),
        }
        coverage_summary[split] = summary
        for category in ["variable", "flat", "missing"]:
            coverage_rows.append({
                "split": split,
                "category": category,
                "rows": summary[category],
                "percent": 100.0 * summary[category] / summary["rows"],
            })
        metric_rows.extend(calculate_subgroup_metrics(split, labels, categories, predictions))
        overall_metrics = {
            model_id: float(average_precision_score(labels, model_predictions))
            for model_id, model_predictions in predictions.items()
        }
        core_predictions = {
            model_id: predictions[model_id]
            for model_id in sorted({item for _, model_a, model_b in PAIR_DEFINITIONS for item in (model_a, model_b)})
        }
        distributions, group_count = cluster_bootstrap_average_precision(
            labels,
            dataset["sequences"],
            core_predictions,
            args.bootstrap_replicates,
            args.bootstrap_seed + split_index,
            args.bootstrap_batch_size,
        )
        bootstrap_rows.extend(
            summarize_paired_bootstrap(split, overall_metrics, distributions, group_count)
        )
        masks = subgroup_masks(categories)
        subgroup_pair = [("p1_mirna_vs_sequence", "p1_mirna", "sequence_only")]
        for subgroup_index, subgroup in enumerate(["p1_missing", "p1_flat", "p1_variable"], start=1):
            mask = masks[subgroup]
            subgroup_predictions = {
                "sequence_only": predictions["sequence_only"][mask],
                "p1_mirna": predictions["p1_mirna"][mask],
            }
            subgroup_distributions, subgroup_group_count = cluster_bootstrap_average_precision(
                labels[mask],
                dataset["sequences"][mask],
                subgroup_predictions,
                args.bootstrap_replicates,
                args.bootstrap_seed + split_index * 10 + subgroup_index,
                args.bootstrap_batch_size,
            )
            subgroup_overall_metrics = {
                model_id: float(average_precision_score(labels[mask], model_predictions))
                for model_id, model_predictions in subgroup_predictions.items()
            }
            bootstrap_rows.extend(
                summarize_paired_bootstrap(
                    split,
                    subgroup_overall_metrics,
                    subgroup_distributions,
                    subgroup_group_count,
                    subgroup=subgroup,
                    pair_definitions=subgroup_pair,
                )
            )
        split_provenance[split] = {
            "dataset": str(dataset_path),
            "cache": str(cache_dir),
            "prediction_paths": {
                model_id: str(path)
                for model_id, path in model_prediction_paths(args.scratch_root, args.old_release, split).items()
            },
            "unique_exact_mature_sequences": group_count,
        }

    write_tsv(args.output_dir / "p1_row_coverage.tsv", coverage_rows)
    write_tsv(args.output_dir / "p1_subgroup_metrics.tsv", metric_rows)
    write_tsv(args.output_dir / "p1_paired_cluster_bootstrap.tsv", bootstrap_rows)

    summary = {
        "analysis": "PHACT consensus parameter-1 evidence audit",
        "training_performed": False,
        "arm_signal": arm_summary,
        "row_coverage": coverage_summary,
        "bootstrap": {
            "method": "paired nonparametric cluster bootstrap over exact mature-miRNA sequences",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "comparisons": bootstrap_rows,
        },
        "provenance": {
            "old_parameter1_arm_table": str(old_arm_table),
            "consensus_parameter1_arm_table": str(new_arm_table),
            "splits": split_provenance,
        },
    }
    (args.output_dir / "p1_evidence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_overview_plot(
        args.output_dir / "p1_evidence_overview.png",
        arm_summary,
        coverage_summary,
        metric_rows,
        bootstrap_rows,
    )
    make_report(
        args.output_dir / "README.md",
        arm_summary,
        coverage_summary,
        metric_rows,
        bootstrap_rows,
    )
    print(json.dumps({
        "status": "complete",
        "output_dir": str(args.output_dir),
        "arm_signal": arm_summary,
        "coverage": coverage_summary,
        "bootstrap_replicates": args.bootstrap_replicates,
    }, indent=2))


if __name__ == "__main__":
    main()
