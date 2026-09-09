#!/usr/bin/env python3
"""Relate Manakov development-test errors to existing models and training evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


METADATA_COLUMNS = (
    "gene",
    "noncodingRNA",
    "noncodingRNA_name",
    "noncodingRNA_fam",
    "mirgenedb_family",
    "feature",
    "label",
    "chr",
    "start",
    "end",
    "strand",
    "dominant_region",
    "regions_present",
    "gene_cluster_ID",
)
KEY_NAMES = (
    "exact_pair",
    "target",
    "target_seed",
    "target_family",
    "cluster_mirna",
    "mirna",
)
SIGNATURE_MODELS = (
    "original_phact",
    "seq_only",
    "conservation_phact",
    "rinalmo",
    "rinalmo_cross",
)


def normalize_sequence(value: str) -> str:
    return str(value).strip().upper().replace("U", "T")


def normalize_category(value: str) -> str:
    normalized = str(value).strip().upper()
    return "NA" if normalized in {"", "NA", "NAN", "NONE"} else normalized


def family_value(row: dict[str, str]) -> str:
    primary = normalize_category(row.get("mirgenedb_family", ""))
    return primary if primary != "NA" else normalize_category(row["noncodingRNA_fam"])


def seed_value(mirna: str) -> str:
    return normalize_sequence(mirna)[1:8]


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("value must use NAME=PATH")
    return name, Path(raw_path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_test_rows(path: Path) -> pd.DataFrame:
    with path.open() as handle:
        header = handle.readline().rstrip("\n\r").split("\t")
    available = set(header)
    required = {"gene", "noncodingRNA", "noncodingRNA_fam", "label"}
    missing = required - available
    if missing:
        raise ValueError(f"Missing test columns: {sorted(missing)}")
    usecols = [column for column in METADATA_COLUMNS if column in available]
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
    )
    for column in METADATA_COLUMNS:
        if column not in frame:
            frame[column] = ""
    frame.insert(0, "test_row_id", np.arange(1, len(frame) + 1, dtype=np.int32))
    frame["label"] = frame["label"].astype(np.int8)
    frame["gene"] = frame["gene"].map(normalize_sequence)
    frame["noncodingRNA"] = frame["noncodingRNA"].map(normalize_sequence)
    frame["analysis_family"] = [
        primary if primary != "NA" else fallback
        for primary, fallback in zip(
            frame["mirgenedb_family"].map(normalize_category),
            frame["noncodingRNA_fam"].map(normalize_category),
            strict=True,
        )
    ]
    frame["analysis_seed"] = frame["noncodingRNA"].map(seed_value)
    frame["gene_cluster_ID"] = frame["gene_cluster_ID"].map(normalize_category)
    return frame


def frame_keys(frame: pd.DataFrame, name: str) -> Iterable[object]:
    if name == "exact_pair":
        return zip(frame["gene"], frame["noncodingRNA"], strict=True)
    if name == "target":
        return iter(frame["gene"])
    if name == "target_seed":
        return zip(frame["gene"], frame["analysis_seed"], strict=True)
    if name == "target_family":
        return zip(frame["gene"], frame["analysis_family"], strict=True)
    if name == "cluster_mirna":
        return zip(frame["gene_cluster_ID"], frame["noncodingRNA"], strict=True)
    if name == "mirna":
        return iter(frame["noncodingRNA"])
    raise ValueError(f"Unknown evidence key: {name}")


def row_keys(row: dict[str, str]) -> dict[str, object]:
    target = normalize_sequence(row["gene"])
    mirna = normalize_sequence(row["noncodingRNA"])
    family = family_value(row)
    cluster = normalize_category(row.get("gene_cluster_ID", ""))
    return {
        "exact_pair": (target, mirna),
        "target": target,
        "target_seed": (target, seed_value(mirna)),
        "target_family": (target, family),
        "cluster_mirna": (cluster, mirna),
        "mirna": mirna,
    }


def scan_training_evidence(
    path: Path,
    test_key_sets: dict[str, set[object]],
) -> tuple[dict[str, dict[object, list[int]]], int]:
    counts: dict[str, dict[object, list[int]]] = {
        name: {} for name in KEY_NAMES
    }
    rows_scanned = 0
    with path.open(newline="") as handle:
        for rows_scanned, row in enumerate(
            csv.DictReader(handle, delimiter="\t"), start=1
        ):
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Invalid training label at row {rows_scanned}")
            for name, key in row_keys(row).items():
                if key not in test_key_sets[name]:
                    continue
                label_counts = counts[name].setdefault(key, [0, 0])
                label_counts[label] += 1
            if rows_scanned % 500_000 == 0:
                print(f"training_rows_scanned={rows_scanned:,}", flush=True)
    return counts, rows_scanned


def add_training_counts(
    frame: pd.DataFrame,
    counts: dict[str, dict[object, list[int]]],
) -> None:
    for name in KEY_NAMES:
        for label, suffix in ((0, "neg"), (1, "pos")):
            frame[f"train_{name}_{suffix}"] = np.fromiter(
                (counts[name].get(key, (0, 0))[label] for key in frame_keys(frame, name)),
                dtype=np.int32,
                count=len(frame),
            )


def scan_gse_evidence(
    sources: list[tuple[str, Path]],
    test_pairs: set[object],
    test_target_seeds: set[object],
) -> tuple[
    dict[object, list[int]],
    dict[object, list[int]],
    dict[object, Counter[str]],
    int,
]:
    pair_counts: dict[object, list[int]] = {}
    target_seed_counts: dict[object, list[int]] = {}
    pair_sources: dict[object, Counter[str]] = defaultdict(Counter)
    rows_scanned = 0
    for source_name, path in sources:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                rows_scanned += 1
                label = int(row["label"])
                keys = row_keys(row)
                pair = keys["exact_pair"]
                if pair in test_pairs:
                    pair_counts.setdefault(pair, [0, 0])[label] += 1
                    pair_sources[pair][source_name] += 1
                target_seed = keys["target_seed"]
                if target_seed in test_target_seeds:
                    target_seed_counts.setdefault(target_seed, [0, 0])[label] += 1
    return pair_counts, target_seed_counts, pair_sources, rows_scanned


def add_gse_counts(
    frame: pd.DataFrame,
    pair_counts: dict[object, list[int]],
    target_seed_counts: dict[object, list[int]],
) -> None:
    for label, suffix in ((0, "neg"), (1, "pos")):
        frame[f"gse_exact_pair_{suffix}"] = np.fromiter(
            (pair_counts.get(key, (0, 0))[label] for key in frame_keys(frame, "exact_pair")),
            dtype=np.int16,
            count=len(frame),
        )
        frame[f"gse_target_seed_{suffix}"] = np.fromiter(
            (
                target_seed_counts.get(key, (0, 0))[label]
                for key in frame_keys(frame, "target_seed")
            ),
            dtype=np.int32,
            count=len(frame),
        )


def load_npz_predictions(path: Path, expected_labels: np.ndarray) -> np.ndarray:
    arrays = np.load(path)
    if "labels" not in arrays or "predictions" not in arrays:
        raise ValueError(f"Prediction NPZ lacks labels/predictions: {path}")
    labels = arrays["labels"].astype(np.int8)
    if not np.array_equal(labels, expected_labels):
        raise ValueError(f"Prediction labels are not test-row aligned: {path}")
    return arrays["predictions"].astype(np.float32)


def load_csv_predictions(
    path: Path, expected_rows: int, id_prefix: str = "test"
) -> np.ndarray:
    predictions: list[float] = []
    with path.open(newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=1):
            expected_id = f"{id_prefix}_{row_index}"
            if row.get("id") != expected_id:
                raise ValueError(f"Unexpected prediction ID at row {row_index}: {path}")
            predictions.append(float(row["prediction"]))
    if len(predictions) != expected_rows:
        raise ValueError(f"Prediction rows do not match test rows: {path}")
    return np.asarray(predictions, dtype=np.float32)


def rank_violation_counts(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    negative_scores = np.sort(predictions[labels == 0])
    positive_scores = np.sort(predictions[labels == 1])
    violations = np.empty(len(labels), dtype=np.int32)
    positive_rows = labels == 1
    violations[positive_rows] = len(negative_scores) - np.searchsorted(
        negative_scores, predictions[positive_rows], side="right"
    )
    violations[~positive_rows] = np.searchsorted(
        positive_scores, predictions[~positive_rows], side="left"
    )
    return violations


def assign_error_signatures(
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    core_models: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    correct = {
        name: (predictions[name] >= 0.5) == labels for name in core_models
    }
    correct_count = np.stack([correct[name] for name in core_models]).sum(axis=0)
    signatures = np.full(len(labels), "mixed_core", dtype=object)
    signatures[correct_count == 0] = "all_core_wrong"
    signatures[correct_count == len(core_models)] = "all_core_correct"

    phact = correct["original_phact"]
    sequence = correct["seq_only"]
    conservation = correct["conservation_phact"]
    rinalmo = correct["rinalmo"] | correct["rinalmo_cross"]
    selectable = (correct_count > 0) & (correct_count < len(core_models))
    rules = (
        (rinalmo & ~phact & ~sequence, "rinalmo_rescue"),
        (sequence & ~phact, "sequence_rescue"),
        (phact & ~sequence, "phact_rescue"),
        (conservation & ~phact, "conservation_rescue"),
        (~conservation & phact, "conservation_hurts"),
    )
    for condition, name in rules:
        assign = selectable & (signatures == "mixed_core") & condition
        signatures[assign] = name
    return signatures, correct_count.astype(np.int8)


def evidence_flags(frame: pd.DataFrame) -> np.ndarray:
    flags = np.full(len(frame), "mixed_or_weak_evidence", dtype=object)
    exact_total = frame["gse_exact_pair_neg"] + frame["gse_exact_pair_pos"]
    gse_support = np.where(
        frame["label"].to_numpy() == 1,
        frame["gse_exact_pair_pos"],
        frame["gse_exact_pair_neg"],
    )
    gse_conflict = np.where(
        frame["label"].to_numpy() == 1,
        frame["gse_exact_pair_neg"],
        frame["gse_exact_pair_pos"],
    )
    flags[(exact_total > 0) & (gse_support > 0) & (gse_conflict == 0)] = (
        "gse_exact_support_diagnostic_only"
    )
    flags[(exact_total > 0) & (gse_conflict > 0) & (gse_support == 0)] = (
        "gse_exact_conflict_diagnostic_only"
    )
    flags[(exact_total > 0) & (gse_conflict > 0) & (gse_support > 0)] = (
        "gse_internally_mixed"
    )

    target_seed_total = frame["train_target_seed_neg"] + frame["train_target_seed_pos"]
    target_seed_positive_rate = np.divide(
        frame["train_target_seed_pos"],
        target_seed_total,
        out=np.zeros(len(frame), dtype=np.float64),
        where=target_seed_total > 0,
    )
    no_exact_gse = exact_total == 0
    flags[
        no_exact_gse
        & (frame["label"] == 0)
        & (target_seed_total >= 3)
        & (target_seed_positive_rate >= 0.8)
    ] = "sampled_negative_in_positive_train_neighborhood"
    flags[
        no_exact_gse
        & (frame["label"] == 1)
        & (target_seed_total >= 3)
        & (target_seed_positive_rate <= 0.2)
    ] = "positive_in_negative_train_neighborhood"
    flags[
        no_exact_gse
        & ((frame["train_target_neg"] + frame["train_target_pos"]) == 0)
    ] = "target_window_unseen_in_train"
    return flags


def model_metrics(
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    records = []
    for name, probability in predictions.items():
        predicted = probability >= 0.5
        records.append(
            {
                "model": name,
                "auprc": average_precision_score(labels, probability),
                "threshold_errors": int((predicted != labels).sum()),
                "false_positives": int(((labels == 0) & predicted).sum()),
                "false_negatives": int(((labels == 1) & ~predicted).sum()),
                "hard_errors_0.9": int(
                    (((labels == 0) & (probability >= 0.9))
                    | ((labels == 1) & (probability <= 0.1))).sum()
                ),
            }
        )
    return pd.DataFrame(records).sort_values("auprc", ascending=False)


def subgroup_metrics(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    model_names: list[str],
    minimum_rows: int,
) -> pd.DataFrame:
    records = []
    group_columns = (
        "analysis_signature",
        "evidence_flag",
        "feature",
        "dominant_region",
        "analysis_family",
        "train_target_seen",
        "train_target_seed_seen",
        "gse_exact_pair_seen",
    )
    labels = frame["label"].to_numpy()
    for group_column in group_columns:
        for group_value, indexes in frame.groupby(group_column, sort=True).indices.items():
            indexes = np.asarray(indexes)
            group_labels = labels[indexes]
            positives = int(group_labels.sum())
            negatives = int(len(indexes) - positives)
            if len(indexes) < minimum_rows or min(positives, negatives) < 10:
                continue
            for model_name in model_names:
                probability = predictions[model_name][indexes]
                records.append(
                    {
                        "group_column": group_column,
                        "group_value": group_value,
                        "model": model_name,
                        "rows": len(indexes),
                        "positives": positives,
                        "positive_rate": positives / len(indexes),
                        "auprc": average_precision_score(group_labels, probability),
                        "threshold_error_rate": float(
                            ((probability >= 0.5) != group_labels).mean()
                        ),
                    }
                )
    return pd.DataFrame(records)


def write_gse_pair_evidence(
    path: Path,
    frame: pd.DataFrame,
    pair_counts: dict[object, list[int]],
    pair_sources: dict[object, Counter[str]],
) -> None:
    pair_to_index = {
        key: index for index, key in enumerate(frame_keys(frame, "exact_pair"))
    }
    fields = (
        "test_row_id",
        "test_label",
        "gse_label_0_rows",
        "gse_label_1_rows",
        "gse_sources",
        "gene",
        "noncodingRNA",
        "noncodingRNA_name",
        "analysis_family",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for pair, counts in sorted(pair_counts.items(), key=lambda item: pair_to_index[item[0]]):
            index = pair_to_index[pair]
            writer.writerow(
                {
                    "test_row_id": int(frame.iloc[index]["test_row_id"]),
                    "test_label": int(frame.iloc[index]["label"]),
                    "gse_label_0_rows": counts[0],
                    "gse_label_1_rows": counts[1],
                    "gse_sources": ";".join(
                        f"{name}:{count}"
                        for name, count in sorted(pair_sources[pair].items())
                    ),
                    "gene": pair[0],
                    "noncodingRNA": pair[1],
                    "noncodingRNA_name": frame.iloc[index]["noncodingRNA_name"],
                    "analysis_family": frame.iloc[index]["analysis_family"],
                }
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_test_rows(args.test)
    labels = frame["label"].to_numpy()
    print(f"test_rows={len(frame):,}", flush=True)

    predictions = {
        name: load_npz_predictions(path, labels) for name, path in args.prediction
    }
    predictions.update(
        {
            name: load_csv_predictions(path, len(frame), args.csv_id_prefix)
            for name, path in args.csv_prediction
        }
    )
    duplicate_names = len(predictions) != len(args.prediction) + len(args.csv_prediction)
    if duplicate_names:
        raise ValueError("Prediction model names must be unique")
    missing_core = set(args.core_model) - predictions.keys()
    if missing_core:
        raise ValueError(f"Missing core predictions: {sorted(missing_core)}")
    missing_signature = set(SIGNATURE_MODELS) - predictions.keys()
    if missing_signature:
        raise ValueError(f"Missing signature predictions: {sorted(missing_signature)}")
    if args.stacker_model not in predictions:
        raise ValueError(f"Missing stacker model: {args.stacker_model}")

    test_key_sets = {name: set(frame_keys(frame, name)) for name in KEY_NAMES}
    training_counts, training_rows = scan_training_evidence(args.train, test_key_sets)
    add_training_counts(frame, training_counts)

    pair_counts, target_seed_counts, pair_sources, gse_rows = scan_gse_evidence(
        args.gse,
        test_key_sets["exact_pair"],
        test_key_sets["target_seed"],
    )
    add_gse_counts(frame, pair_counts, target_seed_counts)

    for name, probability in predictions.items():
        frame[f"pred_{name}"] = probability
    signatures, core_correct_count = assign_error_signatures(
        labels, predictions, args.core_model
    )
    frame["analysis_signature"] = signatures
    frame["core_correct_count"] = core_correct_count
    stacker = predictions[args.stacker_model]
    stacker_predicted = stacker >= 0.5
    frame["stacker_error_type"] = np.where(
        stacker_predicted == labels,
        "correct",
        np.where(labels == 0, "false_positive", "false_negative"),
    )
    frame["stacker_wrong_label_probability"] = np.where(
        labels == 0, stacker, 1.0 - stacker
    )
    violations = rank_violation_counts(labels, stacker)
    opposite_rows = np.where(labels == 1, int((labels == 0).sum()), int((labels == 1).sum()))
    frame["stacker_opposite_rank_violations"] = violations
    frame["stacker_opposite_rank_violation_fraction"] = violations / opposite_rows
    frame["train_target_seen"] = (
        frame["train_target_neg"] + frame["train_target_pos"] > 0
    )
    frame["train_target_seed_seen"] = (
        frame["train_target_seed_neg"] + frame["train_target_seed_pos"] > 0
    )
    frame["gse_exact_pair_seen"] = (
        frame["gse_exact_pair_neg"] + frame["gse_exact_pair_pos"] > 0
    )
    frame["evidence_flag"] = evidence_flags(frame)

    model_table = model_metrics(labels, predictions)
    model_table.to_csv(args.output_dir / "model_metrics.tsv", sep="\t", index=False)
    signature_table = (
        frame.groupby("analysis_signature", as_index=False)
        .agg(
            rows=("label", "size"),
            positives=("label", "sum"),
            stacker_errors=("stacker_error_type", lambda values: (values != "correct").sum()),
            mean_rank_violation_fraction=(
                "stacker_opposite_rank_violation_fraction",
                "mean",
            ),
        )
        .sort_values("rows", ascending=False)
    )
    signature_table["stacker_error_rate"] = (
        signature_table["stacker_errors"] / signature_table["rows"]
    )
    signature_table.to_csv(
        args.output_dir / "error_signature_summary.tsv", sep="\t", index=False
    )
    subgroup_metrics(
        frame,
        predictions,
        [args.stacker_model, *args.core_model],
        args.minimum_subgroup_rows,
    ).to_csv(args.output_dir / "subgroup_metrics.tsv", sep="\t", index=False)

    write_gse_pair_evidence(
        args.output_dir / "gse_exact_test_pair_evidence.tsv",
        frame,
        pair_counts,
        pair_sources,
    )
    frame.to_csv(
        args.output_dir / "test_failure_evidence.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    priority = frame[
        (frame["stacker_error_type"] != "correct")
        & (frame["core_correct_count"] <= 1)
    ].sort_values(
        ["stacker_opposite_rank_violation_fraction", "stacker_wrong_label_probability"],
        ascending=False,
    )
    top_priority = pd.concat(
        [
            group.head(args.top_errors_per_type)
            for _, group in priority.groupby("stacker_error_type", sort=True)
        ]
    ).sort_values(
        "stacker_opposite_rank_violation_fraction", ascending=False
    )
    top_priority.to_csv(
        args.output_dir / "top_consensus_errors.tsv", sep="\t", index=False
    )

    gse_support = int(
        np.where(
            labels == 1,
            frame["gse_exact_pair_pos"],
            frame["gse_exact_pair_neg"],
        ).astype(bool).sum()
    )
    gse_conflict = int(
        np.where(
            labels == 1,
            frame["gse_exact_pair_neg"],
            frame["gse_exact_pair_pos"],
        ).astype(bool).sum()
    )
    summary = {
        "scope": (
            f"Manakov {args.split_name} error analysis. Model-selection claims must "
            "be based on validation; test and leftout analyses are monitoring-only."
        ),
        "split_name": args.split_name,
        "test": str(args.test),
        "test_sha256": sha256(args.test),
        "train": str(args.train),
        "train_sha256": sha256(args.train),
        "test_rows": len(frame),
        "training_rows_scanned": training_rows,
        "gse_rows_scanned": gse_rows,
        "models": {
            row["model"]: {
                key: (int(value) if key != "auprc" else float(value))
                for key, value in row.items()
                if key != "model"
            }
            for row in model_table.to_dict("records")
        },
        "counts": {
            "exact_test_pairs_in_train": int(
                ((frame["train_exact_pair_neg"] + frame["train_exact_pair_pos"]) > 0).sum()
            ),
            "test_pairs_with_exact_gse_evidence": len(pair_counts),
            "exact_gse_evidence_supports_test_label": gse_support,
            "exact_gse_evidence_conflicts_with_test_label": gse_conflict,
            "stacker_threshold_errors": int((stacker_predicted != labels).sum()),
            "stacker_hard_errors_0.9": int(
                (((labels == 0) & (stacker >= 0.9))
                | ((labels == 1) & (stacker <= 0.1))).sum()
            ),
            "all_core_wrong": int((core_correct_count == 0).sum()),
            "all_core_wrong_and_stacker_wrong": int(
                ((core_correct_count == 0) & (stacker_predicted != labels)).sum()
            ),
            "priority_consensus_errors": len(priority),
        },
        "interpretation": (
            "Model disagreement and development-test performance are diagnostic. "
            "A training edit still requires independent evidence. Exact GSE matches "
            "to development-test pairs are flagged diagnostic-only and must not be "
            "copied into training."
        ),
        "outputs": {
            "full_row_table": str(args.output_dir / "test_failure_evidence.tsv.gz"),
            "top_errors": str(args.output_dir / "top_consensus_errors.tsv"),
            "model_metrics": str(args.output_dir / "model_metrics.tsv"),
            "signature_summary": str(args.output_dir / "error_signature_summary.tsv"),
            "subgroup_metrics": str(args.output_dir / "subgroup_metrics.tsv"),
            "gse_exact_pair_evidence": str(
                args.output_dir / "gse_exact_test_pair_evidence.tsv"
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument(
        "--gse", type=parse_named_path, action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument(
        "--prediction",
        type=parse_named_path,
        action="append",
        default=[],
        metavar="NAME=NPZ",
    )
    parser.add_argument(
        "--csv-prediction",
        type=parse_named_path,
        action="append",
        default=[],
        metavar="NAME=CSV",
    )
    parser.add_argument("--core-model", action="append", default=[])
    parser.add_argument("--stacker-model", default="stacker")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--csv-id-prefix", default="test")
    parser.add_argument("--minimum-subgroup-rows", type=int, default=500)
    parser.add_argument("--top-errors-per-type", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.prediction:
        parser.error("at least one --prediction is required")
    if not args.core_model:
        parser.error("at least one --core-model is required")
    if args.minimum_subgroup_rows < 1 or args.top_errors_per_type < 1:
        parser.error("row-count limits must be positive")
    return args


if __name__ == "__main__":
    main()
