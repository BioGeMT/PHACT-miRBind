#!/usr/bin/env python3
"""Find sequence-similar training neighbors for prioritized Manakov test errors."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ANALYSIS_COLUMNS = (
    "test_row_id",
    "gene",
    "noncodingRNA",
    "noncodingRNA_name",
    "analysis_family",
    "label",
    "pred_stacker",
    "stacker_error_type",
    "analysis_signature",
)
NEIGHBOR_COLUMNS = (
    "test_row_id",
    "analysis_set",
    "control_match",
    "test_label",
    "stacker_error_type",
    "analysis_signature",
    "pred_stacker",
    "test_gene",
    "test_noncodingRNA",
    "test_noncodingRNA_name",
    "test_family",
    "neighbor_label",
    "neighbor_relation",
    "neighbor_rank_within_label",
    "shared_unique_target_kmers",
    "longest_shared_target_segment",
    "train_row_number",
    "manakov_row_id",
    "train_label",
    "train_gene",
    "train_noncodingRNA",
    "train_noncodingRNA_name",
    "train_family",
    "train_gene_cluster_ID",
    "augmentation_source",
    "augmentation_source_row_id",
)


def normalize_sequence(value: str) -> str:
    return str(value).strip().upper().replace("U", "T")


def unique_kmers(sequence: str, length: int) -> set[str]:
    sequence = normalize_sequence(sequence)
    return {
        sequence[index : index + length]
        for index in range(len(sequence) - length + 1)
    }


def longest_shared_segment(first: str, second: str, seed_length: int) -> int:
    first = normalize_sequence(first)
    second = normalize_sequence(second)
    longest = 0
    for first_start in range(len(first) - seed_length + 1):
        seed = first[first_start : first_start + seed_length]
        second_start = second.find(seed)
        while second_start >= 0:
            left = 0
            while (
                first_start - left - 1 >= 0
                and second_start - left - 1 >= 0
                and first[first_start - left - 1] == second[second_start - left - 1]
            ):
                left += 1
            right = seed_length
            while (
                first_start + right < len(first)
                and second_start + right < len(second)
                and first[first_start + right] == second[second_start + right]
            ):
                right += 1
            longest = max(longest, left + right)
            second_start = second.find(seed, second_start + 1)
    return longest


def build_kmer_index(
    errors: pd.DataFrame,
    kmer_length: int,
) -> dict[tuple[str, str], list[int]]:
    inverted: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in errors.iterrows():
        mirna = normalize_sequence(row["noncodingRNA"])
        for kmer in unique_kmers(row["gene"], kmer_length):
            inverted[(mirna, kmer)].append(index)
    return dict(inverted)


def family_label_matched_controls(
    errors: pd.DataFrame,
    control_pool: pd.DataFrame,
    random_seed: int,
) -> pd.DataFrame:
    available = control_pool[
        (control_pool["stacker_error_type"] == "correct")
        & ~control_pool["test_row_id"].isin(errors["test_row_id"])
    ]
    selected = []
    deficits: Counter[int] = Counter()
    for (label, family), count in errors.groupby(
        ["label", "analysis_family"], dropna=False
    ).size().items():
        candidates = available[
            (available["label"] == label)
            & (available["analysis_family"].fillna("NA") == (family if pd.notna(family) else "NA"))
        ]
        exact_count = min(count, len(candidates))
        if exact_count:
            exact = candidates.sample(n=exact_count, random_state=random_seed).copy()
            exact["control_match"] = "label_and_family"
            selected.append(exact)
        deficits[int(label)] += count - exact_count
    used_ids = {
        int(row_id)
        for selected_frame in selected
        for row_id in selected_frame["test_row_id"]
    }
    for label, count in deficits.items():
        if not count:
            continue
        fallback_candidates = available[
            (available["label"] == label)
            & ~available["test_row_id"].isin(used_ids)
        ]
        if len(fallback_candidates) < count:
            raise ValueError(
                f"Not enough correct label-matched controls for label={label}: "
                f"need {count}, found {len(fallback_candidates)}"
            )
        fallback = fallback_candidates.sample(n=count, random_state=random_seed).copy()
        fallback["control_match"] = "label_only_fallback"
        selected.append(fallback)
    return pd.concat(selected, ignore_index=True)


def scan_training_neighbors(
    train_path: Path,
    errors: pd.DataFrame,
    inverted: dict[tuple[str, str], list[int]],
    kmer_length: int,
    candidates_per_label: int,
) -> tuple[dict[tuple[int, int], list[tuple[int, int, dict[str, str]]]], int]:
    heaps: dict[tuple[int, int], list[tuple[int, int, dict[str, str]]]] = defaultdict(list)
    relevant_mirnas = set(errors["noncodingRNA"].map(normalize_sequence))
    rows_scanned = 0
    with train_path.open(newline="") as handle:
        for rows_scanned, row in enumerate(
            csv.DictReader(handle, delimiter="\t"), start=1
        ):
            mirna = normalize_sequence(row["noncodingRNA"])
            if mirna not in relevant_mirnas:
                continue
            matches: Counter[int] = Counter()
            for kmer in unique_kmers(row["gene"], kmer_length):
                for error_index in inverted.get((mirna, kmer), ()):
                    matches[error_index] += 1
            if not matches:
                continue
            label = int(row["label"])
            candidate = {
                "train_row_number": str(rows_scanned),
                "manakov_row_id": row.get("manakov_row_id", ""),
                "train_label": str(label),
                "train_gene": normalize_sequence(row["gene"]),
                "train_noncodingRNA": mirna,
                "train_noncodingRNA_name": row.get("noncodingRNA_name", ""),
                "train_family": row.get("mirgenedb_family")
                or row.get("noncodingRNA_fam", ""),
                "train_gene_cluster_ID": row.get("gene_cluster_ID", ""),
                "augmentation_source": row.get("augmentation_source", "") or "manakov",
                "augmentation_source_row_id": row.get("augmentation_source_row_id", ""),
            }
            for error_index, shared_kmers in matches.items():
                heap = heaps[(error_index, label)]
                item = (shared_kmers, rows_scanned, candidate)
                if len(heap) < candidates_per_label:
                    heapq.heappush(heap, item)
                elif item[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, item)
            if rows_scanned % 500_000 == 0:
                print(f"training_rows_scanned={rows_scanned:,}", flush=True)
    return heaps, rows_scanned


def write_outputs(
    output_dir: Path,
    errors: pd.DataFrame,
    heaps: dict[tuple[int, int], list[tuple[int, int, dict[str, str]]]],
    kmer_length: int,
    neighbors_per_label: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    neighbor_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for error_index, error in errors.iterrows():
        label_summaries: dict[int, tuple[int, int]] = {}
        for train_label in (0, 1):
            candidates = []
            for shared_kmers, _, candidate in heaps.get((error_index, train_label), ()):
                shared_segment = longest_shared_segment(
                    error["gene"], candidate["train_gene"], kmer_length
                )
                candidates.append((shared_segment, shared_kmers, candidate))
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            selected = candidates[:neighbors_per_label]
            label_summaries[train_label] = (
                max((item[0] for item in candidates), default=0),
                max((item[1] for item in candidates), default=0),
            )
            for rank, (shared_segment, shared_kmers, candidate) in enumerate(
                selected, start=1
            ):
                neighbor_records.append(
                    {
                        "test_row_id": int(error["test_row_id"]),
                        "analysis_set": error["analysis_set"],
                        "control_match": error.get("control_match", ""),
                        "test_label": int(error["label"]),
                        "stacker_error_type": error["stacker_error_type"],
                        "analysis_signature": error["analysis_signature"],
                        "pred_stacker": float(error["pred_stacker"]),
                        "test_gene": error["gene"],
                        "test_noncodingRNA": error["noncodingRNA"],
                        "test_noncodingRNA_name": error["noncodingRNA_name"],
                        "test_family": error["analysis_family"],
                        "neighbor_label": train_label,
                        "neighbor_relation": (
                            "same_label" if train_label == int(error["label"])
                            else "opposite_label"
                        ),
                        "neighbor_rank_within_label": rank,
                        "shared_unique_target_kmers": shared_kmers,
                        "longest_shared_target_segment": shared_segment,
                        **candidate,
                    }
                )
        same_label = int(error["label"])
        opposite_label = 1 - same_label
        summary_records.append(
            {
                "test_row_id": int(error["test_row_id"]),
                "analysis_set": error["analysis_set"],
                "control_match": error.get("control_match", ""),
                "test_label": same_label,
                "stacker_error_type": error["stacker_error_type"],
                "analysis_signature": error["analysis_signature"],
                "pred_stacker": float(error["pred_stacker"]),
                "same_label_max_shared_segment": label_summaries[same_label][0],
                "opposite_label_max_shared_segment": label_summaries[opposite_label][0],
                "same_label_max_shared_kmers": label_summaries[same_label][1],
                "opposite_label_max_shared_kmers": label_summaries[opposite_label][1],
            }
        )

    neighbors = pd.DataFrame(neighbor_records, columns=NEIGHBOR_COLUMNS)
    neighbor_summary = pd.DataFrame(summary_records)
    neighbors.to_csv(output_dir / "nearest_training_neighbors.tsv", sep="\t", index=False)
    neighbor_summary.to_csv(
        output_dir / "training_neighbor_summary_per_error.tsv", sep="\t", index=False
    )
    return neighbor_summary, neighbors


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    errors = pd.read_csv(
        args.errors,
        sep="\t",
        usecols=ANALYSIS_COLUMNS,
        dtype={"test_row_id": int, "label": int},
    )
    errors["analysis_set"] = "priority_error"
    analysis_rows = [errors]
    if args.control_pool is not None:
        control_pool = pd.read_csv(
            args.control_pool,
            sep="\t",
            usecols=ANALYSIS_COLUMNS,
            dtype={"test_row_id": int, "label": int},
        )
        controls = family_label_matched_controls(
            errors, control_pool, args.random_seed
        )
        controls["analysis_set"] = "matched_correct_control"
        analysis_rows.append(controls)
    examples = pd.concat(analysis_rows, ignore_index=True)
    examples["noncodingRNA"] = examples["noncodingRNA"].map(normalize_sequence)
    examples["gene"] = examples["gene"].map(normalize_sequence)
    inverted = build_kmer_index(examples, args.kmer_length)
    print(
        f"examples={len(examples):,} indexed_mirna_kmers={len(inverted):,}", flush=True
    )
    heaps, rows_scanned = scan_training_neighbors(
        args.train,
        examples,
        inverted,
        args.kmer_length,
        args.candidates_per_label,
    )
    neighbor_summary, neighbors = write_outputs(
        args.output_dir,
        examples,
        heaps,
        args.kmer_length,
        args.neighbors_per_label,
    )
    sampled_negative_review = neighbors[
        (neighbors["analysis_set"] == "priority_error")
        & (neighbors["test_label"] == 1)
        & (neighbors["neighbor_label"] == 0)
        & (neighbors["longest_shared_target_segment"] >= args.review_segment_threshold)
    ].sort_values(
        ["longest_shared_target_segment", "shared_unique_target_kmers"],
        ascending=False,
    ).drop_duplicates("test_row_id")
    sampled_negative_review.to_csv(
        args.output_dir / "candidate_sampled_negative_rows_for_review.tsv",
        sep="\t",
        index=False,
    )
    ambiguous_test_negative_review = neighbors[
        (neighbors["analysis_set"] == "priority_error")
        & (neighbors["test_label"] == 0)
        & (neighbors["neighbor_label"] == 1)
        & (neighbors["longest_shared_target_segment"] >= args.review_segment_threshold)
    ].sort_values(
        ["longest_shared_target_segment", "shared_unique_target_kmers"],
        ascending=False,
    ).drop_duplicates("test_row_id")
    ambiguous_test_negative_review.to_csv(
        args.output_dir / "ambiguous_test_negatives_for_review.tsv",
        sep="\t",
        index=False,
    )
    neighbor_group_comparison(neighbor_summary, args.kmer_length).to_csv(
        args.output_dir / "neighbor_group_comparison.tsv", sep="\t", index=False
    )

    counts = {
        analysis_set: neighbor_count_summary(group, args.kmer_length)
        for analysis_set, group in neighbor_summary.groupby("analysis_set")
    }
    summary = {
        "scope": (
            "Sequence-neighbor diagnostics for prioritized development-test errors. "
            "Neighbors are not automatic removal candidates."
        ),
        "errors": str(args.errors),
        "train": str(args.train),
        "analysis_rows": len(examples),
        "analysis_sets": {
            name: len(group) for name, group in examples.groupby("analysis_set")
        },
        "control_matches": (
            controls["control_match"].value_counts().to_dict()
            if args.control_pool is not None
            else {}
        ),
        "training_rows_scanned": rows_scanned,
        "kmer_length": args.kmer_length,
        "same_mirna_required": True,
        "candidate_neighbors_retained_per_label": args.candidates_per_label,
        "output_neighbors_per_label": args.neighbors_per_label,
        "output_neighbor_rows": len(neighbors),
        "review_segment_threshold": args.review_segment_threshold,
        "candidate_sampled_negative_rows_for_review": len(sampled_negative_review),
        "ambiguous_test_negatives_for_review": len(ambiguous_test_negative_review),
        "counts": counts,
        "interpretation": (
            "A similar opposite-label row can identify contradictory local evidence, "
            "but sampled negatives and cell-context differences must be reviewed before "
            "any row is down-weighted or removed."
        ),
    }
    (args.output_dir / "training_neighbor_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--control-pool", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kmer-length", type=int, default=12)
    parser.add_argument("--candidates-per-label", type=int, default=8)
    parser.add_argument("--neighbors-per-label", type=int, default=3)
    parser.add_argument("--review-segment-threshold", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    if args.kmer_length < 4:
        parser.error("--kmer-length must be at least 4")
    if args.candidates_per_label < args.neighbors_per_label:
        parser.error("--candidates-per-label must be at least --neighbors-per-label")
    if args.review_segment_threshold < args.kmer_length:
        parser.error("--review-segment-threshold must be at least --kmer-length")
    return args


def neighbor_count_summary(
    neighbor_summary: pd.DataFrame,
    kmer_length: int,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for relation in ("same_label", "opposite_label"):
        column = f"{relation}_max_shared_segment"
        for threshold in (kmer_length, 16, 20, 30):
            if threshold < kmer_length:
                continue
            counts[f"rows_with_{relation}_neighbor_segment_ge_{threshold}"] = int(
                (neighbor_summary[column] >= threshold).sum()
            )
    counts["opposite_neighbor_more_similar_than_same_label"] = int(
        (
            neighbor_summary["opposite_label_max_shared_segment"]
            > neighbor_summary["same_label_max_shared_segment"]
        ).sum()
    )
    counts["same_neighbor_more_similar_than_opposite_label"] = int(
        (
            neighbor_summary["same_label_max_shared_segment"]
            > neighbor_summary["opposite_label_max_shared_segment"]
        ).sum()
    )
    counts["equal_best_segment"] = int(
        (
            neighbor_summary["same_label_max_shared_segment"]
            == neighbor_summary["opposite_label_max_shared_segment"]
        ).sum()
    )
    return counts


def neighbor_group_comparison(
    neighbor_summary: pd.DataFrame,
    kmer_length: int,
) -> pd.DataFrame:
    records = []
    for (analysis_set, test_label), group in neighbor_summary.groupby(
        ["analysis_set", "test_label"]
    ):
        record: dict[str, object] = {
            "analysis_set": analysis_set,
            "test_label": int(test_label),
            "rows": len(group),
        }
        for relation in ("same_label", "opposite_label"):
            column = f"{relation}_max_shared_segment"
            for threshold in (kmer_length, 16, 20, 30):
                if threshold < kmer_length:
                    continue
                count = int((group[column] >= threshold).sum())
                record[f"{relation}_segment_ge_{threshold}_rows"] = count
                record[f"{relation}_segment_ge_{threshold}_rate"] = count / len(group)
        opposite_more = int(
            (
                group["opposite_label_max_shared_segment"]
                > group["same_label_max_shared_segment"]
            ).sum()
        )
        record["opposite_neighbor_more_similar_rows"] = opposite_more
        record["opposite_neighbor_more_similar_rate"] = opposite_more / len(group)
        records.append(record)
    return pd.DataFrame(records)


if __name__ == "__main__":
    main()
