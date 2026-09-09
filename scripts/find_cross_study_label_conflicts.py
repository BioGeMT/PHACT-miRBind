#!/usr/bin/env python3
"""Find Manakov training rows contradicted by consistent GSE labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def normalize_sequence(value: str) -> str:
    return value.strip().upper().replace("U", "T")


def pair_key(row: dict[str, str]) -> tuple[str, str]:
    return (
        normalize_sequence(row["gene"]),
        normalize_sequence(row["noncodingRNA"]),
    )


def parse_source(value: str) -> tuple[str, Path]:
    source, separator, raw_path = value.partition("=")
    if not separator or not source or not raw_path:
        raise argparse.ArgumentTypeError("--gse must use SOURCE=PATH")
    return source, Path(raw_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gse_labels: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    gse_sources: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for source_name, source_path in args.gse:
        with source_path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                key = pair_key(row)
                gse_labels[key][row["label"]] += 1
                gse_sources[key][source_name] += 1

    evidence_path = args.output_dir / "manakov_gse_label_conflicts.tsv"
    exclusion_path = args.output_dir / "recommended_manakov_exclude_row_ids.txt"
    summary = Counter()
    exclusions: list[int] = []
    with args.manakov_train.open(newline="") as source, evidence_path.open(
        "w", newline=""
    ) as destination:
        reader = csv.DictReader(source, delimiter="\t")
        fields = [
            "manakov_row_id",
            "manakov_label",
            "gse_label_0_rows",
            "gse_label_1_rows",
            "gse_sources",
            "recommend_exclude",
            "gene",
            "noncodingRNA",
            "noncodingRNA_name",
            "noncodingRNA_fam",
        ]
        writer = csv.DictWriter(
            destination,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in reader:
            summary["manakov_rows_scanned"] += 1
            key = pair_key(row)
            labels = gse_labels.get(key)
            if not labels:
                continue
            summary["manakov_pairs_seen_in_gse"] += 1
            manakov_label = row["label"]
            opposite_label = "0" if manakov_label == "1" else "1"
            if not labels[opposite_label]:
                summary["same_label_only"] += 1
                continue

            summary["rows_with_any_conflict"] += 1
            internally_consistent_opposite = (
                labels[opposite_label] > 0 and labels[manakov_label] == 0
            )
            positive_precedence = manakov_label == "0" and opposite_label == "1"
            recommend_exclude = internally_consistent_opposite and (
                args.conflict_policy == "all_consistent" or positive_precedence
            )
            if recommend_exclude:
                summary["recommended_exclusions"] += 1
                exclusions.append(int(row["manakov_row_id"]))
            elif internally_consistent_opposite and manakov_label == "1":
                summary["retained_manakov_positive_over_gse_negative"] += 1
            else:
                summary["gse_internally_conflicting"] += 1

            writer.writerow(
                {
                    "manakov_row_id": row["manakov_row_id"],
                    "manakov_label": manakov_label,
                    "gse_label_0_rows": labels["0"],
                    "gse_label_1_rows": labels["1"],
                    "gse_sources": ";".join(
                        f"{name}:{count}"
                        for name, count in sorted(gse_sources[key].items())
                    ),
                    "recommend_exclude": int(recommend_exclude),
                    "gene": key[0],
                    "noncodingRNA": key[1],
                    "noncodingRNA_name": row["noncodingRNA_name"],
                    "noncodingRNA_fam": row["noncodingRNA_fam"],
                }
            )

    exclusion_path.write_text("".join(f"{row_id}\n" for row_id in exclusions))
    report = {
        "manakov_train": str(args.manakov_train),
        "gse": {source: str(path) for source, path in args.gse},
        "conflict_policy": args.conflict_policy,
        "criteria": conflict_policy_description(args.conflict_policy),
        "counts": dict(summary),
        "evidence": str(evidence_path),
        "exclusions": str(exclusion_path),
    }
    summary_path = args.output_dir / "cross_study_conflict_summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manakov-train", type=Path, required=True)
    parser.add_argument(
        "--gse",
        type=parse_source,
        action="append",
        required=True,
        metavar="SOURCE=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conflict-policy",
        choices=("positive_precedence", "all_consistent"),
        default="positive_precedence",
        help=(
            "positive_precedence removes only Manakov negatives contradicted by "
            "GSE positives; all_consistent removes every consistently contradicted row."
        ),
    )
    return parser.parse_args()


def conflict_policy_description(policy: str) -> str:
    if policy == "positive_precedence":
        return (
            "Treat observed positives as stronger evidence than sampled negatives: "
            "exclude a Manakov row only when it is negative and every matching GSE "
            "observation is positive."
        )
    return (
        "Exclude whenever every GSE observation of the exact normalized "
        "target-miRNA pair has the opposite label to Manakov."
    )


if __name__ == "__main__":
    main()
