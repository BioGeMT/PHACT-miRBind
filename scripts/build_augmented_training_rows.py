#!/usr/bin/env python
"""Build leakage-safe Manakov and GSE training-row tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def normalize_sequence(value: str) -> str:
    return value.strip().upper().replace("U", "T")


def pair_key(row: dict[str, str]) -> str:
    return (
        normalize_sequence(row["gene"])
        + "\x1f"
        + normalize_sequence(row["noncodingRNA"])
    )


def family_key(row: dict[str, str]) -> str:
    return row["noncodingRNA_fam"].strip().casefold()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_reference(
    path: Path,
    *,
    collect_families: bool = False,
) -> tuple[dict[str, str], set[str], set[str], Counter[str]]:
    pairs: dict[str, str] = {}
    families: set[str] = set()
    mirnas: set[str] = set()
    stats: Counter[str] = Counter()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = pair_key(row)
            label = row["label"]
            if key in pairs and pairs[key] != label:
                stats["conflicting_pair_labels"] += 1
            pairs[key] = label
            stats["rows"] += 1
            if collect_families:
                families.add(family_key(row))
                mirnas.add(normalize_sequence(row["noncodingRNA"]))
    stats["unique_pairs"] = len(pairs)
    stats["unique_families"] = len(families)
    stats["unique_mirnas"] = len(mirnas)
    return pairs, families, mirnas, stats


def read_excluded_row_ids(path: Path | None) -> set[int]:
    if path is None:
        return set()
    excluded: set[int] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            try:
                excluded.add(int(value))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid Manakov row id at {path}:{line_number}: {value}"
                ) from exc
    return excluded


def write_clean_manakov_train(
    input_path: Path,
    output_path: Path,
    excluded_row_ids: set[int],
) -> tuple[dict[str, str], Counter[str]]:
    retained_pairs: dict[str, str] = {}
    stats: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(newline="") as source, output_path.open(
        "w", newline=""
    ) as destination:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames is None or "manakov_row_id" not in reader.fieldnames:
            raise ValueError(f"{input_path} must contain manakov_row_id")
        writer = csv.DictWriter(
            destination,
            fieldnames=reader.fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in reader:
            stats["input_rows"] += 1
            row_id = int(row["manakov_row_id"])
            if row_id in excluded_row_ids:
                stats["excluded_problematic_row_ids"] += 1
                continue
            key = pair_key(row)
            if key in retained_pairs:
                stats["duplicate_pairs"] += 1
                if retained_pairs[key] != row["label"]:
                    stats["conflicting_pair_labels"] += 1
                continue
            retained_pairs[key] = row["label"]
            writer.writerow(row)
            stats["output_rows"] += 1
            stats[f"output_label_{row['label']}"] += 1
    return retained_pairs, stats


def parse_gse_argument(value: str) -> tuple[str, Path]:
    source, separator, raw_path = value.partition("=")
    if not separator or not source or not raw_path:
        raise argparse.ArgumentTypeError("--gse must use SOURCE=PATH")
    return source, Path(raw_path)


def write_filtered_gse_rows(
    sources: list[tuple[str, Path]],
    output_path: Path,
    *,
    manakov_train_pairs: dict[str, str],
    val_pairs: dict[str, str],
    test_pairs: dict[str, str],
    leftout_pairs: dict[str, str],
    leftout_families: set[str],
    leftout_mirnas: set[str],
) -> tuple[Counter[str], dict[str, Counter[str]]]:
    accepted_pairs: dict[str, str] = {}
    totals: Counter[str] = Counter()
    source_stats: dict[str, Counter[str]] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: csv.DictWriter[str] | None = None
    destination = output_path.open("w", newline="")
    try:
        for source_name, source_path in sources:
            stats: Counter[str] = Counter()
            source_stats[source_name] = stats
            with source_path.open(newline="") as source:
                reader = csv.DictReader(source, delimiter="\t")
                if reader.fieldnames is None:
                    raise ValueError(f"Missing header: {source_path}")
                if writer is None:
                    output_fields = [
                        *reader.fieldnames,
                        "augmentation_source",
                        "augmentation_source_row_id",
                    ]
                    writer = csv.DictWriter(
                        destination,
                        fieldnames=output_fields,
                        delimiter="\t",
                        lineterminator="\n",
                    )
                    writer.writeheader()
                elif reader.fieldnames != writer.fieldnames[:-2]:
                    raise ValueError(
                        f"GSE header mismatch in {source_path}: {reader.fieldnames}"
                    )

                for source_row_id, row in enumerate(reader, start=1):
                    stats["input_rows"] += 1
                    key = pair_key(row)
                    mirna = normalize_sequence(row["noncodingRNA"])
                    reasons: list[str] = []
                    if family_key(row) in leftout_families:
                        reasons.append("leftout_family")
                    if mirna in leftout_mirnas:
                        reasons.append("leftout_mirna")
                    if key in test_pairs:
                        reasons.append("manakov_test_pair")
                    if key in leftout_pairs:
                        reasons.append("manakov_leftout_pair")
                    if key in val_pairs:
                        reasons.append("manakov_val_pair")
                    if key in manakov_train_pairs:
                        reasons.append("manakov_train_pair")
                        if manakov_train_pairs[key] != row["label"]:
                            reasons.append("manakov_train_label_conflict")
                    if key in accepted_pairs:
                        reasons.append("prior_gse_pair")
                        if accepted_pairs[key] != row["label"]:
                            reasons.append("prior_gse_label_conflict")

                    if reasons:
                        stats["dropped_rows"] += 1
                        for reason in reasons:
                            stats[f"reason_{reason}"] += 1
                        continue

                    accepted_pairs[key] = row["label"]
                    output_row = dict(row)
                    output_row["augmentation_source"] = source_name
                    output_row["augmentation_source_row_id"] = str(source_row_id)
                    assert writer is not None
                    writer.writerow(output_row)
                    stats["output_rows"] += 1
                    stats[f"output_label_{row['label']}"] += 1

            totals.update(stats)
    finally:
        destination.close()
    totals["unique_output_pairs"] = len(accepted_pairs)
    return totals, source_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manakov-train", type=Path, required=True)
    parser.add_argument("--manakov-val", type=Path, required=True)
    parser.add_argument("--manakov-test", type=Path, required=True)
    parser.add_argument("--manakov-leftout", type=Path, required=True)
    parser.add_argument(
        "--gse",
        type=parse_gse_argument,
        action="append",
        required=True,
        metavar="SOURCE=PATH",
    )
    parser.add_argument("--exclude-manakov-row-ids", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    excluded_row_ids = read_excluded_row_ids(args.exclude_manakov_row_ids)
    val_pairs, _, _, val_stats = read_reference(args.manakov_val)
    test_pairs, _, _, test_stats = read_reference(args.manakov_test)
    leftout_pairs, leftout_families, leftout_mirnas, leftout_stats = read_reference(
        args.manakov_leftout,
        collect_families=True,
    )
    clean_train_path = args.output_dir / "manakov_train_clean.tsv"
    filtered_gse_path = args.output_dir / "gse_augmentation_filtered.tsv"
    manakov_train_pairs, train_stats = write_clean_manakov_train(
        args.manakov_train,
        clean_train_path,
        excluded_row_ids,
    )
    gse_stats, gse_source_stats = write_filtered_gse_rows(
        args.gse,
        filtered_gse_path,
        manakov_train_pairs=manakov_train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        leftout_pairs=leftout_pairs,
        leftout_families=leftout_families,
        leftout_mirnas=leftout_mirnas,
    )

    summary = {
        "inputs": {
            "manakov_train": str(args.manakov_train),
            "manakov_val": str(args.manakov_val),
            "manakov_test": str(args.manakov_test),
            "manakov_leftout": str(args.manakov_leftout),
            "gse": {source: str(path) for source, path in args.gse},
        },
        "evaluation_sha256": {
            "test": sha256(args.manakov_test),
            "leftout": sha256(args.manakov_leftout),
        },
        "excluded_manakov_row_ids": len(excluded_row_ids),
        "reference_stats": {
            "val": dict(val_stats),
            "test": dict(test_stats),
            "leftout": dict(leftout_stats),
        },
        "manakov_train": dict(train_stats),
        "gse_total": dict(gse_stats),
        "gse_sources": {
            source: dict(stats) for source, stats in gse_source_stats.items()
        },
        "outputs": {
            "manakov_train": str(clean_train_path),
            "gse_augmentation": str(filtered_gse_path),
        },
    }
    summary_path = args.output_dir / "augmentation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"manakov_train={clean_train_path}")
    print(f"gse_augmentation={filtered_gse_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
