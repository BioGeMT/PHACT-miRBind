#!/usr/bin/env python3
"""Extract independently supported replacement rows in original cache order."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_conflict_cleaned_phact_caches import read_replacements


def main() -> None:
    args = parse_args()
    replacements = read_replacements(args.conflict_evidence)
    written: set[int] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input_rows.open(newline="") as source, args.output.open(
        "w", newline=""
    ) as destination:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames is None or "manakov_row_id" not in reader.fieldnames:
            raise ValueError("Input rows must contain manakov_row_id")
        writer = csv.DictWriter(
            destination,
            fieldnames=reader.fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in reader:
            row_id = int(row["manakov_row_id"])
            if row_id not in replacements:
                continue
            row["label"] = str(replacements[row_id])
            writer.writerow(row)
            written.add(row_id)

    if written != replacements.keys():
        missing = sorted(replacements.keys() - written)
        raise ValueError(f"Missing replacement rows: {missing}")
    print(f"rows={len(written)} output={args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-rows", type=Path, required=True)
    parser.add_argument("--conflict-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
