#!/usr/bin/env python3
"""Convert mature-coordinate parameter-1 scores to Agentomics profile schema."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    if temp.exists():
        temp.unlink()

    output_fields = [
        "mirgenedb_mature_id",
        "mirna_position_1based",
        "actual_nt",
        "phact_param_1_A",
        "phact_param_1_C",
        "phact_param_1_G",
        "phact_param_1_T",
    ]
    mature_ids: set[str] = set()
    rows_written = 0
    try:
        with args.input.open(newline="") as source, temp.open("w", newline="") as out:
            reader = csv.DictReader(source, delimiter="\t")
            writer = csv.DictWriter(
                out,
                delimiter="\t",
                fieldnames=output_fields,
                lineterminator="\n",
            )
            writer.writeheader()
            for row in reader:
                mature_id = f"{row['pre_mirna']}_{row['arm']}"
                writer.writerow(
                    {
                        "mirgenedb_mature_id": mature_id,
                        "mirna_position_1based": row["arm_position_1based"],
                        "actual_nt": row["actual_nt"],
                        **{
                            f"phact_param_1_{base}": row[f"score_{base}"]
                            for base in "ACGT"
                        },
                    }
                )
                mature_ids.add(mature_id)
                rows_written += 1
    except BaseException:
        if temp.exists():
            temp.unlink()
        raise
    os.replace(temp, args.output)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "score_columns": output_fields[3:],
        "mature_profiles": len(mature_ids),
        "rows": rows_written,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
