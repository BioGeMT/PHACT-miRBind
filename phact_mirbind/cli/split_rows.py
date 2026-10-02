"""CLI for original miRBind-style row splitting."""

from __future__ import annotations

import argparse
from pathlib import Path

from phact_mirbind.data.row_split import split_original_rows


def main() -> None:
    args = parse_args()
    summary = split_original_rows(
        args.input_file,
        args.output_dir,
        output_prefix=args.output_prefix,
        val_fraction=args.val_fraction,
        seed=args.seed,
        include_row_id=args.include_row_id,
        row_id_column=args.row_id_column,
    )
    print(f"train_file={summary['train_file']}")
    print(f"val_file={summary['val_file']}")
    print(f"summary={summary['summary_file']}")
    print(f"train_rows={summary['rows']['train']}")
    print(f"val_rows={summary['rows']['val']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-split a Manakov TSV like original miRBind"
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=str, default="manakov_original_rows")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-row-id", action="store_true")
    parser.add_argument("--row-id-column", type=str, default="manakov_row_id")
    return parser.parse_args()


if __name__ == "__main__":
    main()
