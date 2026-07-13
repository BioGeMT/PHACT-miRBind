"""CLI for building pair-grid plus PHACT score caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phact_mirbind.cache.phact_cache import (
    DEFAULT_MIRNA_PHACT_FILE,
    DEFAULT_TARGET_PHACT_FILE,
    PHACT_REDUCTIONS,
    write_phact_cache_from_tsv,
)


def main() -> None:
    args = parse_args()
    manifest_path = write_phact_cache_from_tsv(
        args.input_file,
        args.output_dir,
        output_prefix=args.output_prefix,
        phact_split=args.phact_split,
        phact_models=args.phact_models,
        target_phact_models=args.target_phact_models,
        mirna_phact_file=args.mirna_phact_file,
        target_phact_file=args.target_phact_file,
        row_id_column=args.row_id_column,
        shard_size=args.shard_size,
        target_length=args.target_length,
        mirna_length=args.mirna_length,
        phact_dtype=args.phact_dtype,
        phact_reduction=args.phact_reduction,
    )
    manifest = json.loads(manifest_path.read_text())
    print(f"manifest={manifest_path}")
    print(f"rows={manifest['row_count']}")
    print(f"positive_rows={manifest['positive_rows']}")
    print(f"phact_models={','.join(manifest['phact_models'])}")
    print(f"target_phact_models={','.join(manifest['target_phact_models'])}")
    print(f"phact_reduction={manifest['phact_reduction']}")
    print(f"missing_fill={manifest['phact_missing']['fill_value']}")
    print(f"shards={len(manifest['shards'])}")
    print(f"elapsed_sec={manifest['elapsed_sec']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Manakov rows plus PHACT row-score TSVs to .pt shards"
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument(
        "--phact-split",
        choices=["train", "test", "leftout"],
        required=True,
        help="Split name used in the PHACT row-score TSVs.",
    )
    parser.add_argument(
        "--phact-models",
        "--phact-model",
        dest="phact_models",
        type=str,
        default="CountNodes_2",
        help="Comma-separated miRNA PHACT models, e.g. CountNodes_3,CountNodes_4.",
    )
    parser.add_argument(
        "--target-phact-models",
        type=str,
        default=None,
        help=(
            "Comma-separated target PHACT models. Defaults to --phact-models "
            "when the target TSV has named phact_<model>_<base> columns; otherwise "
            "defaults to target_score for the current score_A/C/G/T target TSV."
        ),
    )
    parser.add_argument("--mirna-phact-file", type=Path, default=DEFAULT_MIRNA_PHACT_FILE)
    parser.add_argument(
        "--target-phact-file",
        type=Path,
        default=DEFAULT_TARGET_PHACT_FILE,
    )
    parser.add_argument("--row-id-column", type=str, default="manakov_row_id")
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--target-length", type=int, default=50)
    parser.add_argument("--mirna-length", type=int, default=28)
    parser.add_argument("--phact-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--phact-reduction",
        choices=PHACT_REDUCTIONS,
        default="nucleotide",
        help=(
            "nucleotide stores A/C/G/T channels; actual_margin stores "
            "score_actual_nt - max(score_other_3_nts) per model; alt_mean "
            "stores mean(score_other_3_nts) per model."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
