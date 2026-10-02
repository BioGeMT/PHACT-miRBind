"""CLI for building seq-only pair-grid caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phact_mirbind.cache.pair_cache import write_pair_cache_from_tsv


def main() -> None:
    args = parse_args()
    manifest_path = write_pair_cache_from_tsv(
        args.input_file,
        args.output_dir,
        output_prefix=args.output_prefix,
        shard_size=args.shard_size,
        target_length=args.target_length,
        mirna_length=args.mirna_length,
    )
    manifest = json.loads(manifest_path.read_text())
    print(f"manifest={manifest_path}")
    print(f"rows={manifest['row_count']}")
    print(f"positive_rows={manifest['positive_rows']}")
    print(f"shards={len(manifest['shards'])}")
    print(f"elapsed_sec={manifest['elapsed_sec']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Manakov TSV to compact pair-grid .pt shards"
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--target-length", type=int, default=50)
    parser.add_argument("--mirna-length", type=int, default=28)
    return parser.parse_args()


if __name__ == "__main__":
    main()
