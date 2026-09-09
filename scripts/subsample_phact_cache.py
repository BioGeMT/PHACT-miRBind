#!/usr/bin/env python3
"""Create a deterministic label-stratified subset of a cached PHACT dataset."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from phact_mirbind.cache.manifest import find_manifest


def main() -> None:
    args = parse_args()
    manifest_path = find_manifest(args.input_cache)
    manifest = json.loads(manifest_path.read_text())

    labels = []
    for shard_metadata in manifest["shards"]:
        shard = torch.load(
            manifest_path.parent / str(shard_metadata["file"]),
            map_location="cpu",
            weights_only=False,
        )
        labels.append(shard["labels"].cpu().numpy())
    all_labels = np.concatenate(labels).astype(np.int8, copy=False)

    rng = np.random.default_rng(args.seed)
    selected = np.zeros(len(all_labels), dtype=bool)
    selected_counts: dict[str, int] = {}
    requested_counts = {
        0: args.negative_rows if args.negative_rows is not None else args.rows_per_class,
        1: args.positive_rows if args.positive_rows is not None else args.rows_per_class,
    }
    for label in (0, 1):
        candidates = np.flatnonzero(all_labels == label)
        count = min(int(requested_counts[label]), len(candidates))
        if count == 0:
            selected_counts[str(label)] = 0
            continue
        chosen = rng.choice(candidates, size=count, replace=False)
        selected[chosen] = True
        selected_counts[str(label)] = int(count)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_shards: list[dict[str, object]] = []
    global_offset = 0
    for shard_index, shard_metadata in enumerate(manifest["shards"]):
        row_count = int(shard_metadata["rows"])
        keep = torch.from_numpy(selected[global_offset : global_offset + row_count])
        global_offset += row_count
        if not keep.any():
            continue
        shard = torch.load(
            manifest_path.parent / str(shard_metadata["file"]),
            map_location="cpu",
            weights_only=False,
        )
        subset = {
            key: value[keep]
            if torch.is_tensor(value) and value.ndim and value.shape[0] == row_count
            else value
            for key, value in shard.items()
        }
        output_file = f"subset_shard_{shard_index:05d}.pt"
        output_path = args.output_dir / output_file
        torch.save(subset, output_path)
        rows = int(keep.sum())
        positive_rows = int(subset["labels"].sum().item())
        output_shards.append(
            {
                "file": output_file,
                "rows": rows,
                "positive_rows": positive_rows,
                "size_bytes": output_path.stat().st_size,
            }
        )
        print(f"shard={shard_index} selected={rows}", flush=True)

    output_manifest = deepcopy(manifest)
    output_manifest.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "row_count": int(selected.sum()),
            "positive_rows": selected_counts["1"],
            "shards": output_shards,
            "derived_from_manifest": str(manifest_path),
            "subsample": {
                "policy": "uniform without replacement within each binary label",
                "seed": args.seed,
                "requested_rows_by_label": {
                    str(label): int(count)
                    for label, count in requested_counts.items()
                },
                "selected_rows_by_label": selected_counts,
            },
        }
    )
    output_manifest_path = args.output_dir / "subset_manifest.json"
    output_manifest_path.write_text(json.dumps(output_manifest, indent=2) + "\n")
    print(f"manifest={output_manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-class", type=int)
    parser.add_argument("--negative-rows", type=int)
    parser.add_argument("--positive-rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    has_balanced_count = args.rows_per_class is not None
    has_label_counts = (
        args.negative_rows is not None or args.positive_rows is not None
    )
    if has_balanced_count == has_label_counts:
        parser.error(
            "supply either --rows-per-class or both label-specific row counts"
        )
    if has_label_counts and (
        args.negative_rows is None or args.positive_rows is None
    ):
        parser.error("--negative-rows and --positive-rows must be supplied together")
    counts = (
        [args.rows_per_class]
        if has_balanced_count
        else [args.negative_rows, args.positive_rows]
    )
    if any(count is None or count < 0 for count in counts):
        parser.error("requested row counts must be non-negative")
    if has_balanced_count and args.rows_per_class < 1:
        parser.error("--rows-per-class must be positive")
    return args


if __name__ == "__main__":
    main()
