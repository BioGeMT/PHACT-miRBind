#!/usr/bin/env python3
"""Conservatively remove likely false-negative rows from a PHACT cache."""

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
    arrays = np.load(args.predictions)
    labels = arrays["labels"].astype(np.int8)
    predictions = arrays["predictions"].astype(np.float64)
    expected_rows = int(manifest["row_count"])
    if len(labels) != expected_rows or len(predictions) != expected_rows:
        raise ValueError(
            f"Prediction rows ({len(predictions)}) do not match cache ({expected_rows})"
        )
    drop = (labels == 0) & (predictions >= args.negative_threshold)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    output_shards: list[dict[str, object]] = []
    output_positive = 0
    global_offset = 0
    for shard_index, shard_metadata in enumerate(manifest["shards"]):
        row_count = int(shard_metadata["rows"])
        local_drop = torch.from_numpy(
            drop[global_offset : global_offset + row_count]
        )
        keep = ~local_drop
        shard = torch.load(
            manifest_path.parent / str(shard_metadata["file"]),
            map_location="cpu",
        )
        output_shard = {
            key: value[keep]
            if torch.is_tensor(value) and value.ndim and value.shape[0] == row_count
            else value
            for key, value in shard.items()
        }
        output_file = f"filtered_shard_{shard_index:05d}.pt"
        output_path = output_dir / output_file
        torch.save(output_shard, output_path)
        output_rows = int(keep.sum())
        positive_rows = int(output_shard["labels"].sum().item())
        output_positive += positive_rows
        output_shards.append(
            {
                "file": output_file,
                "rows": output_rows,
                "positive_rows": positive_rows,
                "size_bytes": output_path.stat().st_size,
            }
        )
        global_offset += row_count
        print(
            f"shard={shard_index} rows={row_count} dropped={int(local_drop.sum())}",
            flush=True,
        )

    output_manifest = deepcopy(manifest)
    output_manifest.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "row_count": int((~drop).sum()),
            "positive_rows": output_positive,
            "shards": output_shards,
            "derived_from_manifest": str(manifest_path),
            "prediction_filter": {
                "prediction_file": str(args.predictions),
                "policy": "drop sampled-negative rows at or above threshold",
                "negative_threshold": args.negative_threshold,
                "dropped_rows": int(drop.sum()),
                "dropped_positive_rows": int(labels[drop].sum()),
            },
        }
    )
    output_manifest_path = output_dir / "filtered_manifest.json"
    output_manifest_path.write_text(json.dumps(output_manifest, indent=2) + "\n")
    print(f"manifest={output_manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--negative-threshold", type=float, default=0.99)
    args = parser.parse_args()
    if not 0.5 < args.negative_threshold <= 1.0:
        parser.error("--negative-threshold must be in (0.5, 1.0]")
    return args


if __name__ == "__main__":
    main()
