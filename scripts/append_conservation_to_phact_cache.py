#!/usr/bin/env python3
"""Append normalized target conservation tracks to an aligned PHACT cache."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from phact_mirbind.cache.manifest import find_manifest
from phact_mirbind.cache.pair_cache import PairSchema
from phact_mirbind.data.conservation import (
    ConservationSchema,
    conservation_matrix,
    parse_conservation_features,
)
from phact_mirbind.data.pair_encoding import encode_pair_indices
from phact_mirbind.data.sequences import normalize_sequence


def main() -> None:
    args = parse_args()
    features = parse_conservation_features(args.features)
    manifest_path = find_manifest(args.input_cache)
    manifest = json.loads(manifest_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)

    output_shards: list[dict[str, object]] = []
    total_rows = 0
    total_positive = 0
    with args.input_rows.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        pair_schema = PairSchema.from_header(header)
        conservation_schema = ConservationSchema.from_header(header, features)

        for shard_index, shard_metadata in enumerate(manifest["shards"]):
            row_count = int(shard_metadata["rows"])
            shard = torch.load(
                manifest_path.parent / str(shard_metadata["file"]),
                map_location="cpu",
                weights_only=False,
            )
            conservation = np.empty(
                (row_count, int(manifest["target_length"]), len(features)),
                dtype=np.float16,
            )
            labels = np.empty(row_count, dtype=np.float32)
            boundary_rows: dict[int, list[str]] = {}
            for local_index in range(row_count):
                try:
                    row = next(reader)
                except StopIteration as exc:
                    raise ValueError(
                        f"Input TSV ended before cache row {total_rows + local_index}"
                    ) from exc
                conservation[local_index] = conservation_matrix(
                    row,
                    conservation_schema.feature_indices,
                    features,
                    int(manifest["target_length"]),
                ).astype(np.float16)
                labels[local_index] = float(row[pair_schema.label_idx])
                if local_index in {0, row_count - 1}:
                    boundary_rows[local_index] = row

            np.testing.assert_array_equal(labels, shard["labels"].cpu().numpy())
            for local_index, row in boundary_rows.items():
                expected_pair = encode_pair_indices(
                    normalize_sequence(
                        row[pair_schema.gene_idx], int(manifest["target_length"])
                    ),
                    normalize_sequence(
                        row[pair_schema.mirna_idx], int(manifest["mirna_length"])
                    ),
                    target_length=int(manifest["target_length"]),
                    mirna_length=int(manifest["mirna_length"]),
                )
                np.testing.assert_array_equal(
                    expected_pair,
                    shard["pair_indices"][local_index].cpu().numpy(),
                )

            output_shard = dict(shard)
            output_shard["target_phact"] = torch.cat(
                [
                    shard["target_phact"],
                    torch.from_numpy(conservation).to(shard["target_phact"].dtype),
                ],
                dim=2,
            )
            output_file = f"extended_shard_{shard_index:05d}.pt"
            output_path = args.output_dir / output_file
            torch.save(output_shard, output_path)
            positive_rows = int(output_shard["labels"].sum().item())
            output_shards.append(
                {
                    "file": output_file,
                    "rows": row_count,
                    "positive_rows": positive_rows,
                    "size_bytes": output_path.stat().st_size,
                }
            )
            total_rows += row_count
            total_positive += positive_rows
            print(f"shard={shard_index} rows={row_count}", flush=True)

        try:
            next(reader)
        except StopIteration:
            pass
        else:
            raise ValueError("Input TSV contains more rows than the PHACT cache")

    expected_rows = int(manifest["row_count"])
    if total_rows != expected_rows:
        raise ValueError(f"Wrote {total_rows} rows; expected {expected_rows}")

    output_manifest = deepcopy(manifest)
    counts = dict(output_manifest.get("phact_axis_channel_counts", {}))
    counts["target"] = int(counts.get("target", 0)) + len(features)
    total_counts = dict(output_manifest.get("phact_axis_total_channel_counts", {}))
    total_counts["target"] = int(total_counts.get("target", counts["target"])) + len(
        features
    )
    target_models = list(output_manifest.get("target_phact_models", []))
    target_models.extend(f"conservation_{feature}" for feature in features)
    score_order = list(output_manifest.get("phact_score_channel_order", []))
    added_channels = [f"target_conservation_{feature}" for feature in features]
    score_order.extend(added_channels)
    channel_order = list(output_manifest.get("phact_channel_order", []))
    target_missing_index = next(
        (
            index
            for index, name in enumerate(channel_order)
            if name.startswith("target_") and name.endswith("_missing")
        ),
        len(channel_order),
    )
    channel_order[target_missing_index:target_missing_index] = added_channels
    source_columns = dict(output_manifest.get("source_columns", {}))
    source_columns["target_conservation"] = dict(
        zip(features, conservation_schema.feature_columns)
    )
    output_manifest.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_file": str(args.input_rows),
            "source_columns": source_columns,
            "phact_axis_channel_counts": counts,
            "phact_axis_total_channel_counts": total_counts,
            "target_phact_models": target_models,
            "phact_score_channel_order": score_order,
            "phact_channel_order": channel_order,
            "row_count": total_rows,
            "positive_rows": total_positive,
            "shards": output_shards,
            "derived_from_manifest": str(manifest_path),
            "appended_target_conservation": {
                "features": list(features),
                "normalization": {
                    "phylop": "clamp(score / 10, -1, 1)",
                    "phastcons": "raw score, missing filled with 0.5",
                },
            },
        }
    )
    output_manifest_path = args.output_dir / "extended_manifest.json"
    output_manifest_path.write_text(json.dumps(output_manifest, indent=2) + "\n")
    print(f"manifest={output_manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--input-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", default="phylop,phastcons")
    return parser.parse_args()


if __name__ == "__main__":
    main()
