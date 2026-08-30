#!/usr/bin/env python3
"""Remove cross-study conflicts from a PHACT cache and emit replacements."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import torch

from phact_mirbind.cache.manifest import find_manifest


def main() -> None:
    args = parse_args()
    manifest_path = find_manifest(args.input_cache)
    manifest = json.loads(manifest_path.read_text())
    replacements = read_replacements(args.conflict_evidence)
    replacement_indices, scanned_rows = locate_replacements(args.input_rows, replacements)
    expected_rows = int(manifest["row_count"])
    if scanned_rows != expected_rows:
        raise ValueError(
            f"Input TSV has {scanned_rows} rows but cache manifest has {expected_rows}"
        )
    if len(replacement_indices) != len(replacements):
        missing = sorted(replacements.keys() - replacement_indices.keys())
        raise ValueError(f"Did not find {len(missing)} replacement row ids: {missing[:10]}")

    clean_dir = args.output_dir / "manakov_clean"
    replacement_dir = args.output_dir / "gse_conflict_replacements"
    clean_dir.mkdir(parents=True, exist_ok=False)
    replacement_dir.mkdir(parents=True, exist_ok=False)

    clean_shards: list[dict[str, object]] = []
    replacement_parts: dict[str, list[torch.Tensor]] = {}
    clean_positive = 0
    global_offset = 0
    for shard_index, shard_metadata in enumerate(manifest["shards"]):
        input_shard_path = manifest_path.parent / str(shard_metadata["file"])
        shard = torch.load(input_shard_path, map_location="cpu")
        row_count = int(shard_metadata["rows"])
        local_replacements = {
            global_index - global_offset: new_label
            for global_index, new_label in replacement_indices.items()
            if global_offset <= global_index < global_offset + row_count
        }
        keep = torch.ones(row_count, dtype=torch.bool)
        if local_replacements:
            local_indices = torch.tensor(sorted(local_replacements), dtype=torch.long)
            keep[local_indices] = False
            for key, value in shard.items():
                if torch.is_tensor(value) and value.ndim and value.shape[0] == row_count:
                    replacement_parts.setdefault(key, []).append(value[local_indices])
            new_labels = torch.tensor(
                [local_replacements[index] for index in sorted(local_replacements)],
                dtype=shard["labels"].dtype,
            )
            replacement_parts["labels"][-1] = new_labels

        clean_shard = {
            key: value[keep]
            if torch.is_tensor(value) and value.ndim and value.shape[0] == row_count
            else value
            for key, value in shard.items()
        }
        clean_file = f"clean_shard_{shard_index:05d}.pt"
        clean_path = clean_dir / clean_file
        torch.save(clean_shard, clean_path)
        clean_rows = int(keep.sum())
        shard_positive = int(clean_shard["labels"].sum().item())
        clean_positive += shard_positive
        clean_shards.append(
            {
                "file": clean_file,
                "rows": clean_rows,
                "positive_rows": shard_positive,
                "size_bytes": clean_path.stat().st_size,
            }
        )
        global_offset += row_count
        print(
            f"shard={shard_index} rows={row_count} removed={len(local_replacements)}",
            flush=True,
        )

    replacement_shard = {
        key: torch.cat(parts, dim=0) for key, parts in replacement_parts.items()
    }
    replacement_shard["cache_version"] = int(manifest["cache_version"])
    replacement_path = replacement_dir / "replacement_shard_00000.pt"
    torch.save(replacement_shard, replacement_path)
    replacement_count = int(replacement_shard["labels"].shape[0])
    replacement_positive = int(replacement_shard["labels"].sum().item())

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    clean_manifest = deepcopy(manifest)
    clean_manifest.update(
        {
            "created_utc": now,
            "source_file": str(args.clean_source_file or args.input_rows),
            "row_count": expected_rows - replacement_count,
            "positive_rows": clean_positive,
            "shards": clean_shards,
            "derived_from_manifest": str(manifest_path),
            "removed_cross_study_conflicts": replacement_count,
            "conflict_evidence": str(args.conflict_evidence),
        }
    )
    clean_manifest_path = clean_dir / "clean_manifest.json"
    clean_manifest_path.write_text(json.dumps(clean_manifest, indent=2) + "\n")

    replacement_manifest = deepcopy(manifest)
    replacement_manifest.update(
        {
            "created_utc": now,
            "source_file": str(args.conflict_evidence),
            "source_columns": {
                **manifest.get("source_columns", {}),
                "label": "consistent_opposite_gse_label",
            },
            "row_count": replacement_count,
            "positive_rows": replacement_positive,
            "shard_size": replacement_count,
            "shards": [
                {
                    "file": replacement_path.name,
                    "rows": replacement_count,
                    "positive_rows": replacement_positive,
                    "size_bytes": replacement_path.stat().st_size,
                }
            ],
            "derived_from_manifest": str(manifest_path),
            "replacement_for_cross_study_conflicts": replacement_count,
            "conflict_evidence": str(args.conflict_evidence),
        }
    )
    replacement_manifest_path = replacement_dir / "replacement_manifest.json"
    replacement_manifest_path.write_text(
        json.dumps(replacement_manifest, indent=2) + "\n"
    )
    print(f"clean_manifest={clean_manifest_path}")
    print(f"replacement_manifest={replacement_manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--input-rows", type=Path, required=True)
    parser.add_argument("--conflict-evidence", type=Path, required=True)
    parser.add_argument("--clean-source-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_replacements(path: Path) -> dict[int, int]:
    replacements: dict[int, int] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["recommend_exclude"] != "1":
                continue
            manakov_label = int(row["manakov_label"])
            opposite_label = 1 - manakov_label
            if int(row[f"gse_label_{opposite_label}_rows"]) < 1:
                raise ValueError(f"Missing opposite-label evidence for {row}")
            replacements[int(row["manakov_row_id"])] = opposite_label
    if not replacements:
        raise ValueError(f"No recommended replacements found in {path}")
    return replacements


def locate_replacements(
    input_rows: Path,
    replacements: dict[int, int],
) -> tuple[dict[int, int], int]:
    by_index: dict[int, int] = {}
    with input_rows.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "manakov_row_id" not in reader.fieldnames:
            raise ValueError(f"{input_rows} must contain manakov_row_id")
        row_count = 0
        for global_index, row in enumerate(reader):
            row_count += 1
            row_id = int(row["manakov_row_id"])
            if row_id not in replacements:
                continue
            new_label = replacements[row_id]
            if int(row["label"]) == new_label:
                raise ValueError(f"Replacement label equals Manakov label for row {row_id}")
            by_index[global_index] = new_label
    return by_index, row_count


if __name__ == "__main__":
    main()
