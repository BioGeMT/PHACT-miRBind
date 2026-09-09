#!/usr/bin/env python3
"""Rebuild conventional tracks on exactly the P1 cache rows, checking every pair."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workspace import historical_path

import csv
import json

import numpy as np
import torch

from run_condition import PHACT_CACHE, ROOT, EXPECTED_ROWS, sha256
from phact_mirbind.cache.manifest import find_manifest
from phact_mirbind.data.conservation import ConservationSchema, conservation_matrix


def encoded_pairs(guide_sequences, target_sequences):
    lookup = np.full(256, 4, dtype=np.uint8)
    for index, base in enumerate(b"ATCG"):
        lookup[base] = index
    axes = []
    for sequences, length in ((guide_sequences, 28), (target_sequences, 50)):
        data = b"".join(sequence.upper().replace("U", "T")[:length].ljust(length, "N").encode("ascii")
                        for sequence in sequences)
        axes.append(lookup[np.frombuffer(data, dtype=np.uint8).reshape(-1, length)])
    guide, target = axes
    return np.where((guide[:, :, None] < 4) & (target[:, None, :] < 4),
                    4 * guide[:, :, None] + target[:, None, :], 17).astype(np.uint8)


def prepare(split):
    torch.set_num_threads(2)
    source_manifest = find_manifest(PHACT_CACHE / split)
    source = json.loads(source_manifest.read_text())
    output = ROOT / "conservation_cache" / split
    output.mkdir(parents=True, exist_ok=False)
    features = ("phylop", "phastcons")
    manifest = dict(cache_version=1, cache_type="conservation", source_file=source["source_file"],
                    target_length=50, mirna_length=28, conservation_features=list(features),
                    row_count=EXPECTED_ROWS[split], shards=[],
                    source_manifest_sha256=sha256(source_manifest),
                    conservation_normalization={"phylop": "clamp(score / 10, -1, 1)",
                                                "phastcons": "raw; missing fill 0.5"})
    with historical_path(source["source_file"]).open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        schema = ConservationSchema.from_header(header, features)
        for entry in source["shards"]:
            rows = entry["rows"]
            shard_path = source_manifest.parent / entry["file"]
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            tracks = np.empty((rows, 50, 2), dtype=np.float16)
            labels = np.empty(rows, dtype=np.float32)
            guides, targets = [], []
            for index in range(rows):
                row = next(reader)
                tracks[index] = conservation_matrix(row, schema.feature_indices, features, 50)
                labels[index] = float(row[schema.label_idx])
                guides.append(row[schema.mirna_idx])
                targets.append(row[schema.gene_idx])
            if not np.array_equal(labels, shard["labels"].numpy()):
                raise ValueError(f"Label/order mismatch: {shard_path}")
            if not np.array_equal(encoded_pairs(guides, targets), shard["pair_indices"].numpy()):
                raise ValueError(f"Sequence/order mismatch: {shard_path}")
            if not np.isfinite(tracks).all():
                raise ValueError(f"Non-finite conservation: {shard_path}")
            destination = output / entry["file"]
            torch.save({"cache_version": 1, "pair_indices": shard["pair_indices"],
                        "labels": shard["labels"], "conservation": torch.from_numpy(tracks)}, destination)
            manifest["shards"].append(dict(file=entry["file"], rows=rows,
                                           sha256=sha256(destination),
                                           source_sha256=sha256(shard_path)))
            print(f"{split}: checked and prepared {entry['file']} ({rows} rows)", flush=True)
        if next(reader, None) is not None:
            raise ValueError(f"Unconsumed source rows: {split}")
    manifest["all_labels_and_pair_grids_match_source_tsv"] = True
    (output / f"{split}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(prepare, EXPECTED_ROWS))
