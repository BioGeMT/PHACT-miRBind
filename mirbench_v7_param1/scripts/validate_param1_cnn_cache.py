#!/usr/bin/env python3
"""Validate parameter-1 CNN caches and unchanged inherited inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


SPLITS = ("train", "val", "test", "leftout")
UNCHANGED_KEYS = (
    "pair_indices",
    "target_phact",
    "mirna_phact_missing",
    "target_phact_missing",
    "labels",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def update_tensor_hash(hasher: hashlib._Hash, tensor: torch.Tensor) -> None:
    contiguous = tensor.detach().cpu().contiguous()
    hasher.update(str(contiguous.dtype).encode())
    hasher.update(str(tuple(contiguous.shape)).encode())
    hasher.update(contiguous.numpy().tobytes())


def validate_split(cache_root: Path, reference_root: Path, split: str) -> dict[str, object]:
    split_dir = cache_root / split
    reference_dir = reference_root / split
    manifest_path = split_dir / f"{split}_manifest.json"
    reference_manifest_path = reference_dir / f"{split}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    reference_manifest = json.loads(reference_manifest_path.read_text())

    expected_score_order = [
        "mirna_param_1_A",
        "mirna_param_1_C",
        "mirna_param_1_G",
        "mirna_param_1_T",
        "target_target_score_A",
        "target_target_score_C",
        "target_target_score_G",
        "target_target_score_T",
    ]
    if manifest["cache_type"] != "phact":
        raise ValueError(f"Unexpected cache type for {split}: {manifest['cache_type']}")
    if manifest["phact_models"] != ["param_1"]:
        raise ValueError(f"Unexpected miRNA models for {split}: {manifest['phact_models']}")
    if manifest["target_phact_models"] != ["target_score"]:
        raise ValueError(f"Unexpected target models for {split}: {manifest['target_phact_models']}")
    if manifest["phact_score_channel_order"] != expected_score_order:
        raise ValueError(f"Unexpected score order for {split}: {manifest['phact_score_channel_order']}")
    if manifest["row_count"] != reference_manifest["row_count"]:
        raise ValueError(f"Row-count mismatch for {split}")
    if manifest["positive_rows"] != reference_manifest["positive_rows"]:
        raise ValueError(f"Positive-count mismatch for {split}")

    shards = sorted(split_dir.glob(f"{split}_shard_*.pt"))
    reference_shards = sorted(reference_dir.glob(f"{split}_shard_*.pt"))
    if len(shards) != len(reference_shards) or len(shards) != len(manifest["shards"]):
        raise ValueError(f"Shard-count mismatch for {split}")

    hashers = {key: hashlib.sha256() for key in UNCHANGED_KEYS}
    observed_rows = 0
    mirna_min = float("inf")
    mirna_max = -float("inf")
    mirna_finite = True
    for shard_path, reference_path in zip(shards, reference_shards, strict=True):
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        for key in UNCHANGED_KEYS:
            if not torch.equal(shard[key], reference[key]):
                raise ValueError(f"Inherited tensor changed: split={split} shard={shard_path.name} key={key}")
            update_tensor_hash(hashers[key], shard[key])
        mirna = shard["mirna_phact"]
        if mirna.ndim != 3 or tuple(mirna.shape[1:]) != (28, 4):
            raise ValueError(f"Bad miRNA tensor shape in {shard_path}: {tuple(mirna.shape)}")
        mirna_finite = mirna_finite and bool(torch.isfinite(mirna).all())
        mirna_min = min(mirna_min, float(mirna.min()))
        mirna_max = max(mirna_max, float(mirna.max()))
        observed_rows += int(mirna.shape[0])
        del shard, reference, mirna

    if observed_rows != manifest["row_count"]:
        raise ValueError(f"Observed rows differ from manifest for {split}")
    if not mirna_finite:
        raise ValueError(f"Non-finite miRNA PHACT tensor in {split}")
    return {
        "row_count": observed_rows,
        "positive_rows": int(manifest["positive_rows"]),
        "shard_count": len(shards),
        "mirna_phact_min_float16": mirna_min,
        "mirna_phact_max_float16": mirna_max,
        "inherited_tensor_sha256": {key: hasher.hexdigest() for key, hasher in hashers.items()},
    }


def main() -> None:
    args = parse_args()
    result = {
        "status": "passed",
        "cache_root": str(args.cache_root),
        "reference_root": str(args.reference_root),
        "splits": {
            split: validate_split(args.cache_root, args.reference_root, split)
            for split in SPLITS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
