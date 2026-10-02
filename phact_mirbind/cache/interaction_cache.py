"""Convert interaction-array tables to the compact representation used by trainers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from phact_mirbind.cache.manifest import CACHE_VERSION
from phact_mirbind.data.columns import BASES
from phact_mirbind.data.conservation import parse_conservation_features
from phact_mirbind.data.interaction_table import iter_interactions
from phact_mirbind.data.pair_encoding import encode_pair_indices


def source_key(path: Path, settings: dict) -> str:
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interaction_cache(
    path: str | Path, cache_root: str | Path, *, mode: str = "both",
    conservation_features: tuple[str, ...] = (), shard_size: int = 25_000,
) -> Path:
    """Validate and stream a table into reusable, content-addressed tensor shards."""
    path = Path(path).resolve()
    if mode not in {"mirna", "target", "both"}:
        raise ValueError(f"Unknown PHACT mode: {mode}")
    if conservation_features and mode == "mirna":
        raise ValueError("Target conservation features require mode target or both")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    settings = {"array_schema": 1, "mode": mode, "conservation_features": conservation_features,
                "shard_size": shard_size}
    key = source_key(path, settings)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / key
    manifest_path = destination / "arrays_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if all((destination / shard["file"]).is_file() for shard in manifest["shards"]):
            return manifest_path
        raise FileNotFoundError(f"Incomplete cached input: {destination}")
    if destination.exists():
        raise FileExistsError(f"Incomplete cached input: {destination}")

    required_axes = ("mirna", "target") if mode == "both" else (mode,)
    target_count = 4 + len(conservation_features)
    missing_count = 1 + len(conservation_features)
    mirna_names = [f"mirna_param_1_{base}" for base in BASES]
    target_names = [f"target_target_score_{base}" for base in BASES]
    target_names += [f"target_conservation_{feature}" for feature in conservation_features]
    target_missing_names = ["target_target_score_missing"] + [
        f"target_conservation_{feature}_missing" for feature in conservation_features
    ]
    with tempfile.TemporaryDirectory(prefix=".building-", dir=cache_root) as temporary:
        output = Path(temporary) / "cache"
        output.mkdir()
        arrays = {
            "pair_indices": np.empty((shard_size, 28, 50), dtype=np.uint8),
            "mirna_phact": np.empty((shard_size, 28, 4), dtype=np.float16),
            "target_phact": np.empty((shard_size, 50, target_count), dtype=np.float16),
            "mirna_phact_missing": np.empty((shard_size, 28, 1), dtype=np.uint8),
            "target_phact_missing": np.empty((shard_size, 50, missing_count), dtype=np.uint8),
            "labels": np.empty(shard_size, dtype=np.float32),
        }
        shards = []
        rows = positives = shard_rows = 0

        def flush() -> None:
            filename = f"arrays_{len(shards):05d}.pt"
            torch.save({name: torch.from_numpy(value[:shard_rows].copy())
                        for name, value in arrays.items()}, output / filename)
            shards.append({"file": filename, "rows": shard_rows})

        with (output / "ids.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["id"])
            for item in iter_interactions(path, required_axes=required_axes,
                                          conservation_features=conservation_features):
                i = shard_rows
                arrays["pair_indices"][i] = encode_pair_indices(
                    item.gene, item.mirna, target_length=50, mirna_length=28,
                )
                for axis, scores in (("mirna", item.mirna_phact), ("target", item.target_phact)):
                    if (np.abs(scores) > np.finfo(np.float16).max).any():
                        raise ValueError(f"{path}, id {item.id}: {axis} scores exceed the compact float16 range")
                    arrays[f"{axis}_phact"][i, :, :4] = np.nan_to_num(scores, nan=0.5)
                    arrays[f"{axis}_phact_missing"][i, :, 0] = np.isnan(scores).all(axis=1)
                for j, feature in enumerate(conservation_features, 4):
                    scores = item.conservation[feature]
                    missing = np.isnan(scores)
                    if feature == "phylop":
                        scores = np.clip(scores / 10., -1., 1.)
                    arrays["target_phact"][i, :, j] = np.nan_to_num(
                        scores, nan=0. if feature == "phylop" else 0.5,
                    )
                    arrays["target_phact_missing"][i, :, j - 3] = missing
                arrays["labels"][i] = item.label
                writer.writerow([item.id])
                rows += 1
                positives += int(item.label)
                shard_rows += 1
                if shard_rows == shard_size:
                    flush()
                    shard_rows = 0
                    print(f"Prepared {rows:,} input rows from {path.name}", flush=True)
            if shard_rows:
                flush()
        if not rows:
            raise ValueError(f"{path}: no interactions")
        manifest = {
            "cache_version": CACHE_VERSION, "cache_type": "phact",
            "source_file": str(path), "source_key": key, **settings,
            "target_length": 50, "mirna_length": 28, "phact_dtype": "float16",
            "phact_reduction": "nucleotide", "phact_model": "param_1",
            "phact_models": ["param_1"], "target_phact_models": ["target_score"],
            "phact_axis_channel_counts": {"mirna": 4, "target": target_count},
            "phact_axis_missingness_channel_counts": {"mirna": 1, "target": missing_count},
            "phact_axis_total_channel_counts": {"mirna": 5, "target": target_count + missing_count},
            "phact_score_channel_order": mirna_names + target_names,
            "phact_channel_order": mirna_names + ["mirna_param_1_missing"] + target_names + target_missing_names,
            "phact_missingness_channel_order": ["mirna_param_1_missing"] + target_missing_names,
            "phact_missing": {"fill_value": 0.5, "partial_missing": "error", "mask_value": 1},
            "row_count": rows, "positive_rows": positives, "shard_size": shard_size,
            "shards": shards, "ids_file": "ids.tsv",
        }
        (output / manifest_path.name).write_text(json.dumps(manifest, indent=2) + "\n")
        output.rename(destination)
    return manifest_path


def add_table_inputs(parser: argparse.ArgumentParser) -> None:
    for phase in ("train", "val", "test", "leftout"):
        group = parser.add_mutually_exclusive_group(required=phase in {"train", "val"})
        group.add_argument(f"--{phase}-cache", type=Path)
        group.add_argument(f"--{phase}-file", type=Path, help="CSV/TSV with JSON score arrays")
    parser.add_argument("--input-cache-dir", type=Path,
                        help="Reusable compact inputs; defaults to OUTPUT_DIR/input_cache")
    parser.add_argument("--conservation-features", default="",
                        help="Optional target tracks to include: phylop,phastcons (table inputs)")


def resolve_table_inputs(args: argparse.Namespace, *, mode: str) -> None:
    features = parse_conservation_features(args.conservation_features) if args.conservation_features else ()
    if len(features) != len(set(features)):
        raise ValueError("Duplicate conservation features")
    if features and any(getattr(args, f"{phase}_cache") for phase in ("train", "val", "test", "leftout")):
        raise ValueError("--conservation-features applies to table inputs; cached inputs already define their channels")
    root = args.input_cache_dir or args.output_dir / "input_cache"
    for phase in ("train", "val", "test", "leftout"):
        table = getattr(args, f"{phase}_file")
        if table is not None:
            setattr(args, f"{phase}_cache", interaction_cache(
                table, root, mode=mode, conservation_features=features,
            ))
