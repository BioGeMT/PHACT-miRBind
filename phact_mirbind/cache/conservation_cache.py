"""Compact .pt cache writer for pair grids plus target conservation tracks."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from phact_mirbind.cache.manifest import CACHE_VERSION, write_cache_log_header
from phact_mirbind.cache.pair_cache import PairSchema
from phact_mirbind.data.conservation import (
    CONSERVATION_FEATURES,
    ConservationSchema,
    conservation_matrix,
    parse_conservation_features,
)
from phact_mirbind.data.pair_encoding import encode_pair_indices
from phact_mirbind.data.sequences import normalize_sequence


def write_conservation_cache_from_tsv(
    input_file: str | Path,
    output_dir: str | Path,
    *,
    output_prefix: str,
    shard_size: int = 100_000,
    target_length: int = 50,
    mirna_length: int = 28,
    conservation_features: str | list[str] | tuple[str, ...] = CONSERVATION_FEATURES,
    conservation_dtype: str = "float16",
) -> Path:
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = parse_conservation_features(conservation_features)

    if conservation_dtype not in {"float16", "float32"}:
        raise ValueError("conservation_dtype must be float16 or float32")

    log_path = output_dir / f"{output_prefix}_cache_log.tsv"
    manifest_path = output_dir / f"{output_prefix}_manifest.json"
    shards: list[dict[str, object]] = []
    total_rows = 0
    total_positive = 0
    shard_idx = 0
    shard_started = time.perf_counter()
    started = time.perf_counter()

    pairs = np.empty((shard_size, mirna_length, target_length), dtype=np.uint8)
    conservation = np.empty(
        (shard_size, target_length, len(features)),
        dtype=conservation_dtype,
    )
    labels = np.empty((shard_size,), dtype=np.float32)
    write_cache_log_header(log_path)

    with input_file.open("r", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        pair_schema = PairSchema.from_header(header)
        conservation_schema = ConservationSchema.from_header(header, features)
        shard_rows = 0
        shard_positive = 0
        for row in reader:
            target_sequence = normalize_sequence(row[pair_schema.gene_idx], target_length)
            mirna_sequence = normalize_sequence(row[pair_schema.mirna_idx], mirna_length)
            pairs[shard_rows] = encode_pair_indices(
                target_sequence,
                mirna_sequence,
                target_length=target_length,
                mirna_length=mirna_length,
            ).astype(np.uint8)
            conservation[shard_rows] = conservation_matrix(
                row,
                conservation_schema.feature_indices,
                features,
                target_length,
            ).astype(conservation_dtype)
            label = float(row[pair_schema.label_idx])
            labels[shard_rows] = label

            shard_rows += 1
            shard_positive += int(label)
            total_rows += 1
            total_positive += int(label)

            if shard_rows == shard_size:
                shards.append(
                    _flush_shard(
                        output_dir,
                        output_prefix,
                        shard_idx,
                        pairs,
                        conservation,
                        labels,
                        shard_rows,
                        shard_positive,
                        shard_started,
                        log_path,
                    )
                )
                shard_idx += 1
                shard_rows = 0
                shard_positive = 0
                shard_started = time.perf_counter()

        if shard_rows:
            shards.append(
                _flush_shard(
                    output_dir,
                    output_prefix,
                    shard_idx,
                    pairs,
                    conservation,
                    labels,
                    shard_rows,
                    shard_positive,
                    shard_started,
                    log_path,
                )
            )

    manifest = {
        "cache_version": CACHE_VERSION,
        "cache_type": "conservation",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": str(input_file),
        "target_length": target_length,
        "mirna_length": mirna_length,
        "source_columns": {
            "target_sequence": pair_schema.gene_column,
            "mirna_sequence": pair_schema.mirna_column,
            "label": pair_schema.label_column,
            "conservation": dict(zip(features, conservation_schema.feature_columns)),
        },
        "pair_dtype": "uint8",
        "conservation_dtype": conservation_dtype,
        "conservation_features": list(features),
        "conservation_normalization": {
            "phylop": "clamp(score / 10, -1, 1)",
            "phastcons": "raw score, missing filled with 0.5",
        },
        "label_dtype": "float32",
        "row_count": total_rows,
        "positive_rows": total_positive,
        "shard_size": shard_size,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "shards": shards,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def _flush_shard(
    output_dir: Path,
    output_prefix: str,
    shard_idx: int,
    pairs: np.ndarray,
    conservation: np.ndarray,
    labels: np.ndarray,
    row_count: int,
    positive_rows: int,
    shard_started: float,
    log_path: Path,
) -> dict[str, object]:
    shard_name = f"{output_prefix}_shard_{shard_idx:05d}.pt"
    shard_path = output_dir / shard_name
    torch.save(
        {
            "cache_version": CACHE_VERSION,
            "pair_indices": torch.from_numpy(pairs[:row_count].copy()),
            "conservation": torch.from_numpy(conservation[:row_count].copy()),
            "labels": torch.from_numpy(labels[:row_count].copy()),
        },
        shard_path,
    )
    elapsed = time.perf_counter() - shard_started
    size_bytes = shard_path.stat().st_size
    with log_path.open("a") as handle:
        handle.write(
            "\t".join(
                [
                    str(shard_idx),
                    shard_name,
                    str(row_count),
                    str(positive_rows),
                    str(size_bytes),
                    f"{elapsed:.3f}",
                ]
            )
            + "\n"
        )
    print(
        f"cached shard {shard_idx}: rows={row_count:,} "
        f"size={size_bytes / (1024 ** 3):.2f} GiB elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "file": shard_name,
        "rows": row_count,
        "positive_rows": positive_rows,
        "size_bytes": size_bytes,
        "elapsed_sec": round(elapsed, 3),
    }
