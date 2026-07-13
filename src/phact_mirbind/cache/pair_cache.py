"""Compact .pt cache writer for seq-only pair grids."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from phact_mirbind.cache.manifest import CACHE_VERSION, write_cache_log_header
from phact_mirbind.data.columns import (
    LABEL_COLUMN,
    TARGET_SEQUENCE_COLUMN,
    column_index,
    resolve_mirna_column,
)
from phact_mirbind.data.pair_encoding import encode_pair_indices
from phact_mirbind.data.sequences import normalize_sequence


@dataclass(frozen=True)
class PairSchema:
    gene_idx: int
    mirna_idx: int
    label_idx: int
    gene_column: str
    mirna_column: str
    label_column: str

    @classmethod
    def from_header(cls, header: list[str]) -> "PairSchema":
        columns = column_index(header)
        missing_required = [
            column
            for column in (TARGET_SEQUENCE_COLUMN, LABEL_COLUMN)
            if column not in columns
        ]
        if missing_required:
            raise ValueError(f"Missing required columns: {', '.join(missing_required)}")

        mirna_column = resolve_mirna_column(columns)
        return cls(
            gene_idx=columns[TARGET_SEQUENCE_COLUMN],
            mirna_idx=columns[mirna_column],
            label_idx=columns[LABEL_COLUMN],
            gene_column=TARGET_SEQUENCE_COLUMN,
            mirna_column=mirna_column,
            label_column=LABEL_COLUMN,
        )


def write_pair_cache_from_tsv(
    input_file: str | Path,
    output_dir: str | Path,
    *,
    output_prefix: str,
    shard_size: int = 100_000,
    target_length: int = 50,
    mirna_length: int = 28,
) -> Path:
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"{output_prefix}_cache_log.tsv"
    manifest_path = output_dir / f"{output_prefix}_manifest.json"
    shards: list[dict[str, object]] = []
    total_rows = 0
    total_positive = 0
    shard_idx = 0
    shard_started = time.perf_counter()
    started = time.perf_counter()

    pairs = np.empty((shard_size, mirna_length, target_length), dtype=np.uint8)
    labels = np.empty((shard_size,), dtype=np.float32)
    write_cache_log_header(log_path)

    with input_file.open("r", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        schema = PairSchema.from_header(next(reader))
        shard_rows = 0
        shard_positive = 0
        for row in reader:
            target_sequence = normalize_sequence(row[schema.gene_idx], target_length)
            mirna_sequence = normalize_sequence(row[schema.mirna_idx], mirna_length)
            pairs[shard_rows] = encode_pair_indices(
                target_sequence,
                mirna_sequence,
                target_length=target_length,
                mirna_length=mirna_length,
            ).astype(np.uint8)
            label = float(row[schema.label_idx])
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
                    labels,
                    shard_rows,
                    shard_positive,
                    shard_started,
                    log_path,
                )
            )

    manifest = {
        "cache_version": CACHE_VERSION,
        "cache_type": "pair",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": str(input_file),
        "target_length": target_length,
        "mirna_length": mirna_length,
        "source_columns": {
            "target_sequence": schema.gene_column,
            "mirna_sequence": schema.mirna_column,
            "label": schema.label_column,
        },
        "pair_dtype": "uint8",
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
