#!/usr/bin/env python3
"""Export row-aligned predictions from a trained sequence-only CNN."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from phact_mirbind.cli.predict_seq import encode_batch_pair_indices, load_model
from phact_mirbind.data.columns import (
    TARGET_SEQUENCE_COLUMN,
    resolve_mirna_column,
)
from phact_mirbind.training.loop import resolve_device


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)

    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    target_sequences: list[str] = []
    mirna_sequences: list[str] = []
    batch_labels: list[int] = []
    started = time.perf_counter()

    with args.input_file.open() as handle:
        header = handle.readline().rstrip("\n\r").split("\t")
        columns = {name: index for index, name in enumerate(header)}
        if TARGET_SEQUENCE_COLUMN not in columns or "label" not in columns:
            raise ValueError("Input TSV must contain gene and label columns")
        mirna_column = resolve_mirna_column(columns)

        for row_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n\r").split("\t")
            target_sequences.append(fields[columns[TARGET_SEQUENCE_COLUMN]])
            mirna_sequences.append(fields[columns[mirna_column]])
            batch_labels.append(int(fields[columns["label"]]))
            if len(batch_labels) == args.batch_size:
                append_batch(
                    model,
                    device,
                    target_sequences,
                    mirna_sequences,
                    batch_labels,
                    labels,
                    predictions,
                )
                if args.progress_every and row_number % args.progress_every < args.batch_size:
                    print(f"rows={row_number:,}", flush=True)
                target_sequences.clear()
                mirna_sequences.clear()
                batch_labels.clear()

    if batch_labels:
        append_batch(
            model,
            device,
            target_sequences,
            mirna_sequences,
            batch_labels,
            labels,
            predictions,
        )

    labels_array = np.concatenate(labels).astype(np.int8)
    predictions_array = np.concatenate(predictions).astype(np.float32)
    metrics = {
        "checkpoint": str(args.checkpoint),
        "input_file": str(args.input_file),
        "rows": int(len(labels_array)),
        "positive_rows": int(labels_array.sum()),
        "auprc": float(average_precision_score(labels_array, predictions_array)),
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        labels=labels_array,
        predictions=predictions_array,
    )
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2), flush=True)


def append_batch(
    model: torch.nn.Module,
    device: torch.device,
    targets: list[str],
    mirnas: list[str],
    batch_labels: list[int],
    labels: list[np.ndarray],
    predictions: list[np.ndarray],
) -> None:
    pair_indices = encode_batch_pair_indices(
        targets,
        mirnas,
        target_length=int(model.target_length),
        mirna_length=int(model.mirna_length),
    ).to(device, non_blocking=True)
    with torch.inference_mode():
        probability = model.predict_proba(pair_indices).float().cpu().numpy()
    labels.append(np.asarray(batch_labels, dtype=np.int8))
    predictions.append(probability)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


if __name__ == "__main__":
    main()
