#!/usr/bin/env python3
"""Export ViennaRNA RNAduplex features for miRNA/target TSV pairs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator

import numpy as np


DUPLEX_PATTERN = re.compile(
    r"^(?P<structure>\S+&\S+)\s+"
    r"(?P<target_start>\d+),(?P<target_end>\d+)\s+:\s+"
    r"(?P<mirna_start>\d+),(?P<mirna_end>\d+)\s+"
    r"\(\s*(?P<energy>-?\d+(?:\.\d+)?)\)$"
)
FEATURE_NAMES = (
    "energy",
    "paired_bases",
    "target_start",
    "target_end",
    "target_span",
    "mirna_start",
    "mirna_end",
    "mirna_span",
)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    batches = read_batches(args.input, args.batch_size)
    worker_args = (
        (args.rnaduplex, batch, args.allow_lonely_pairs) for batch in batches
    )
    labels: list[np.ndarray] = []
    features: list[np.ndarray] = []
    row_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for batch_index, (batch_labels, batch_features) in enumerate(
            executor.map(run_batch, worker_args), start=1
        ):
            labels.append(batch_labels)
            features.append(batch_features)
            row_count += len(batch_labels)
            if args.progress_every and batch_index % args.progress_every == 0:
                print(
                    f"batches={batch_index:,} rows={row_count:,} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    label_array = np.concatenate(labels).astype(np.int8, copy=False)
    feature_array = np.concatenate(features).astype(np.float32, copy=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        labels=label_array,
        features=feature_array,
        feature_names=np.asarray(FEATURE_NAMES),
    )
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rnaduplex": str(args.rnaduplex),
        "allow_lonely_pairs": args.allow_lonely_pairs,
        "rows": int(len(label_array)),
        "positive_rows": int(label_array.sum()),
        "feature_names": list(FEATURE_NAMES),
        "seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rnaduplex", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument(
        "--allow-lonely-pairs",
        action="store_true",
        help="Do not pass --noLP to RNAduplex.",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if not args.rnaduplex.is_file():
        parser.error(f"RNAduplex executable does not exist: {args.rnaduplex}")
    if args.batch_size < 1 or args.workers < 1:
        parser.error("batch-size and workers must be positive")
    return args


def read_batches(
    path: Path,
    batch_size: int,
) -> Iterator[list[tuple[str, str, int]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"gene", "noncodingRNA", "label"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        batch: list[tuple[str, str, int]] = []
        for row in reader:
            batch.append(
                (
                    normalize_sequence(row["gene"]),
                    normalize_sequence(row["noncodingRNA"]),
                    int(row["label"]),
                )
            )
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def run_batch(
    args: tuple[Path, list[tuple[str, str, int]], bool],
) -> tuple[np.ndarray, np.ndarray]:
    executable, batch, allow_lonely_pairs = args
    command = [str(executable)]
    if not allow_lonely_pairs:
        command.append("--noLP")
    sequence_input = "\n".join(
        sequence for target, mirna, _ in batch for sequence in (target, mirna)
    )
    completed = subprocess.run(
        command,
        input=sequence_input + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    lines = completed.stdout.splitlines()
    if len(lines) != len(batch):
        raise RuntimeError(
            f"RNAduplex returned {len(lines)} rows for a batch of {len(batch)}"
        )
    labels = np.fromiter((row[2] for row in batch), dtype=np.int8)
    features = np.asarray([parse_duplex_line(line) for line in lines], dtype=np.float32)
    return labels, features


def parse_duplex_line(line: str) -> tuple[float, ...]:
    match = DUPLEX_PATTERN.fullmatch(line.strip())
    if match is None:
        raise ValueError(f"Could not parse RNAduplex output: {line!r}")
    values = match.groupdict()
    target_structure, mirna_structure = values["structure"].split("&", maxsplit=1)
    target_start = int(values["target_start"])
    target_end = int(values["target_end"])
    mirna_start = int(values["mirna_start"])
    mirna_end = int(values["mirna_end"])
    paired_bases = sum(base in "()" for base in target_structure)
    if paired_bases != sum(base in "()" for base in mirna_structure):
        raise ValueError(f"Unbalanced RNAduplex structure: {values['structure']!r}")
    return (
        float(values["energy"]),
        float(paired_bases),
        float(target_start),
        float(target_end),
        float(target_end - target_start + 1),
        float(mirna_start),
        float(mirna_end),
        float(mirna_end - mirna_start + 1),
    )


def normalize_sequence(sequence: str) -> str:
    return sequence.strip().upper().replace("T", "U")


if __name__ == "__main__":
    main()
