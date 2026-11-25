"""Predict interaction probabilities with a saved five-fold model."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from phact_mirbind.cache.fold_cache import bases, model_inputs, target_scores
from phact_mirbind.training.cross_validation import (
    bf16_dtype,
    load_model,
    predict_probabilities,
)


def read_interactions(path, variant):
    """Return row IDs, base IDs and, for target-PHACT, target score channels."""
    ids, mirnas, targets, scores = [], [], [], []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if len(row["gene"]) != 50:
                raise ValueError(f"{row['id']}: target sequence must have 50 nucleotides")
            ids.append(row["id"])
            mirnas.append(bases(row["noncodingRNA"], 28))
            targets.append(bases(row["gene"], 50))
            if variant == "target_phact":
                scores.append(target_scores(row))
    return (
        ids,
        np.stack(mirnas),
        np.stack(targets),
        np.stack(scores) if scores else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="TSV to create; must not exist"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    device = torch.device(args.device)
    model, variant = load_model(args.checkpoint, device)
    ids, mirnas, targets, scores = read_interactions(args.input, variant)

    def input_batches():
        for start in range(0, len(ids), args.batch_size):
            rows = slice(start, start + args.batch_size)
            yield model_inputs(
                torch.from_numpy(mirnas[rows]).long().to(device),
                torch.from_numpy(targets[rows]).long().to(device),
                None if scores is None else torch.from_numpy(scores[rows]).to(device),
            )

    probabilities = predict_probabilities(
        model, input_batches(), device, bf16_dtype(device)
    )
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "probability"])
        writer.writerows(zip(ids, probabilities.tolist()))


if __name__ == "__main__":
    main()
