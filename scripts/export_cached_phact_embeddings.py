#!/usr/bin/env python3
"""Export frozen PHACT-CNN latent features from a compact cache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from phact_mirbind.cache.dataloaders import (
    CachedPhactPairwiseIterableDataset,
    make_phact_collate,
)
from phact_mirbind.models.phact import PairwisePhactCNN


def main() -> None:
    args = parse_args()
    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_params = dict(checkpoint["model_params"])
    model_params["filter_sizes"] = tuple(model_params["filter_sizes"])
    model_params["kernel_sizes"] = tuple(model_params["kernel_sizes"])
    model = PairwisePhactCNN(**model_params)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    channel_mode = str(checkpoint.get("phact_channel_mode", "both"))
    interaction_mode = str(checkpoint.get("phact_interaction_mode", "none"))
    interaction_mirna_channels = int(checkpoint.get("interaction_mirna_channels", 4))
    interaction_target_channels = int(checkpoint.get("interaction_target_channels", 4))
    dataset = compatible_dataset(
        args.cache,
        channel_mode,
        interaction_mode,
        interaction_mirna_channels,
        interaction_target_channels,
        int(model_params["phact_channel_count"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=make_phact_collate(
            channel_mode,
            interaction_mode,
            interaction_mirna_channels,
            interaction_target_channels,
        ),
    )

    labels: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (pairs, channels, label) in enumerate(loader, start=1):
            pairs = pairs.to(device, non_blocking=True)
            channels = channels.to(device, non_blocking=True)
            latent = model.encode(pairs, channels)
            batch_logits = model.fc2(latent).squeeze(-1)
            labels.append(label.numpy())
            embeddings.append(latent.cpu().numpy().astype(np.float16))
            logits.append(batch_logits.cpu().numpy().astype(np.float32))
            if args.progress_every and batch_index % args.progress_every == 0:
                print(
                    f"batches={batch_index:,} rows={sum(len(x) for x in labels):,}",
                    flush=True,
                )

    labels_array = np.concatenate(labels).astype(np.int8)
    embeddings_array = np.concatenate(embeddings)
    logits_array = np.concatenate(logits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        labels=labels_array,
        embeddings=embeddings_array,
        logits=logits_array,
    )
    metrics = {
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "rows": len(labels_array),
        "embedding_dim": embeddings_array.shape[1],
        "seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2), flush=True)


def compatible_dataset(
    cache: Path,
    channel_mode: str,
    interaction_mode: str,
    interaction_mirna_channels: int,
    interaction_target_channels: int,
    expected_channels: int,
) -> CachedPhactPairwiseIterableDataset:
    interaction_channels = (
        interaction_mirna_channels * interaction_target_channels
        if interaction_mode == "outer"
        else 0
    )
    for include_missingness in (False, True):
        candidate = CachedPhactPairwiseIterableDataset(
            cache,
            include_missingness=include_missingness,
            shuffle_mode="none",
        )
        count = candidate.phact_channel_count_for_mode(channel_mode)
        if count + interaction_channels == expected_channels:
            return candidate
    raise ValueError("checkpoint channel count is incompatible with cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    main()
