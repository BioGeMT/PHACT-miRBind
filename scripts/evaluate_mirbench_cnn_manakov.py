#!/usr/bin/env python3
"""Evaluate the official miRBench Manakov CNN through a PyTorch weight reader."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from phact_mirbind.cache.dataloaders import (
    CachedPairwiseIterableDataset,
    pair_collate,
)


class MirBenchCnnManakov(nn.Module):
    """PyTorch equivalent of ``miRBenchCNN_Manakov_v7.keras``."""

    def __init__(self) -> None:
        super().__init__()
        channels = (1, 32, 64, 96, 128, 160, 192)
        self.convs = nn.ModuleList(
            nn.Conv2d(channels[i], channels[i + 1], 5, padding=2)
            for i in range(6)
        )
        self.conv_bns = nn.ModuleList(
            nn.BatchNorm2d(channel, eps=0.001) for channel in channels[1:]
        )
        self.fc1 = nn.Linear(192, 192)
        self.bn1 = nn.BatchNorm1d(192, eps=0.001)
        self.fc2 = nn.Linear(192, 160)
        self.bn2 = nn.BatchNorm1d(160, eps=0.001)
        self.out = nn.Linear(160, 1)

    def forward(self, pair_indices: torch.Tensor) -> torch.Tensor:
        complement_indices = torch.tensor(
            (1, 4, 11, 14), device=pair_indices.device
        )
        pair_grid = pair_indices[:, :20].transpose(1, 2)
        x = torch.isin(pair_grid, complement_indices).float().unsqueeze(1)
        for conv, batch_norm in zip(self.convs, self.conv_bns):
            x = F.max_pool2d(
                batch_norm(F.leaky_relu(conv(x), negative_slope=0.3)),
                kernel_size=2,
                stride=2,
                ceil_mode=True,
            )
        x = x.flatten(1)
        x = self.bn1(F.leaky_relu(self.fc1(x), negative_slope=0.3))
        x = self.bn2(F.leaky_relu(self.fc2(x), negative_slope=0.3))
        return self.out(x).squeeze(-1)


def main() -> None:
    args = parse_args()
    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)
    model = MirBenchCnnManakov()
    load_keras_weights(model, args.model)
    model.to(device).eval()
    dataset = CachedPairwiseIterableDataset(args.cache, shuffle_mode="none")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=pair_collate,
    )

    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (pairs, label) in enumerate(loader, start=1):
            probability = torch.sigmoid(model(pairs.to(device, non_blocking=True)))
            labels.append(label.numpy())
            predictions.append(probability.cpu().numpy())
            if args.progress_every and batch_index % args.progress_every == 0:
                print(f"batches={batch_index:,}", flush=True)
    labels_array = np.concatenate(labels).astype(np.int8)
    predictions_array = np.concatenate(predictions).astype(np.float32)
    metrics = {
        "model": str(args.model),
        "cache": str(args.cache),
        "rows": len(labels_array),
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


def load_keras_weights(model: MirBenchCnnManakov, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive, tempfile.NamedTemporaryFile() as tmp:
        tmp.write(archive.read("model.weights.h5"))
        tmp.flush()
        with h5py.File(tmp.name) as weights:
            root = weights["_layer_checkpoint_dependencies"]
            for index, (conv, batch_norm) in enumerate(
                zip(model.convs, model.conv_bns)
            ):
                suffix = "" if index == 0 else f"_{2 * index}"
                copy_conv(conv, root[f"conv2d{suffix}"]["vars"])
                copy_batch_norm(
                    batch_norm,
                    root[f"batch_normalization{suffix}"]["vars"],
                )
            copy_dense(model.fc1, root["dense"]["vars"])
            copy_batch_norm(model.bn1, root["batch_normalization_12"]["vars"])
            copy_dense(model.fc2, root["dense_2"]["vars"])
            copy_batch_norm(model.bn2, root["batch_normalization_14"]["vars"])
            copy_dense(model.out, root["dense_4"]["vars"])


def copy_conv(layer: nn.Conv2d, variables: h5py.Group) -> None:
    layer.weight.data.copy_(
        torch.from_numpy(np.asarray(variables["0"]).transpose(3, 2, 0, 1))
    )
    layer.bias.data.copy_(torch.from_numpy(np.asarray(variables["1"])))


def copy_dense(layer: nn.Linear, variables: h5py.Group) -> None:
    layer.weight.data.copy_(torch.from_numpy(np.asarray(variables["0"]).T))
    layer.bias.data.copy_(torch.from_numpy(np.asarray(variables["1"])))


def copy_batch_norm(layer: nn.modules.batchnorm._BatchNorm, variables: h5py.Group) -> None:
    layer.weight.data.copy_(torch.from_numpy(np.asarray(variables["0"])))
    layer.bias.data.copy_(torch.from_numpy(np.asarray(variables["1"])))
    layer.running_mean.data.copy_(torch.from_numpy(np.asarray(variables["2"])))
    layer.running_var.data.copy_(torch.from_numpy(np.asarray(variables["3"])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    main()
