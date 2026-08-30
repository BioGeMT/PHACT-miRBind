#!/usr/bin/env python3
"""Export predictions from a trained PHACT CNN for a compact cache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from phact_mirbind.cache.dataloaders import CachedPhactPairwiseIterableDataset
from phact_mirbind.models.phact import PairwisePhactCNN
from phact_mirbind.data.columns import PADDING_PAIR_INDEX


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
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
    if args.mc_dropout_passes > 1:
        torch.manual_seed(args.seed)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    channel_mode = str(checkpoint.get("phact_channel_mode", "both"))
    interaction_mode = str(checkpoint.get("phact_interaction_mode", "none"))
    interaction_mirna_channels = int(
        checkpoint.get("interaction_mirna_channels", 4)
    )
    interaction_target_channels = int(
        checkpoint.get("interaction_target_channels", 4)
    )
    target_shifts = parse_target_shifts(args.target_shifts)
    if interaction_mode != "none" and target_shifts != (0,):
        raise ValueError("target-shift inference does not support interaction channels")
    dataset = CachedPhactPairwiseIterableDataset(args.cache, shuffle_mode="none")
    collate = make_checkpoint_compatible_collate(
        dataset,
        channel_mode,
        int(model_params["phact_channel_count"]),
        interaction_mode,
        interaction_mirna_channels,
        interaction_target_channels,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    labels: list[np.ndarray] = []
    predictions: dict[int, list[np.ndarray]] = {
        shift: [] for shift in target_shifts
    }
    weighted_loss = 0.0
    row_count = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (pair_indices, channels, label) in enumerate(loader, start=1):
            pair_indices = pair_indices.to(device, non_blocking=True)
            channels = channels.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            batch_predictions: dict[int, np.ndarray] = {}
            batch_logits: dict[int, torch.Tensor] = {}
            for shift in target_shifts:
                shifted_pairs, shifted_channels = shift_target_inputs(
                    pair_indices,
                    channels,
                    shift,
                    dataset,
                    channel_mode,
                    int(model_params["phact_channel_count"]),
                )
                logits_samples = [
                    model(shifted_pairs, shifted_channels)
                    for _ in range(args.mc_dropout_passes)
                ]
                probability_tensor = torch.stack(
                    [torch.sigmoid(logits) for logits in logits_samples]
                ).mean(dim=0)
                logits = torch.logit(probability_tensor.clamp(1e-6, 1 - 1e-6))
                batch_logits[shift] = logits
                batch_predictions[shift] = probability_tensor.float().cpu().numpy()
            batch_labels = label.cpu().numpy()
            labels.append(batch_labels)
            for shift in target_shifts:
                predictions[shift].append(batch_predictions[shift])
            batch_rows = int(label.numel())
            reference_shift = 0 if 0 in batch_logits else target_shifts[0]
            weighted_loss += float(
                F.binary_cross_entropy_with_logits(
                    batch_logits[reference_shift], label
                )
            ) * batch_rows
            row_count += batch_rows
            if args.progress_every and batch_index % args.progress_every == 0:
                print(
                    f"batches={batch_index:,} rows={row_count:,} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    labels_array = np.concatenate(labels).astype(np.int8)
    prediction_matrix = np.stack(
        [np.concatenate(predictions[shift]) for shift in target_shifts]
    ).astype(np.float32)
    reference_shift = 0 if 0 in target_shifts else target_shifts[0]
    predictions_array = prediction_matrix[target_shifts.index(reference_shift)]
    auprc_by_shift = {
        str(shift): float(
            average_precision_score(labels_array, prediction_matrix[index])
        )
        for index, shift in enumerate(target_shifts)
    }
    metrics = {
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "rows": row_count,
        "positive_rows": int(labels_array.sum()),
        "loss": weighted_loss / row_count,
        "auprc": auprc_by_shift[str(reference_shift)],
        "target_shifts": list(target_shifts),
        "auprc_by_target_shift": auprc_by_shift,
        "seconds": time.perf_counter() - started,
        "channel_mode": channel_mode,
        "interaction_mode": interaction_mode,
        "mc_dropout_passes": args.mc_dropout_passes,
        "model_channel_count": int(model_params["phact_channel_count"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        labels=labels_array,
        predictions=predictions_array,
        shift_predictions=prediction_matrix,
        target_shifts=np.asarray(target_shifts, dtype=np.int16),
    )
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mc-dropout-passes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-shifts",
        default="0",
        help="Comma-separated target-axis padding shifts, e.g. -2,-1,0,1,2.",
    )
    args = parser.parse_args()
    if args.mc_dropout_passes < 1:
        parser.error("--mc-dropout-passes must be positive")
    return args


def parse_target_shifts(value: str) -> tuple[int, ...]:
    shifts = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    if not shifts:
        raise ValueError("target shifts cannot be empty")
    return shifts


def shift_target_inputs(
    pair_indices: torch.Tensor,
    channels: torch.Tensor,
    shift: int,
    dataset: CachedPhactPairwiseIterableDataset,
    channel_mode: str,
    expected_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if shift == 0:
        return pair_indices, channels
    if abs(shift) >= pair_indices.shape[-1]:
        raise ValueError("absolute target shift must be smaller than target length")

    shifted_pairs = torch.full_like(pair_indices, PADDING_PAIR_INDEX)
    shifted_channels = channels.clone()
    if shift > 0:
        shifted_pairs[:, :, shift:] = pair_indices[:, :, :-shift]
    else:
        shifted_pairs[:, :, :shift] = pair_indices[:, :, -shift:]

    target_names = checkpoint_target_channel_names(
        dataset,
        channel_mode,
        expected_channels,
    )
    if not target_names:
        return shifted_pairs, shifted_channels
    target_start = channels.shape[1] - len(target_names)
    target_slice = channels[:, target_start:]
    fills = torch.tensor(
        [target_channel_fill(name, dataset.manifest) for name in target_names],
        dtype=channels.dtype,
        device=channels.device,
    ).view(1, -1, 1, 1)
    shifted_target = fills.expand_as(target_slice).clone()
    if shift > 0:
        shifted_target[:, :, :, shift:] = target_slice[:, :, :, :-shift]
    else:
        shifted_target[:, :, :, :shift] = target_slice[:, :, :, -shift:]
    shifted_channels[:, target_start:] = shifted_target
    return shifted_pairs, shifted_channels


def checkpoint_target_channel_names(
    dataset: CachedPhactPairwiseIterableDataset,
    channel_mode: str,
    expected_channels: int,
) -> list[str]:
    if channel_mode == "mirna":
        return []
    score_count = dataset.phact_channel_count_for_mode(channel_mode)
    score_only_count = (
        int(dataset.manifest["phact_axis_channel_counts"]["mirna"])
        if channel_mode in ("mirna", "both")
        else 0
    ) + (
        int(dataset.manifest["phact_axis_channel_counts"]["target"])
        if channel_mode in ("target", "both")
        else 0
    )
    key = (
        "phact_channel_order"
        if expected_channels == score_count
        else "phact_score_channel_order"
    )
    if expected_channels not in (score_count, score_only_count):
        raise ValueError("checkpoint channel count is incompatible with target shifts")
    return [
        name
        for name in dataset.manifest[key]
        if name.startswith("target_")
    ]


def target_channel_fill(name: str, manifest: dict[str, object]) -> float:
    if name.endswith("_missing"):
        return 1.0
    if name.endswith("conservation_phylop"):
        return 0.0
    if name.endswith("conservation_phastcons"):
        return 0.5
    return float(manifest.get("phact_missing", {}).get("fill_value", 0.5))


def make_checkpoint_compatible_collate(
    dataset: CachedPhactPairwiseIterableDataset,
    channel_mode: str,
    expected_channels: int,
    interaction_mode: str = "none",
    interaction_mirna_channels: int = 4,
    interaction_target_channels: int = 4,
):
    score_counts = {
        "mirna": int(dataset.manifest["phact_axis_channel_counts"]["mirna"]),
        "target": int(dataset.manifest["phact_axis_channel_counts"]["target"]),
    }
    total_counts = {
        "mirna": dataset.mirna_phact_channel_count,
        "target": dataset.target_phact_channel_count,
    }
    axes = ("mirna", "target") if channel_mode == "both" else (channel_mode,)
    score_channel_count = sum(score_counts[axis] for axis in axes)
    total_channel_count = sum(total_counts[axis] for axis in axes)
    interaction_channel_count = (
        interaction_mirna_channels * interaction_target_channels
        if interaction_mode == "outer"
        else 0
    )
    expected_base_channels = expected_channels - interaction_channel_count
    if expected_base_channels == score_channel_count:
        include_missingness = False
    elif expected_base_channels == total_channel_count:
        include_missingness = True
    else:
        raise ValueError(
            f"Checkpoint expects {expected_channels} channels, while cache offers "
            f"{score_channel_count} score-only or {total_channel_count} total channels"
        )

    def collate(batch):
        pair_indices = torch.stack([item[0] for item in batch]).long()
        mirna = torch.stack([item[1] for item in batch]).float()
        target = torch.stack([item[2] for item in batch]).float()
        if not include_missingness:
            mirna = mirna[:, :, : score_counts["mirna"]]
            target = target[:, :, : score_counts["target"]]
        labels = torch.stack([item[3] for item in batch]).float()
        batch_size, mirna_length = mirna.shape[:2]
        target_length = target.shape[1]
        mirna_channels = (
            mirna.permute(0, 2, 1)
            .unsqueeze(3)
            .expand(batch_size, mirna.shape[2], mirna_length, target_length)
        )
        target_channels = (
            target.permute(0, 2, 1)
            .unsqueeze(2)
            .expand(batch_size, target.shape[2], mirna_length, target_length)
        )
        if channel_mode == "mirna":
            channels = mirna_channels.contiguous()
        elif channel_mode == "target":
            channels = target_channels.contiguous()
        else:
            channels = torch.cat([mirna_channels, target_channels], dim=1).contiguous()
        if interaction_mode == "outer":
            mirna_scores = (
                mirna[:, :, :interaction_mirna_channels]
                .permute(0, 2, 1)
                .unsqueeze(2)
                .unsqueeze(4)
            )
            target_scores = (
                target[:, :, :interaction_target_channels]
                .permute(0, 2, 1)
                .unsqueeze(1)
                .unsqueeze(3)
            )
            interactions = (mirna_scores * target_scores).reshape(
                batch_size,
                interaction_channel_count,
                mirna_length,
                target_length,
            )
            channels = torch.cat([channels, interactions], dim=1).contiguous()
        return pair_indices, channels, labels

    return collate


if __name__ == "__main__":
    main()
