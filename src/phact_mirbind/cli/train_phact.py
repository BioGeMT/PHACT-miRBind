"""Training CLI for seq + PHACT miRBind."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from phact_mirbind.cache.dataloaders import (
    CachedPhactPairwiseIterableDataset,
    PHACT_CHANNEL_MODES,
    make_phact_collate,
)
from phact_mirbind.cache.manifest import find_manifest
from phact_mirbind.models.phact import PairwisePhactCNN
from phact_mirbind.training.loop import (
    parse_int_tuple,
    resolve_device,
    set_seed,
    train_model,
)
from phact_mirbind.training.logging import json_ready


def main() -> None:
    args = parse_args()
    try:
        validate_cache_paths(args)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    set_seed(args.seed)
    device = resolve_device(args.device)
    train_dataset = dataset(
        args.train_cache,
        args.max_train_rows,
        args.cache_shuffle_mode,
        args.seed,
    )
    val_dataset = dataset(args.val_cache, args.max_eval_rows, "none", args.seed)
    test_dataset = dataset(args.test_cache, args.max_eval_rows, "none", args.seed)
    leftout_dataset = dataset(args.leftout_cache, args.max_eval_rows, "none", args.seed)

    loaders = {
        "train": loader(
            train_dataset,
            args.batch_size,
            args.num_workers,
            device,
            args.phact_channel_mode,
        ),
        "val": loader(
            val_dataset,
            args.batch_size,
            args.num_workers,
            device,
            args.phact_channel_mode,
        ),
        "test": loader(
            test_dataset,
            args.batch_size,
            args.num_workers,
            device,
            args.phact_channel_mode,
        ),
        "leftout": loader(
            leftout_dataset,
            args.batch_size,
            args.num_workers,
            device,
            args.phact_channel_mode,
        ),
    }

    filter_sizes = parse_int_tuple(args.filter_sizes)
    kernel_sizes = parse_int_tuple(args.kernel_sizes)
    model_params = {
        "num_pair_classes": train_dataset.num_pair_classes,
        "phact_channel_count": train_dataset.phact_channel_count_for_mode(
            args.phact_channel_mode
        ),
        "target_length": train_dataset.target_length,
        "mirna_length": train_dataset.mirna_length,
        "embedding_dim": args.embedding_dim,
        "dropout_rate": args.dropout_rate,
        "filter_sizes": filter_sizes,
        "kernel_sizes": kernel_sizes,
    }
    model = PairwisePhactCNN(**model_params).to(device)
    phact_models = train_dataset.manifest.get(
        "phact_models",
        [train_dataset.manifest.get("phact_model", "unknown")],
    )
    target_phact_models = train_dataset.manifest.get("target_phact_models", [])
    phact_reduction = train_dataset.manifest.get("phact_reduction", "nucleotide")
    missing_fill = train_dataset.manifest.get("phact_missing", {}).get("fill_value")
    summary = {
        "model_type": "pairwise_seq_phact",
        "model_params": json_ready(model_params),
        "phact_models": phact_models,
        "target_phact_models": target_phact_models,
        "phact_reduction": phact_reduction,
        "phact_channel_mode": args.phact_channel_mode,
        "phact_missing_fill_value": missing_fill,
        "seed": args.seed,
        "train_cache": str(args.train_cache),
        "val_cache": str(args.val_cache),
        "test_cache": str(args.test_cache),
        "leftout_cache": str(args.leftout_cache),
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "cache_shuffle_mode": args.cache_shuffle_mode,
    }
    print(f"miRNA PHACT models: {','.join(phact_models)}")
    if target_phact_models:
        print(f"target PHACT models: {','.join(target_phact_models)}")
    print(f"PHACT reduction: {phact_reduction}")
    print(f"PHACT channel mode: {args.phact_channel_mode}")
    print(f"PHACT missing fill: {missing_fill}")
    train_model(
        model=model,
        loaders=loaders,
        optimizer=Adam(model.parameters(), lr=args.learning_rate),
        criterion=nn.BCEWithLogitsLoss(),
        device=device,
        output_dir=args.output_dir,
        checkpoint_prefix="pairwise_phact_model",
        summary=summary,
        num_epochs=args.num_epochs,
        patience=args.patience,
        progress_every=args.progress_every,
        show_progress=args.progress_bar,
        extra_checkpoint={
            "phact_models": phact_models,
            "target_phact_models": target_phact_models,
            "phact_reduction": phact_reduction,
            "phact_channel_mode": args.phact_channel_mode,
            "phact_missing_fill_value": missing_fill,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train miRBind CNN with miRNA and target PHACT channels"
    )
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--leftout-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phact"))
    parser.add_argument(
        "--phact-channel-mode",
        choices=PHACT_CHANNEL_MODES,
        default="both",
        help="Use only miRNA PHACT channels, only target PHACT channels, or both.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--filter-sizes", type=str, default="128,64,32")
    parser.add_argument("--kernel-sizes", type=str, default="6,3,3")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-eval-rows", type=int, default=None)
    parser.add_argument(
        "--cache-shuffle-mode",
        choices=["global", "shard"],
        default="global",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show an in-place batch progress bar",
    )
    return parser.parse_args()


def validate_cache_paths(args: argparse.Namespace) -> None:
    for name in ["train", "val", "test", "leftout"]:
        cache_path = getattr(args, f"{name}_cache")
        try:
            find_manifest(cache_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing {name} cache manifest: {cache_path}") from exc


def dataset(
    cache_path: Path,
    max_rows: int | None,
    shuffle_mode: str,
    seed: int,
) -> CachedPhactPairwiseIterableDataset:
    return CachedPhactPairwiseIterableDataset(
        cache_path,
        max_rows=max_rows,
        shuffle_mode=shuffle_mode,
        shuffle_seed=seed,
    )


def loader(
    dataset_obj: CachedPhactPairwiseIterableDataset,
    batch_size: int,
    num_workers: int,
    device,
    phact_channel_mode: str,
) -> DataLoader:
    return DataLoader(
        dataset_obj,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=make_phact_collate(phact_channel_mode),
    )


if __name__ == "__main__":
    main()
