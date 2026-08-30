"""Training CLI for seq + PHACT miRBind."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from phact_mirbind.cache.dataloaders import (
    CachedPhactPairwiseIterableDataset,
    CompositeCachedPhactPairwiseIterableDataset,
    PHACT_CHANNEL_MODES,
    PHACT_INTERACTION_MODES,
    make_phact_collate,
)
from phact_mirbind.cache.manifest import find_manifest
from phact_mirbind.models.phact import PairwisePhactCNN
from phact_mirbind.training.loop import (
    configure_new_input_channels_only,
    load_widened_state_dict,
    parse_int_tuple,
    resolve_device,
    set_seed,
    train_model,
)
from phact_mirbind.training.logging import json_ready
from phact_mirbind.training.losses import (
    AsymmetricLabelSmoothingBCEWithLogitsLoss,
)


def main() -> None:
    args = parse_args()
    try:
        validate_cache_paths(args)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    set_seed(args.seed)
    device = resolve_device(args.device)
    train_dataset = train_dataset_from_caches(
        [args.train_cache, *args.additional_train_cache],
        args.max_train_rows,
        args.cache_shuffle_mode,
        args.seed,
        not args.exclude_missingness,
        args.train_sample_weights,
    )
    val_dataset = dataset(
        args.val_cache,
        args.max_eval_rows,
        "none",
        args.seed,
        not args.exclude_missingness,
    )
    test_dataset = dataset(
        args.test_cache,
        args.max_eval_rows,
        "none",
        args.seed,
        not args.exclude_missingness,
    )
    loaders = {
        "train": loader(
            train_dataset,
            args.batch_size,
            args.num_workers,
            device,
            args.phact_channel_mode,
            args.phact_interaction_mode,
            args.interaction_mirna_channels,
            args.interaction_target_channels,
            args.training_target_shift_max,
        ),
        "val": loader(
            val_dataset,
            args.batch_size,
            args.num_workers,
            device,
            args.phact_channel_mode,
            args.phact_interaction_mode,
            args.interaction_mirna_channels,
            args.interaction_target_channels,
            0,
        ),
        "test": loader(
            test_dataset,
            args.batch_size,
            args.num_workers,
            device,
            args.phact_channel_mode,
            args.phact_interaction_mode,
            args.interaction_mirna_channels,
            args.interaction_target_channels,
            0,
        ),
    }
    if args.leftout_cache is not None:
        leftout_dataset = dataset(
            args.leftout_cache,
            args.max_eval_rows,
            "none",
            args.seed,
            not args.exclude_missingness,
        )
        loaders["leftout"] = loader(
            leftout_dataset,
            args.batch_size,
            args.num_workers,
            device,
            args.phact_channel_mode,
            args.phact_interaction_mode,
            args.interaction_mirna_channels,
            args.interaction_target_channels,
            0,
        )

    filter_sizes = parse_int_tuple(args.filter_sizes)
    kernel_sizes = parse_int_tuple(args.kernel_sizes)
    phact_channel_count = train_dataset.phact_channel_count_for_mode(
        args.phact_channel_mode
    )
    if args.phact_interaction_mode == "outer":
        phact_channel_count += (
            args.interaction_mirna_channels * args.interaction_target_channels
        )
    model_params = {
        "num_pair_classes": train_dataset.num_pair_classes,
        "phact_channel_count": phact_channel_count,
        "target_length": train_dataset.target_length,
        "mirna_length": train_dataset.mirna_length,
        "embedding_dim": args.embedding_dim,
        "dropout_rate": args.dropout_rate,
        "filter_sizes": filter_sizes,
        "kernel_sizes": kernel_sizes,
    }
    model = PairwisePhactCNN(**model_params).to(device)
    checkpoint = None
    new_input_trainable_weights = None
    if args.initial_checkpoint is not None:
        checkpoint = torch.load(
            args.initial_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if args.initial_checkpoint_mode == "strict":
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        else:
            load_widened_state_dict(model, checkpoint["model_state_dict"])
    if args.freeze_new_input_only:
        if checkpoint is None or args.initial_checkpoint_mode != "widen":
            raise SystemExit(
                "--freeze-new-input-only requires a widened initial checkpoint"
            )
        new_input_trainable_weights = configure_new_input_channels_only(
            model,
            checkpoint["model_state_dict"],
        )
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
        "phact_interaction_mode": args.phact_interaction_mode,
        "interaction_mirna_channels": args.interaction_mirna_channels,
        "interaction_target_channels": args.interaction_target_channels,
        "phact_missing_fill_value": missing_fill,
        "phact_missingness_included": not args.exclude_missingness,
        "initial_checkpoint": (
            str(args.initial_checkpoint) if args.initial_checkpoint is not None else None
        ),
        "initial_checkpoint_mode": args.initial_checkpoint_mode,
        "freeze_new_input_only": args.freeze_new_input_only,
        "new_input_trainable_weights": new_input_trainable_weights,
        "seed": args.seed,
        "train_cache": str(args.train_cache),
        "additional_train_caches": [
            str(path) for path in args.additional_train_cache
        ],
        "train_sample_weights": (
            str(args.train_sample_weights)
            if args.train_sample_weights is not None
            else None
        ),
        "val_cache": str(args.val_cache),
        "test_cache": str(args.test_cache),
        "leftout_cache": (
            str(args.leftout_cache) if args.leftout_cache is not None else None
        ),
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "cache_shuffle_mode": args.cache_shuffle_mode,
        "negative_label_smoothing": args.negative_label_smoothing,
        "positive_label_smoothing": args.positive_label_smoothing,
        "focal_gamma": args.focal_gamma,
        "training_target_shift_max": args.training_target_shift_max,
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
        optimizer=Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
        ),
        criterion=AsymmetricLabelSmoothingBCEWithLogitsLoss(
            negative_smoothing=args.negative_label_smoothing,
            positive_smoothing=args.positive_label_smoothing,
            focal_gamma=args.focal_gamma,
        ),
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
            "phact_interaction_mode": args.phact_interaction_mode,
            "interaction_mirna_channels": args.interaction_mirna_channels,
            "interaction_target_channels": args.interaction_target_channels,
            "phact_missing_fill_value": missing_fill,
        },
        final_eval_phases=(
            ("test", "leftout") if args.leftout_cache is not None else ("test",)
        ),
        amp_dtype=(
            torch.bfloat16 if args.bfloat16 and device.type == "cuda" else None
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train miRBind CNN with miRNA and target PHACT channels"
    )
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument(
        "--additional-train-cache",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional compatible PHACT cache to mix into training. "
            "May be supplied more than once."
        ),
    )
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--leftout-cache", type=Path)
    parser.add_argument(
        "--train-sample-weights",
        type=Path,
        help="NPZ containing one non-negative weight per combined training row.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phact"))
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument(
        "--initial-checkpoint-mode",
        choices=("strict", "widen"),
        default="strict",
        help=(
            "Load an exact architecture (strict), or initialize a wider filter "
            "stack from the overlapping checkpoint units (widen)."
        ),
    )
    parser.add_argument(
        "--freeze-new-input-only",
        action="store_true",
        help=(
            "Freeze a widened checkpoint and train only weights connected to "
            "new first-convolution input channels."
        ),
    )
    parser.add_argument(
        "--exclude-missingness",
        action="store_true",
        help="Use score channels only, for compatibility with legacy checkpoints.",
    )
    parser.add_argument(
        "--phact-channel-mode",
        choices=PHACT_CHANNEL_MODES,
        default="both",
        help="Use only miRNA PHACT channels, only target PHACT channels, or both.",
    )
    parser.add_argument(
        "--phact-interaction-mode",
        choices=PHACT_INTERACTION_MODES,
        default="none",
        help="Append cross-axis PHACT interaction channels.",
    )
    parser.add_argument("--interaction-mirna-channels", type=int, default=4)
    parser.add_argument("--interaction-target-channels", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--negative-label-smoothing", type=float, default=0.0)
    parser.add_argument("--positive-label-smoothing", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument(
        "--training-target-shift-max",
        type=int,
        default=0,
        help="Randomly shift each training batch along the target axis by up to N bases.",
    )
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--filter-sizes", type=str, default="128,64,32")
    parser.add_argument("--kernel-sizes", type=str, default="6,3,3")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Use BF16 autocast on CUDA.",
    )
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
    args = parser.parse_args()
    for name in ("negative_label_smoothing", "positive_label_smoothing"):
        value = getattr(args, name)
        if not 0.0 <= value < 0.5:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 0.5)")
    if args.training_target_shift_max < 0:
        parser.error("--training-target-shift-max must be non-negative")
    if args.focal_gamma < 0:
        parser.error("--focal-gamma must be non-negative")
    return args


def validate_cache_paths(args: argparse.Namespace) -> None:
    for name in ["train", "val", "test", "leftout"]:
        cache_path = getattr(args, f"{name}_cache")
        if cache_path is None:
            continue
        try:
            find_manifest(cache_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing {name} cache manifest: {cache_path}") from exc
    for cache_path in args.additional_train_cache:
        try:
            find_manifest(cache_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing additional train cache manifest: {cache_path}"
            ) from exc
    if args.initial_checkpoint is not None and not args.initial_checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing initial checkpoint: {args.initial_checkpoint}"
        )
    if args.train_sample_weights is not None and not args.train_sample_weights.is_file():
        raise FileNotFoundError(
            f"Missing train sample weights: {args.train_sample_weights}"
        )


def dataset(
    cache_path: Path,
    max_rows: int | None,
    shuffle_mode: str,
    seed: int,
    include_missingness: bool = True,
) -> CachedPhactPairwiseIterableDataset:
    return CachedPhactPairwiseIterableDataset(
        cache_path,
        max_rows=max_rows,
        shuffle_mode=shuffle_mode,
        shuffle_seed=seed,
        include_missingness=include_missingness,
    )


def train_dataset_from_caches(
    cache_paths: list[Path],
    max_rows: int | None,
    shuffle_mode: str,
    seed: int,
    include_missingness: bool = True,
    sample_weights: Path | None = None,
) -> CachedPhactPairwiseIterableDataset:
    if len(cache_paths) == 1:
        dataset_obj = dataset(
            cache_paths[0],
            max_rows,
            shuffle_mode,
            seed,
            include_missingness,
        )
    else:
        dataset_obj = CompositeCachedPhactPairwiseIterableDataset(
            cache_paths,
            max_rows=max_rows,
            shuffle_mode=shuffle_mode,
            shuffle_seed=seed,
            include_missingness=include_missingness,
        )
    if sample_weights is not None:
        dataset_obj.set_sample_weights(sample_weights)
    return dataset_obj


def loader(
    dataset_obj: CachedPhactPairwiseIterableDataset,
    batch_size: int,
    num_workers: int,
    device,
    phact_channel_mode: str,
    phact_interaction_mode: str,
    interaction_mirna_channels: int,
    interaction_target_channels: int,
    target_shift_max: int,
) -> DataLoader:
    return DataLoader(
        dataset_obj,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=make_phact_collate(
            phact_channel_mode,
            phact_interaction_mode,
            interaction_mirna_channels,
            interaction_target_channels,
            target_shift_max,
            target_compact_fill_values(dataset_obj),
        ),
    )


def target_compact_fill_values(
    dataset_obj: CachedPhactPairwiseIterableDataset,
) -> tuple[float, ...]:
    """Return neutral padding values in compact target-channel order."""
    order_key = (
        "phact_channel_order"
        if dataset_obj.include_missingness
        else "phact_score_channel_order"
    )
    target_names = [
        name
        for name in dataset_obj.manifest[order_key]
        if str(name).startswith("target_")
    ]
    default_fill = float(
        dataset_obj.manifest.get("phact_missing", {}).get("fill_value", 0.5)
    )
    return tuple(
        1.0
        if str(name).endswith("_missing")
        else 0.0
        if str(name).endswith("conservation_phylop")
        else 0.5
        if str(name).endswith("conservation_phastcons")
        else default_fill
        for name in target_names
    )


if __name__ == "__main__":
    main()
