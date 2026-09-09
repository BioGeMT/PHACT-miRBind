"""Fully fine-tune RiNALMo-micro with PHACT pooling and frozen miRBind."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import torch.nn as nn
from multimolecule import RiNALMoModel, RnaTokenizer
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from phact_mirbind.cache.dataloaders import (
    CachedPhactPairwiseIterableDataset,
    CompositeCachedPhactPairwiseIterableDataset,
    rinalmo_phact_collate,
)
from phact_mirbind.cache.manifest import find_manifest
from phact_mirbind.cli.predict_seq import load_model as load_mirbind
from phact_mirbind.models.phact import PairwisePhactCNN
from phact_mirbind.models.rinalmo_phact import PairwiseRinalmoPhactFusion
from phact_mirbind.models.rinalmo_phact import RINALMO_ENCODING_MODES
from phact_mirbind.training.logging import json_ready
from phact_mirbind.training.loop import resolve_device, set_seed, train_model

DEFAULT_RINALMO_MODEL = "multimolecule/rinalmo-micro"
PAIR_BASE_TOKENS = ("A", "U", "C", "G")


def main() -> None:
    args = parse_args()
    validate_paths(args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    datasets = {
        "train": make_train_dataset(
            [args.train_cache, *args.additional_train_cache],
            args.max_train_rows,
            args.cache_shuffle_mode,
            args.seed,
        ),
        "val": make_dataset(args.val_cache, args.max_eval_rows, "none", args.seed),
        "test": make_dataset(args.test_cache, args.max_eval_rows, "none", args.seed),
        "leftout": make_dataset(
            args.leftout_cache,
            args.max_eval_rows,
            "none",
            args.seed,
        ),
    }
    validate_dataset_shapes(datasets)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=rinalmo_phact_collate,
        )
        for name, dataset in datasets.items()
    }

    tokenizer = RnaTokenizer.from_pretrained(args.rinalmo_model)
    rinalmo = RiNALMoModel.from_pretrained(
        args.rinalmo_model,
        add_pooling_layer=False,
    )
    if args.gradient_checkpointing:
        rinalmo.gradient_checkpointing_enable()
    mirbind = load_mirbind(args.mirbind_checkpoint, torch.device("cpu"))
    train_dataset = datasets["train"]
    phact_baseline, phact_baseline_mode = load_phact_baseline(
        args.phact_baseline_checkpoint
    )
    model = PairwiseRinalmoPhactFusion(
        rinalmo=rinalmo,
        mirbind=mirbind,
        mirna_phact_channels=train_dataset.mirna_phact_channel_count,
        target_phact_channels=train_dataset.target_phact_channel_count,
        base_token_ids=tuple(
            tokenizer.convert_tokens_to_ids(token) for token in PAIR_BASE_TOKENS
        ),
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.eos_token_id,
        layers_to_mix=args.layers_to_mix,
        attention_dim=args.attention_dim,
        pair_projection_dim=args.pair_projection_dim,
        classifier_dim=args.classifier_dim,
        dropout_rate=args.dropout_rate,
        encoding_mode=args.encoding_mode,
        phact_baseline=phact_baseline,
        phact_baseline_channel_mode=phact_baseline_mode,
        phact_baseline_score_channels=(
            int(train_dataset.manifest["phact_axis_channel_counts"]["mirna"]),
            int(train_dataset.manifest["phact_axis_channel_counts"]["target"]),
        ),
        correction_scale=args.correction_scale,
    ).to(device)
    if args.initial_checkpoint is not None:
        initial_checkpoint = torch.load(
            args.initial_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(initial_checkpoint["model_state_dict"], strict=True)
    if args.freeze_rinalmo:
        model.freeze_rinalmo()
    optimizer = build_optimizer(
        model,
        head_learning_rate=args.head_learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        layerwise_lr_decay=args.layerwise_lr_decay,
        weight_decay=args.weight_decay,
    )

    phact_models = train_dataset.manifest.get("phact_models", [])
    target_phact_models = train_dataset.manifest.get("target_phact_models", [])
    summary = {
        "model_type": "pairwise_rinalmo_phact_fusion",
        "model_params": {
            "rinalmo_model": args.rinalmo_model,
            "rinalmo_hidden_size": rinalmo.config.hidden_size,
            "rinalmo_layers": rinalmo.config.num_hidden_layers,
            "layers_to_mix": args.layers_to_mix,
            "mirna_phact_channels": train_dataset.mirna_phact_channel_count,
            "target_phact_channels": train_dataset.target_phact_channel_count,
            "attention_dim": args.attention_dim,
            "pair_projection_dim": args.pair_projection_dim,
            "classifier_dim": args.classifier_dim,
            "dropout_rate": args.dropout_rate,
            "encoding_mode": args.encoding_mode,
        },
        "rinalmo_fully_trainable": not args.freeze_rinalmo,
        "rinalmo_gradient_checkpointing": args.gradient_checkpointing,
        "mirbind_frozen": True,
        "mirbind_checkpoint": str(args.mirbind_checkpoint),
        "phact_baseline_checkpoint": (
            str(args.phact_baseline_checkpoint)
            if args.phact_baseline_checkpoint is not None
            else None
        ),
        "phact_baseline_channel_mode": phact_baseline_mode,
        "phact_baseline_score_channels": list(model.phact_baseline_score_channels),
        "correction_scale": args.correction_scale,
        "initial_checkpoint": (
            str(args.initial_checkpoint) if args.initial_checkpoint is not None else None
        ),
        "phact_models": phact_models,
        "target_phact_models": target_phact_models,
        "phact_reduction": train_dataset.manifest.get(
            "phact_reduction",
            "nucleotide",
        ),
        "head_learning_rate": args.head_learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "layerwise_lr_decay": args.layerwise_lr_decay,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "seed": args.seed,
        "train_cache": str(args.train_cache),
        "additional_train_caches": [
            str(path) for path in args.additional_train_cache
        ],
        "val_cache": str(args.val_cache),
        "test_cache": str(args.test_cache),
        "leftout_cache": str(args.leftout_cache),
    }
    print(f"RiNALMo model: {args.rinalmo_model}")
    trainable_rinalmo_layers = 0 if args.freeze_rinalmo else rinalmo.config.num_hidden_layers
    print(f"RiNALMo layers fully trainable: {trainable_rinalmo_layers}")
    print(f"Frozen miRBind checkpoint: {args.mirbind_checkpoint}")
    if args.initial_checkpoint is not None:
        print(f"Initialized fusion model from: {args.initial_checkpoint}")
    if args.phact_baseline_checkpoint is not None:
        print(f"Frozen PHACT residual baseline: {args.phact_baseline_checkpoint}")
    print(f"miRNA PHACT channels: {train_dataset.mirna_phact_channel_count}")
    print(f"target PHACT channels: {train_dataset.target_phact_channel_count}")
    train_model(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        criterion=nn.BCEWithLogitsLoss(),
        device=device,
        output_dir=args.output_dir,
        checkpoint_prefix="pairwise_rinalmo_phact_model",
        summary=json_ready(summary),
        num_epochs=args.num_epochs,
        patience=args.patience,
        progress_every=args.progress_every,
        show_progress=args.progress_bar,
        amp_dtype=torch.bfloat16 if args.bfloat16 and device.type == "cuda" else None,
        grad_clip_norm=args.grad_clip_norm,
        extra_checkpoint={
            "rinalmo_model": args.rinalmo_model,
            "mirbind_checkpoint": str(args.mirbind_checkpoint),
            "phact_models": phact_models,
            "target_phact_models": target_phact_models,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--leftout-cache", type=Path, required=True)
    parser.add_argument("--mirbind-checkpoint", type=Path, required=True)
    parser.add_argument("--phact-baseline-checkpoint", type=Path)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Initialize the full RiNALMo-PHACT model from a prior checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rinalmo_phact"))
    parser.add_argument("--rinalmo-model", default=DEFAULT_RINALMO_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-6)
    parser.add_argument("--layerwise-lr-decay", type=float, default=0.85)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--layers-to-mix", type=int, default=4)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--pair-projection-dim", type=int, default=256)
    parser.add_argument("--classifier-dim", type=int, default=128)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--correction-scale", type=float, default=0.25)
    parser.add_argument("--freeze-rinalmo", action="store_true")
    parser.add_argument(
        "--encoding-mode",
        choices=RINALMO_ENCODING_MODES,
        default="independent",
        help="Encode RNAs independently or jointly with cross-molecule attention.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-eval-rows", type=int, default=None)
    parser.add_argument(
        "--cache-shuffle-mode",
        choices=["global", "shard"],
        default="shard",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--no-bfloat16",
        dest="bfloat16",
        action="store_false",
        help="Disable BF16 autocast on CUDA.",
    )
    parser.set_defaults(bfloat16=True)
    return parser.parse_args()


def make_dataset(
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


def make_train_dataset(
    cache_paths: list[Path],
    max_rows: int | None,
    shuffle_mode: str,
    seed: int,
) -> CachedPhactPairwiseIterableDataset:
    if len(cache_paths) == 1:
        return make_dataset(cache_paths[0], max_rows, shuffle_mode, seed)
    return CompositeCachedPhactPairwiseIterableDataset(
        cache_paths,
        max_rows=max_rows,
        shuffle_mode=shuffle_mode,
        shuffle_seed=seed,
    )


def validate_paths(args: argparse.Namespace) -> None:
    if not args.mirbind_checkpoint.is_file():
        raise FileNotFoundError(f"Missing miRBind checkpoint: {args.mirbind_checkpoint}")
    if args.initial_checkpoint is not None and not args.initial_checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing initial checkpoint: {args.initial_checkpoint}"
        )
    if (
        args.phact_baseline_checkpoint is not None
        and not args.phact_baseline_checkpoint.is_file()
    ):
        raise FileNotFoundError(
            f"Missing PHACT baseline checkpoint: {args.phact_baseline_checkpoint}"
        )
    for name in ("train", "val", "test", "leftout"):
        find_manifest(getattr(args, f"{name}_cache"))
    for cache_path in args.additional_train_cache:
        find_manifest(cache_path)


def validate_dataset_shapes(
    datasets: dict[str, CachedPhactPairwiseIterableDataset],
) -> None:
    train = datasets["train"]
    expected = (
        train.mirna_length,
        train.target_length,
        train.mirna_phact_channel_count,
        train.target_phact_channel_count,
    )
    for name, dataset in datasets.items():
        actual = (
            dataset.mirna_length,
            dataset.target_length,
            dataset.mirna_phact_channel_count,
            dataset.target_phact_channel_count,
        )
        if actual != expected:
            raise ValueError(f"{name} cache shape {actual} does not match train {expected}")


def build_optimizer(
    model: PairwiseRinalmoPhactFusion,
    *,
    head_learning_rate: float,
    backbone_learning_rate: float,
    layerwise_lr_decay: float,
    weight_decay: float,
) -> Optimizer:
    if not 0 < layerwise_lr_decay <= 1:
        raise ValueError("layerwise_lr_decay must be in (0, 1]")
    layer_count = int(model.rinalmo.config.num_hidden_layers)
    grouped_backbone: dict[float, list[nn.Parameter]] = {}
    for name, parameter in model.rinalmo.named_parameters():
        if not parameter.requires_grad:
            continue
        layer_match = re.match(r"encoder\.layer\.(\d+)\.", name)
        if layer_match:
            depth_from_top = layer_count - 1 - int(layer_match.group(1))
        elif name.startswith("embeddings."):
            depth_from_top = layer_count
        else:
            depth_from_top = 0
        learning_rate = backbone_learning_rate * (
            layerwise_lr_decay**depth_from_top
        )
        grouped_backbone.setdefault(learning_rate, []).append(parameter)

    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("rinalmo.")
        and not name.startswith("mirbind.")
    ]
    parameter_groups = [
        {"params": parameters, "lr": learning_rate}
        for learning_rate, parameters in grouped_backbone.items()
    ]
    parameter_groups.append({"params": head_parameters, "lr": head_learning_rate})
    return AdamW(parameter_groups, weight_decay=weight_decay)


def load_phact_baseline(
    checkpoint_path: Path | None,
) -> tuple[PairwisePhactCNN | None, str]:
    if checkpoint_path is None:
        return None, "both"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_params = dict(checkpoint["model_params"])
    model_params["filter_sizes"] = tuple(model_params["filter_sizes"])
    model_params["kernel_sizes"] = tuple(model_params["kernel_sizes"])
    model = PairwisePhactCNN(**model_params)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, str(checkpoint.get("phact_channel_mode", "both"))


if __name__ == "__main__":
    main()
