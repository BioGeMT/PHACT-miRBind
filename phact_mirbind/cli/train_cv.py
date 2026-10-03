"""Train sequence-only or target-PHACT miRBind2 on one fixed fold."""

import argparse
from pathlib import Path

import torch

from phact_mirbind.cache.fold_cache import FoldInputs
from phact_mirbind.training.cross_validation import train_fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=["seq_only", "target_phact"], required=True
    )
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    inputs = FoldInputs(
        args.cache_dir, args.folds_dir, args.variant, device=args.device
    )
    print(
        train_fold(
            inputs,
            args.output_dir,
            fold=args.fold,
            seed=args.seed,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            patience=args.patience,
        )
    )


if __name__ == "__main__":
    main()
