"""Prepare and verify conservation-aware Manakov cross-validation folds."""

import argparse
from pathlib import Path

from phact_mirbind.data.fold_features import prepare_fold_features
from phact_mirbind.data.folds import build_conservation_folds, verify_conservation_folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Use existing features.npz and feature_audit.json.",
    )
    args = parser.parse_args()
    if not args.reuse_features:
        prepare_fold_features(args.data_root, args.output_dir)
    build_conservation_folds(args.output_dir, seed=args.seed)
    print(verify_conservation_folds(args.output_dir, args.data_root), flush=True)


if __name__ == "__main__":
    main()
