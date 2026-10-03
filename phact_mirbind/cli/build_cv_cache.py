"""Build a shared compact input cache for the saved five-fold assignments."""

import argparse
from pathlib import Path

from phact_mirbind.cache.fold_cache import prepare_training_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_training_cache(args.data_root, args.folds_dir, args.output_dir))


if __name__ == "__main__":
    main()
