#!/usr/bin/env python3
"""Train a compact, interpretable seed/region LightGBM baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


BASES = "ATCG"
COMPLEMENT = str.maketrans("ATCG", "TAGC")
CATEGORICAL_COLUMNS = ("feature", "dominant_region", "regions_present")
INPUT_COLUMNS = (
    "gene",
    "noncodingRNA",
    "label",
    *CATEGORICAL_COLUMNS,
)


def main() -> None:
    args = parse_args()
    rows = {
        "train": read_rows(args.train),
        "val": read_rows(args.val),
        "test": read_rows(args.test),
        "leftout": read_rows(args.leftout),
    }
    category_maps = {
        column: {
            value: index
            for index, value in enumerate(
                sorted(rows["train"][column].fillna("NA").unique())
            )
        }
        for column in CATEGORICAL_COLUMNS
    }
    features = {
        split: build_features(frame, category_maps)
        for split, frame in rows.items()
    }
    labels = {
        split: frame["label"].to_numpy(dtype=np.int8)
        for split, frame in rows.items()
    }
    categorical_features = [
        features["train"].columns.get_loc(f"{column}_code")
        for column in CATEGORICAL_COLUMNS
    ]
    train_set = lgb.Dataset(
        features["train"],
        label=labels["train"],
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        features["val"],
        label=labels["val"],
        categorical_feature=categorical_features,
        reference=train_set,
        free_raw_data=False,
    )
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "average_precision",
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_data_in_leaf": args.min_data_in_leaf,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "num_threads": args.num_threads,
            "seed": args.seed,
            "verbosity": -1,
        },
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(25),
        ],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "seed_metadata_model.txt"
    model.save_model(model_path)
    report: dict[str, object] = {
        "model": str(model_path),
        "best_iteration": model.best_iteration,
        "feature_names": list(features["train"].columns),
        "splits": {},
    }
    for split in ("val", "test", "leftout"):
        prediction = model.predict(
            features[split],
            num_iteration=model.best_iteration,
        ).astype(np.float32)
        prediction_path = args.output_dir / f"{split}_predictions.npz"
        np.savez_compressed(
            prediction_path,
            labels=labels[split],
            predictions=prediction,
        )
        report["splits"][split] = {
            "rows": len(prediction),
            "auprc": float(average_precision_score(labels[split], prediction)),
            "prediction_file": str(prediction_path),
        }
        print(
            f"{split}: auprc={report['splits'][split]['auprc']:.8f}",
            flush=True,
        )
    report_path = args.output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--leftout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=200)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--num-threads", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", usecols=INPUT_COLUMNS)


def build_features(
    frame: pd.DataFrame,
    category_maps: dict[str, dict[str, int]],
) -> pd.DataFrame:
    targets = frame["gene"].str.upper().str.replace("U", "T", regex=False).tolist()
    mirnas = (
        frame["noncodingRNA"]
        .str.upper()
        .str.replace("U", "T", regex=False)
        .tolist()
    )
    seed_features = np.asarray(
        [seed_match_features(target, mirna) for target, mirna in zip(targets, mirnas)],
        dtype=np.float32,
    )
    result = pd.DataFrame(
        seed_features,
        columns=(
            "seed6_count",
            "seed7_count",
            "seed8_count",
            "seed7_min_mismatches",
            "target_gc",
            "mirna_gc",
            "mirna_length",
        ),
    )
    for column, mapping in category_maps.items():
        result[f"{column}_code"] = (
            frame[column].fillna("NA").map(mapping).fillna(-1).astype(np.int32)
        )
    return result


def seed_match_features(target: str, mirna: str) -> tuple[float, ...]:
    seed6 = reverse_complement(mirna[1:7])
    seed7 = reverse_complement(mirna[1:8])
    seed8 = reverse_complement(mirna[:8])
    min_mismatches = len(seed7)
    if seed7 and len(target) >= len(seed7):
        min_mismatches = min(
            sum(left != right for left, right in zip(seed7, target[offset:]))
            for offset in range(len(target) - len(seed7) + 1)
        )
    return (
        float(target.count(seed6)) if seed6 else 0.0,
        float(target.count(seed7)) if seed7 else 0.0,
        float(target.count(seed8)) if seed8 else 0.0,
        float(min_mismatches),
        gc_fraction(target),
        gc_fraction(mirna),
        float(len(mirna)),
    )


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def gc_fraction(sequence: str) -> float:
    return (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)


if __name__ == "__main__":
    main()
