#!/usr/bin/env python3
"""Fit a nonlinear classifier on frozen PHACT-CNN latent features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score


def main() -> None:
    args = parse_args()
    arrays = {
        split: np.load(path)
        for split, path in {
            "train": args.train,
            "val": args.val,
            "test": args.test,
            "leftout": args.leftout,
        }.items()
    }
    features = {split: feature_matrix(data) for split, data in arrays.items()}
    labels = {split: data["labels"] for split, data in arrays.items()}
    train_set = lgb.Dataset(features["train"], label=labels["train"])
    val_set = lgb.Dataset(features["val"], label=labels["val"], reference=train_set)
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
            "lambda_l2": 2.0,
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
    model_path = args.output_dir / "phact_embedding_model.txt"
    model.save_model(model_path)
    report: dict[str, object] = {
        "model": str(model_path),
        "best_iteration": model.best_iteration,
        "splits": {},
    }
    for split in ("val", "test", "leftout"):
        prediction = model.predict(
            features[split], num_iteration=model.best_iteration
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
        print(f"{split}: auprc={report['splits'][split]['auprc']:.8f}")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


def feature_matrix(data: np.lib.npyio.NpzFile) -> np.ndarray:
    return np.column_stack(
        [data["embeddings"].astype(np.float32), data["logits"].astype(np.float32)]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--leftout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-data-in-leaf", type=int, default=500)
    parser.add_argument("--num-boost-round", type=int, default=400)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--num-threads", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
