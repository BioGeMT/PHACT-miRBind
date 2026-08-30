#!/usr/bin/env python3
"""Evaluate Agentomics-format prediction CSVs against held-out labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--leftout-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate(labels_path: Path, predictions_path: Path) -> dict[str, object]:
    labels = pd.read_csv(labels_path, usecols=["id", "label"])
    predictions = pd.read_csv(predictions_path, usecols=["id", "probability_1"])
    if labels["id"].duplicated().any() or predictions["id"].duplicated().any():
        raise ValueError(f"Duplicate ids in {labels_path} or {predictions_path}")
    merged = labels.merge(predictions, on="id", how="outer", indicator=True, validate="one_to_one")
    missing = {
        str(key): int(value)
        for key, value in merged.loc[merged["_merge"] != "both", "_merge"].value_counts().items()
        if int(value) > 0
    }
    if missing:
        raise ValueError(f"Label/prediction id mismatch for {predictions_path}: {missing}")
    y_true = merged["label"].astype(int).to_numpy()
    probability = merged["probability_1"].astype(float).to_numpy()
    if not ((probability >= 0.0) & (probability <= 1.0)).all():
        raise ValueError(f"Probabilities outside [0, 1] in {predictions_path}")
    predicted = (probability >= 0.5).astype(int)
    return {
        "n": int(len(merged)),
        "positive": int(y_true.sum()),
        "auprc": float(average_precision_score(y_true, probability)),
        "auroc": float(roc_auc_score(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probability)),
        "accuracy_at_0p5": float(accuracy_score(y_true, predicted)),
        "precision_at_0p5": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_at_0p5": float(recall_score(y_true, predicted, zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    result = {
        "model": args.model_name,
        "test": evaluate(args.test_labels, args.predictions_dir / "test_predictions.csv"),
        "leftout": evaluate(args.leftout_labels, args.predictions_dir / "leftout_predictions.csv"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
