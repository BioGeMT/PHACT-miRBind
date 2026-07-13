"""Metric helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def average_precision(
    labels: list[np.ndarray],
    probabilities: list[np.ndarray],
) -> float:
    if not labels:
        return float("nan")

    label_array = np.concatenate(labels)
    probability_array = np.concatenate(probabilities)
    if len(np.unique(label_array)) < 2:
        return float("nan")
    return float(average_precision_score(label_array, probability_array))
