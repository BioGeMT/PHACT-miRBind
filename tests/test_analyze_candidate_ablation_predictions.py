import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "analyze_candidate_ablation_predictions.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_candidate_ablation_predictions", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_validation_error_threshold_minimizes_total_errors() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    predictions = np.array([0.1, 0.6, 0.4, 0.9], dtype=np.float32)

    threshold = MODULE.validation_error_threshold(labels, predictions)
    counts = MODULE.confusion_counts(labels, predictions, threshold)

    assert threshold == float(predictions[2])
    assert counts["errors"] == 1


def test_validation_error_threshold_can_choose_all_negative() -> None:
    labels = np.array([0, 0, 0, 1], dtype=np.int8)
    predictions = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float32)

    threshold = MODULE.validation_error_threshold(labels, predictions)
    counts = MODULE.confusion_counts(labels, predictions, threshold)

    assert threshold > predictions.max()
    assert counts["errors"] == 1
