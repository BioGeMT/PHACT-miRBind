#!/usr/bin/env python3
"""Allocate pooled AP differences to positive rows, retaining cross-guide ranking."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workspace import WORKSPACE


import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = WORKSPACE / "runs/followup-2026-09-06/analysis"


def positive_row_contributions(labels, predictions):
    order = np.argsort(-predictions, kind='mergesort')
    sorted_predictions = predictions[order]
    sorted_labels = labels[order]
    ends = np.flatnonzero(np.r_[sorted_predictions[1:] != sorted_predictions[:-1], True])
    counts = np.diff(np.r_[-1, ends])
    precision = np.cumsum(sorted_labels)[ends] / (ends + 1)
    sorted_contributions = np.repeat(precision, counts) * sorted_labels / labels.sum()
    contributions = np.empty(len(labels))
    contributions[order] = sorted_contributions
    assert abs(contributions.sum() - average_precision_score(labels, predictions)) < 1e-12
    return contributions


def main():
    labels = np.array([1, 0, 1, 0, 1])
    predictions = np.array([0.8, 0.8, 0.4, 0.3, 0.3])
    positive_row_contributions(labels, predictions)
    results = {'sequence': [], 'family': []}
    for split in ['test', 'leftout']:
        frame = pd.read_csv(ROOT / f'{split}_joined_predictions.tsv.gz', sep='\t')
        labels = frame.label.to_numpy()
        target = positive_row_contributions(labels, frame.target.to_numpy())
        for baseline in ['sequence_model', 'conservation_both']:
            contributions = positive_row_contributions(labels, frame[baseline].to_numpy())
            frame['pooled_ap_delta_contribution'] = target - contributions
            frame['prediction_delta'] = frame.target - frame[baseline]
            for grouping in results:
                rows = []
                for name, group in frame.groupby(grouping):
                    rows.append({'split': split, 'baseline': baseline, grouping: name,
                                 'rows': len(group), 'positives': int(group.label.sum()),
                                 'pooled_ap_delta_contribution': group.pooled_ap_delta_contribution.sum(),
                                 'mean_positive_prediction_delta': group.loc[group.label == 1, 'prediction_delta'].mean(),
                                 'mean_negative_prediction_delta': group.loc[group.label == 0, 'prediction_delta'].mean()})
                assert abs(sum(row['pooled_ap_delta_contribution'] for row in rows) - (target - contributions).sum()) < 1e-12
                results[grouping].extend(rows)
    for grouping, rows in results.items():
        pd.DataFrame(rows).to_csv(ROOT / f'{grouping}_pooled_ap_contributions.tsv', sep='\t', index=False, na_rep='NA')
    print('Pooled AP contribution sums and tied-score example verified.')


if __name__ == '__main__':
    main()
