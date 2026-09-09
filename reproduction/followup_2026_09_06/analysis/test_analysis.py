"""Checks for row matching, undefined groups and weighted AP used in this analysis."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import analyze_saved_predictions as analysis


class AnalysisChecks(unittest.TestCase):
    def test_prediction_ids_define_order_and_duplicates_fail(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / 'predictions.csv'
            path.write_text('id,prediction\ntest_2,0.8\ntest_1,0.1\n')
            actual = analysis.previous.load_prediction_csv(path, 'test', 2)
            np.testing.assert_array_equal(actual, [0.1, 0.8])
            path.write_text('id,prediction\ntest_1,0.8\ntest_1,0.1\n')
            with self.assertRaises(ValueError):
                analysis.previous.load_prediction_csv(path, 'test', 2)

    def test_one_class_groups_are_na(self):
        frame = pd.DataFrame({'label': [1, 1], 'sequence': ['AA', 'AA'],
                              **{model: [0.2, 0.7] for model in analysis.MODELS}})
        result = analysis.metrics(frame)
        self.assertEqual(result['ap_status'], 'one_class')
        self.assertTrue(all(np.isnan(result[f'ap_{model}']) for model in analysis.MODELS))

    def test_profile_coverage_and_missing_values(self):
        scores = np.zeros((3, 2, 4))
        scores[:, 0, 0] = 1
        scores[:, 1, 0] = 0.2
        missing = np.array([[False, True], [True, True], [False, False]])
        result = analysis.profile_properties(scores, missing, ['AA'] * 3, 2)
        np.testing.assert_array_equal(result['category'], ['flat', 'missing', 'variable'])
        np.testing.assert_array_equal(result['fraction'], [0.5, 0, 1])
        self.assertTrue(np.isnan(result['range'][1]))
        self.assertAlmostEqual(result['range'][2], 0.8)

    def test_weighted_ap_matches_repeated_rows_with_ties(self):
        labels = np.array([1, 0, 1, 0, 1])
        predictions = np.array([0.8, 0.8, 0.4, 0.3, 0.3])
        counts = np.array([2, 0, 3, 1, 2])
        calculator = analysis.previous.WeightedAveragePrecision(labels, predictions)
        observed = calculator.calculate(counts[None, :].astype(float))[0]
        expected = average_precision_score(np.repeat(labels, counts), np.repeat(predictions, counts))
        self.assertAlmostEqual(observed, expected, places=12)


if __name__ == '__main__':
    unittest.main()
