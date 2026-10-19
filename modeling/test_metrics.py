"""Synthetic tests: no real or held-out labels are read."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from metrics import (evaluate_predictions, fractional_top_k, top20_recall,
                     paired_difference, window_analysis, correlation)
from run_experiments import baseline_predictions, development_targets


class MetricTests(unittest.TestCase):
    def test_macro_is_environment_equal(self):
        data = pd.DataFrame({"Env": ["a"]*9+["b"], "y_true": [0]*10,
                             "prediction": [1]*9+[3]})
        summary, _ = evaluate_predictions(data)
        self.assertAlmostEqual(summary["macro_rmse"], 2)
        self.assertAlmostEqual(summary["pooled_rmse"], np.sqrt(1.8))
        self.assertAlmostEqual(summary["macro_bias"], 2)

    def test_constant_prediction_has_chance_recall_and_undefined_correlation(self):
        self.assertAlmostEqual(top20_recall(np.arange(10), np.ones(10)), .2)
        self.assertTrue(np.isnan(correlation(np.arange(10), np.ones(10))))
        self.assertTrue(np.isnan(correlation([1], [1])))

    def test_fractional_ties_and_permutation_invariance(self):
        values = [10, 9, 9, 9, 1]
        np.testing.assert_allclose(fractional_top_k(values, 2), [1, 1/3, 1/3, 1/3, 0])
        observed, predicted = np.array([2, 2, 3, 3, 4, 4]), np.array([4, 5, 5, 3, 2, 2])
        order = [5, 0, 3, 1, 4, 2]
        self.assertEqual(top20_recall(observed, predicted), top20_recall(observed[order], predicted[order]))

    def test_perfect_ranking_and_reverse(self):
        values = np.arange(10)
        self.assertEqual(top20_recall(values, values), 1)
        self.assertEqual(top20_recall(values, -values), 0)
        self.assertAlmostEqual(correlation(values, -values, ranked=True), -1)

    def test_paired_bootstrap_sign_order_and_reproducibility(self):
        reference = pd.DataFrame({"Env": ["b", "a", "c"], "rmse": [3., 2., 4.]})
        candidate = pd.DataFrame({"Env": ["a", "c", "b"], "rmse": [1., 3., 2.]})
        first = paired_difference(reference, candidate)
        self.assertEqual(first, paired_difference(reference, candidate))
        self.assertEqual(first["estimate"], -1)
        self.assertEqual(first["ci95_high"], -1)
        with self.assertRaises(ValueError):
            paired_difference(reference, candidate.iloc[:2])

    def test_sufficiency_rule(self):
        windows = {h: pd.DataFrame({"Env": ["a", "b"], "rmse": [v, v]})
                   for h, v in zip([0, 30, 60, 90, 120], [10, 9, 8.2, 8.1, 8])}
        _, result = window_analysis(windows)
        self.assertEqual(result["earliest_sufficient_window"], 60)
        for h in [30, 60, 90, 120]:
            windows[h]["rmse"] = [9.9, 9.9]
        _, result = window_analysis(windows)
        self.assertIsNone(result["earliest_sufficient_window"])
        windows[120]["rmse"] = [11, 11]
        table, result = window_analysis(windows)
        self.assertTrue(table.gain_retention.isna().all())
        self.assertFalse(result["W120_gain_supported"])

    def test_nan_input_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_predictions(pd.DataFrame({"Env": ["a"], "y_true": [1], "prediction": [np.nan]}))

    def test_historical_site_baseline_unknown_site_falls_back_to_training_mean(self):
        train = pd.DataFrame({"Experiment_Code": ["a", "a", "b"]})
        future = pd.DataFrame({"Experiment_Code": ["a", "b", "unknown"]})
        prediction = baseline_predictions("HistoricalSiteMean", train, np.array([1., 3., 8.]), future)
        np.testing.assert_allclose(prediction, [2., 8., 4.])

    def test_development_targets_key_alignment_and_no_holdout_read(self):
        samples = {"train": pd.DataFrame({"sample_id": ["b", "a"]}),
                   "validation": pd.DataFrame({"sample_id": ["c"]})}
        sources = [pd.DataFrame({"sample_id": ["a", "b"], "Yield_Mg_ha": [1., 2.]}),
                   pd.DataFrame({"sample_id": ["c"], "Yield_Mg_ha": [3.]})]
        with patch("run_experiments.pd.read_csv", side_effect=sources) as reader:
            values = development_targets(samples)
        np.testing.assert_array_equal(values["train"], [2., 1.])
        np.testing.assert_array_equal(values["validation"], [3.])
        self.assertEqual(reader.call_count, 2)
        self.assertTrue(all("holdout" not in str(call.args[0]) for call in reader.call_args_list))


if __name__ == "__main__":
    unittest.main()
