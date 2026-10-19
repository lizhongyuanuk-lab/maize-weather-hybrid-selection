"""Synthetic boundary checks; never read or modify the real input/output data.

Run from the project root: python -m unittest preparation.test_prepare_data -v
"""
import unittest

import numpy as np
import pandas as pd

from preparation import prepare_data as prep


class PreparationBoundaryTests(unittest.TestCase):
    @staticmethod
    def traits(dates, yields):
        return pd.DataFrame({
            "Env": ["E_2022"] * len(dates), "Hybrid": ["H1"] * len(dates),
            "Year": [2022] * len(dates), "Date_Planted": dates,
            "Yield_Mg_ha": yields,
        })

    @staticmethod
    def registry(samples, bridge):
        environments = samples[["Env", "Year"]].drop_duplicates().copy()
        environments["Experiment_Code"] = "E"
        environments["Latitude"] = 40.0
        environments["Longitude"] = -90.0
        coverage = pd.DataFrame([
            {"Env": env, "Planting_Date": date, "horizon_days": horizon,
             "calendar_complete": True}
            for env, date in bridge[["Env", "Planting_Date"]].drop_duplicates().itertuples(index=False, name=None)
            for horizon in prep.WINDOWS
        ])
        return prep.assemble_registry(
            samples, environments, samples[["Hybrid"]].drop_duplicates(), bridge, coverage,
        )

    def test_replicate_target_mean_ignores_missing_and_invalid_yields(self):
        source = self.traits(["2022-05-01"] * 6, [8.0, 12.0, np.nan, 0.0, -1.0, np.inf])
        samples, bridge, membership = prep.aggregate_traits(source)
        self.assertEqual(samples.loc[0, "Yield_Mg_ha"], 10.0)
        self.assertEqual(samples.loc[0, "raw_plot_count"], 6)
        self.assertEqual(samples.loc[0, "labeled_plot_count"], 2)
        self.assertEqual(bridge.loc[0, "plot_count"], 6)
        self.assertEqual(membership["target_status"].tolist(),
                         ["observed", "observed", "missing", "invalid", "invalid", "invalid"])

    def test_date_resolution_uses_unique_input_dates_not_yield_availability(self):
        source = pd.DataFrame({
            "Env": ["E_2022"] * 3 + ["F_2022"] * 3,
            "Hybrid": ["H1", "H1", "H2", "H3", "H3", "H4"],
            "Date_Planted": ["2022-05-01", None, None, "2022-05-01", "2022-05-08", None],
            "Yield_Mg_ha": [np.nan, 10.0, np.nan, 8.0, np.nan, 11.0],
        })
        original = prep.resolve_planting_dates(source)
        changed = source.copy()
        changed["Yield_Mg_ha"] = [100.0, np.nan, 99.0, np.nan, 200.0, np.nan]
        modified = prep.resolve_planting_dates(changed)
        pd.testing.assert_frame_equal(original[["Planting_Date", "planting_source"]],
                                      modified[["Planting_Date", "planting_source"]])
        self.assertEqual(original.loc[1, "planting_source"], "unique_env_hybrid")
        self.assertEqual(original.loc[2, "planting_source"], "unique_environment")
        self.assertTrue(pd.isna(original.loc[5, "Planting_Date"]))

    def test_planting_weights_use_all_plots_independent_of_yields(self):
        dates = ["2022-05-01", "2022-05-01", "2022-05-08"]
        _, first, _ = prep.aggregate_traits(self.traits(dates, [8.0, 10.0, 12.0]))
        _, second, _ = prep.aggregate_traits(self.traits(dates, [np.nan, 100.0, np.nan]))
        columns = ["sample_id", "Env", "Planting_Date", "plot_count", "total_plots", "planting_weight"]
        pd.testing.assert_frame_equal(first[columns], second[columns])
        np.testing.assert_allclose(first["planting_weight"], [2 / 3, 1 / 3])

    def test_mismatched_label_date_distribution_is_excluded(self):
        dates = ["2022-05-01", "2022-05-08"]
        samples, bridge, _ = prep.aggregate_traits(self.traits(dates, [10.0, np.nan]))
        registry = self.registry(samples, bridge)
        self.assertTrue(registry.loc[0, "planting_weight_target_mismatch"])
        self.assertFalse(registry.loc[0, "eligible_common"])
        self.assertEqual(registry.loc[0, "first_exclusion_reason"], "planting_target_weight_mismatch")
        balanced, balanced_bridge, _ = prep.aggregate_traits(self.traits(dates, [10.0, 12.0]))
        self.assertTrue(self.registry(balanced, balanced_bridge).loc[0, "eligible_common"])

    def test_weather_window_includes_day_zero_excludes_day_thirty(self):
        planting = pd.Timestamp("2022-05-01")
        dates = pd.date_range(planting - pd.Timedelta(days=1), periods=122)
        weather = pd.DataFrame({"Env": "E_2022", "Date": dates})
        for variable in prep.CORE_WEATHER:
            weather[variable] = 1.0
        weather.loc[weather["Date"].eq(planting + pd.Timedelta(days=30)), "T2M"] = np.nan
        bridge = pd.DataFrame({"Env": ["E_2022"], "Planting_Date": [planting]})
        windows = prep.window_coverage(bridge, weather).set_index("horizon_days")
        self.assertEqual(windows.loc[30, "present_days"], 30)
        self.assertEqual(windows.loc[30, "core_missing_days"], 0)
        self.assertEqual(windows.loc[60, "core_missing_days"], 1)
        without_day_zero = weather.loc[~weather["Date"].eq(planting)]
        missing = prep.window_coverage(bridge, without_day_zero).set_index("horizon_days")
        self.assertEqual(missing.loc[30, "missing_calendar_days"], 1)
        self.assertFalse(missing.loc[30, "calendar_complete"])

    def test_test_input_eligibility_does_not_require_observed_labels(self):
        samples, bridge, _ = prep.aggregate_traits(self.traits(["2022-05-01"], [10.0]))
        samples["Env"] = "E_2024"
        samples["Year"] = 2024
        samples["sample_id"] = prep.sample_ids(samples)
        samples["Planting_Date_Min"] = pd.Timestamp("2024-05-01")
        samples["Planting_Date_Max"] = pd.Timestamp("2024-05-01")
        samples["Planting_DOY"] = pd.Timestamp("2024-05-01").dayofyear
        bridge["sample_id"] = samples.loc[0, "sample_id"]
        bridge["Env"] = "E_2024"
        bridge["Planting_Date"] = pd.Timestamp("2024-05-01")
        samples["labeled_plot_count"] = np.nan
        samples["Yield_Mg_ha"] = np.nan
        without_labels = self.registry(samples, bridge)
        samples["labeled_plot_count"] = 1
        samples["Yield_Mg_ha"] = 100.0
        with_labels = self.registry(samples, bridge)
        self.assertTrue(without_labels.loc[0, "eligible_common"])
        self.assertEqual(without_labels.loc[0, "split"], "test")
        pd.testing.assert_series_equal(without_labels["eligible_common"], with_labels["eligible_common"])

    def test_split_boundaries_and_out_of_range_years(self):
        splits = prep.assign_split(pd.Series([2014, 2022, 2023, 2024]))
        self.assertEqual(splits.tolist(), ["train", "train", "validation", "test"])
        for invalid in (2013, 2025, np.nan):
            with self.subTest(year=invalid), self.assertRaisesRegex(ValueError, "Unexpected sample year"):
                prep.assign_split(pd.Series([invalid]))

    def test_duplicate_or_missing_join_keys_are_rejected(self):
        for keys, message in ((["E_2022", "E_2022"], "Duplicate"),
                              (["E_2022", None], "Missing")):
            with self.subTest(keys=keys), self.assertRaisesRegex(ValueError, message):
                prep.unique(pd.DataFrame({"Env": keys}), ["Env"], "metadata")


if __name__ == "__main__":
    unittest.main()
