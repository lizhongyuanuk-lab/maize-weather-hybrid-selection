"""Synthetic regression tests for preprocessing boundaries; no real targets read.

Run from project root:
    PYTHONPATH=.runtime:./modeling python -m unittest modeling/test_features.py -v
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import build_features as features


def daily(start="2022-05-01", periods=8):
    defaults = [20.0, 25.0, 15.0, 0.0, 5.0, 60.0, 2.0]
    return pd.DataFrame(
        {name: value for name, value in zip(features.WEATHER, defaults)},
        index=pd.date_range(start, periods=periods),
    )


def feature_dict(values):
    return dict(zip(features.WEATHER_NAMES, values))


def genotype_fixture():
    return pd.DataFrame({
        "Hybrid": ["a", "b", "c", "new"],
        "m1": [0.0, 1.0, 2.0, 1000.0],
        "m2": [np.nan, 2.0, 0.0, -1000.0],
        "m3": [1.0, 2.0, 0.0, 999.0],
    })


def material_samples():
    return {
        "train": pd.DataFrame({"Hybrid": ["a", "a", "b", "c"]}),
        "validation": pd.DataFrame({"Hybrid": ["new"]}),
        "test": pd.DataFrame({"Hybrid": ["new"]}),
    }


def write_tiny_inputs(root):
    """Create a self-contained no-label fixture under TemporaryDirectory."""
    prep = root / "preparation/outputs"
    for child in ("samples", "dimensions", "links"):
        (prep / child).mkdir(parents=True, exist_ok=True)
    frames = {
        "train": pd.DataFrame({
            "sample_id": ["t-a", "t-b", "t-c"],
            "Env": ["train_2022"] * 3, "Hybrid": ["a", "b", "c"],
            "Year": [2022] * 3, "Experiment_Code": ["known"] * 3,
            "Latitude": [40.0] * 3, "Longitude": [-90.0] * 3,
            "Planting_DOY": [121.75, 121, 121],
        }),
        "validation": pd.DataFrame({
            "sample_id": ["v-a"], "Env": ["validation_2023"], "Hybrid": ["a"],
            "Year": [2023], "Experiment_Code": ["known"],
            "Latitude": [41.0], "Longitude": [-91.0], "Planting_DOY": [121],
        }),
        "test": pd.DataFrame({
            "sample_id": ["x-new"], "Env": ["test_2024"], "Hybrid": ["new"],
            "Year": [2024], "Experiment_Code": ["unknown"],
            "Latitude": [np.nan], "Longitude": [np.nan], "Planting_DOY": [122],
        }),
    }
    for split, frame in frames.items():
        frame.to_csv(prep / f"samples/{split}.csv", index=False)
    genotype_fixture().to_csv(prep / "dimensions/genotypes.csv.gz", index=False)
    weather_frames = []
    links = []
    for split, frame in frames.items():
        year = int(frame.Year.iloc[0])
        planting = pd.Timestamp(year=year, month=5, day=1)
        env = frame.Env.iloc[0]
        weather = daily(planting, 6)
        weather["PRECTOTCORR"] = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0]
        weather.index.name = "Date"
        weather = weather.reset_index()
        weather["Env"] = env
        weather_frames.append(weather)
        for sid in frame.sample_id:
            if sid == "t-a":
                links.extend([
                    [sid, env, planting, 0.25],
                    [sid, env, planting + pd.Timedelta(days=1), 0.75],
                ])
            else:
                links.append([sid, env, planting, 1.0])
    pd.concat(weather_frames).to_csv(prep / "dimensions/weather_daily.csv.gz", index=False)
    bridge = pd.DataFrame(links, columns=["sample_id", "Env", "Planting_Date", "planting_weight"])
    bridge.to_csv(prep / "links/sample_plantings.csv.gz", index=False)
    declared = bridge[["Env", "Planting_Date"]].drop_duplicates()
    declared["horizon_days"] = 3
    declared.to_csv(prep / "links/weather_windows.csv", index=False)
    (prep / "manifest.json").write_text("{}\n")
    return prep


class FeatureBoundaryTests(unittest.TestCase):
    def test_outside_window_never_changes_features(self):
        frame = daily()
        planting = frame.index[1]
        medians = frame.median()
        frame.loc[frame.index[3], "T2M"] = np.nan
        expected, expected_audit = features.aggregate_window(frame, planting, 3, medians)
        changed = frame.copy()
        changed.loc[changed.index < planting, :] = -1e8
        changed.loc[changed.index >= planting + pd.Timedelta(days=3), :] = 1e8
        actual, audit = features.aggregate_window(changed, planting, 3, medians)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(audit, expected_audit)
        self.assertEqual(audit["end_exclusive"], "2022-05-05")

    def test_interior_interpolation_and_edge_median_no_extrapolation(self):
        frame = daily(periods=7)
        frame["T2M"] = [900, np.nan, 10, np.nan, 30, np.nan, 800]
        medians = daily().median()
        medians["T2M"] = 7
        values, audit = features.aggregate_window(frame, frame.index[1], 5, medians)
        got = feature_dict(values)
        # Independently expected sequence: [7, 10, 20, 30, 7].
        self.assertAlmostEqual(got["temperature_mean"], 14.8)
        self.assertAlmostEqual(got["temperature_sd"], np.std([7, 10, 20, 30, 7]))
        self.assertEqual(audit["interpolated_cells"], 1)
        self.assertEqual(audit["median_filled_cells"], 2)
        self.assertEqual(got["missing_fraction_T2M"], 0.6)
        self.assertEqual(got["available_weather_days"], 2)

    def test_agronomic_thresholds_gdd_and_dry_spell(self):
        frame = daily(periods=7)
        frame["T2M_MAX"] = [29, 30, 35, 40, 8, 15, 20]
        frame["T2M_MIN"] = [8, 12, 20, 25, 0, 5, 10]
        frame["PRECTOTCORR"] = [0, 0.5, 1, 10, 0, 0, 0]
        values, _ = features.aggregate_window(frame, frame.index[0], 7, daily().median())
        got = feature_dict(values)
        self.assertEqual(len(values), 30)
        for key, expected in {
            "hot30_days": 3, "hot35_days": 2, "gdd_celsius_sum": 60.5,
            "precipitation_sum": 11.5, "wet_days": 2, "heavy_rain_days": 1,
            "dry_days": 5, "longest_dry_spell": 3, "wet_day_intensity": 5.5,
        }.items():
            self.assertAlmostEqual(got[key], expected, msg=key)

    def test_weather_median_uses_distinct_training_env_dates_only(self):
        frame = daily(periods=4)
        frame["T2M"] = [1.0, 100.0, 2.0, 999.0]
        other = daily(periods=4)
        other["T2M"] = -999.0
        bridge = pd.DataFrame({
            "sample_id": ["a", "b", "a", "v"],
            "Env": ["A", "A", "A", "V"],
            "Planting_Date": [frame.index[0], frame.index[0], frame.index[1], frame.index[0]],
        })
        got, count = features.training_weather_medians(bridge, {"A": frame, "V": other}, {"a", "b"}, 2)
        self.assertEqual(count, 3)
        self.assertEqual(got["T2M"], 2.0)
        other.loc[:, :] = 1e9
        frame.loc[frame.index[3], :] = -1e9
        changed, changed_count = features.training_weather_medians(bridge, {"A": frame, "V": other}, {"a", "b"}, 2)
        pd.testing.assert_series_equal(got, changed)
        self.assertEqual(count, changed_count)

    def test_genotype_fit_ignores_validation_values_and_train_repeats(self):
        samples = material_samples()
        genotype = genotype_fixture()
        projection, fitted, metadata = features.genotype_features(samples, genotype, 2)
        self.assertEqual(metadata["fit_unique_hybrids"], 3)
        self.assertEqual(metadata["fit_hybrids"], ["a", "b", "c"])
        np.testing.assert_allclose(fitted["imputer"].statistics_, [1, 1, 1])
        changed = genotype.copy()
        changed.loc[changed.Hybrid == "new", ["m1", "m2", "m3"]] = 1e7
        unique = {**samples, "train": samples["train"].drop_duplicates().copy()}
        _, fitted_after, _ = features.genotype_features(unique, changed, 2)
        np.testing.assert_allclose(fitted["pca"].components_, fitted_after["pca"].components_)
        np.testing.assert_allclose(fitted["pca"].mean_, fitted_after["pca"].mean_)
        np.testing.assert_array_equal(projection["train"][0], projection["train"][1])

    def test_static_unknown_site_train_medians_and_empty_validation(self):
        train = pd.DataFrame({
            "Latitude": [10.0, 30.0], "Longitude": [100.0, 140.0],
            "Planting_DOY": [100.0, 120.0], "Experiment_Code": ["A", "B"],
        })
        test = pd.DataFrame({
            "Latitude": [np.nan], "Longitude": [np.nan], "Planting_DOY": [np.nan],
            "Experiment_Code": ["NEW"],
        })
        actual, fitted, names = features.static_features({"train": train, "validation": train.iloc[:0], "test": test})
        np.testing.assert_array_equal(actual["test"][0], [20, 120, 110, 1, 1, 0, 0])
        self.assertEqual(actual["validation"].shape, (0, len(names)))
        self.assertEqual(list(fitted["encoder"].categories_[0]), ["A", "B"])

    def test_duplicate_genotype_keys_fail_explicitly(self):
        duplicate = pd.concat([genotype_fixture(), genotype_fixture().iloc[:1]])
        with self.assertRaisesRegex(ValueError, "Duplicate genotype keys"):
            features.genotype_features(material_samples(), duplicate, 2)

    def test_complete_development_path_weights_nonlinear_features_after_aggregation(self):
        with tempfile.TemporaryDirectory(prefix="maize_feature_test_") as folder:
            root = Path(folder)
            prep = write_tiny_inputs(root)
            with patch.object(features, "PROJECT", root), patch.object(features, "PREP", prep), patch.object(features, "HORIZONS", (0, 3)):
                metadata = features.build("development", n_components=2)
            out = root / "modeling/features/development"
            columns = json.loads((out / "W3_columns.json").read_text())["columns"]
            with np.load(out / "W3.npz") as arrays:
                # Date 1 dry-run=2; date 2 dry-run=1. Weight summaries, not trajectories.
                self.assertEqual(arrays["X_train"][0, columns.index("longest_dry_spell")], 1.25)
                self.assertEqual(arrays["X_train"].shape[0], 3)
                self.assertEqual(arrays["X_validation"].shape[0], 1)
            self.assertTrue(all(metadata["checks"].values()))
            self.assertFalse(any("targets" in key for key in metadata["input_hashes"]))

    def test_final_path_has_empty_validation_and_includes_2023_fit(self):
        with tempfile.TemporaryDirectory(prefix="maize_feature_final_test_") as folder:
            root = Path(folder)
            prep = write_tiny_inputs(root)
            frozen = root / "modeling/results/development/frozen_selection.json"
            frozen.parent.mkdir(parents=True)
            frozen.write_text("{}\n")
            with patch.object(features, "PROJECT", root), patch.object(features, "PREP", prep), patch.object(features, "HORIZONS", (0, 3)):
                metadata = features.build("final", n_components=2)
            self.assertEqual(metadata["fit_years"][1], 2023)
            self.assertEqual(metadata["rows"], {"train": 4, "validation": 0, "test": 1})
            for horizon in (0, 3):
                with np.load(root / f"modeling/features/final/W{horizon}.npz") as arrays:
                    self.assertEqual(arrays["X_validation"].shape, (0, arrays["X_train"].shape[1]))
                    self.assertTrue(np.isfinite(arrays["X_test"]).all())

    def test_final_path_requires_preexisting_freeze(self):
        with tempfile.TemporaryDirectory(prefix="maize_feature_lock_test_") as folder:
            with patch.object(features, "PROJECT", Path(folder)):
                with self.assertRaisesRegex(ValueError, "Freeze development"):
                    features.build("final", n_components=2)


if __name__ == "__main__":
    unittest.main()
