"""Synthetic-only rolling feature tests; no research data, labels or models read."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import build_rolling_features as rolling


def make_fixture(folder):
    prep = folder / "prepared"
    root = folder / "rolling"
    for child in ("samples", "dimensions", "links"):
        (prep / child).mkdir(parents=True, exist_ok=True)
    root.mkdir()
    (root / "PROTOCOL.md").write_text("Synthetic rolling protocol v1\n")
    records, weather_rows, bridge_rows, declared_rows = [], [], [], []
    newer = {2019: "validation-new", 2020: "test-new", 2021: "future-21", 2022: "future-22", 2023: "future-23"}
    for year in range(2014, 2024):
        env = f"ENV_{year}"
        planting = pd.Timestamp(year=year, month=5, day=1)
        materials = ["a", "b", newer.get(year, "c")]
        for material in materials:
            sid = f"{year}-{material}"
            records.append({"sample_id": sid, "Env": env, "Hybrid": material, "Year": year,
                            "Experiment_Code": "TRAIN" if year <= 2018 else f"SITE_{year}",
                            "Latitude": 40.0 if year <= 2018 else 1000.0,
                            "Longitude": -90.0 if year <= 2018 else 1000.0,
                            "Planting_DOY": 121.0})
            if year > 2020:
                # Malformed future dates must be filtered before date conversion.
                bridge_rows.append([sid, env, "future-not-a-date", 1.0])
            elif sid == "2014-a":
                bridge_rows.extend([[sid, env, planting, 0.25],
                                    [sid, env, planting + pd.Timedelta(days=1), 0.75]])
            else:
                bridge_rows.append([sid, env, planting, 1.0])
        if year <= 2020:
            for day, rain in enumerate([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]):
                values = [float(year - 2013), 25.0, 15.0, rain, 5.0, 60.0, 2.0]
                weather_rows.append({"Env": env, "Date": planting + pd.Timedelta(days=day),
                                     **dict(zip(rolling.original.WEATHER, values))})
            dates = [planting] + ([planting + pd.Timedelta(days=1)] if year == 2014 else [])
            declared_rows.extend([[env, date, 3] for date in dates])
        else:
            weather_rows.append({"Env": env, "Date": "future-not-a-date",
                                 **{column: -999 for column in rolling.original.WEATHER}})
            declared_rows.append([env, "future-not-a-date", 3])
    pool = pd.DataFrame(records)
    pool.loc[pool.Year <= 2022].to_csv(prep / "samples/train.csv", index=False)
    pool.loc[pool.Year == 2023].to_csv(prep / "samples/validation.csv", index=False)
    # There is intentionally no samples/test.csv, targets directory or holdout.
    genotype = pd.DataFrame({
        "Hybrid": ["a", "b", "c", "validation-new", "test-new", "future-21", "future-22", "future-23"],
        "m1": [0.0, 1.0, 2.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        "m2": [np.nan, 2.0, 0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        "m3": [1.0, 2.0, 0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
    })
    genotype.to_csv(prep / "dimensions/genotypes.csv.gz", index=False)
    pd.DataFrame(weather_rows).to_csv(prep / "dimensions/weather_daily.csv.gz", index=False)
    pd.DataFrame(bridge_rows, columns=["sample_id", "Env", "Planting_Date", "planting_weight"]).to_csv(
        prep / "links/sample_plantings.csv.gz", index=False)
    pd.DataFrame(declared_rows, columns=["Env", "Planting_Date", "horizon_days"]).to_csv(
        prep / "links/weather_windows.csv", index=False)
    (prep / "manifest.json").write_text("{}\n")
    return prep, root, pool


class RollingFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="maize_rolling_feature_test_")
        self.addCleanup(self.temporary.cleanup)
        self.prep, self.root, self.pool = make_fixture(Path(self.temporary.name))
        self.patch_prep = patch.object(rolling, "PREP", self.prep)
        self.patch_horizons = patch.object(rolling, "HORIZONS", (0, 3))
        self.patch_prep.start()
        self.patch_horizons.start()
        self.addCleanup(self.patch_prep.stop)
        self.addCleanup(self.patch_horizons.stop)

    def build(self, mode="development"):
        return rolling.build_fold(2020, mode, n_components=2, root=self.root)

    def out(self, mode="development"):
        return self.root / "features/2020" / mode

    def freeze(self):
        path = self.root / "results/2020/development/frozen_selection.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"test_year": 2020, "frozen": True}))
        return path

    def test_all_prescribed_fold_year_boundaries_and_source_order(self):
        for year in rolling.TEST_YEARS:
            for mode in ("development", "final"):
                parts = rolling.select_fold_samples(self.pool, year, mode)
                last = year - (2 if mode == "development" else 1)
                self.assertEqual(set(parts["train"].Year), set(range(2014, last + 1)))
                self.assertEqual(set(parts["test"].Year), {year})
                self.assertEqual(parts["test"].sample_id.tolist(), self.pool.loc[self.pool.Year == year].sample_id.tolist())
                if mode == "development":
                    self.assertEqual(set(parts["validation"].Year), {year - 1})
                else:
                    self.assertEqual(len(parts["validation"]), 0)
                self.assertTrue(all((part.Year <= year).all() for part in parts.values()))

    def test_future_rows_filtered_before_genotype_and_weather_processing(self):
        genotype_call, window_years = [], []
        real_genotype = rolling.original.genotype_features
        real_window = rolling.original.aggregate_window
        real_read = pd.read_csv

        def checked_genotype(samples, genotype, n_components):
            genotype_call.append(set(genotype.Hybrid))
            self.assertEqual(set(samples["train"].Hybrid), {"a", "b", "c"})
            return real_genotype(samples, genotype, n_components)

        def checked_window(frame, planting, horizon, medians):
            window_years.append(int(frame.index.year.max()))
            self.assertLessEqual(frame.index.year.max(), 2020)
            return real_window(frame, planting, horizon, medians)

        def label_guard(path, *args, **kwargs):
            name = str(path)
            self.assertNotIn("/targets/", name)
            self.assertNotIn("/holdout/", name)
            self.assertFalse(name.endswith("samples/test.csv"))
            return real_read(path, *args, **kwargs)

        with patch.object(rolling.original, "genotype_features", side_effect=checked_genotype), \
                patch.object(rolling.original, "aggregate_window", side_effect=checked_window), \
                patch.object(pd, "read_csv", side_effect=label_guard):
            result = self.build()
        self.assertEqual(genotype_call, [{"a", "b", "c", "validation-new", "test-new"}])
        self.assertEqual(max(window_years), 2020)
        self.assertEqual(result["rows"], {"train": 15, "validation": 3, "test": 3})
        self.assertEqual(result["fit_years"], [2014, 2018])
        self.assertEqual(result["genotype"]["fit_hybrids"], ["a", "b", "c"])
        self.assertEqual(result["weather_fitting"]["3"]["medians"]["T2M"], 3.0)
        self.assertEqual(result["weather_fitting"]["3"]["training_distinct_env_dates"], 16)
        fit = joblib.load(self.out() / "preprocessing.joblib")
        np.testing.assert_array_equal(fit["genotype"]["imputer"].statistics_, [1, 1, 1])
        self.assertEqual(list(fit["static"]["encoder"].categories_[0]), ["TRAIN"])
        self.assertEqual(fit["static"]["medians"].Latitude, 40.0)

    def test_mixture_weights_nonlinear_summary_and_output_shapes(self):
        result = self.build()
        names = json.loads((self.out() / "W3_columns.json").read_text())["columns"]
        with np.load(self.out() / "W3.npz") as data:
            self.assertEqual(data["X_train"][0, names.index("longest_dry_spell")], 1.25)
            self.assertEqual(data["X_train"].shape, (15, len(names)))
            self.assertEqual(data["X_validation"].shape, (3, len(names)))
            self.assertEqual(data["X_test"].shape, (3, len(names)))
            self.assertTrue(all(np.isfinite(data[key]).all() for key in data.files))
        self.assertEqual(set(result["outputs"]), rolling.expected_outputs())
        self.assertEqual(result["test_year"], 2020)
        self.assertEqual(result["mode"], "development")
        self.assertTrue(all(result["checks"].values()))

    def test_final_requires_its_own_fold_freeze(self):
        with self.assertRaisesRegex(ValueError, "Freeze this fold"):
            self.build("final")
        wrong = self.root / "results/2021/development/frozen_selection.json"
        wrong.parent.mkdir(parents=True)
        wrong.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Freeze this fold"):
            self.build("final")
        self.assertFalse(self.out("final").exists())

    def test_final_refits_only_through_prior_year_and_keeps_validation_empty(self):
        freeze = self.freeze()
        result = self.build("final")
        self.assertEqual(result["fit_years"], [2014, 2019])
        self.assertEqual(result["rows"], {"train": 18, "validation": 0, "test": 3})
        self.assertEqual(result["genotype"]["fit_hybrids"], ["a", "b", "c", "validation-new"])
        self.assertIn(str(freeze.resolve()), result["input_hashes"])
        fit = joblib.load(self.out("final") / "preprocessing.joblib")
        self.assertEqual(fit["genotype"]["imputer"].statistics_[0], 25.75)
        for horizon in (0, 3):
            with np.load(self.out("final") / f"W{horizon}.npz") as data:
                self.assertEqual(data["X_validation"].shape, (0, data["X_train"].shape[1]))
                self.assertEqual(data["X_test"].shape[0], 3)

    def test_verified_complete_cache_skips_fitting_and_preserves_artifacts(self):
        first = self.build()
        before = {path.name: (path.stat().st_mtime_ns, rolling.original.digest(path)) for path in self.out().iterdir()}
        with patch.object(rolling.original, "genotype_features", side_effect=AssertionError("Should not refit")):
            second = self.build()
        self.assertEqual(first, second)
        after = {path.name: (path.stat().st_mtime_ns, rolling.original.digest(path)) for path in self.out().iterdir()}
        self.assertEqual(before, after)

    def test_changed_protocol_and_changed_source_are_rejected_without_overwrite(self):
        self.build()
        manifest = self.out() / "feature_manifest.json"
        original_hash = rolling.original.digest(manifest)
        protocol = self.root / "PROTOCOL.md"
        old_text = protocol.read_text()
        protocol.write_text(old_text + "Changed rule\n")
        with self.assertRaisesRegex(ValueError, "source/code/protocol/configuration changed"):
            self.build()
        self.assertEqual(rolling.original.digest(manifest), original_hash)
        protocol.write_text(old_text)
        source = self.prep / "samples/train.csv"
        frame = pd.read_csv(source)
        frame.loc[0, "Latitude"] = 99.0
        frame.to_csv(source, index=False)
        with self.assertRaisesRegex(ValueError, "source/code/protocol/configuration changed"):
            self.build()
        self.assertEqual(rolling.original.digest(manifest), original_hash)

    def test_changed_code_or_component_count_is_not_silently_reused(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "source/code/protocol/configuration changed"):
            rolling.build_fold(2020, "development", n_components=1, root=self.root)
        alternate = Path(self.temporary.name) / "different_feature_code.py"
        alternate.write_text("# Deliberately changed code fingerprint\n")
        with patch.object(rolling, "SCRIPT", alternate):
            with self.assertRaisesRegex(ValueError, "source/code/protocol/configuration changed"):
                self.build()

    def test_changed_final_freeze_is_rejected(self):
        freeze = self.freeze()
        self.build("final")
        before = rolling.original.digest(self.out("final") / "feature_manifest.json")
        freeze.write_text('{"test_year":2020,"changed":true}\n')
        with self.assertRaisesRegex(ValueError, "source/code/protocol/configuration changed"):
            self.build("final")
        self.assertEqual(rolling.original.digest(self.out("final") / "feature_manifest.json"), before)

    def test_tampered_output_is_rejected(self):
        self.build()
        artifact = self.out() / "W0.npz"
        artifact.write_bytes(artifact.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed: W0.npz"):
            self.build()
        self.assertTrue(artifact.read_bytes().endswith(b"changed"))

    def test_partial_output_or_other_build_lock_is_preserved(self):
        self.out().mkdir(parents=True)
        marker = self.out() / "partial.txt"
        marker.write_text("preserve this partial result")
        with self.assertRaisesRegex(ValueError, "Incomplete feature directory"):
            self.build()
        self.assertEqual(marker.read_text(), "preserve this partial result")
        lock = self.out().parent / ".development.lock"
        lock.write_text("other builder")
        with self.assertRaisesRegex(ValueError, "unresolved lock"):
            self.build()
        self.assertEqual(lock.read_text(), "other builder")

    def test_midbuild_source_change_does_not_publish_partial_features(self):
        real_generate = rolling.generate_fold

        def changing_source(*args, **kwargs):
            result = real_generate(*args, **kwargs)
            path = self.root / "PROTOCOL.md"
            path.write_text(path.read_text() + "Concurrent change\n")
            return result

        with patch.object(rolling, "generate_fold", side_effect=changing_source):
            with self.assertRaisesRegex(ValueError, "Inputs changed during build"):
                self.build()
        self.assertFalse(self.out().exists())
        self.assertEqual(len(list(self.out().parent.glob(".development-building-*"))), 1)
        self.assertFalse((self.out().parent / ".development.lock").exists())

    def test_invalid_fold_and_label_columns_are_rejected(self):
        for year, mode, components in ((2024, "development", 2), (2020, "bad", 2), (2020, "development", 0)):
            with self.assertRaises(ValueError):
                rolling.build_fold(year, mode, components, self.root)
        with self.assertRaisesRegex(ValueError, "Target/prediction columns"):
            rolling.select_fold_samples(self.pool.assign(y_true=1.0), 2020, "development")


if __name__ == "__main__":
    unittest.main()
