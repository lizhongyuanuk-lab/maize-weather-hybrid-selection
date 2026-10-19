"""Synthetic-only segmented-feature tests: never train or read research data."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import build_segmented_features as segment


def weather_frame(planting, days=122):
    values = np.tile([20., 25., 15., 0., 5., 60., 2.], (days, 1))
    values[:, 0] += np.arange(days) / 10
    values[29:31, 3] = 5.
    return pd.DataFrame(values, index=pd.date_range(planting, periods=days), columns=segment.original.WEATHER)


def make_fixture(folder):
    prep, cumulative, root = folder / "prepared", folder / "cumulative", folder / "segmented"
    (prep / "dimensions").mkdir(parents=True)
    (prep / "links").mkdir()
    root.mkdir()
    (root / "PROTOCOL.md").write_text("Synthetic segmented-weather protocol\n")
    rows, bridges, weather, daily = [], [], [], {}
    for year in range(2014, 2021):
        env, planting = f"ENV_{year}", pd.Timestamp(year=year, month=5, day=1)
        for material in ("a", "b"):
            sid = f"{year}-{material}"
            rows.append({"sample_id": sid, "Env": env, "Hybrid": material, "Year": year,
                         "Experiment_Code": "TRAIN" if year < 2019 else f"SITE_{year}",
                         "Latitude": 40., "Longitude": -90., "Planting_DOY": 121.})
            if sid == "2014-a":
                bridges.extend([[sid, env, planting, .25], [sid, env, planting + pd.Timedelta(days=1), .75]])
            else:
                bridges.append([sid, env, planting, 1.])
        frame = weather_frame(planting)
        frame.loc[frame.index[[0, 10]], "T2M"] = np.nan
        daily[env] = frame
        weather.append(frame.rename_axis("Date").reset_index().assign(Env=env))
    pool = pd.DataFrame(rows)
    bridge = pd.DataFrame(bridges, columns=["sample_id", "Env", "Planting_Date", "planting_weight"])
    # Future rows are deliberately invalid and must be excluded before parsing.
    weather.append(pd.DataFrame([{"Env": "FUTURE", "Date": "bad-future-date", **{n: -999 for n in segment.original.WEATHER}}]))
    pd.concat(weather, ignore_index=True).to_csv(prep / "dimensions/weather_daily.csv.gz", index=False)
    pd.concat([bridge, pd.DataFrame([["future", "FUTURE", "bad-future-date", 1.]], columns=bridge.columns)]).to_csv(
        prep / "links/sample_plantings.csv.gz", index=False)
    for mode in ("development", "final"):
        source = cumulative / "features/2020" / mode
        source.mkdir(parents=True)
        train_end = 2018 if mode == "development" else 2019
        samples = {"train": pool.loc[pool.Year <= train_end],
                   "validation": pool.loc[pool.Year == 2019] if mode == "development" else pool.iloc[:0],
                   "test": pool.loc[pool.Year == 2020]}
        base = {}
        for split, frame in samples.items():
            frame.to_csv(source / f"{split}_samples.csv", index=False)
            base[split] = np.column_stack([frame.Year.to_numpy() - 2014, (frame.Hybrid == "b").to_numpy()]).astype(np.float32)
        (source / "preprocessing.joblib").write_bytes(b"synthetic pre-fitted artifact: must only byte-copy, never load or fit")
        names = ["PC001", "PC002"]
        np.savez_compressed(source / "W0.npz", **{f"X_{s}": a for s, a in base.items()})
        (source / "W0_columns.json").write_text(json.dumps({"columns": names, "weather_columns": [], "horizon_days": 0}))
        fitting = {}
        pairs = bridge[["Env", "Planting_Date"]].drop_duplicates().sort_values(["Env", "Planting_Date"])
        for horizon in segment.HORIZONS[1:]:
            medians, count = segment.original.training_weather_medians(bridge, daily, set(samples["train"].sample_id), horizon)
            records = [[env, planting, *segment.original.aggregate_window(daily[env], planting, horizon, medians)[0]]
                       for env, planting in pairs.itertuples(index=False, name=None)]
            pair = pd.DataFrame(records, columns=["Env", "Planting_Date", *segment.original.WEATHER_NAMES])
            linked = bridge.merge(pair, on=["Env", "Planting_Date"], validate="many_to_one")
            weighted = linked[segment.original.WEATHER_NAMES].multiply(linked.planting_weight, axis=0)
            weighted["sample_id"] = linked.sample_id
            by_sample = weighted.groupby("sample_id")[segment.original.WEATHER_NAMES].sum()
            arrays = {f"X_{s}": np.column_stack([base[s], by_sample.loc[f.sample_id].to_numpy()]).astype(np.float32) for s, f in samples.items()}
            np.savez_compressed(source / f"W{horizon}.npz", **arrays)
            (source / f"W{horizon}_columns.json").write_text(json.dumps({"columns": names + segment.original.WEATHER_NAMES,
                "weather_columns": segment.original.WEATHER_NAMES, "horizon_days": horizon}))
            fitting[str(horizon)] = {"medians": medians.to_dict(), "training_distinct_env_dates": count}
        code = segment.PROJECT / "modeling/build_rolling_features.py"
        manifest = {"mode": mode, "test_year": 2020, "fit_years": [2014, train_end],
                    "rows": {s: len(f) for s, f in samples.items()}, "checks": {"synthetic": True},
                    "weather_fitting": fitting, "genotype": {"components": 2, "fit_unique_hybrids": 2},
                    "input_hashes": segment.fingerprints([prep / "dimensions/weather_daily.csv.gz", prep / "links/sample_plantings.csv.gz"]),
                    "script_sha256": segment.original.digest(code),
                    "outputs": {p.name: segment.original.digest(p) for p in source.iterdir()}}
        (source / "feature_manifest.json").write_text(json.dumps(manifest, allow_nan=False))
    return prep, cumulative, root


class SegmentWindowTests(unittest.TestCase):
    def setUp(self):
        self.planting = pd.Timestamp("2020-05-01")
        self.daily = weather_frame(self.planting)
        self.medians = pd.Series(dict(zip(segment.original.WEATHER, [19., 25., 15., 0., 5., 60., 2.])))

    def test_clipping_before_interpolation_and_single_block_exactness(self):
        changed = self.daily.copy()
        changed.iloc[60:] = 9999.
        changed.loc[self.planting - pd.Timedelta(days=1)] = -999.
        np.testing.assert_array_equal(segment.aggregate_segmented_window(self.daily, self.planting, 60, self.medians)[0],
                                      segment.aggregate_segmented_window(changed, self.planting, 60, self.medians)[0])
        np.testing.assert_array_equal(segment.aggregate_segmented_window(self.daily, self.planting, 30, self.medians)[0],
                                      segment.original.aggregate_window(self.daily, self.planting, 30, self.medians)[0])

    def test_interpolation_crosses_internal_block_boundary_not_horizon_edge(self):
        frame = self.daily.copy()
        frame.T2M = 10.
        frame.loc[frame.index[30], "T2M"] = 30.
        frame.loc[frame.index[29], "T2M"] = np.nan
        sixty, diagnostics = segment.aggregate_segmented_window(frame, self.planting, 60, self.medians)
        thirty, short_diag = segment.aggregate_segmented_window(frame, self.planting, 30, self.medians)
        self.assertAlmostEqual(sixty[0], (29 * 10 + 20) / 30)
        self.assertAlmostEqual(thirty[0], (29 * 10 + 19) / 30)
        self.assertEqual(diagnostics["interpolated_cells"], 1)
        self.assertEqual(short_diag["median_filled_cells"], 1)

    def test_outer_missing_uses_inherited_median_and_global_quality(self):
        frame = self.daily.copy()
        frame.loc[frame.index[[0, 59]], "T2M"] = np.nan
        values, diagnostics = segment.aggregate_segmented_window(frame, self.planting, 60, self.medians)
        self.assertEqual(diagnostics["median_filled_cells"], 2)
        self.assertEqual(diagnostics["interpolated_cells"], 0)
        self.assertEqual(values[-9], 2 / 60)
        self.assertEqual(values[-1], 58)
        for horizon, count in ((30, 30), (60, 51), (90, 72), (120, 93)):
            segmented, _ = segment.aggregate_segmented_window(frame, self.planting, horizon, self.medians)
            cumulative, _ = segment.original.aggregate_window(frame, self.planting, horizon, self.medians)
            self.assertEqual(len(segmented), count)
            np.testing.assert_array_equal(segmented[-9:], cumulative[-9:])


class SegmentedBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="maize_segmented_feature_test_")
        self.addCleanup(self.temporary.cleanup)
        self.prep, self.cumulative, self.root = make_fixture(Path(self.temporary.name))
        self.patch_prep = patch.object(segment, "PREP", self.prep)
        self.patch_prep.start()
        self.addCleanup(self.patch_prep.stop)

    def source(self, mode="development"):
        return self.cumulative / "features/2020" / mode

    def out(self, mode="development"):
        return self.root / "features/2020" / mode

    def build(self, mode="development"):
        return segment.build_fold(2020, mode, self.root, self.cumulative)

    def freeze(self):
        path = self.root / "results/2020/development/frozen_selection.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"test_year":2020,"synthetic":true}')
        return path

    def test_exact_reuse_all_quality_base_and_weighted_nonlinear_features(self):
        before = {p.name: (p.stat().st_mtime_ns, segment.original.digest(p)) for p in self.source().iterdir()}
        real_read = pd.read_csv
        def guard(path, *args, **kwargs):
            self.assertNotIn("/targets/", str(path))
            self.assertNotIn("/holdout/", str(path))
            self.assertNotIn("genotypes", str(path))
            return real_read(path, *args, **kwargs)
        with patch.object(segment.original, "genotype_features", side_effect=AssertionError("No PCA fitting")), \
                patch.object(segment.original, "static_features", side_effect=AssertionError("No static fitting")), \
                patch.object(segment.original, "training_weather_medians", side_effect=AssertionError("No new median fitting")), \
                patch.object(pd, "read_csv", side_effect=guard):
            manifest = self.build()
        for name in segment.COPIED:
            self.assertEqual((self.out() / name).read_bytes(), (self.source() / name).read_bytes())
        self.assertEqual(manifest["weather_feature_count_by_horizon"], {"30": 30, "60": 51, "90": 72, "120": 93})
        self.assertEqual(manifest["representation"], "segmented_30d")
        self.assertTrue(all(manifest["checks"].values()))
        for horizon in segment.HORIZONS[2:]:
            with np.load(self.out() / f"W{horizon}.npz") as new, np.load(self.source() / f"W{horizon}.npz") as old, np.load(self.source() / "W0.npz") as base:
                for split in segment.SPLITS:
                    key = f"X_{split}"
                    np.testing.assert_array_equal(new[key][:, :2], base[key])
                    np.testing.assert_array_equal(new[key][:, -9:], old[key][:, -9:])
                columns = json.loads((self.out() / f"W{horizon}_columns.json").read_text())["columns"]
                self.assertEqual(new["X_train"][0, columns.index("D000_030_longest_dry_spell")], 28.25)
        self.assertEqual(before, {p.name: (p.stat().st_mtime_ns, segment.original.digest(p)) for p in self.source().iterdir()})

    def test_final_gate_and_empty_validation(self):
        with self.assertRaisesRegex(ValueError, "Freeze this segmented fold"):
            self.build("final")
        freeze = self.freeze()
        manifest = self.build("final")
        self.assertEqual(manifest["fit_years"], [2014, 2019])
        self.assertEqual(manifest["rows"]["validation"], 0)
        self.assertIn(str(freeze.resolve()), manifest["input_hashes"])
        for horizon in segment.HORIZONS:
            with np.load(self.out("final") / f"W{horizon}.npz") as data:
                self.assertEqual(data["X_validation"].shape[0], 0)
                self.assertEqual(data["X_validation"].shape[1], data["X_train"].shape[1])

    def test_complete_cache_is_no_compute_and_changed_protocol_is_rejected(self):
        first = self.build()
        before = {p.name: (p.stat().st_mtime_ns, segment.original.digest(p)) for p in self.out().iterdir()}
        with patch.object(segment, "generate_fold", side_effect=AssertionError("No recomputation")):
            self.assertEqual(first, self.build())
        self.assertEqual(before, {p.name: (p.stat().st_mtime_ns, segment.original.digest(p)) for p in self.out().iterdir()})
        (self.root / "PROTOCOL.md").write_text("changed protocol")
        with self.assertRaisesRegex(ValueError, "source/code/protocol/configuration changed"):
            self.build()

    def test_changed_source_and_output_are_rejected(self):
        self.build()
        artifact = self.out() / "W60.npz"
        artifact.write_bytes(artifact.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "Cached feature artifact changed"):
            self.build()
        source = self.source() / "W0.npz"
        source.write_bytes(source.read_bytes() + b"source tamper")
        with self.assertRaisesRegex(ValueError, "Cumulative feature output changed"):
            self.build()

    def test_changed_weather_input_and_final_freeze_are_rejected(self):
        freeze = self.freeze()
        self.build("final")
        freeze.write_text('{"test_year":2020,"changed":true}')
        with self.assertRaisesRegex(ValueError, "source/code/protocol/configuration changed"):
            self.build("final")
        weather = self.prep / "dimensions/weather_daily.csv.gz"
        weather.write_bytes(weather.read_bytes() + b"changed source")
        with self.assertRaisesRegex(ValueError, "Cumulative source input changed"):
            self.build()

    def test_inconsistent_global_quality_fails_even_if_source_output_rehashed(self):
        source = self.source()
        with np.load(source / "W60.npz") as data:
            arrays = {name: data[name] for name in data.files}
        arrays["X_test"][0, -1] -= 1
        np.savez_compressed(source / "W60.npz", **arrays)
        manifest = json.loads((source / "feature_manifest.json").read_text())
        manifest["outputs"]["W60.npz"] = segment.original.digest(source / "W60.npz")
        (source / "feature_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Whole-horizon quality fields changed"):
            self.build()
        self.assertFalse(self.out().exists())

    def test_wrong_W30_values_fail_exact_recomputation_even_if_rehashed(self):
        source = self.source()
        with np.load(source / "W30.npz") as data:
            arrays = {name: data[name] for name in data.files}
        arrays["X_test"][0, 2] += .25
        np.savez_compressed(source / "W30.npz", **arrays)
        manifest = json.loads((source / "feature_manifest.json").read_text())
        manifest["outputs"]["W30.npz"] = segment.original.digest(source / "W30.npz")
        (source / "feature_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "W30 exact float32 equivalence failed"):
            self.build()
        self.assertFalse(self.out().exists())
        self.assertEqual(len(list(self.out().parent.glob(".development-building-*"))), 1)

    def test_partial_cache_and_foreign_lock_preserved(self):
        self.out().mkdir(parents=True)
        marker = self.out() / "partial.txt"
        marker.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "Incomplete feature directory"):
            self.build()
        self.assertEqual(marker.read_text(), "preserve")
        lock = self.out().parent / ".development.lock"
        lock.write_text("foreign lock")
        with self.assertRaisesRegex(ValueError, "unresolved lock"):
            self.build()
        self.assertEqual(lock.read_text(), "foreign lock")

    def test_midbuild_change_never_publishes_and_removes_only_owned_lock(self):
        real_generate = segment.generate_fold
        def changed(*args, **kwargs):
            result = real_generate(*args, **kwargs)
            (self.root / "PROTOCOL.md").write_text("changed while building")
            return result
        with patch.object(segment, "generate_fold", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Inputs changed during build"):
                self.build()
        self.assertFalse(self.out().exists())
        self.assertFalse((self.out().parent / ".development.lock").exists())

    def test_source_adapter_and_invalid_parameters(self):
        self.assertEqual(segment.source_feature_dir(2024, "final", self.cumulative), segment.PROJECT / "modeling/features/final")
        self.assertEqual(segment.source_feature_dir(2020, "development", self.cumulative), self.source().resolve())
        for year, mode in ((2019, "development"), (2025, "final"), (2020, "wrong")):
            with self.assertRaises(ValueError):
                segment.source_feature_dir(year, mode, self.cumulative)


if __name__ == "__main__":
    unittest.main()
