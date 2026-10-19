"""Synthetic rolling-runner tests; no production training or held-out data read."""
import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_rolling_experiments as rolling


class TinyEstimator:
    fit_calls = 0

    def fit(self, x, y):
        TinyEstimator.fit_calls += 1
        self.mean = float(np.mean(y))
        self.coef_ = np.zeros(x.shape[1])
        self.feature_importances_ = np.ones(x.shape[1])/x.shape[1]
        self.named_steps = {"ridge": self}
        return self

    def predict(self, x):
        return self.mean + .1*x[:, 0]

    def save_model(self, path):
        Path(path).write_text(json.dumps({"synthetic_test_mean": self.mean}))


def sample(year, offset=0):
    return pd.DataFrame({"sample_id": [f"{year}_{i}" for i in range(4)],
                         "Env": [f"A_{year}"]*2+[f"B_{year}"]*2,
                         "Hybrid": ["h0", "h1", "h2", f"h{year}"], "Year": year,
                         "Experiment_Code": ["A", "A", "B", "B"]})


def fixture(base):
    root, prep = base/"rolling", base/"prep"
    root.mkdir()
    (root/"PROTOCOL.md").write_text("Synthetic test-only fixed protocol; no scientific result.\n")
    (prep/"targets").mkdir(parents=True)
    (prep/"manifest.json").write_text("{}")
    frames = {year: sample(year) for year in [2018, 2019, 2020]}
    labels = pd.concat([f[["sample_id"]].assign(Yield_Mg_ha=np.arange(4)+year-2010) for year, f in frames.items()])
    labels.to_csv(prep/"targets/train.csv", index=False)
    labels.iloc[:0].to_csv(prep/"targets/validation.csv", index=False)
    for stage in ["development", "final"]:
        folder = root/"features/2020"/stage
        folder.mkdir(parents=True)
        samples = {"train": frames[2018] if stage == "development" else pd.concat([frames[2018], frames[2019]], ignore_index=True),
                   "validation": frames[2019] if stage == "development" else frames[2019].iloc[:0],
                   "test": frames[2020]}
        for split, frame in samples.items():
            frame.to_csv(folder/f"{split}_samples.csv", index=False)
        for horizon in rolling.HORIZONS:
            np.savez_compressed(folder/f"W{horizon}.npz", **{f"X_{s}": np.arange(len(f)*2, dtype=np.float32).reshape(len(f), 2) for s, f in samples.items()})
            (folder/f"W{horizon}_columns.json").write_text(json.dumps({"columns": ["a", "b"]}))
        manifest = {"mode": stage, "test_year": 2020,
                    "fit_years": [2014, 2018 if stage == "development" else 2019],
                    "checks": {"synthetic_fixture": True},
                    "input_hashes": {str((root/"PROTOCOL.md").resolve()): rolling.digest(root/"PROTOCOL.md")},
                    "outputs": {p.name: rolling.digest(p) for p in folder.iterdir() if p.is_file()}}
        (folder/"feature_manifest.json").write_text(json.dumps(manifest))
    return root, prep


class RollingTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.redirect = redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def test_historical_target_filter_preserves_order_without_test_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            root, prep = fixture(Path(temp))
            source = pd.read_csv(prep/"targets/train.csv")
            source.loc[source.sample_id.str.startswith("2020"), "Yield_Mg_ha"] = np.nan
            source.to_csv(prep/"targets/train.csv", index=False)
            values = rolling.targets_for_ids(["2018_3", "2019_0"], prep)
            np.testing.assert_equal(values, [11., 9.])
            with self.assertRaises(ValueError):
                rolling.targets_for_ids(["2020_0"], prep)

    def test_temporal_boundaries_and_2024_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root, prep = fixture(Path(temp))
            directory = root/"features/2020/development"
            loaded = rolling.load_fold_samples(directory, 2020, "development")
            self.assertEqual(set(loaded["validation"].Year), {2019})
            bad = loaded["train"].copy()
            bad["Year"] = 2019
            bad.to_csv(directory/"train_samples.csv", index=False)
            with self.assertRaises(ValueError):
                rolling.load_fold_samples(directory, 2020, "development")
            with self.assertRaises(ValueError):
                rolling.run_stage(2024, "development", root, prepared_dir=prep)

    def test_complete_run_resume_and_final_prediction_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root, prep = fixture(Path(temp))
            TinyEstimator.fit_calls = 0
            with patch.object(rolling, "estimator", side_effect=lambda name, params: TinyEstimator()):
                frozen = rolling.run_stage(2020, "development", root, prepared_dir=prep)
                self.assertEqual(TinyEstimator.fit_calls, 30)
                self.assertEqual(len(frozen["configs"]), 15)
                self.assertEqual(frozen["validation_year"], 2019)
                rolling.run_stage(2020, "development", root, prepared_dir=prep)
                self.assertEqual(TinyEstimator.fit_calls, 30)
                final = rolling.run_stage(2020, "final", root, prepared_dir=prep)
                self.assertEqual(TinyEstimator.fit_calls, 45)
                self.assertEqual(final["preset_models_scored"], 17)
                self.assertTrue(final["retrospective_backtest"])
                self.assertFalse(final["original_2024_holdout_read"])
                rolling.run_stage(2020, "final", root, prepared_dir=prep)
                self.assertEqual(TinyEstimator.fit_calls, 45)
            events = [json.loads(line)["event"] for line in (root/"results/2020/final/progress.jsonl").read_text().splitlines()]
            self.assertLess(events.index("all_predictions_locked"), events.index("historical_test_targets_joined"))
            rows = pd.read_csv(root/"results/2020/final/summary_metrics.csv")
            self.assertEqual(len(rows), 17)
            self.assertEqual(set(rows.test_year), {2020})
            groups = pd.read_csv(root/"results/2020/final/subgroup_metrics.csv")
            self.assertIn("unseen", set(groups.loc[groups.group_type == "hybrid", "group"]))
            # Cached units are immutable evidence, not a reason to retrain silently.
            prediction = root/"results/2020/final/predictions/Ridge_W0.csv"
            prediction.write_text(prediction.read_text()+"\n")
            with self.assertRaises(ValueError):
                rolling.run_stage(2020, "final", root, prepared_dir=prep)

    def test_changed_protocol_rejects_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root, prep = fixture(Path(temp))
            with patch.object(rolling, "estimator", side_effect=lambda name, params: TinyEstimator()):
                rolling.run_stage(2020, "development", root, prepared_dir=prep)
            (root/"PROTOCOL.md").write_text("Different protocol")
            with self.assertRaises(ValueError):
                rolling.run_stage(2020, "development", root, prepared_dir=prep)

    def test_partial_stage_resumes_completed_units_without_refitting(self):
        with tempfile.TemporaryDirectory() as temp:
            root, prep = fixture(Path(temp))
            TinyEstimator.fit_calls = 0
            def fail_after_first_unit(name, params):
                if TinyEstimator.fit_calls == 2:
                    raise RuntimeError("Synthetic interruption after one completed unit")
                return TinyEstimator()
            with patch.object(rolling, "estimator", side_effect=fail_after_first_unit):
                with self.assertRaises(RuntimeError):
                    rolling.run_stage(2020, "development", root, prepared_dir=prep)
            self.assertEqual(TinyEstimator.fit_calls, 2)
            first = root/"results/2020/development/units/Ridge_W0.json"
            first_hash = rolling.digest(first)
            with patch.object(rolling, "estimator", side_effect=lambda name, params: TinyEstimator()):
                rolling.run_stage(2020, "development", root, prepared_dir=prep)
            self.assertEqual(TinyEstimator.fit_calls, 30)
            self.assertEqual(rolling.digest(first), first_hash)

    def test_inputs_changed_during_run_cannot_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root, prep = fixture(Path(temp))
            changed = False
            def mutate_during_fit(name, params):
                nonlocal changed
                if not changed:
                    (root/"PROTOCOL.md").write_text("Mutation during execution")
                    changed = True
                return TinyEstimator()
            with patch.object(rolling, "estimator", side_effect=mutate_during_fit):
                with self.assertRaisesRegex(ValueError, "input changed during execution"):
                    rolling.run_stage(2020, "development", root, prepared_dir=prep)
            self.assertFalse((root/"results/2020/development/development_completion_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
