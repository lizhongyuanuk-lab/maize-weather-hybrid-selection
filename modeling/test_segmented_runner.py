"""Synthetic-only segmented-runner tests; no real model fitting or label access."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_segmented_experiments as runner
from segmented_sources import SourceFold, SourceVerifier, source_fold


class SmallModel:
    fit_calls = 0

    def __init__(self, penalty=0.):
        self.penalty = penalty

    def fit(self, x, y):
        SmallModel.fit_calls += 1
        self.mean = float(np.mean(y))
        self.coef_ = np.zeros(x.shape[1])
        self.feature_importances_ = np.ones(x.shape[1])/x.shape[1]
        self.named_steps = {"ridge": self}
        return self

    def predict(self, x):
        return self.mean+.01*x[:, 0]+self.penalty

    def save_model(self, path):
        Path(path).write_text(json.dumps({"test_only_mean": self.mean, "penalty": self.penalty}))


def estimator(name, params):
    return SmallModel(100.*runner.CANDIDATES[name].index(params))


def frame(year):
    return pd.DataFrame({"sample_id": [f"{year}_{i}" for i in range(4)],
                         "Env": [f"A_{year}"]*2+[f"B_{year}"]*2,
                         "Hybrid": ["h0", "h1", "h2", f"h{year}"], "Year": year,
                         "Experiment_Code": ["A", "A", "B", "B"]})


def fixture(base, year=2020):
    base = Path(base).resolve()
    root, cumulative, prepared = base/"segmented", base/"cumulative", base/"prepared"
    root.mkdir()
    cumulative.mkdir()
    (root/"PROTOCOL.md").write_text("Synthetic fixed segmented protocol, not a research result.\n")
    (prepared/"targets").mkdir(parents=True)
    (prepared/"manifest.json").write_text("{}")
    frames = {y: frame(y) for y in (year-2, year-1, year)}
    labels = {y: pd.DataFrame({"sample_id": frames[y].sample_id, "Yield_Mg_ha": np.arange(4)+8.+y-(year-2)}) for y in frames}
    historical = [labels[year-2], labels[year-1]]+([] if year == 2024 else [labels[year]])
    pd.concat(historical, ignore_index=True).to_csv(prepared/"targets/train.csv", index=False)
    labels[year].iloc[:0].to_csv(prepared/"targets/validation.csv", index=False)
    if year == 2024:
        (prepared/"holdout").mkdir()
        labels[year].iloc[:3].to_csv(prepared/"holdout/test_targets.csv", index=False)
    source_objects = {}
    source_configs = []
    development_units = {}
    source_baselines = {}
    for stage in ("development", "final"):
        feature_directory = cumulative/"features"/str(year)/stage
        directory = cumulative/"results"/str(year)/stage
        new_features = root/"features"/str(year)/stage
        for folder in (feature_directory, new_features, directory/"units", directory/"models", directory/"predictions", directory/"candidate_predictions"):
            folder.mkdir(parents=True)
        samples = {"train": frames[year-2] if stage == "development" else pd.concat([frames[year-2], frames[year-1]], ignore_index=True),
                   "validation": frames[year-1] if stage == "development" else frames[year-1].iloc[:0], "test": frames[year]}
        arrays = {s: np.arange(len(f)*2, dtype=np.float32).reshape(len(f), 2) for s, f in samples.items()}
        for split, samples_frame in samples.items():
            samples_frame.to_csv(feature_directory/f"{split}_samples.csv", index=False)
            shutil.copyfile(feature_directory/f"{split}_samples.csv", new_features/f"{split}_samples.csv")
        (feature_directory/"preprocessing.joblib").write_text("Synthetic preprocessing artifact; never deserialized")
        shutil.copyfile(feature_directory/"preprocessing.joblib", new_features/"preprocessing.joblib")
        for horizon in runner.HORIZONS:
            old = {s: x if horizon == 0 else np.column_stack([x, np.tile(np.arange(30, dtype=np.float32), (len(x), 1))]) for s, x in arrays.items()}
            columns = ["a", "b"]+([] if horizon == 0 else [f"physical_{i}" for i in range(21)]+[f"quality_{i}" for i in range(9)])
            np.savez_compressed(feature_directory/f"W{horizon}.npz", **{f"X_{s}": a for s, a in old.items()})
            (feature_directory/f"W{horizon}_columns.json").write_text(json.dumps({"columns": columns}))
            if horizon in (0, 30):
                for suffix in (".npz", "_columns.json"):
                    shutil.copyfile(feature_directory/f"W{horizon}{suffix}", new_features/f"W{horizon}{suffix}")
            else:
                new = {s: np.column_stack([x[:, :2], *[x[:, 2:23]+segment for segment in range(horizon//30)], x[:, -9:]]) for s, x in old.items()}
                new_columns = ["a", "b"]+[f"segment_{segment}_physical_{i}" for segment in range(horizon//30) for i in range(21)]+columns[-9:]
                np.savez_compressed(new_features/f"W{horizon}.npz", **{f"X_{s}": a for s, a in new.items()})
                (new_features/f"W{horizon}_columns.json").write_text(json.dumps({"columns": new_columns}))
        manifest = {"mode": stage, "test_year": year, "fit_years": [2014, year-(2 if stage == "development" else 1)],
                    "representation": "segmented_30d", "physical_weather_features": 21, "quality_feature_count": 9,
                    "base_feature_count": 2, "checks": {"synthetic_test_only": True},
                    "input_hashes": {str(root/"PROTOCOL.md"): runner.digest(root/"PROTOCOL.md")},
                    "outputs": {p.name: runner.digest(p) for p in new_features.iterdir() if p.is_file()}}
        runner.write_json(manifest, new_features/"feature_manifest.json")
        train_y = runner.targets_for_ids(samples["train"].sample_id, prepared)
        target_split = "validation" if stage == "development" else "test"
        target_samples = samples[target_split]
        target_labels = labels[year-1] if stage == "development" else labels[year].iloc[:3 if year == 2024 else 4]
        unit_records = {}
        selected_rows = []
        for model in runner.BASELINES:
            predictions = runner.prediction_frame(target_samples, runner.baseline_predictions(model, samples["train"], train_y, target_samples))
            predictions.to_csv(directory/"predictions"/f"{model}.csv", index=False)
            evaluated = predictions.merge(target_labels, on="sample_id").rename(columns={"Yield_Mg_ha": "y_true"})
            summary, _ = runner.score(evaluated.drop(columns="y_true"), evaluated.y_true.to_numpy())
            if stage == "development":
                source_baselines[model] = {"model": model, "window": 0, "candidate": "baseline", "fit_seconds": 0., **summary}
                selected_rows.append(source_baselines[model])
            else:
                evaluated.to_csv(directory/"predictions"/f"{model}_W0_evaluated.csv", index=False)
        for horizon in runner.HORIZONS:
            for model in runner.CANDIDATES:
                candidates = []
                chosen = int(horizon in runner.NEW_WINDOWS)
                for index, params in enumerate(runner.CANDIDATES[model]):
                    # Source prefers candidate 1 for new horizons; new runner must
                    # independently prefer 0 in this synthetic representation.
                    penalty = 100. if horizon in runner.NEW_WINDOWS and index == 0 else 0.
                    prediction = runner.prediction_frame(target_samples, np.mean(train_y)+.01*arrays[target_split][:, 0]+penalty)
                    evaluated = prediction.merge(target_labels, on="sample_id").rename(columns={"Yield_Mg_ha": "y_true"})
                    summary, env = runner.score(evaluated.drop(columns="y_true"), evaluated.y_true.to_numpy())
                    entry = {"model": model, "window": horizon, "candidate": index, "params": params, "fit_seconds": 1., **summary}
                    candidates.append(entry)
                    if stage == "development":
                        candidate_path = directory/"candidate_predictions"/f"{model}_W{horizon}_c{index}.csv"
                        prediction.to_csv(candidate_path, index=False)
                        env.to_csv(candidate_path.with_name(candidate_path.stem+"_environments.csv"), index=False)
                    if index == chosen:
                        selected_prediction, selected_entry = prediction, entry
                prediction_path = directory/"predictions"/f"{model}_W{horizon}.csv"
                selected_prediction.to_csv(prediction_path, index=False)
                model_path = directory/"models"/f"{model}_W{horizon}{'.cbm' if model == 'CatBoost' else '.joblib'}"
                model_path.write_text("Synthetic source artifact, never loaded as a real model")
                unit = {"model": model, "window": horizon, "selected": selected_entry, "candidates": candidates,
                        "fit_seconds": 1., "artifacts": runner.hash_paths([prediction_path, model_path], cumulative)}
                runner.write_json(unit, directory/"units"/f"{model}_W{horizon}.json")
                unit_records[(model, horizon)] = unit
                if stage == "development":
                    development_units[(model, horizon)] = unit
                    selected_rows.append(selected_entry)
                    source_configs.append({"model": model, "window": horizon, "candidate": chosen,
                                           "params": runner.CANDIDATES[model][chosen], "validation_macro_rmse": selected_entry["macro_rmse"]})
        if stage == "development":
            pd.DataFrame(selected_rows).to_csv(directory/"selected_metrics.csv", index=False)
        completion = {"n_scored": len(labels[year])-(1 if year == 2024 else 0), "n_prediction_inputs": 4}
        if year == 2024:
            completion["holdout_sha256"] = runner.digest(prepared/"holdout/test_targets.csv")
        source_objects[stage] = SourceFold(year, stage, feature_directory, directory,
            cumulative/"results"/str(year)/"development", cumulative/"results"/str(year)/"final",
            {"configs": source_configs}, completion, unit_records, development_units, source_baselines, {}, cumulative)
    hashes = {str(p): runner.digest(p) for p in cumulative.rglob("*") if p.is_file()}
    for source in source_objects.values():
        source.hashes = dict(hashes)
    return root, cumulative, prepared, source_objects


class SegmentedRunnerTests(unittest.TestCase):
    def setUp(self):
        SmallModel.fit_calls = 0
        redirect = redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def run_fold(self, stage, root, cumulative, prepared, sources):
        with patch.object(runner, "source_fold", side_effect=lambda year, phase, cumulative_root: sources[phase]), \
                patch.object(runner, "estimator", side_effect=estimator):
            return runner.run_stage(sources[stage].year, stage, root, cumulative, prepared_dir=prepared)

    def test_new_fit_counts_independent_selection_reuse_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root, cumulative, prepared, sources = fixture(Path(temp))
            frozen = self.run_fold("development", root, cumulative, prepared, sources)
            self.assertEqual(SmallModel.fit_calls, 18)
            self.assertEqual(len(frozen["configs"]), 15)
            self.assertEqual(sum(c["reused"] for c in frozen["configs"]), 6)
            self.assertTrue(all(c["candidate"] == 0 for c in frozen["configs"]))
            self.assertTrue(all(c["candidate"] == 1 for c in sources["development"].freeze["configs"] if c["window"] in runner.NEW_WINDOWS))
            self.run_fold("development", root, cumulative, prepared, sources)
            self.assertEqual(SmallModel.fit_calls, 18)
            completed = self.run_fold("final", root, cumulative, prepared, sources)
            self.assertEqual(SmallModel.fit_calls, 27)
            self.assertEqual(completed["n_new_model_fits"], 9)
            self.assertEqual(completed["n_reused_units"], 8)
            self.assertEqual(completed["preset_models_scored"], 17)
            self.run_fold("final", root, cumulative, prepared, sources)
            self.assertEqual(SmallModel.fit_calls, 27)
            for stage in ("development", "final"):
                for model, horizon in [(m, h) for m in runner.CANDIDATES for h in (0, 30)]+[(m, 0) for m in runner.BASELINES]:
                    original = sources[stage].prediction_path(model, horizon)
                    copied = root/"results/2020"/stage/"predictions"/original.name
                    self.assertEqual(runner.digest(original), runner.digest(copied))
            self.assertEqual(len(pd.read_csv(root/"results/2020/development/candidate_metrics.csv")), 30)
            self.assertEqual(len(list((root/"results/2020/final/models").iterdir())), 9)
            rows = pd.read_csv(root/"results/2020/final/summary_metrics.csv")
            self.assertEqual(len(rows), 17)
            self.assertEqual(int(rows.reused.sum()), 8)

    def test_partial_stage_keeps_completed_new_unit(self):
        with tempfile.TemporaryDirectory() as temp:
            root, cumulative, prepared, sources = fixture(Path(temp))
            def fail_after_unit(name, params):
                if SmallModel.fit_calls == 2:
                    raise RuntimeError("Synthetic interruption")
                return estimator(name, params)
            with patch.object(runner, "source_fold", return_value=sources["development"]), patch.object(runner, "estimator", side_effect=fail_after_unit):
                with self.assertRaisesRegex(RuntimeError, "Synthetic"):
                    runner.run_stage(2020, "development", root, cumulative, prepared_dir=prepared)
            first = root/"results/2020/development/units/Ridge_W60.json"
            saved_hash = runner.digest(first)
            self.run_fold("development", root, cumulative, prepared, sources)
            self.assertEqual(SmallModel.fit_calls, 18)
            self.assertEqual(runner.digest(first), saved_hash)

    def test_2024_missing_targets_and_no_outcomes_before_prediction_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root, cumulative, prepared, sources = fixture(Path(temp), 2024)
            self.run_fold("development", root, cumulative, prepared, sources)
            read_csv = pd.read_csv
            target_reads = []
            def guarded_read(path, *args, **kwargs):
                if Path(path) in (prepared/"holdout/test_targets.csv", sources["final"].evaluated_reference):
                    lock = runner.read_json(root/"results/2024/final/prediction_lock_before_holdout.json")
                    self.assertEqual(len(lock["prediction_hashes"]), 17)
                    target_reads.append(str(path))
                return read_csv(path, *args, **kwargs)
            with patch.object(pd, "read_csv", side_effect=guarded_read):
                completed = self.run_fold("final", root, cumulative, prepared, sources)
            self.assertEqual(completed["n_prediction_inputs"], 4)
            self.assertEqual(completed["n_scored"], 3)
            self.assertEqual(completed["n_without_observed_target"], 1)
            self.assertEqual(len(target_reads), 2)
            events = [json.loads(line)["event"] for line in (root/"results/2024/final/progress.jsonl").read_text().splitlines()]
            self.assertLess(events.index("all_predictions_locked"), events.index("historical_2024_targets_opened_after_prediction_lock"))
            for path in (root/"results/2024/final/predictions").glob("*_evaluated.csv"):
                self.assertEqual(len(pd.read_csv(path)), 3)
            target_path = prepared/"holdout/test_targets.csv"
            target_path.write_text(target_path.read_text()+"\n")
            with self.assertRaisesRegex(ValueError, "source artifact changed"):
                self.run_fold("final", root, cumulative, prepared, sources)
            self.assertEqual(SmallModel.fit_calls, 27)

    def test_stage_lock_rejects_reentry_and_releases(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"stage"
            with runner.stage_lock(out):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with runner.stage_lock(out):
                        self.fail("Second stage lock acquired")
            with runner.stage_lock(out):
                pass

    def test_changed_source_or_cached_prediction_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root, cumulative, prepared, sources = fixture(Path(temp))
            self.run_fold("development", root, cumulative, prepared, sources)
            path = root/"results/2020/development/predictions/Ridge_W60.csv"
            path.write_text(path.read_text()+"\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                self.run_fold("development", root, cumulative, prepared, sources)
            self.assertEqual(SmallModel.fit_calls, 18)
        with tempfile.TemporaryDirectory() as temp:
            root, cumulative, prepared, sources = fixture(Path(temp))
            self.run_fold("development", root, cumulative, prepared, sources)
            source_path = sources["development"].prediction_path("Ridge", 0)
            source_path.write_text(source_path.read_text()+"\n")
            with self.assertRaises(ValueError):
                self.run_fold("final", root, cumulative, prepared, sources)

    def test_global_quality_and_static_changes_rejected_even_if_rehashed(self):
        for column in (0, -1):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as temp:
                root, cumulative, prepared, sources = fixture(Path(temp))
                folder = root/"features/2020/development"
                with np.load(folder/"W60.npz") as saved:
                    arrays = {key: saved[key] for key in saved.files}
                arrays["X_train"][0, column] += 1
                np.savez_compressed(folder/"W60.npz", **arrays)
                manifest = runner.read_json(folder/"feature_manifest.json")
                manifest["outputs"]["W60.npz"] = runner.digest(folder/"W60.npz")
                runner.write_json(manifest, folder/"feature_manifest.json")
                with self.assertRaises(ValueError):
                    self.run_fold("development", root, cumulative, prepared, sources)
                self.assertEqual(SmallModel.fit_calls, 0)

    def test_preprocessing_must_be_listed_and_byte_identical(self):
        for mutation in ("omit", "change"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root, cumulative, prepared, sources = fixture(Path(temp))
                folder = root/"features/2020/development"
                manifest = runner.read_json(folder/"feature_manifest.json")
                if mutation == "omit":
                    del manifest["outputs"]["preprocessing.joblib"]
                else:
                    (folder/"preprocessing.joblib").write_text("Changed synthetic preprocessing")
                    manifest["outputs"]["preprocessing.joblib"] = runner.digest(folder/"preprocessing.joblib")
                runner.write_json(manifest, folder/"feature_manifest.json")
                with self.assertRaises(ValueError):
                    self.run_fold("development", root, cumulative, prepared, sources)
                self.assertEqual(SmallModel.fit_calls, 0)

    def test_source_hash_verifier_rejects_modified_or_escaping_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root/"model"
            artifact.write_text("Synthetic model")
            expected = runner.digest(artifact)
            verify = SourceVerifier()
            self.assertEqual(verify.check(artifact, expected), expected)
            artifact.write_text("Modified model")
            with self.assertRaises(ValueError):
                verify.check(artifact, expected)
            manifest = root/"manifest.json"
            runner.write_json({"artifacts": {"../outside": "wrong"}}, manifest)
            with self.assertRaisesRegex(ValueError, "Escaping"):
                verify.artifacts(manifest, root)

    def test_source_adapter_rejects_failed_audit_without_opening_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            runner.write_json({"status": "FAIL"}, root/"validation_checks.json")
            with patch.object(pd, "read_csv", side_effect=AssertionError("No outcome/table should open")):
                with self.assertRaisesRegex(ValueError, "PASS audit"):
                    source_fold(2020, "development", root, project=root)
                with self.assertRaises(ValueError):
                    source_fold(2025, "development", root, project=root)


if __name__ == "__main__":
    unittest.main()
