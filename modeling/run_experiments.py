"""Frozen two-stage multi-model experiment; never tune using 2024 outcomes.

Development: train 2014-2022, select on 2023. Final: refit 2014-2023,
persist every preset prediction before opening the holdout target file once.
Completed model/window units are resumable with input/code hash checks.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from metrics import evaluate_predictions, bootstrap_mean, window_analysis

PROJECT = Path(__file__).resolve().parents[1]
PREP = PROJECT / "preparation/outputs"
SEED = 20261018
HORIZONS = (0, 30, 60, 90, 120)
CANDIDATES = {
    "Ridge": [{"alpha": 100.0}, {"alpha": 10000.0}],
    "ExtraTrees": [{"n_estimators": 160, "max_depth": 18, "min_samples_leaf": leaf,
                    "max_features": .8, "n_jobs": 4, "random_state": SEED} for leaf in (5, 20)],
    "CatBoost": [{"iterations": 600, "depth": depth, "learning_rate": .05,
                  "l2_leaf_reg": 5, "thread_count": 4, "random_seed": SEED,
                  "loss_function": "RMSE", "verbose": False, "allow_writing_files": False}
                 for depth in (6, 8)],
}
BASELINES = ("GlobalMean", "HistoricalSiteMean")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def event(out, name, **details):
    record = {"utc": now(), "event": name, **details}
    with (out / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(clean_json(record), ensure_ascii=False)+"\n")
    print(json.dumps(clean_json(record), ensure_ascii=False), flush=True)


def versions():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name)
        for name in ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib"]}}


def feature_hashes(directory):
    manifest_path = directory / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["outputs"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Feature hash differs from manifest: {name}")
    files = [manifest_path, *[directory/f"{split}_samples.csv" for split in ("train", "validation", "test")],
             *[directory/f"W{h}.npz" for h in HORIZONS],
             *[directory/f"W{h}_columns.json" for h in HORIZONS],
             PREP/"targets/train.csv", PREP/"targets/validation.csv", PREP/"manifest.json",
             PROJECT/"modeling/PROTOCOL_LOCK.md", Path(__file__), Path(__file__).with_name("metrics.py")]
    return {str(p.relative_to(PROJECT)): digest(p) for p in files}


def load_samples(directory, stage):
    samples = {s: pd.read_csv(directory/f"{s}_samples.csv") for s in ("train", "validation", "test")}
    for name, frame in samples.items():
        if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
            raise ValueError(f"Invalid {name} sample IDs")
        if "Yield_Mg_ha" in frame or "y_true" in frame:
            raise ValueError("Labels found in feature sample metadata")
    allowed = {"train": set(range(2014, 2023 if stage == "development" else 2024)),
               "validation": {2023} if stage == "development" else set(), "test": {2024}}
    for name, frame in samples.items():
        if set(frame.Year.unique())-allowed[name]:
            raise ValueError(f"Unexpected years in {name}")
    if not len(samples["train"]) or not len(samples["test"]):
        raise ValueError("Empty training or prediction cohort")
    if stage == "development" and not len(samples["validation"]):
        raise ValueError("Development requires validation data")
    keys = [set(s.sample_id) for s in samples.values()]
    if any(keys[i] & keys[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("Sample overlap across time splits")
    return samples


def development_targets(samples):
    """Only train/validation files; no function here opens holdout labels."""
    targets = pd.concat([pd.read_csv(PREP/"targets/train.csv"), pd.read_csv(PREP/"targets/validation.csv")])
    if targets.sample_id.duplicated().any():
        raise ValueError("Duplicate development labels")
    lookup = targets.set_index("sample_id").Yield_Mg_ha
    arrays = {}
    for split in ("train", "validation"):
        arrays[split] = lookup.reindex(samples[split].sample_id).to_numpy(float)
        if not np.isfinite(arrays[split]).all():
            raise ValueError(f"Missing or invalid {split} development labels")
    return arrays


def arrays_for(directory, horizon, samples):
    with np.load(directory/f"W{horizon}.npz", allow_pickle=False) as saved:
        arrays = {split: saved[f"X_{split}"] for split in samples}
    count = len(json.loads((directory/f"W{horizon}_columns.json").read_text())["columns"])
    for split, array in arrays.items():
        if array.shape != (len(samples[split]), count) or not np.isfinite(array).all():
            raise ValueError(f"Invalid feature dimensions or values: W{horizon} {split}")
    return arrays


def estimator(name, params):
    if name == "Ridge":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        return make_pipeline(StandardScaler(), Ridge(**params, solver="cholesky"))
    if name == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(**params)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params)
    raise ValueError(name)


def prediction_frame(samples, predictions):
    predictions = np.asarray(predictions, float)
    if predictions.shape != (len(samples),) or not np.isfinite(predictions).all():
        raise ValueError("Prediction length or finite-value check failed")
    frame = samples[["sample_id", "Env", "Hybrid", "Year", "Experiment_Code"]].copy()
    frame["prediction"] = predictions
    return frame


def score(predictions, observed):
    frame = predictions.copy()
    frame["y_true"] = observed
    summary, environments = evaluate_predictions(frame)
    confidence = bootstrap_mean(environments.rmse.to_numpy())
    summary.update({"macro_rmse_ci95_low": confidence["ci95_low"],
                    "macro_rmse_ci95_high": confidence["ci95_high"]})
    return summary, environments


def baseline_predictions(name, train_samples, train_y, target_samples):
    global_mean = float(np.mean(train_y))
    if name == "GlobalMean":
        return np.full(len(target_samples), global_mean)
    lookup = pd.DataFrame({"site": train_samples.Experiment_Code.to_numpy(), "y": train_y}).groupby("site").y.mean()
    return target_samples.Experiment_Code.map(lookup).fillna(global_mean).to_numpy(float)


def persist_model(model, name, path):
    if name == "CatBoost":
        path = path.with_suffix(".cbm")
        model.save_model(str(path))
    else:
        path = path.with_suffix(".joblib")
        joblib.dump(model, path, compress=3)
    return path


def model_effects(model, name, columns, path):
    if name == "Ridge":
        values = model.named_steps["ridge"].coef_
        kind = "coefficient_per_training_standard_deviation_not_causal"
    else:
        values = model.feature_importances_
        kind = "model_internal_importance_not_causal"
    pd.DataFrame({"feature": columns, "value": values, "absolute_value": np.abs(values),
                  "interpretation": kind}).sort_values("absolute_value", ascending=False).to_csv(path, index=False)


def verify_cached_record(path):
    record = json.loads(path.read_text())
    for relative, expected in record["artifacts"].items():
        if digest(PROJECT/relative) != expected:
            raise ValueError(f"Cached artifact was modified: {relative}")
    return record


def initialize_run(stage, directory, out, frozen=None):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("predictions", "models", "units", "candidate_predictions", "feature_effects"):
        (out/name).mkdir(exist_ok=True)
    manifest = {"stage": stage, "inputs": feature_hashes(directory), "candidates": CANDIDATES,
                "seed": SEED, "versions": versions()}
    if frozen:
        manifest["frozen_selection_sha256"] = digest(frozen)
    previous = out/"run_manifest.json"
    if previous.exists():
        if json.loads(previous.read_text()) != manifest:
            raise ValueError("Run inputs/code/configuration changed. Preserve existing results and select a new --output directory; do not silently reuse stale results.")
    else:
        write_json(manifest, previous)
    return manifest


def development(directory, out):
    manifest = initialize_run("development", directory, out)
    samples, selected, all_candidates = load_samples(directory, "development"), [], []
    targets = development_targets(samples)
    for name in BASELINES:
        predictions = prediction_frame(samples["validation"], baseline_predictions(name, samples["train"], targets["train"], samples["validation"]))
        summary, environments = score(predictions, targets["validation"])
        predictions.to_csv(out/"predictions"/f"{name}.csv", index=False)
        environments.to_csv(out/f"{name}_environment_metrics.csv", index=False)
        selected.append({"model": name, "window": 0, "candidate": "baseline", "fit_seconds": 0., **summary})
    configurations = []
    for horizon in HORIZONS:
        arrays = arrays_for(directory, horizon, samples)
        columns = json.loads((directory/f"W{horizon}_columns.json").read_text())["columns"]
        for name, candidates in CANDIDATES.items():
            unit = out/"units"/f"{name}_W{horizon}.json"
            if unit.exists():
                record = verify_cached_record(unit)
                event(out, "resume_completed_unit", model=name, window=horizon)
            else:
                candidate_records, best, best_model, best_predictions, best_environments = [], None, None, None, None
                for index, params in enumerate(candidates):
                    candidate_id = f"{name}_W{horizon}_c{index}"
                    event(out, "fit_start", model=name, window=horizon, candidate=index, params=params)
                    model = estimator(name, params)
                    start = time.monotonic()
                    with threadpool_limits(limits=4):
                        model.fit(arrays["train"], targets["train"])
                        fit_seconds = time.monotonic()-start
                        predictions = prediction_frame(samples["validation"], model.predict(arrays["validation"]))
                    summary, environments = score(predictions, targets["validation"])
                    candidate_record = {"model": name, "window": horizon, "candidate": index,
                                        "params": params, "fit_seconds": fit_seconds, **summary}
                    candidate_records.append(candidate_record)
                    predictions.to_csv(out/"candidate_predictions"/f"{candidate_id}.csv", index=False)
                    environments.to_csv(out/"candidate_predictions"/f"{candidate_id}_environments.csv", index=False)
                    event(out, "fit_complete", **candidate_record)
                    if best is None or summary["macro_rmse"] < best["macro_rmse"]:
                        best, best_model, best_predictions, best_environments = candidate_record, model, predictions, environments
                prediction_path = out/"predictions"/f"{name}_W{horizon}.csv"
                environment_path = out/f"{name}_W{horizon}_environment_metrics.csv"
                best_predictions.to_csv(prediction_path, index=False)
                best_environments.to_csv(environment_path, index=False)
                model_path = persist_model(best_model, name, out/"models"/f"{name}_W{horizon}")
                effects_path = out/"feature_effects"/f"{name}_W{horizon}.csv"
                model_effects(best_model, name, columns, effects_path)
                record = {"selected": best, "candidates": candidate_records,
                          "artifacts": {str(p.relative_to(PROJECT)): digest(p) for p in
                                        [prediction_path, environment_path, model_path, effects_path]}}
                write_json(record, unit)
                del best_model, model
            selected.append(record["selected"])
            all_candidates.extend(record["candidates"])
            choice = record["selected"]
            configurations.append({"model": name, "window": horizon, "candidate": choice["candidate"],
                                   "params": choice["params"], "validation_macro_rmse": choice["macro_rmse"]})
            pd.DataFrame(selected).drop(columns="params", errors="ignore").to_csv(out/"selected_metrics.csv", index=False)
            pd.DataFrame(all_candidates).assign(params=lambda f: f.params.map(json.dumps)).to_csv(out/"candidate_metrics.csv", index=False)
        del arrays
    winner = min(configurations, key=lambda c: c["validation_macro_rmse"])
    frozen = {"schema_version": 1, "frozen_at_utc": now(), "selection_basis": "2023 environment-equal RMSE only",
              "holdout_labels_read": False, "seed": SEED, "windows": list(HORIZONS),
              "configs": configurations, "recommended_by_validation": winner,
              "baselines": list(BASELINES), "development_run_manifest_sha256": digest(out/"run_manifest.json"),
              "development_input_hashes": manifest["inputs"], "versions": manifest["versions"],
              "bootstrap_repeats": 2000,
              "final_policy": "refit using 2014-2023; write all 17 prediction files, then read 2024 observed targets once; never retune on final results",
              "limitations": ["single validation and final year", "four large input-source equivalence checks pending",
                              "small two-candidate search budget; selection uncertainty not included in bootstrap"]}
    freeze_path = out/"frozen_selection.json"
    if freeze_path.exists():
        old = json.loads(freeze_path.read_text())
        frozen["frozen_at_utc"] = old["frozen_at_utc"]
        if old != frozen:
            raise ValueError("Existing freeze differs; do not overwrite a locked selection")
    else:
        write_json(frozen, freeze_path)
    event(out, "development_complete_selection_frozen", recommended=winner, freeze=str(freeze_path))
    return frozen


def validate_freeze(path):
    frozen = json.loads(path.read_text())
    if frozen["holdout_labels_read"] or frozen["windows"] != list(HORIZONS):
        raise ValueError("Invalid selection freeze")
    expected = {(m, h) for m in CANDIDATES for h in HORIZONS}
    if {(c["model"], c["window"]) for c in frozen["configs"]} != expected or len(frozen["configs"]) != 15:
        raise ValueError("Freeze must specify exactly 15 model-window configurations")
    for config in frozen["configs"]:
        if config["params"] != CANDIDATES[config["model"]][config["candidate"]]:
            raise ValueError("Frozen configuration differs from preset candidates")
    for relative, expected_hash in frozen["development_input_hashes"].items():
        if digest(PROJECT/relative) != expected_hash:
            raise ValueError(f"Frozen development input changed: {relative}")
    return frozen


def subgroup_metrics(frame, training_samples, model, window):
    groups = {
        "hybrid": frame.Hybrid.isin(set(training_samples.Hybrid)),
        "site": frame.Experiment_Code.isin(set(training_samples.Experiment_Code))}
    rows = []
    for kind, seen in groups.items():
        for flag, label in [(True, "seen"), (False, "unseen")]:
            part = frame.loc[seen == flag]
            if len(part):
                summary, _ = evaluate_predictions(part)
                rows.append({"model": model, "window": window, "group_type": kind, "group": label, **summary})
    return rows


def final(directory, out, freeze_path):
    frozen = validate_freeze(freeze_path)
    initialize_run("final", directory, out, freeze_path)
    completion = out/"final_evaluation_manifest.json"
    if completion.exists():
        record = verify_cached_record(completion)
        event(out, "final_already_complete_no_new_holdout_read")
        return record
    samples, predictions_to_score, fit_times = load_samples(directory, "final"), [], {}
    targets = development_targets(samples)
    for name in BASELINES:
        path = out/"predictions"/f"{name}.csv"
        prediction_frame(samples["test"], baseline_predictions(name, samples["train"], targets["train"], samples["test"])).to_csv(path, index=False)
        predictions_to_score.append((name, 0, path))
        fit_times[(name, 0)] = 0.
    for horizon in HORIZONS:
        arrays = arrays_for(directory, horizon, samples)
        columns = json.loads((directory/f"W{horizon}_columns.json").read_text())["columns"]
        for config in [c for c in frozen["configs"] if c["window"] == horizon]:
            name = config["model"]
            unit = out/"units"/f"{name}_W{horizon}.json"
            path = out/"predictions"/f"{name}_W{horizon}.csv"
            if unit.exists():
                record = verify_cached_record(unit)
                event(out, "resume_completed_final_prediction", model=name, window=horizon)
            else:
                event(out, "final_fit_start", model=name, window=horizon, params=config["params"])
                model = estimator(name, config["params"])
                start = time.monotonic()
                with threadpool_limits(limits=4):
                    model.fit(arrays["train"], targets["train"])
                    fit_seconds = time.monotonic()-start
                    predicted = model.predict(arrays["test"])
                prediction_frame(samples["test"], predicted).to_csv(path, index=False)
                model_path = persist_model(model, name, out/"models"/f"{name}_W{horizon}")
                effects_path = out/"feature_effects"/f"{name}_W{horizon}.csv"
                model_effects(model, name, columns, effects_path)
                record = {"model": name, "window": horizon, "fit_seconds": fit_seconds,
                          "frozen_selection_sha256": digest(freeze_path),
                          "artifacts": {str(p.relative_to(PROJECT)): digest(p) for p in [path, model_path, effects_path]}}
                write_json(record, unit)
                event(out, "final_fit_complete", model=name, window=horizon, fit_seconds=fit_seconds)
                del model
            predictions_to_score.append((name, horizon, path))
            fit_times[(name, horizon)] = record["fit_seconds"]
        del arrays
    if len(predictions_to_score) != 17:
        raise ValueError("All 15 models and 2 baseline predictions required before opening test labels")
    prediction_hashes = {str(path.relative_to(PROJECT)): digest(path) for _, _, path in predictions_to_score}
    prediction_lock = out/"prediction_lock_before_holdout.json"
    lock = {"frozen_selection_sha256": digest(freeze_path), "prediction_hashes": prediction_hashes,
            "test_sample_ids_sha256": digest(directory/"test_samples.csv")}
    if prediction_lock.exists() and json.loads(prediction_lock.read_text()) != lock:
        raise ValueError("Predictions changed after holdout prediction lock")
    write_json(lock, prediction_lock)
    event(out, "all_predictions_persisted_before_holdout", count=17, prediction_lock_sha256=digest(prediction_lock))

    # The sole read of 2024 outcome data in the experiment runner. This point is
    # strictly after all frozen-model fits, predictions, and their hash lock.
    label_path = PREP/"holdout/test_targets.csv"
    event(out, "holdout_labels_opened_for_preset_evaluation", sha256=digest(label_path))
    labels = pd.read_csv(label_path)
    if labels.sample_id.duplicated().any() or not set(labels.sample_id).issubset(set(samples["test"].sample_id)):
        raise ValueError("Unexpected or duplicate holdout label keys")
    if not np.isfinite(labels.Yield_Mg_ha.to_numpy(float)).all():
        raise ValueError("Invalid holdout labels")
    summaries, all_environments, all_groups, weather_tables, sufficiency = [], [], [], [], {}
    per_model = {m: {} for m in CANDIDATES}
    for name, horizon, path in predictions_to_score:
        predictions = pd.read_csv(path)
        if predictions.sample_id.tolist() != samples["test"].sample_id.tolist():
            raise ValueError("Final predictions changed cohort/order")
        evaluated = predictions.merge(labels[["sample_id", "Yield_Mg_ha"]], on="sample_id", how="inner", validate="one_to_one").rename(columns={"Yield_Mg_ha": "y_true"})
        summary, environments = score(evaluated.drop(columns="y_true"), evaluated.y_true.to_numpy())
        summaries.append({"model": name, "window": horizon, "fit_seconds": fit_times[(name, horizon)],
                          "n_prediction_inputs": len(predictions), "n_without_observed_target": len(predictions)-len(evaluated), **summary})
        environments.insert(0, "window", horizon)
        environments.insert(0, "model", name)
        all_environments.append(environments)
        all_groups.extend(subgroup_metrics(evaluated, samples["train"], name, horizon))
        evaluated.to_csv(out/"predictions"/f"{name}_W{horizon}_evaluated.csv", index=False)
        if name in per_model:
            per_model[name][horizon] = environments
    for name, windows in per_model.items():
        table, sufficient = window_analysis(windows)
        table.insert(0, "model", name)
        weather_tables.append(table)
        sufficiency[name] = sufficient
    pd.DataFrame(summaries).to_csv(out/"summary_metrics.csv", index=False)
    pd.concat(all_environments, ignore_index=True).to_csv(out/"environment_metrics.csv", index=False)
    pd.DataFrame(all_groups).to_csv(out/"subgroup_metrics.csv", index=False)
    pd.concat(weather_tables, ignore_index=True).to_csv(out/"weather_increment_comparisons.csv", index=False)
    write_json({"models": sufficiency, "recommended_by_2023_validation": frozen["recommended_by_validation"],
                "test_result_selection_permitted": False,
                "interpretation": "These are final-year descriptive operational thresholds, not parameter selection or proof of universality. Bootstrap intervals are unadjusted for multiple comparisons."}, out/"earliest_sufficient_windows.json")
    outputs = [p for p in out.iterdir() if p.is_file() and p.name not in ("progress.jsonl", completion.name)]
    outputs.extend((out/"predictions").glob("*.csv"))
    final_manifest = {"completed_at_utc": now(), "holdout_sha256": digest(label_path),
                      "frozen_selection_sha256": digest(freeze_path), "n_scored": len(labels),
                      "n_prediction_inputs": len(samples["test"]), "preset_models_scored": 17,
                      "recommended_by_validation": frozen["recommended_by_validation"],
                      "artifacts": {str(p.relative_to(PROJECT)): digest(p) for p in outputs}}
    write_json(final_manifest, completion)
    event(out, "final_evaluation_complete", n_scored=len(labels), models=17)
    return final_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["development", "final"])
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze", type=Path, default=PROJECT/"modeling/results/development/frozen_selection.json")
    args = parser.parse_args()
    feature_directory = args.features or PROJECT/"modeling/features"/args.stage
    output_directory = args.output or PROJECT/"modeling/results"/args.stage
    if args.stage == "development":
        development(feature_directory, output_directory)
    else:
        final(feature_directory, output_directory, args.freeze)
