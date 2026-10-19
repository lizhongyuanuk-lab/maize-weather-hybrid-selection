"""Reload 15 final models and replay locked predictions, without fitting/labels.

Only final feature matrices, sample identifiers, model files, unlabeled prediction
CSVs, and JSON manifests are opened. Evaluated CSVs and all target files are
deliberately excluded, even from byte-hash reads. Original artifacts are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
FINAL = PROJECT / "modeling/results/final"
FEATURES = PROJECT / "modeling/features/final"
FREEZE = PROJECT / "modeling/results/development/frozen_selection.json"
OUTPUT = PROJECT / "modeling/prediction_replay_report.json"
TOLERANCE = 1e-8


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(PROJECT))


def checked_hash(path, expected, verified):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch: {relative(path)}")
    verified[relative(path)] = actual
    return actual


def load_json(path):
    return json.loads(path.read_text())


def save_report(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temporary.replace(output)


def replay(output=OUTPUT):
    started = time.monotonic()
    report = {"status": "FAIL", "stage": "final_saved_model_prediction_replay",
              "checked_at_utc": None, "absolute_tolerance": TOLERANCE,
              "relative_tolerance": 0.0, "thread_limit": 2,
              "model_count_expected": 15, "model_count_checked": 0,
              "prediction_values_checked": 0, "max_abs_error": 0.0,
              "no_labels_read": True, "no_fit_called": True, "original_artifacts_modified": False,
              "labels_policy": "No targets, holdout-label files, evaluated prediction CSVs, or score tables opened, including for hashing.",
              "script_sha256": sha256(Path(__file__)), "verified_hashes": {}, "models": [], "failure": None}
    verified = report["verified_hashes"]
    try:
        completion_path = FINAL / "final_evaluation_manifest.json"
        if not completion_path.exists():
            raise FileNotFoundError("Final completion manifest is required before replay")
        completion = load_json(completion_path)
        completion_hash = sha256(completion_path)
        report["audited_manifest_path"] = relative(completion_path)
        report["audited_manifest_sha256"] = completion_hash
        checked_hash(FREEZE, completion["frozen_selection_sha256"], verified)
        run_path = FINAL / "run_manifest.json"
        checked_hash(run_path, completion["artifacts"][relative(run_path)], verified)
        run = load_json(run_path)
        if run["stage"] != "final" or run["frozen_selection_sha256"] != completion["frozen_selection_sha256"]:
            raise ValueError("Final run is not bound to the frozen configuration")
        lock_path = FINAL / "prediction_lock_before_holdout.json"
        checked_hash(lock_path, completion["artifacts"][relative(lock_path)], verified)
        lock = load_json(lock_path)
        if lock["frozen_selection_sha256"] != completion["frozen_selection_sha256"]:
            raise ValueError("Prediction lock uses another freeze")
        freeze = load_json(FREEZE)
        expected = {(m, w) for m in ["Ridge", "ExtraTrees", "CatBoost"] for w in [0, 30, 60, 90, 120]}
        if len(freeze["configs"]) != 15 or {(x["model"], x["window"]) for x in freeze["configs"]} != expected:
            raise ValueError("Freeze is not the expected 15 model-window matrix")
        feature_manifest_path = FEATURES / "feature_manifest.json"
        checked_hash(feature_manifest_path, run["inputs"][relative(feature_manifest_path)], verified)
        feature_manifest = load_json(feature_manifest_path)
        if feature_manifest["mode"] != "final":
            raise ValueError("Final model replay requires final-stage features")
        sample_path = FEATURES / "test_samples.csv"
        checked_hash(sample_path, run["inputs"][relative(sample_path)], verified)
        if verified[relative(sample_path)] != lock["test_sample_ids_sha256"]:
            raise ValueError("Test input cohort differs from prediction lock")
        samples = pd.read_csv(sample_path)
        if samples.sample_id.duplicated().any() or samples.sample_id.isna().any():
            raise ValueError("Invalid test sample identifiers")
        if any(c in samples for c in ["y_true", "Yield_Mg_ha", "target"]):
            raise ValueError("Label column found in supposedly unlabeled sample metadata")
        if len(samples) != completion["n_prediction_inputs"]:
            raise ValueError("Test cohort size differs from completion manifest")
        report["test_input_rows"] = len(samples)
        for horizon in [0, 30, 60, 90, 120]:
            matrix_path = FEATURES / f"W{horizon}.npz"
            checked_hash(matrix_path, run["inputs"][relative(matrix_path)], verified)
            if verified[relative(matrix_path)] != feature_manifest["outputs"][matrix_path.name]:
                raise ValueError("Feature matrix does not match feature manifest")
            with np.load(matrix_path, allow_pickle=False) as matrices:
                features = matrices["X_test"]
            if features.shape[0] != len(samples) or not np.isfinite(features).all():
                raise ValueError("Invalid test feature matrix")
            for config in [c for c in freeze["configs"] if c["window"] == horizon]:
                name = config["model"]
                unit_path = FINAL / "units" / f"{name}_W{horizon}.json"
                unit = load_json(unit_path)
                verified[relative(unit_path)] = sha256(unit_path)
                if unit["model"] != name or unit["window"] != horizon or unit["frozen_selection_sha256"] != completion["frozen_selection_sha256"]:
                    raise ValueError("Saved-model unit disagrees with frozen identity")
                suffix = ".cbm" if name == "CatBoost" else ".joblib"
                model_path = FINAL / "models" / f"{name}_W{horizon}{suffix}"
                prediction_path = FINAL / "predictions" / f"{name}_W{horizon}.csv"
                checked_hash(model_path, unit["artifacts"][relative(model_path)], verified)
                checked_hash(prediction_path, unit["artifacts"][relative(prediction_path)], verified)
                for reference in [lock["prediction_hashes"], completion["artifacts"]]:
                    if verified[relative(prediction_path)] != reference[relative(prediction_path)]:
                        raise ValueError("Prediction hash differs from pre-holdout/final lock")
                recorded = pd.read_csv(prediction_path)
                if set(recorded.columns) != {"sample_id", "Env", "Hybrid", "Year", "Experiment_Code", "prediction"}:
                    raise ValueError("Unexpected columns in unlabeled prediction CSV")
                for column in ["sample_id", "Env", "Hybrid", "Year", "Experiment_Code"]:
                    if not recorded[column].equals(samples[column]):
                        raise ValueError(f"Prediction rows do not align with feature input: {column}")
                if not np.isfinite(recorded.prediction.to_numpy()).all():
                    raise ValueError("Non-finite locked prediction")
                tick = time.monotonic()
                with threadpool_limits(limits=2):
                    if name == "CatBoost":
                        from catboost import CatBoostRegressor
                        model = CatBoostRegressor()
                        model.load_model(str(model_path))
                        params = model.get_all_params()
                        for key in ["iterations", "depth", "l2_leaf_reg", "random_seed"]:
                            if params[key] != config["params"][key]:
                                raise ValueError(f"Reloaded CatBoost parameter differs: {key}")
                        predicted = model.predict(features, thread_count=2)
                    else:
                        model = joblib.load(model_path)
                        if name == "Ridge":
                            if model.named_steps["ridge"].alpha != config["params"]["alpha"]:
                                raise ValueError("Reloaded Ridge alpha differs from freeze")
                        else:
                            actual = model.get_params()
                            for key, value in config["params"].items():
                                if actual[key] != value:
                                    raise ValueError(f"Reloaded Extra Trees parameter differs: {key}")
                            model.set_params(n_jobs=2)  # In-memory prediction parallelism only.
                        predicted = model.predict(features)
                predicted = np.asarray(predicted, float)
                if predicted.shape != (len(samples),) or not np.isfinite(predicted).all():
                    raise ValueError("Reloaded model produced invalid predictions")
                absolute_error = np.abs(predicted-recorded.prediction.to_numpy())
                maximum = float(absolute_error.max())
                passed = bool(np.all(absolute_error <= TOLERANCE))
                entry = {"model": name, "window": horizon, "prediction_rows": len(predicted),
                         "max_abs_error": maximum, "mean_abs_error": float(absolute_error.mean()),
                         "within_tolerance": passed, "reload_and_predict_seconds": time.monotonic()-tick}
                report["models"].append(entry)
                report["model_count_checked"] += 1
                report["prediction_values_checked"] += len(predicted)
                report["max_abs_error"] = max(report["max_abs_error"], maximum)
                print(json.dumps(entry), flush=True)
                if not passed:
                    raise ValueError(f"Replay mismatch for {name} W{horizon}: max abs error {maximum:.12g}")
                del model
            del features
        if report["model_count_checked"] != 15:
            raise ValueError("Not all 15 saved models were replayed")
        if sha256(completion_path) != completion_hash:
            raise ValueError("Completion manifest changed during replay")
        for path, expected_hash in verified.items():
            if sha256(PROJECT / path) != expected_hash:
                raise ValueError(f"Input changed during replay: {path}")
        report["status"] = "PASS"
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
    report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = time.monotonic()-started
    save_report(report, output)
    print(json.dumps({key: report[key] for key in ["status", "model_count_checked", "prediction_values_checked", "max_abs_error", "failure"]}), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError(report["failure"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    replay(args.output)
