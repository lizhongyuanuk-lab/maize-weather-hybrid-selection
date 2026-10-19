"""Isolated, resumable segmented-weather representation experiment.

Only W60/W90/W120 fit new models: 18 development + 9 final fits per year.
W0/W30 and both baselines reuse hash-bound cumulative predictions. All results
are retrospective exploratory comparisons, never new blind confirmation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from metrics import window_analysis
from run_experiments import (BASELINES, CANDIDATES, HORIZONS, PROJECT, PREP, SEED,
    arrays_for, baseline_predictions, digest, estimator, event, model_effects, now,
    persist_model, prediction_frame, score, subgroup_metrics, versions, write_json)
from run_rolling_experiments import load_fold_samples, targets_for_ids
from segmented_sources import IDENTITY, YEARS, read_json, source_fold

DEFAULT_ROOT = PROJECT/"modeling/segmented_v1"
DEFAULT_CUMULATIVE_ROOT = PROJECT/"modeling/rolling_v1"
NEW_WINDOWS = (60, 90, 120)
REPRESENTATION = "segmented_30d"
CAVEAT = "Exploratory retrospective representation experiment designed after cumulative test results were inspected; not a new blind test or a new algorithm."


@contextmanager
def stage_lock(out):
    """Non-blocking process lock; automatic release on failure/process exit."""
    out.mkdir(parents=True, exist_ok=True)
    with (out/".stage.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Segmented stage is already running: {out}") from exc
        try:
            lock.seek(0)
            lock.truncate()
            lock.write(json.dumps({"pid": os.getpid(), "acquired_at_utc": now()}))
            lock.flush()
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def hash_paths(paths, base):
    base = Path(base).resolve()
    return {str(Path(p).resolve().relative_to(base)): digest(p) for p in paths}


def verify_unit(path, root, run_hash):
    record = read_json(path)
    if record["run_manifest_sha256"] != run_hash or Path(record["artifact_path_base"]).resolve() != root:
        raise ValueError("Cached segmented unit belongs to a different run/root")
    for relative, expected in record["artifacts"].items():
        path = (root/relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError(f"Cached segmented artifact changed: {relative}")
    if record.get("source_artifacts") and record.get("source_artifact_path_base") != "absolute":
        raise ValueError("External source artifact base must be explicit")
    for source, expected in record.get("source_artifacts", {}).items():
        if not Path(source).is_absolute() or digest(source) != expected:
            raise ValueError(f"Reused source artifact changed: {source}")
    return record


def verify_run_inputs(out, run_hash):
    path = out/"run_manifest.json"
    if digest(path) != run_hash:
        raise ValueError("Segmented run manifest changed")
    for name, expected in read_json(path)["inputs"].items():
        if not Path(name).is_absolute() or digest(name) != expected:
            raise ValueError(f"Segmented run input changed: {name}")


def verify_features(directory, source, year, stage):
    """Check exact common representation and base/global-quality invariants."""
    manifest_path = directory/"feature_manifest.json"
    feature = read_json(manifest_path)
    if (feature["mode"] != stage or feature["test_year"] != year
            or feature["fit_years"] != [2014, year-(2 if stage == "development" else 1)]
            or feature.get("representation") != REPRESENTATION
            or feature.get("physical_weather_features") != 21
            or feature.get("quality_feature_count") != 9
            or not feature.get("checks") or not all(feature["checks"].values())):
        raise ValueError("Segmented feature identity, period or preparation checks differ")
    hashes = {str(manifest_path): digest(manifest_path)}
    for name, expected in feature["outputs"].items():
        path = (directory/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(directory) or digest(path) != expected:
            raise ValueError(f"Segmented feature artifact mismatch: {name}")
        hashes[str(path)] = expected
    required = {f"W{h}{suffix}" for h in HORIZONS for suffix in (".npz", "_columns.json")}
    required |= {f"{s}_samples.csv" for s in ("train", "validation", "test")}
    required.add("preprocessing.joblib")
    if not required.issubset(feature["outputs"]):
        raise ValueError("Segmented feature inventory incomplete")
    for name, expected in feature["input_hashes"].items():
        if not Path(name).is_absolute() or digest(name) != expected:
            raise ValueError(f"Segmented feature input changed: {name}")
        hashes[name] = expected
    for filename in ["W0.npz", "W30.npz", "W0_columns.json", "W30_columns.json",
                     "train_samples.csv", "validation_samples.csv", "test_samples.csv", "preprocessing.joblib"]:
        if digest(directory/filename) != digest(source.feature_directory/filename):
            raise ValueError(f"Shared features/metadata must be byte-identical: {filename}")
    samples = load_fold_samples(directory, year, stage)
    base_columns = read_json(directory/"W0_columns.json")["columns"]
    base_count = len(base_columns)
    if feature["base_feature_count"] != base_count:
        raise ValueError("Base feature count differs")
    base = arrays_for(directory, 0, samples)
    dimensions = {0: base_count}
    for horizon in HORIZONS[1:]:
        current = arrays_for(directory, horizon, samples)
        old = arrays_for(source.feature_directory, horizon, samples)
        columns = read_json(directory/f"W{horizon}_columns.json")["columns"]
        old_columns = read_json(source.feature_directory/f"W{horizon}_columns.json")["columns"]
        expected_count = base_count+21*(horizon//30)+9
        if len(columns) != expected_count or columns[:base_count] != base_columns or columns[-9:] != old_columns[-9:]:
            raise ValueError("Segmented physical/global-quality column structure differs")
        for split in samples:
            if current[split].dtype != np.float32:
                raise ValueError("Segmented arrays must retain float32 representation")
            if not np.array_equal(current[split][:, :base_count], base[split]):
                raise ValueError("Segmented base/PCA/static values differ from source W0")
            if not np.array_equal(current[split][:, -9:], old[split][:, -9:]):
                raise ValueError("Whole-horizon quality features changed")
        dimensions[horizon] = expected_count
    return samples, hashes, dimensions


def initialize(year, stage, root, directory, out, source, feature_hashes, prepared_dir, freeze_path):
    files = [Path(__file__), Path(__file__).with_name("segmented_sources.py"),
             Path(__file__).with_name("run_experiments.py"), Path(__file__).with_name("run_rolling_experiments.py"),
             Path(__file__).with_name("metrics.py"), root/"PROTOCOL.md", prepared_dir/"manifest.json",
             prepared_dir/"targets/train.csv", prepared_dir/"targets/validation.csv"]
    if stage == "final":
        files.append(freeze_path)
    inputs = {**source.hashes, **feature_hashes, **{str(p.resolve()): digest(p) for p in files}}
    manifest = {"schema_version": 1, "experiment": "segmented_v1", "representation": REPRESENTATION,
                "test_year": year, "validation_year": year-1, "stage": stage,
                "artifact_path_base": str(root), "source_input_path_base": "absolute", "inputs": inputs,
                "candidates": CANDIDATES, "seed": SEED, "versions": versions(),
                "new_windows": list(NEW_WINDOWS), "shared_windows": [0, 30],
                "expected_new_model_fits": 18 if stage == "development" else 9,
                "expected_reused_units": 8, "retrospective_backtest": True, "interpretation": CAVEAT}
    if stage == "final":
        manifest["frozen_selection_sha256"] = digest(freeze_path)
    path = out/"run_manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Segmented source/code/protocol/inputs/config changed; preserve results and use a new root")
    if not path.exists():
        write_json(manifest, path)
    for name in ("models", "predictions", "units", "candidate_predictions", "feature_effects"):
        (out/name).mkdir(exist_ok=True)
    return manifest, digest(path)


def validate_freeze(path, root, year):
    frozen = read_json(path)
    if frozen["test_year"] != year or frozen["validation_year"] != year-1 or frozen["representation"] != REPRESENTATION:
        raise ValueError("Segmented freeze identity differs")
    expected = {(m, h) for m in CANDIDATES for h in HORIZONS}
    if len(frozen["configs"]) != 15 or {(c["model"], c["window"]) for c in frozen["configs"]} != expected:
        raise ValueError("Segmented freeze must contain all 15 configurations")
    for config in frozen["configs"]:
        if (config["candidate"] not in (0, 1) or config["params"] != CANDIDATES[config["model"]][config["candidate"]]
                or config["reused"] != (config["window"] in (0, 30))):
            raise ValueError("Frozen parameters or reuse policy differ")
    if frozen["recommended_by_validation"] != min(frozen["configs"], key=lambda c: c["validation_macro_rmse"]):
        raise ValueError("Segmented recommendation not selected by validation-only RMSE")
    run_hash = digest(path.parent/"run_manifest.json")
    if run_hash != frozen["development_run_manifest_sha256"]:
        raise ValueError("Segmented development manifest changed")
    completed = verify_unit(path.parent/"development_completion_manifest.json", root, run_hash)
    if completed["artifacts"].get(str(path.relative_to(root))) != digest(path):
        raise ValueError("Segmented freeze not bound to completed development")
    verify_run_inputs(path.parent, run_hash)
    return frozen


def complete(out, root, year, stage, run_hash, **metadata):
    verify_run_inputs(out, run_hash)
    for path, expected in metadata.get("source_artifacts", {}).items():
        if digest(path) != expected:
            raise ValueError("Scoring target source changed during evaluation")
    units = list((out/"units").glob("*.json"))
    if len(units) != 17:
        raise ValueError("Exactly 17 completed units required")
    records = [verify_unit(p, root, run_hash) for p in units]
    if sum(r["reused"] for r in records) != 8:
        raise ValueError("Exactly eight shared units required")
    filename = "development_completion_manifest.json" if stage == "development" else "final_evaluation_manifest.json"
    outputs = [p for p in out.rglob("*") if p.is_file() and p.name not in (filename, "progress.jsonl", ".stage.lock")
               and not p.name.endswith(".tmp")]
    record = {"completed_at_utc": now(), "stage": stage, "test_year": year,
              "representation": REPRESENTATION, "artifact_path_base": str(root), "run_manifest_sha256": run_hash,
              "artifacts": hash_paths(outputs, root), "n_new_model_fits": 18 if stage == "development" else 9,
              "n_reused_units": 8, "retrospective_backtest": True, "interpretation": CAVEAT, **metadata}
    write_json(record, out/filename)
    return record


def reuse_unit(name, horizon, source, samples, train_y, validation_y, out, root, run_hash, dimensions, frozen_hash):
    source_path = source.prediction_path(name, horizon)
    target = "validation" if source.stage == "development" else "test"
    frame = pd.read_csv(source_path)
    if set(frame.columns) != set(IDENTITY+["prediction"]) or not frame[IDENTITY].equals(samples[target][IDENTITY]):
        raise ValueError("Reused prediction cohort/order/metadata differs")
    if not np.isfinite(frame.prediction.to_numpy()).all():
        raise ValueError("Reused predictions must be finite")
    source_artifacts = {str(source_path): digest(source_path)}
    baseline = name in BASELINES
    if baseline:
        expected = baseline_predictions(name, samples["train"], train_y, samples[target])
        if not np.allclose(frame.prediction.to_numpy(), expected, atol=1e-12, rtol=1e-12):
            raise ValueError("Reused baseline differs from same-fold training-only calculation")
        selected = dict(source.baseline_selected[name]) if source.stage == "development" else {}
        source_fit_seconds, candidates = 0., []
    else:
        dev_unit = source.development_units[(name, horizon)]
        selected = dict(dev_unit["selected"])
        candidates = [dict(c, fit_seconds=0., source_fit_seconds=c["fit_seconds"], reused=True,
                           fit_kind="reused_identical_features", representation=REPRESENTATION) for c in dev_unit["candidates"]]
        original_unit = source.units[(name, horizon)]
        source_fit_seconds = original_unit.get("fit_seconds", selected["fit_seconds"])
        for relative, expected in original_unit["artifacts"].items():
            source_artifacts[str((source.artifact_base/relative).resolve())] = expected
        for directory in {source.directory, source.development_directory}:
            path = directory/"units"/f"{name}_W{horizon}.json"
            source_artifacts[str(path)] = digest(path)
        for candidate in dev_unit["candidates"]:
            for suffix in (".csv", "_environments.csv"):
                path = source.development_directory/"candidate_predictions"/f"{name}_W{horizon}_c{candidate['candidate']}{suffix}"
                source_artifacts[str(path)] = digest(path)
    prediction_path = out/"predictions"/source_path.name
    shutil.copyfile(source_path, prediction_path)
    selected.update(model=name, window=horizon, fit_seconds=0., source_fit_seconds=source_fit_seconds,
                    reused=True, fit_kind="reused_identical_features", representation=REPRESENTATION,
                    n_features=0 if baseline else dimensions[horizon])
    paths = [prediction_path]
    if source.stage == "development":
        summary, environment = score(frame, validation_y)
        if not np.isclose(summary["macro_rmse"], selected["macro_rmse"], atol=1e-10, rtol=1e-10):
            raise ValueError("Reused validation score differs from source selection")
        environment_path = out/f"{name}_W{horizon}_environment_metrics.csv"
        environment.to_csv(environment_path, index=False)
        paths.append(environment_path)
    record = {"model": name, "window": horizon, "test_year": source.year, "stage": source.stage,
              "representation": REPRESENTATION, "reused": True, "fit_kind": "reused_identical_features",
              "fit_seconds": 0., "source_fit_seconds": source_fit_seconds,
              "selected": selected, "candidates": candidates, "n_features": 0 if baseline else dimensions[horizon],
              "artifact_path_base": str(root), "run_manifest_sha256": run_hash,
              "artifacts": hash_paths(paths, root), "source_artifact_path_base": "absolute",
              "source_artifacts": source_artifacts}
    if frozen_hash:
        record["frozen_selection_sha256"] = frozen_hash
    write_json(record, out/"units"/f"{name}_W{horizon}.json")
    event(out, "source_unit_reused_no_fit", model=name, window=horizon, stage=source.stage, test_year=source.year)
    return record


def scoring_targets(year, samples, prepared_dir, source, out):
    """Called only after the new 17-prediction lock, including for 2024."""
    if year == 2024:
        path = prepared_dir/"holdout/test_targets.csv"
        target_hashes = {str(path): source.completion["holdout_sha256"]}
        if digest(path) != target_hashes[str(path)]:
            raise ValueError("Original 2024 target source changed")
        event(out, "historical_2024_targets_opened_after_prediction_lock", path=str(path), sha256=digest(path))
        labels = pd.read_csv(path, usecols=["sample_id", "Yield_Mg_ha"])
        if labels.sample_id.duplicated().any() or not set(labels.sample_id).issubset(set(samples.sample_id)):
            raise ValueError("2024 label cohort is duplicated or outside prediction inputs")
        values = pd.to_numeric(labels.Yield_Mg_ha, errors="coerce")
        labels = labels.loc[np.isfinite(values) & (values > 0)].copy()
        labels["Yield_Mg_ha"] = values.loc[labels.index]
    else:
        target_hashes = {str(prepared_dir/"targets"/name): digest(prepared_dir/"targets"/name)
                         for name in ("train.csv", "validation.csv")}
        labels = pd.DataFrame({"sample_id": samples.sample_id, "Yield_Mg_ha": targets_for_ids(samples.sample_id, prepared_dir)})
        event(out, "historical_test_targets_joined", test_year=year, n=len(labels))
    # Reuse precisely the cumulative scoring intersection; these values are only
    # opened here, after locking all new predictions, never during selection.
    reference = pd.read_csv(source.evaluated_reference)
    if reference.sample_id.duplicated().any() or len(labels) != source.completion["n_scored"]:
        raise ValueError("Scoring support changed from cumulative source")
    joined = samples[IDENTITY].merge(labels, on="sample_id", how="inner", validate="one_to_one")
    a, b = joined.sort_values("sample_id"), reference.sort_values("sample_id")
    if not a[IDENTITY].reset_index(drop=True).equals(b[IDENTITY].reset_index(drop=True)):
        raise ValueError("Scoring identities differ from completed cumulative experiment")
    if not np.array_equal(a.Yield_Mg_ha.to_numpy(), b.y_true.to_numpy()):
        raise ValueError("Scoring observations differ from completed cumulative experiment")
    return labels, target_hashes


def run_stage(test_year, stage, root=DEFAULT_ROOT, cumulative_root=DEFAULT_CUMULATIVE_ROOT, *, prepared_dir=PREP):
    if test_year not in YEARS or stage not in ("development", "final"):
        raise ValueError("Only 2020-2024 and development/final are allowed")
    root, cumulative_root, prepared_dir = Path(root).resolve(), Path(cumulative_root).resolve(), Path(prepared_dir).resolve()
    if root.is_relative_to(cumulative_root) or cumulative_root.is_relative_to(root):
        raise ValueError("Segmented output root must be isolated from cumulative root")
    for old in (PROJECT/"modeling/results", PROJECT/"modeling/features"):
        if root.is_relative_to(old) or old.is_relative_to(root):
            raise ValueError("Segmented output root must not overlap original artifacts")
    out = root/"results"/str(test_year)/stage
    with stage_lock(out):
        return _run_stage(test_year, stage, root, cumulative_root, prepared_dir, out)


def _run_stage(year, stage, root, cumulative_root, prepared_dir, out):
    source = source_fold(year, stage, cumulative_root)
    directory = root/"features"/str(year)/stage
    freeze_path = root/"results"/str(year)/"development/frozen_selection.json"
    frozen = validate_freeze(freeze_path, root, year) if stage == "final" else None
    samples, feature_hashes, dimensions = verify_features(directory, source, year, stage)
    manifest, run_hash = initialize(year, stage, root, directory, out, source, feature_hashes, prepared_dir, freeze_path)
    completion_path = out/("development_completion_manifest.json" if stage == "development" else "final_evaluation_manifest.json")
    if completion_path.exists():
        completed = verify_unit(completion_path, root, run_hash)
        event(out, "resume_completed_stage_no_fit", test_year=year, stage=stage)
        return read_json(freeze_path) if stage == "development" else completed
    train_y = targets_for_ids(samples["train"].sample_id, prepared_dir)
    validation_y = targets_for_ids(samples["validation"].sample_id, prepared_dir) if stage == "development" else None
    frozen_hash = digest(freeze_path) if frozen else None
    selected_rows, candidate_rows, configs, prediction_records = [], [], [], []
    pairs = [(name, 0) for name in BASELINES]+[(name, h) for h in HORIZONS for name in CANDIDATES]
    active_horizon, arrays = None, None
    for name, horizon in pairs:
        unit_path = out/"units"/f"{name}_W{horizon}.json"
        if unit_path.exists():
            record = verify_unit(unit_path, root, run_hash)
            if record["model"] != name or record["window"] != horizon or record["reused"] != (horizon not in NEW_WINDOWS):
                raise ValueError("Cached segmented unit identity differs")
            event(out, "resume_completed_unit_no_fit", test_year=year, stage=stage, model=name, window=horizon)
        elif horizon not in NEW_WINDOWS:
            record = reuse_unit(name, horizon, source, samples, train_y, validation_y, out, root, run_hash, dimensions, frozen_hash)
        else:
            if active_horizon != horizon:
                arrays, active_horizon = arrays_for(directory, horizon, samples), horizon
            columns = read_json(directory/f"W{horizon}_columns.json")["columns"]
            choices = list(enumerate(CANDIDATES[name])) if stage == "development" else [
                (c["candidate"], c["params"]) for c in frozen["configs"] if c["model"] == name and c["window"] == horizon]
            best, best_model, best_frame, best_environment, candidates, artifacts = None, None, None, None, [], []
            for index, params in choices:
                event(out, "fit_start", test_year=year, stage=stage, model=name, window=horizon, candidate=index, params=params)
                model = estimator(name, params)
                start = time.monotonic()
                with threadpool_limits(limits=4):
                    model.fit(arrays["train"], train_y)
                    seconds = time.monotonic()-start
                    target = "validation" if stage == "development" else "test"
                    frame = prediction_frame(samples[target], model.predict(arrays[target]))
                entry = {"model": name, "window": horizon, "candidate": index, "params": params,
                         "fit_seconds": seconds, "source_fit_seconds": 0., "reused": False,
                         "fit_kind": "new_segmented_fit", "representation": REPRESENTATION, "n_features": dimensions[horizon]}
                environment = None
                if stage == "development":
                    summary, environment = score(frame, validation_y)
                    entry.update(summary)
                    candidate_path = out/"candidate_predictions"/f"{name}_W{horizon}_c{index}.csv"
                    candidate_environment = candidate_path.with_name(candidate_path.stem+"_environments.csv")
                    frame.to_csv(candidate_path, index=False)
                    environment.to_csv(candidate_environment, index=False)
                    artifacts.extend([candidate_path, candidate_environment])
                    candidates.append(entry)
                if best is None or (stage == "development" and entry["macro_rmse"] < best["macro_rmse"]):
                    best, best_model, best_frame, best_environment = entry, model, frame, environment
                event(out, "fit_complete", test_year=year, stage=stage, **entry)
            prediction_path = out/"predictions"/f"{name}_W{horizon}.csv"
            best_frame.to_csv(prediction_path, index=False)
            model_path = persist_model(best_model, name, out/"models"/f"{name}_W{horizon}")
            effects_path = out/"feature_effects"/f"{name}_W{horizon}.csv"
            model_effects(best_model, name, columns, effects_path)
            artifacts.extend([prediction_path, model_path, effects_path])
            if best_environment is not None:
                path = out/f"{name}_W{horizon}_environment_metrics.csv"
                best_environment.to_csv(path, index=False)
                artifacts.append(path)
            record = {"model": name, "window": horizon, "test_year": year, "stage": stage,
                      "representation": REPRESENTATION, "reused": False, "fit_kind": "new_segmented_fit",
                      "selected": best, "candidates": candidates, "fit_seconds": best["fit_seconds"],
                      "source_fit_seconds": 0., "n_features": dimensions[horizon],
                      "artifact_path_base": str(root), "run_manifest_sha256": run_hash, "artifacts": hash_paths(artifacts, root)}
            if frozen_hash:
                record["frozen_selection_sha256"] = frozen_hash
            write_json(record, unit_path)
            del model, best_model
        filename = f"{name}.csv" if name in BASELINES else f"{name}_W{horizon}.csv"
        prediction_records.append((record, out/"predictions"/filename))
        if stage == "development":
            selected_rows.append(record["selected"])
            candidate_rows.extend(record["candidates"])
            if name in CANDIDATES:
                choice = record["selected"]
                configs.append({"model": name, "window": horizon, "candidate": choice["candidate"], "params": choice["params"],
                                "validation_macro_rmse": choice["macro_rmse"], "reused": record["reused"]})
            pd.DataFrame(selected_rows).drop(columns="params", errors="ignore").assign(test_year=year, validation_year=year-1).to_csv(out/"selected_metrics.csv", index=False)
            if candidate_rows:
                pd.DataFrame(candidate_rows).assign(params=lambda f: f.params.map(json.dumps), test_year=year).to_csv(out/"candidate_metrics.csv", index=False)
    if stage == "development":
        verify_run_inputs(out, run_hash)
        frozen = {"schema_version": 1, "frozen_at_utc": now(), "representation": REPRESENTATION,
                  "test_year": year, "validation_year": year-1, "windows": list(HORIZONS), "seed": SEED,
                  "configs": configs, "recommended_by_validation": min(configs, key=lambda c: c["validation_macro_rmse"]),
                  "selection_basis": f"{year-1} environment-equal RMSE only; independent candidates for new W60/W90/W120 representation",
                  "baselines": list(BASELINES), "artifact_path_base": str(root), "development_run_manifest_sha256": run_hash,
                  "development_input_hashes": manifest["inputs"], "retrospective_backtest": True,
                  "bootstrap_repeats": 2000, "n_new_model_fits": 18, "n_reused_units": 8,
                  "versions": manifest["versions"], "interpretation": CAVEAT}
        if freeze_path.exists():
            previous = read_json(freeze_path)
            frozen["frozen_at_utc"] = previous["frozen_at_utc"]
            if previous != frozen:
                raise ValueError("Existing segmented selection freeze differs")
        else:
            write_json(frozen, freeze_path)
        complete(out, root, year, stage, run_hash, frozen_selection_sha256=digest(freeze_path))
        event(out, "development_selection_frozen", test_year=year, recommended=frozen["recommended_by_validation"])
        return frozen
    if len(prediction_records) != 17:
        raise ValueError("All 17 predictions required before connecting historical test outcomes")
    verify_run_inputs(out, run_hash)
    for path in (out/"units").glob("*.json"):
        verify_unit(path, root, run_hash)
    lock = {"test_year": year, "representation": REPRESENTATION, "artifact_path_base": str(root),
            "run_manifest_sha256": run_hash, "frozen_selection_sha256": frozen_hash,
            "prediction_hashes": hash_paths([p for _, p in prediction_records], root),
            "test_sample_ids_sha256": digest(directory/"test_samples.csv"), "retrospective_backtest": True}
    lock_path = out/"prediction_lock_before_holdout.json"
    if lock_path.exists() and read_json(lock_path) != lock:
        raise ValueError("Segmented predictions changed after lock")
    write_json(lock, lock_path)
    event(out, "all_predictions_locked", test_year=year, count=17, prediction_lock_sha256=digest(lock_path))
    labels, target_hashes = scoring_targets(year, samples["test"], prepared_dir, source, out)
    write_json({"test_year": year, "connected_after_prediction_lock": True,
                "prediction_lock_sha256": digest(lock_path), "target_source_path_base": "absolute",
                "target_source_hashes": target_hashes,
                "cumulative_scoring_reference_path": str(source.evaluated_reference),
                "cumulative_scoring_reference_sha256": digest(source.evaluated_reference),
                "n_prediction_inputs": len(samples["test"]), "n_scored": len(labels),
                "n_without_observed_target": len(samples["test"])-len(labels)}, out/"evaluation_target_receipt.json")
    summaries, environments, groups, tables, sufficient = [], [], [], [], {}
    by_model = {m: {} for m in CANDIDATES}
    for record, path in prediction_records:
        name, horizon = record["model"], record["window"]
        predictions = pd.read_csv(path)
        if not predictions[IDENTITY].equals(samples["test"][IDENTITY]):
            raise ValueError("Segmented prediction cohort/order changed")
        frame = predictions.merge(labels, on="sample_id", how="inner", validate="one_to_one").rename(columns={"Yield_Mg_ha": "y_true"})
        summary, environment = score(frame.drop(columns="y_true"), frame.y_true.to_numpy())
        summaries.append({"test_year": year, "model": name, "window": horizon, "representation": REPRESENTATION,
                          "fit_kind": record["fit_kind"], "reused": record["reused"], "fit_seconds": record["fit_seconds"],
                          "source_fit_seconds": record["source_fit_seconds"], "n_features": record["n_features"],
                          "n_prediction_inputs": len(predictions), "n_without_observed_target": len(predictions)-len(frame), **summary})
        environments.append(environment.assign(test_year=year, model=name, window=horizon, representation=REPRESENTATION))
        groups.extend(subgroup_metrics(frame, samples["train"], name, horizon))
        frame.to_csv(out/"predictions"/f"{name}_W{horizon}_evaluated.csv", index=False)
        if name in by_model:
            by_model[name][horizon] = environment
    for name, windows in by_model.items():
        table, conclusion = window_analysis(windows)
        tables.append(table.assign(test_year=year, model=name, representation=REPRESENTATION))
        sufficient[name] = conclusion
    pd.DataFrame(summaries).to_csv(out/"summary_metrics.csv", index=False)
    pd.concat(environments, ignore_index=True).to_csv(out/"environment_metrics.csv", index=False)
    pd.DataFrame(groups).assign(test_year=year, representation=REPRESENTATION).to_csv(out/"subgroup_metrics.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv(out/"weather_increment_comparisons.csv", index=False)
    write_json({"test_year": year, "validation_year": year-1, "representation": REPRESENTATION, "models": sufficient,
                "recommended_by_validation": frozen["recommended_by_validation"], "test_result_selection_permitted": False,
                "retrospective_backtest": True, "interpretation": CAVEAT,
                "site_group_definition": "Experiment_Code proxy, not verified physical-site identity"}, out/"earliest_sufficient_windows.json")
    completed = complete(out, root, year, stage, run_hash, n_scored=len(labels), n_prediction_inputs=len(samples["test"]),
                         n_without_observed_target=len(samples["test"])-len(labels), preset_models_scored=17,
                         source_artifact_path_base="absolute", source_artifacts=target_hashes,
                         frozen_selection_sha256=frozen_hash, recommended_by_validation=frozen["recommended_by_validation"])
    event(out, "final_evaluation_complete", test_year=year, n_scored=len(labels), models=17)
    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, choices=YEARS)
    parser.add_argument("--stage", required=True, choices=("development", "final"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cumulative-root", type=Path, default=DEFAULT_CUMULATIVE_ROOT)
    args = parser.parse_args()
    run_stage(args.year, args.stage, args.root, args.cumulative_root)
