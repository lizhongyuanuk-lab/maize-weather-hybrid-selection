"""Retrospective 2020-2023 rolling backtests, isolated from the original 2024 run.

Historical target files contain fold test years. They were already available to
this project: these folds are NOT fresh blind tests. Only allowed fitting IDs are
joined for development/refitting; test-year targets are joined after all preset
predictions are persisted and hash-locked. The 2024 holdout file is never opened.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from metrics import window_analysis
from run_experiments import (BASELINES, CANDIDATES, HORIZONS, SEED, arrays_for,
    baseline_predictions, digest, estimator, event, model_effects, now, persist_model,
    prediction_frame, score, subgroup_metrics, versions, write_json)

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "modeling/rolling_v1"
PREP = PROJECT / "preparation/outputs"
HISTORICAL_CAVEAT = "Retrospective backtest on previously available historical outcomes, not a new blind test; 2024 original experiment remains unchanged."


def load_fold_samples(directory, year, stage):
    samples = {s: pd.read_csv(directory/f"{s}_samples.csv") for s in ("train", "validation", "test")}
    allowed = {"train": set(range(2014, year-1 if stage == "development" else year)),
               "validation": {year-1} if stage == "development" else set(), "test": {year}}
    for split, frame in samples.items():
        if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
            raise ValueError(f"Invalid {split} sample IDs")
        if {"Yield_Mg_ha", "y_true", "target"} & set(frame.columns):
            raise ValueError("Label in model-facing sample metadata")
        if not set(frame.Year.unique()).issubset(allowed[split]):
            raise ValueError(f"Year leakage in {split}: {frame.Year.unique()}")
        if split != "validation" or stage == "development":
            if not len(frame):
                raise ValueError(f"Empty {split}")
    if stage == "final" and len(samples["validation"]):
        raise ValueError("Final validation must be empty")
    keys = [set(f.sample_id) for f in samples.values()]
    if any(keys[i] & keys[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("Overlapping temporal sample IDs")
    return samples


def targets_for_ids(ids, prepared_dir=PREP):
    """Filter each historical CSV chunk to allowed IDs before joining any values."""
    ids = pd.Index(ids)
    if ids.has_duplicates or ids.isna().any():
        raise ValueError("Target request IDs must be unique and nonmissing")
    if not len(ids):
        return np.empty(0, dtype=float)
    allowed, selected = set(ids), []
    for filename in ["train.csv", "validation.csv"]:
        for chunk in pd.read_csv(Path(prepared_dir)/"targets"/filename,
                                 usecols=["sample_id", "Yield_Mg_ha"], chunksize=50000):
            part = chunk.loc[chunk.sample_id.isin(allowed)]
            if len(part):
                selected.append(part)
    if not selected:
        raise ValueError("No allowed historical labels found")
    labels = pd.concat(selected, ignore_index=True)
    if labels.sample_id.duplicated().any():
        raise ValueError("Duplicate requested historical targets")
    result = pd.to_numeric(labels.set_index("sample_id").Yield_Mg_ha, errors="coerce").reindex(ids).to_numpy(float)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("Missing/nonfinite/nonpositive requested historical targets")
    return result


def artifact_hashes(paths, root):
    return {str(p.resolve().relative_to(root)): digest(p) for p in paths}


def verify_record(path, root, run_hash=None):
    record = json.loads(path.read_text())
    if Path(record["artifact_path_base"]).resolve() != root:
        raise ValueError("Cached artifact root differs")
    if run_hash and record["run_manifest_sha256"] != run_hash:
        raise ValueError("Cached unit belongs to a different input/code/configuration run")
    for relative, expected in record["artifacts"].items():
        artifact = (root/relative).resolve()
        if not artifact.is_relative_to(root) or digest(artifact) != expected:
            raise ValueError(f"Cached artifact mismatch: {relative}")
    return record


def verify_run_inputs(out, run_hash):
    path = out/"run_manifest.json"
    if digest(path) != run_hash:
        raise ValueError("Run manifest changed during execution")
    for input_path, expected in json.loads(path.read_text())["inputs"].items():
        if digest(Path(input_path)) != expected:
            raise ValueError(f"Run input changed during execution: {input_path}")


def initialize(year, stage, root, directory, out, prepared_dir, freeze_path=None):
    feature_manifest_path = directory/"feature_manifest.json"
    features = json.loads(feature_manifest_path.read_text())
    if (features["mode"] != stage or features["test_year"] != year
            or features["fit_years"] != [2014, year-2 if stage == "development" else year-1]):
        raise ValueError("Feature fitting period does not match rolling fold")
    if not features.get("checks") or not all(features["checks"].values()):
        raise ValueError("Feature preparation checks are absent or failed")
    for name, expected in features["outputs"].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError(f"Feature manifest output mismatch: {name}")
    required = [feature_manifest_path, root/"PROTOCOL.md", Path(__file__),
                Path(__file__).with_name("run_experiments.py"), Path(__file__).with_name("metrics.py"),
                prepared_dir/"manifest.json", prepared_dir/"targets/train.csv", prepared_dir/"targets/validation.csv"]
    required += [directory/f"{s}_samples.csv" for s in ["train", "validation", "test"]]
    required += [directory/f"W{w}{suffix}" for w in HORIZONS for suffix in [".npz", "_columns.json"]]
    for source, expected in features["input_hashes"].items():
        source = Path(source)
        if not source.is_absolute() or digest(source) != expected:
            raise ValueError(f"Feature source/code/protocol hash changed: {source}")
        required.append(source)
    if freeze_path:
        required.append(freeze_path)
    manifest = {"schema_version": 1, "stage": stage, "test_year": year, "validation_year": year-1,
                "artifact_path_base": str(root), "seed": SEED, "candidates": CANDIDATES,
                "inputs": {str(p.resolve()): digest(p) for p in required}, "versions": versions(),
                "retrospective_backtest": True, "original_2024_holdout_read": False,
                "historical_target_access": HISTORICAL_CAVEAT}
    if freeze_path:
        manifest["frozen_selection_sha256"] = digest(freeze_path)
    out.mkdir(parents=True, exist_ok=True)
    path = out/"run_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Inputs, code, protocol, versions or config changed; preserve results and use a new version root")
    if not path.exists():
        write_json(manifest, path)
    for name in ["predictions", "models", "units", "candidate_predictions", "feature_effects"]:
        (out/name).mkdir(exist_ok=True)
    return manifest, digest(path)


def validate_selection(path, root, year):
    frozen = json.loads(path.read_text())
    if frozen["test_year"] != year or frozen["validation_year"] != year-1:
        raise ValueError("Wrong fold in selection freeze")
    expected = {(m, w) for m in CANDIDATES for w in HORIZONS}
    if len(frozen["configs"]) != 15 or {(c["model"], c["window"]) for c in frozen["configs"]} != expected:
        raise ValueError("Freeze does not contain all 15 combinations")
    for c in frozen["configs"]:
        if c["params"] != CANDIDATES[c["model"]][c["candidate"]]:
            raise ValueError("Frozen parameters not in preset candidate list")
    if frozen["recommended_by_validation"] != min(frozen["configs"], key=lambda c: c["validation_macro_rmse"]):
        raise ValueError("Recommendation does not match validation-only selection")
    completion = verify_record(path.parent/"development_completion_manifest.json", root)
    if completion["artifacts"].get(str(path.relative_to(root))) != digest(path):
        raise ValueError("Selection is not bound to completed development")
    development_manifest = path.parent/"run_manifest.json"
    if digest(development_manifest) != frozen["development_run_manifest_sha256"]:
        raise ValueError("Development manifest changed")
    for input_path, expected_hash in json.loads(development_manifest.read_text())["inputs"].items():
        if digest(Path(input_path)) != expected_hash:
            raise ValueError(f"Frozen development input changed: {input_path}")
    return frozen


def complete_stage(out, root, year, stage, run_hash, **metadata):
    verify_run_inputs(out, run_hash)
    for unit in (out/"units").glob("*.json"):
        verify_record(unit, root, run_hash)
    filename = "development_completion_manifest.json" if stage == "development" else "final_evaluation_manifest.json"
    paths = [p for p in out.rglob("*") if p.is_file() and p.name not in [filename, "progress.jsonl"] and not p.name.endswith(".tmp")]
    record = {"completed_at_utc": now(), "test_year": year, "stage": stage,
              "artifact_path_base": str(root), "run_manifest_sha256": run_hash,
              "retrospective_backtest": True, "original_2024_holdout_read": False,
              "artifacts": artifact_hashes(paths, root), **metadata}
    write_json(record, out/filename)
    return record


def run_stage(test_year, stage, root=DEFAULT_ROOT, *, prepared_dir=PREP):
    if test_year not in [2020, 2021, 2022, 2023] or stage not in ["development", "final"]:
        raise ValueError("Only rolling test years 2020-2023 and development/final stages are allowed; 2024 is reused unchanged")
    root, prepared_dir = Path(root).resolve(), Path(prepared_dir).resolve()
    if root == PROJECT/"modeling/results" or root == PROJECT/"modeling":
        raise ValueError("Use an isolated rolling version root")
    directory, out = root/"features"/str(test_year)/stage, root/"results"/str(test_year)/stage
    freeze_path = root/"results"/str(test_year)/"development/frozen_selection.json"
    frozen = validate_selection(freeze_path, root, test_year) if stage == "final" else None
    manifest, run_hash = initialize(test_year, stage, root, directory, out, prepared_dir, freeze_path if frozen else None)
    completion_path = out/("development_completion_manifest.json" if stage == "development" else "final_evaluation_manifest.json")
    if completion_path.exists():
        completed = verify_record(completion_path, root, run_hash)
        event(out, "resume_completed_stage_no_fit", test_year=test_year, stage=stage)
        return json.loads(freeze_path.read_text()) if stage == "development" else completed
    samples = load_fold_samples(directory, test_year, stage)
    train_y = targets_for_ids(samples["train"].sample_id, prepared_dir)
    validation_y = targets_for_ids(samples["validation"].sample_id, prepared_dir) if stage == "development" else None
    target_split = "validation" if stage == "development" else "test"
    selected_rows, candidates_rows, configurations, predictions_to_score = [], [], [], []
    fit_times = {}
    for name in BASELINES:
        unit_path = out/"units"/f"{name}_W0.json"
        prediction_path = out/"predictions"/f"{name}.csv"
        if unit_path.exists():
            record = verify_record(unit_path, root, run_hash)
        else:
            frame = prediction_frame(samples[target_split], baseline_predictions(name, samples["train"], train_y, samples[target_split]))
            frame.to_csv(prediction_path, index=False)
            paths = [prediction_path]
            record = {"model": name, "window": 0, "fit_seconds": 0., "test_year": test_year}
            if stage == "development":
                summary, environment = score(frame, validation_y)
                record["selected"] = {"model": name, "window": 0, "candidate": "baseline", "fit_seconds": 0., **summary}
                environment_path = out/f"{name}_environment_metrics.csv"
                environment.to_csv(environment_path, index=False)
                paths.append(environment_path)
            record.update(artifact_path_base=str(root), run_manifest_sha256=run_hash, artifacts=artifact_hashes(paths, root))
            write_json(record, unit_path)
        if stage == "development":
            selected_rows.append(record["selected"])
        predictions_to_score.append((name, 0, prediction_path))
        fit_times[(name, 0)] = 0.
    for horizon in HORIZONS:
        arrays = arrays_for(directory, horizon, samples)
        columns = json.loads((directory/f"W{horizon}_columns.json").read_text())["columns"]
        for name in CANDIDATES:
            unit_path = out/"units"/f"{name}_W{horizon}.json"
            prediction_path = out/"predictions"/f"{name}_W{horizon}.csv"
            if unit_path.exists():
                record = verify_record(unit_path, root, run_hash)
                event(out, "resume_completed_unit_no_fit", test_year=test_year, stage=stage, model=name, window=horizon)
            else:
                choices = list(enumerate(CANDIDATES[name])) if stage == "development" else [
                    (c["candidate"], c["params"]) for c in frozen["configs"] if c["model"] == name and c["window"] == horizon]
                best, best_model, best_frame, best_environment, unit_candidates, artifacts = None, None, None, None, [], []
                for index, params in choices:
                    event(out, "fit_start", test_year=test_year, stage=stage, model=name, window=horizon, candidate=index, params=params)
                    model = estimator(name, params)
                    start = time.monotonic()
                    with threadpool_limits(limits=4):
                        model.fit(arrays["train"], train_y)
                        seconds = time.monotonic()-start
                        frame = prediction_frame(samples[target_split], model.predict(arrays[target_split]))
                    entry = {"model": name, "window": horizon, "candidate": index, "params": params, "fit_seconds": seconds}
                    if stage == "development":
                        summary, environment = score(frame, validation_y)
                        entry.update(summary)
                        candidate_path = out/"candidate_predictions"/f"{name}_W{horizon}_c{index}.csv"
                        candidate_environment = candidate_path.with_name(candidate_path.stem+"_environments.csv")
                        frame.to_csv(candidate_path, index=False)
                        environment.to_csv(candidate_environment, index=False)
                        artifacts.extend([candidate_path, candidate_environment])
                        unit_candidates.append(entry)
                    else:
                        environment = None
                    if best is None or (stage == "development" and entry["macro_rmse"] < best["macro_rmse"]):
                        best, best_model, best_frame, best_environment = entry, model, frame, environment
                    event(out, "fit_complete", test_year=test_year, stage=stage, **entry)
                best_frame.to_csv(prediction_path, index=False)
                model_path = persist_model(best_model, name, out/"models"/f"{name}_W{horizon}")
                effects_path = out/"feature_effects"/f"{name}_W{horizon}.csv"
                model_effects(best_model, name, columns, effects_path)
                artifacts.extend([prediction_path, model_path, effects_path])
                if best_environment is not None:
                    env_path = out/f"{name}_W{horizon}_environment_metrics.csv"
                    best_environment.to_csv(env_path, index=False)
                    artifacts.append(env_path)
                record = {"model": name, "window": horizon, "test_year": test_year,
                          "selected": best, "candidates": unit_candidates, "fit_seconds": best["fit_seconds"],
                          "artifact_path_base": str(root), "run_manifest_sha256": run_hash,
                          "artifacts": artifact_hashes(artifacts, root)}
                if frozen:
                    record["frozen_selection_sha256"] = digest(freeze_path)
                write_json(record, unit_path)
                del model, best_model
            if stage == "development":
                selected_rows.append(record["selected"])
                candidates_rows.extend(record["candidates"])
                choice = record["selected"]
                configurations.append({"model": name, "window": horizon, "candidate": choice["candidate"],
                                       "params": choice["params"], "validation_macro_rmse": choice["macro_rmse"]})
                pd.DataFrame(selected_rows).drop(columns="params", errors="ignore").assign(test_year=test_year, validation_year=test_year-1).to_csv(out/"selected_metrics.csv", index=False)
                pd.DataFrame(candidates_rows).assign(params=lambda f: f.params.map(json.dumps), test_year=test_year).to_csv(out/"candidate_metrics.csv", index=False)
            predictions_to_score.append((name, horizon, prediction_path))
            fit_times[(name, horizon)] = record["fit_seconds"]
        del arrays
    if stage == "development":
        verify_run_inputs(out, run_hash)
        frozen = {"schema_version": 1, "test_year": test_year, "validation_year": test_year-1,
                  "frozen_at_utc": now(), "selection_basis": f"{test_year-1} environment-equal RMSE only",
                  "seed": SEED, "windows": list(HORIZONS), "configs": configurations,
                  "recommended_by_validation": min(configurations, key=lambda c: c["validation_macro_rmse"]),
                  "baselines": list(BASELINES), "development_run_manifest_sha256": run_hash,
                  "development_input_hashes": manifest["inputs"], "artifact_path_base": str(root),
                  "retrospective_backtest": True, "original_2024_holdout_read": False,
                  "historical_target_access": HISTORICAL_CAVEAT, "fold_test_outcomes_joined_for_selection": False,
                  "bootstrap_repeats": 2000, "versions": manifest["versions"]}
        if freeze_path.exists():
            previous = json.loads(freeze_path.read_text())
            frozen["frozen_at_utc"] = previous["frozen_at_utc"]
            if previous != frozen:
                raise ValueError("Existing frozen choice differs; preserve it")
        else:
            write_json(frozen, freeze_path)
        complete_stage(out, root, test_year, stage, run_hash, frozen_selection_sha256=digest(freeze_path))
        event(out, "development_selection_frozen", test_year=test_year, recommended=frozen["recommended_by_validation"])
        return frozen
    if len(predictions_to_score) != 17:
        raise ValueError("All 17 preset predictions must exist before historical test scoring")
    verify_run_inputs(out, run_hash)
    for unit in (out/"units").glob("*.json"):
        verify_record(unit, root, run_hash)
    lock = {"test_year": test_year, "retrospective_backtest": True,
            "artifact_path_base": str(root), "run_manifest_sha256": run_hash,
            "frozen_selection_sha256": digest(freeze_path),
            "prediction_hashes": artifact_hashes([p for _, _, p in predictions_to_score], root),
            "test_sample_ids_sha256": digest(directory/"test_samples.csv")}
    lock_path = out/"prediction_lock_before_holdout.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Predictions changed after historical test prediction lock")
    write_json(lock, lock_path)
    event(out, "all_predictions_locked", test_year=test_year, count=17, prediction_lock_sha256=digest(lock_path))
    test_y = targets_for_ids(samples["test"].sample_id, prepared_dir)
    event(out, "historical_test_targets_joined", test_year=test_year, n=len(test_y), caveat=HISTORICAL_CAVEAT)
    summaries, environments, groups, tables, sufficient = [], [], [], [], {}
    by_model = {m: {} for m in CANDIDATES}
    for name, horizon, path in predictions_to_score:
        frame = pd.read_csv(path)
        if frame.sample_id.tolist() != samples["test"].sample_id.tolist():
            raise ValueError("Prediction cohort/order mismatch")
        frame["y_true"] = test_y
        summary, env = score(frame.drop(columns="y_true"), test_y)
        summaries.append({"test_year": test_year, "model": name, "window": horizon,
                          "fit_seconds": fit_times[(name, horizon)], "n_prediction_inputs": len(frame),
                          "n_without_observed_target": 0, **summary})
        environments.append(env.assign(test_year=test_year, model=name, window=horizon))
        groups.extend(subgroup_metrics(frame, samples["train"], name, horizon))
        frame.to_csv(out/"predictions"/f"{name}_W{horizon}_evaluated.csv", index=False)
        if name in by_model:
            by_model[name][horizon] = env
    for name, window_metrics in by_model.items():
        table, conclusion = window_analysis(window_metrics)
        tables.append(table.assign(test_year=test_year, model=name))
        sufficient[name] = conclusion
    pd.DataFrame(summaries).to_csv(out/"summary_metrics.csv", index=False)
    pd.concat(environments, ignore_index=True).to_csv(out/"environment_metrics.csv", index=False)
    pd.DataFrame(groups).assign(test_year=test_year).to_csv(out/"subgroup_metrics.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv(out/"weather_increment_comparisons.csv", index=False)
    write_json({"test_year": test_year, "validation_year": test_year-1, "models": sufficient,
                "recommended_by_validation": frozen["recommended_by_validation"],
                "test_result_selection_permitted": False, "retrospective_backtest": True,
                "site_group_definition": "Experiment_Code proxy; not guaranteed unique physical site identity",
                "interpretation": HISTORICAL_CAVEAT+" Environment intervals are not multiplicity adjusted and do not include model-selection uncertainty."}, out/"earliest_sufficient_windows.json")
    completed = complete_stage(out, root, test_year, stage, run_hash,
                              n_scored=len(test_y), n_prediction_inputs=len(samples["test"]), preset_models_scored=17,
                              frozen_selection_sha256=digest(freeze_path),
                              recommended_by_validation=frozen["recommended_by_validation"])
    event(out, "final_evaluation_complete", test_year=test_year, n_scored=len(test_y), models=17)
    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, choices=[2020, 2021, 2022, 2023])
    parser.add_argument("--stage", required=True, choices=["development", "final"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    run_stage(args.year, args.stage, args.root)
