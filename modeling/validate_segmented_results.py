"""Independent audit of the completed segmented-weather representation study.

No fitting and no production scorer/feature-builder imports.  Reuses only the
separately implemented rolling validator's arithmetic and file-checking helpers.
All new folds and analysis must be complete before this audit reads outcomes.
Physical weather calculations below are independent but sampled; base/quality
matrix equality, source hashes and prediction replay cover their full inventories.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import traceback
import unittest

import joblib
import numpy as np
import pandas as pd

from validate_rolling_results import (
    Audit, AuditError, PROJECT, PREP, YEARS, HORIZONS, MODELS, BASELINES,
    SEED, REPEATS, ORIGINAL_2024_SHA256, EXPECTED_COVERAGE, PRESET, IDENTITY,
    digest, read_json, timestamp, numerical_metrics, compare_record, paired_rows,
    annual_stability, cross_year_stability, read_predictions, baseline,
    within_year_window_rules,
)

ROOT = PROJECT / "modeling/segmented_v1"
CUMULATIVE_ROOT = PROJECT / "modeling/rolling_v1"
CUMULATIVE_ANALYSIS_SHA256 = "28b56456a512dff413b5d8a343daa2710df130f58e1fadd206f0126aaa0e575d"
PAIRS = {(m, h) for m in MODELS for h in HORIZONS}
CONFIGS = PAIRS | {(m, 0) for m in BASELINES}
SHARED = {(m, h) for m in MODELS for h in (0, 30)} | {(m, 0) for m in BASELINES}
NEW = PAIRS - SHARED
REFERENCES = ("cumulative", "no_weather")
SPLITS = ("train", "validation", "test")
WEATHER = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "RH2M", "WS2M"]
PHYSICAL = ["temperature_mean", "temperature_sd", "tmax_mean", "tmax_max", "tmin_mean", "tmin_min",
            "hot30_days", "hot35_days", "gdd_celsius_sum", "precipitation_sum", "wet_days",
            "heavy_rain_days", "dry_days", "longest_dry_spell", "wet_day_intensity", "solar_sum",
            "solar_mean", "humidity_mean", "humidity_sd", "wind_mean", "diurnal_range_mean"]
QUALITY = [f"missing_fraction_{c}" for c in WEATHER] + ["any_interpolation", "available_weather_days"]
COPIED = {"preprocessing.joblib", "W0.npz", "W0_columns.json", "W30.npz", "W30_columns.json",
          *[f"{s}_samples.csv" for s in SPLITS]}
FEATURE_OUTPUTS = COPIED | {"weather_imputation_audit.csv"} | {
    f"W{h}{suffix}" for h in (60, 90, 120)
    for suffix in (".npz", "_columns.json", "_environment_planting_features.csv")}


def raw_name(model, horizon):
    return f"{model}.csv" if model in BASELINES else f"{model}_W{horizon}.csv"


def source_dirs(cumulative_root, year, stage):
    if year == 2024:
        return PROJECT / "modeling/features" / stage, PROJECT / "modeling/results" / stage
    return cumulative_root / f"features/{year}/{stage}", cumulative_root / f"results/{year}/{stage}"


def renamed(value):
    def key(name):
        return name.replace("_W0", "_reference").replace("_weather", "_segmented")
    if isinstance(value, pd.DataFrame):
        return value.rename(columns={c: key(c) for c in value.columns})
    return {key(k): v for k, v in value.items()}


def weather_names(horizon):
    return (PHYSICAL if horizon == 30 else [f"D{start:03}_{start+30:03}_{name}"
            for start in range(0, horizon, 30) for name in PHYSICAL]) + QUALITY


def independent_weather(daily, planting, horizon, medians):
    """Scalar boundary-aware interpolation and explicit physical arithmetic.

    Never calls pandas.interpolate or a production aggregation function. Cropping
    precedes interpolation, including for a missing endpoint of a short horizon.
    """
    dates = pd.date_range(pd.Timestamp(planting), periods=horizon)
    raw = daily.reindex(dates)[WEATHER].to_numpy(float)
    clean = raw.copy()
    missing = np.isnan(raw)
    interpolated = 0
    for column, name in enumerate(WEATHER):
        valid = np.flatnonzero(~missing[:, column])
        if len(valid):
            for row in np.flatnonzero(missing[:, column]):
                if valid[0] < row < valid[-1]:
                    right = int(valid[np.searchsorted(valid, row)])
                    left = int(valid[np.searchsorted(valid, row) - 1])
                    fraction = (row - left) / (right - left)
                    clean[row, column] = raw[left, column] + fraction * (raw[right, column] - raw[left, column])
                    interpolated += 1
        clean[np.isnan(clean[:, column]), column] = float(medians[name])
    if not np.isfinite(clean).all() or (clean[:, 3] < 0).any():
        raise AuditError("Invalid independently cleaned weather")
    result = []
    for offset in range(0, horizon, 30):
        t, tx, tn, rain, solar, rh, wind = clean[offset:offset+30].T
        n = len(t)
        mean = lambda a: float(sum(a) / n)
        sd = lambda a: float(np.sqrt(sum((a - mean(a)) ** 2) / n))
        longest = current = 0
        for amount in rain:
            current = current + 1 if amount < 1 else 0
            longest = max(longest, current)
        wet = [amount for amount in rain if amount >= 1]
        gdd = sum(max((min(high, 30) + max(low, 10)) / 2 - 10, 0) for high, low in zip(tx, tn))
        result.extend([mean(t), sd(t), mean(tx), max(tx), mean(tn), min(tn),
            sum(tx >= 30), sum(tx >= 35), gdd, sum(rain), len(wet), sum(rain >= 10),
            sum(rain < 1), longest, sum(wet) / len(wet) if wet else 0., sum(solar),
            mean(solar), mean(rh), sd(rh), mean(wind), mean(tx - tn)])
    result.extend([*np.sum(missing, axis=0) / horizon, float(interpolated > 0),
                   float(np.sum(~np.any(missing, axis=1)))])
    return np.asarray(result, float)


def screen(record):
    gain, upper = record["relative_gain_of_year_equal_rmse_percent"], record["crossed_year_site_ci95_high"]
    components = {"all_five_years": record["n_years"] == 5,
                  "gain_at_least_2_percent": bool(np.isfinite(gain) and gain >= 2.),
                  "at_least_four_years_improve": record["years_improved"] >= 4,
                  "crossed_interval_upper_below_zero": bool(np.isfinite(upper) and upper < 0)}
    return {**components, "exploratory_engineering_signal": all(components.values()),
            "meaning": "prespecified_retrospective_screen_not_confirmatory_significance"}


def complete_groups(rows, cross):
    keys = ["model", "window", "reference", "group_type", "group"]
    axes = [MODELS, HORIZONS[1:], REFERENCES, ("hybrid", "site_code"), ("seen", "unseen")]
    if not cross:
        keys, axes = ["Year", *keys], [YEARS, *axes]
    index = pd.MultiIndex.from_product(axes, names=keys)
    result = pd.DataFrame(rows).set_index(keys).reindex(index).reset_index()
    for column in (("n_years", "n_environment_years", "n_site_codes") if cross else
                   ("n_samples", "n_environments", "n_site_codes")):
        result[column] = result[column].fillna(0).astype(int)
    defaults = {"evaluated_years": "", "missing_prespecified_years": ";".join(map(str, YEARS)),
                "bootstrap_status": "empty_subgroup", "interpretation": "empty_subgroup"} if cross else {
                    "interval_status": "empty_subgroup"}
    for column, value in defaults.items():
        result[column] = result[column].fillna(value)
    return result


def readiness(audit, root, cumulative_root):
    for year in YEARS:
        for stage, name in (("development", "development_completion_manifest.json"),
                            ("final", "final_evaluation_manifest.json")):
            audit.require((root / f"results/{year}/{stage}/{name}").is_file(),
                          f"All folds complete before reading outcomes: {year} {stage}")
    audit.require((root / "analysis/analysis_manifest.json").is_file(), "Segmented analysis is complete")
    audit.require(audit.hash(PROJECT / "modeling/results/final/final_evaluation_manifest.json") == ORIGINAL_2024_SHA256,
                  "Original 2024 completion unchanged")
    path = cumulative_root / "analysis/analysis_manifest.json"
    audit.require(audit.hash(path) == CUMULATIVE_ANALYSIS_SHA256, "Cumulative evidence matches frozen protocol anchor")
    old = audit.artifacts(path, cumulative_root)
    audit.hash_map(old["source_hashes"], PROJECT, "Cumulative controlling source")
    receipt = read_json(cumulative_root / "validation_checks.json")
    audit.require(receipt["status"] == "PASS" and receipt["audited_manifest_sha256"] == audit.hash(path)
                  and Path(receipt["audited_manifest_path"]).resolve() == path.resolve()
                  and receipt["original_2024_manifest_sha256"] == ORIGINAL_2024_SHA256,
                  "Independent cumulative validation is bound to exact source analysis")
    return old


def audit_feature(audit, root, cumulative_root, year, stage, daily, bridge, physical_samples):
    source, _ = source_dirs(cumulative_root, year, stage)
    directory = root / f"features/{year}/{stage}"
    manifest_path = directory / "feature_manifest.json"
    original_path = source / "feature_manifest.json"
    record, original = read_json(manifest_path), read_json(original_path)
    name = f"{year} {stage} features"
    end = year - (2 if stage == "development" else 1)
    audit.require(record["test_year"] == year and record["mode"] == stage and record["fit_years"] == [2014, end],
                  name + ": fold/time identity")
    audit.require(record["genotype"] == original["genotype"] and record["weather_fitting"] == original["weather_fitting"],
                  name + ": unchanged fitted PCA and horizon-specific training medians")
    audit.require(record["protocol_sha256"] == audit.hash(root / "PROTOCOL.md") and
                  record["script_sha256"] == audit.hash(PROJECT / "modeling/build_segmented_features.py"), name + ": protocol/code")
    audit.require(all(record["checks"].values()), name + ": generation checks passed")
    inputs = audit.hash_map(record["input_hashes"], PROJECT, name + " input")
    old_inputs = audit.hash_map(original["input_hashes"], PROJECT, name + " cumulative input")
    for p in (original_path, source / "preprocessing.joblib", PREP / "dimensions/weather_daily.csv.gz",
              PREP / "links/sample_plantings.csv.gz", root / "PROTOCOL.md",
              PROJECT / "modeling/build_segmented_features.py", PROJECT / "modeling/build_features.py"):
        audit.require(inputs.get(str(p.resolve())) == audit.hash(p), name + f": binds {p}")
    audit.require(all(inputs.get(p) == sha for p, sha in old_inputs.items()), name + ": cumulative input chain")
    audit.require(not any("/targets/" in p or "/holdout/" in p for p in inputs), name + ": label-free input inventory")
    audit.require(set(record["outputs"]) == FEATURE_OUTPUTS, name + ": exact output inventory")
    audit.hash_map(original["outputs"], source, name + " cumulative output")
    audit.hash_map(record["outputs"], directory, name + " output")
    cumulative = record["cumulative_source"]
    audit.require(Path(cumulative["feature_directory"]).resolve() == source.resolve()
                  and Path(cumulative["feature_manifest_path"]).resolve() == original_path.resolve()
                  and cumulative["feature_manifest_sha256"] == audit.hash(original_path), name + ": explicit source binding")
    audit.require(set(cumulative["copied_artifact_hashes"]) == COPIED, name + ": copy inventory")
    samples = {}
    for filename in COPIED:
        audit.require(audit.hash(directory / filename) == audit.hash(source / filename)
                      == cumulative["copied_artifact_hashes"][filename], name + f": byte reuse {filename}")
    for split in SPLITS:
        frame = pd.read_csv(directory / f"{split}_samples.csv")
        expected_years = (set(range(2014, end + 1)) if split == "train" else
                          ({year - 1} if stage == "development" else set()) if split == "validation" else {year})
        audit.require(set(frame.Year) == expected_years and frame.sample_id.is_unique,
                      name + f": unique temporal {split} cohort")
        audit.require(not {"y_true", "Yield_Mg_ha", "prediction", "target"}.intersection(frame.columns),
                      name + f": no label column in {split}")
        audit.require(record["rows"][split] == len(frame), name + f": {split} row count")
        samples[split] = frame
    pool = pd.concat(samples.values(), ignore_index=True)
    audit.require(pool.sample_id.is_unique, name + ": no cross-split overlap")
    scoped = bridge.loc[bridge.sample_id.isin(set(pool.sample_id))].copy()
    audit.require(set(scoped.sample_id) == set(pool.sample_id), name + ": planting bridge complete")
    audit.require(np.isfinite(scoped.planting_weight).all() and (scoped.planting_weight > 0).all(),
                  name + ": finite positive planting weights")
    lookup = pool.set_index("sample_id")
    audit.require((scoped.Env.to_numpy() == scoped.sample_id.map(lookup.Env).to_numpy()).all()
                  and (scoped.Planting_Date.dt.year.to_numpy() == scoped.sample_id.map(lookup.Year).to_numpy()).all(),
                  name + ": exact planting environment/year identities")
    audit.close(scoped.groupby("sample_id").planting_weight.sum(), np.ones(pool.shape[0]), name + ": planting weights")
    # Fixed first/middle/last IDs per split, plus a multi-date sample where available.
    selected_ids = set()
    for frame in samples.values():
        if len(frame):
            selected_ids.update(frame.sample_id.iloc[[0, len(frame)//2, len(frame)-1]])
    multiple = scoped.groupby("sample_id").size()
    if (multiple > 1).any():
        selected_ids.add(sorted(multiple.index[multiple > 1])[0])
    chosen = scoped.loc[scoped.sample_id.isin(selected_ids)]
    pairs = chosen[["Env", "Planting_Date"]].drop_duplicates().sort_values(["Env", "Planting_Date"])
    base_columns = read_json(directory / "W0_columns.json")["columns"]
    with np.load(directory / "W0.npz", allow_pickle=False) as archive:
        base = {split: archive[f"X_{split}"] for split in SPLITS}
    for horizon in HORIZONS[1:]:
        columns = read_json(directory / f"W{horizon}_columns.json")
        names = weather_names(horizon)
        audit.require(columns["columns"] == base_columns + names and columns["weather_columns"] == names,
                      name + f": W{horizon} exact column definitions")
        audit.require(record["weather_feature_count_by_horizon"][str(horizon)] == 21*(horizon//30)+9,
                      name + f": W{horizon} dimension")
        medians = original["weather_fitting"][str(horizon)]["medians"]
        recomputed = {(env, date): independent_weather(daily[env], date, horizon, medians)
                      for env, date in pairs.itertuples(index=False, name=None)}
        if horizon > 30:
            pair_file = pd.read_csv(directory / f"W{horizon}_environment_planting_features.csv", parse_dates=["Planting_Date"])
            audit.require(not pair_file.duplicated(["Env", "Planting_Date"]).any(), name + ": unique physical pair keys")
            expected_pairs = scoped[["Env", "Planting_Date"]].drop_duplicates()
            audit.require(set(map(tuple, pair_file[["Env", "Planting_Date"]].to_numpy())) ==
                          set(map(tuple, expected_pairs.to_numpy())), name + f": W{horizon} pair coverage")
            pair_file = pair_file.set_index(["Env", "Planting_Date"])
            for pair, values in recomputed.items():
                audit.close(pair_file.loc[pair, names].to_numpy(float), values,
                            name + f": independent sampled physical W{horizon} {pair}")
        with np.load(directory / f"W{horizon}.npz", allow_pickle=False) as new, np.load(source / f"W{horizon}.npz", allow_pickle=False) as old:
            audit.require(set(new.files) == {f"X_{s}" for s in SPLITS}, name + ": matrix split inventory")
            for split, frame in samples.items():
                x, original_x = new[f"X_{split}"], old[f"X_{split}"]
                audit.require(x.dtype == np.float32 and x.shape == (len(frame), len(base_columns)+len(names))
                              and np.isfinite(x).all(), name + f": finite full W{horizon} {split} shape")
                audit.require(np.array_equal(x[:, :len(base_columns)], base[split]) and
                              np.array_equal(original_x[:, :len(base_columns)], base[split]), name + f": full base equality W{horizon} {split}")
                audit.require(np.array_equal(x[:, -9:], original_x[:, -9:]), name + f": full quality equality W{horizon} {split}")
                if horizon == 30:
                    audit.require(np.array_equal(x, original_x), name + f": full W30 identity {split}")
                for row, sid in enumerate(frame.sample_id):
                    if sid not in selected_ids:
                        continue
                    part = chosen.loc[chosen.sample_id == sid]
                    expected = np.sum([recomputed[(r.Env, r.Planting_Date)] * r.planting_weight for r in part.itertuples()], axis=0).astype(np.float32)
                    audit.require(np.allclose(x[row, len(base_columns):], expected, atol=1e-5, rtol=1e-6),
                                  name + f": sampled independently weighted W{horizon} {sid}")
        physical_samples.append({"year": year, "stage": stage, "window": horizon,
                                 "sample_ids": sorted(selected_ids), "n_env_planting_pairs": len(pairs)})
    audit.read_receipts[str(manifest_path)] = audit.hash(manifest_path)
    return samples


def audit_stage(audit, root, cumulative_root, year, stage, samples):
    out = root / f"results/{year}/{stage}"
    features = root / f"features/{year}/{stage}"
    _, source = source_dirs(cumulative_root, year, stage)
    completion_path = out / ("development_completion_manifest.json" if stage == "development" else "final_evaluation_manifest.json")
    completed = audit.artifacts(completion_path, root)
    run_path = out / "run_manifest.json"
    run = read_json(run_path)
    context = f"{year} {stage}"
    audit.require(run["test_year"] == year and run["validation_year"] == year-1 and run["stage"] == stage
                  and run["candidates"] == PRESET and run["seed"] == SEED
                  and run["representation"] == "segmented_30d", context + ": run identity/budget")
    audit.require(completed["run_manifest_sha256"] == audit.hash(run_path)
                  and completed["test_year"] == year and completed["stage"] == stage,
                  context + ": completion binds correct run")
    audit.require(completed["n_new_model_fits"] == (18 if stage == "development" else 9)
                  and completed["n_reused_units"] == 8, context + ": logical fitting/reuse counts")
    inputs = audit.hash_map(run["inputs"], PROJECT, context + " run input")
    old_run = read_json(source / "run_manifest.json")
    old_inputs = audit.hash_map(old_run["inputs"], PROJECT, context + " original run input")
    audit.require(all(inputs.get(p) == sha for p, sha in old_inputs.items()), context + ": includes verified original run input chain")
    old_features, _ = source_dirs(cumulative_root, year, stage)
    audit.require(old_inputs.get(str((old_features / "feature_manifest.json").resolve())) ==
                  audit.hash(old_features / "feature_manifest.json"), context + ": original run binds original features")
    _, old_development = source_dirs(cumulative_root, year, "development")
    old_freeze_path = old_development / "frozen_selection.json"
    old_freeze = read_json(old_freeze_path)
    audit.require(old_freeze["development_run_manifest_sha256"] == audit.hash(old_development / "run_manifest.json"),
                  context + ": original frozen selection binds development source")
    if stage == "final":
        audit.require(old_run["frozen_selection_sha256"] == audit.hash(old_freeze_path), context + ": original final binds original freeze")
    for p in [root / "PROTOCOL.md", features / "feature_manifest.json", source / "run_manifest.json",
              PREP / "targets/train.csv", PREP / "targets/validation.csv", PREP / "manifest.json",
              cumulative_root / "analysis/analysis_manifest.json", cumulative_root / "validation_checks.json",
              *[PROJECT / "modeling" / n for n in ("run_segmented_experiments.py", "segmented_sources.py",
                   "run_experiments.py", "run_rolling_experiments.py", "metrics.py")]]:
        audit.require(inputs.get(str(p.resolve())) == audit.hash(p), context + f": required input {p}")
    feature = read_json(features / "feature_manifest.json")
    for p, sha in feature["input_hashes"].items():
        audit.require(inputs.get(str(Path(p).resolve())) == sha, context + ": feature source chain included")
    for p, sha in feature["outputs"].items():
        audit.require(inputs.get(str((features / p).resolve())) == sha, context + ": feature output bound")
    expected_units = {f"{m}_W{h}" for m, h in CONFIGS}
    audit.require({p.stem for p in (out / "units").glob("*.json")} == expected_units, context + ": exact 17 units")
    records, required = {}, [run_path]
    expected_models = {f"{m}_W{h}{'.cbm' if m == 'CatBoost' else '.joblib'}" for m, h in NEW}
    audit.require({p.name for p in (out / "models").iterdir() if p.is_file()} == expected_models,
                  context + ": only nine new saved models, no refitted shared models")
    for model, horizon in sorted(CONFIGS):
        path = out / "units" / f"{model}_W{horizon}.json"
        unit = audit.artifacts(path, root)
        shared = (model, horizon) in SHARED
        audit.require((unit["model"], unit["window"], unit["test_year"], unit["stage"]) == (model, horizon, year, stage)
                      and unit["run_manifest_sha256"] == audit.hash(run_path)
                      and unit["reused"] == shared, context + f": unit identity {model} W{horizon}")
        audit.require(unit["fit_kind"] == ("reused_identical_features" if shared else "new_segmented_fit"),
                      context + ": truthful fit-kind metadata")
        expected_features = 0 if model in BASELINES else len(read_json(features / f"W{horizon}_columns.json")["columns"])
        audit.require(unit["n_features"] == expected_features, context + ": recorded feature count")
        prediction = out / "predictions" / raw_name(model, horizon)
        audit.bound_paths(unit, [prediction], root, context + ": unit binds prediction")
        required += [path, prediction]
        if shared:
            audit.require(unit["fit_seconds"] == 0 and unit["source_artifact_path_base"] == "absolute",
                          context + ": reuse has zero new fit seconds and explicit source base")
            source_hashes = audit.hash_map(unit["source_artifacts"], PROJECT, context + " shared source")
            old_prediction = source / "predictions" / raw_name(model, horizon)
            audit.require(source_hashes.get(str(old_prediction.resolve())) == audit.hash(old_prediction)
                          == audit.hash(prediction), context + ": raw prediction byte reuse")
            if model in MODELS:
                source_unit = source / "units" / f"{model}_W{horizon}.json"
                audit.require(source_hashes.get(str(source_unit.resolve())) == audit.hash(source_unit), context + ": reused unit source")
                old_unit = read_json(source_unit)
                old_base = PROJECT if year == 2024 else cumulative_root
                extension = ".cbm" if model == "CatBoost" else ".joblib"
                old_model = source / "models" / f"{model}_W{horizon}{extension}"
                audit.require(source_hashes.get(str(old_model.resolve())) == audit.hash(old_model), context + ": reused model reference")
                audit.hash_map(old_unit["artifacts"], old_base, context + " source-unit artifact")
        else:
            model_path = out / "models" / f"{model}_W{horizon}{'.cbm' if model == 'CatBoost' else '.joblib'}"
            audit.bound_paths(unit, [model_path], root, context + ": binds new saved model")
            audit.require(np.isfinite(unit["fit_seconds"]) and unit["fit_seconds"] >= 0,
                          context + ": finite nonnegative fitting time")
            if stage == "development":
                for candidate in (0, 1):
                    for suffix in (".csv", "_environments.csv"):
                        candidate_path = out / "candidate_predictions" / f"{model}_W{horizon}_c{candidate}{suffix}"
                        required.append(candidate_path)
                        audit.bound_paths(unit, [candidate_path], root, context + ": new candidate artifact bound")
        records[(model, horizon)] = unit
    if stage == "development":
        required += [out / n for n in ("frozen_selection.json", "candidate_metrics.csv", "selected_metrics.csv")]
    else:
        required += [out / n for n in ("summary_metrics.csv", "environment_metrics.csv", "subgroup_metrics.csv",
                      "earliest_sufficient_windows.json", "weather_increment_comparisons.csv", "prediction_lock_before_holdout.json",
                      "evaluation_target_receipt.json")]
        required += [out / "predictions" / f"{m}_W{h}_evaluated.csv" for m, h in CONFIGS]
        receipt = read_json(out / "evaluation_target_receipt.json")
        expected_targets = {str(PREP / "holdout/test_targets.csv")} if year == 2024 else {
            str(PREP / f"targets/{s}.csv") for s in ("train", "validation")}
        audit.require(receipt["test_year"] == year and receipt["connected_after_prediction_lock"] is True
                      and receipt["prediction_lock_sha256"] == audit.hash(out / "prediction_lock_before_holdout.json")
                      and receipt["target_source_path_base"] == "absolute"
                      and set(receipt["target_source_hashes"]) == expected_targets,
                      context + ": exact post-lock target receipt")
        audit.hash_map(receipt["target_source_hashes"], PROJECT, context + " evaluation target source")
        audit.require(completed["source_artifact_path_base"] == "absolute" and
                      completed["source_artifacts"] == receipt["target_source_hashes"],
                      context + ": completion binds exact scoring target sources")
        reference = source / "predictions/GlobalMean_W0_evaluated.csv"
        audit.require(Path(receipt["cumulative_scoring_reference_path"]).resolve() == reference.resolve()
                      and receipt["cumulative_scoring_reference_sha256"] == audit.hash(reference),
                      context + ": scoring intersection source binding")
        for key in ("n_prediction_inputs", "n_scored", "n_without_observed_target"):
            audit.require(receipt[key] == completed[key], context + f": receipt/completion {key}")
    audit.bound_paths(completed, required, root, context + ": required completion artifacts")
    events = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines() if line.strip()]
    audit.read_receipts[str(out / "progress.jsonl")] = audit.hash(out / "progress.jsonl")
    fit_events = [e for e in events if e["event"] == "fit_complete"]
    expected = {(m, h, c) for m, h in NEW for c in (0, 1)} if stage == "development" else {
        (m, h, records[(m, h)]["selected"]["candidate"]) for m, h in NEW}
    audit.require({(e["model"], e["window"], e["candidate"]) for e in fit_events} == expected,
                  context + ": all and only planned new logical fits completed")
    audit.require(all((e["model"], e["window"]) in NEW for e in events if e["event"] in ("fit_start", "fit_complete")),
                  context + ": no shared-window fit event")
    if len(fit_events) != len(expected):
        audit.notes.append(f"{context}: {len(fit_events)} completed fit events for {len(expected)} logical fits; retries are not hidden.")
    audit.read_receipts[str(run_path)] = audit.hash(run_path)
    return completed, run, records, events


def audit_lock(audit, root, year, completed, events):
    out = root / f"results/{year}/final"
    feature_dir = root / f"features/{year}/final"
    freeze = root / f"results/{year}/development/frozen_selection.json"
    lock_path = out / "prediction_lock_before_holdout.json"
    lock = read_json(lock_path)
    audit.require(lock["test_year"] == year and lock["frozen_selection_sha256"] == audit.hash(freeze)
                  == completed["frozen_selection_sha256"], f"{year}: lock/freezing/completion chain")
    audit.require(lock["run_manifest_sha256"] == audit.hash(out / "run_manifest.json") and
                  lock["test_sample_ids_sha256"] == audit.hash(feature_dir / "test_samples.csv"), f"{year}: lock binds run/test identity")
    locked = audit.hash_map(lock["prediction_hashes"], root, f"{year} locked predictions")
    expected = {str((out / "predictions" / raw_name(m, h)).resolve()) for m, h in CONFIGS}
    audit.require(set(locked) == expected, f"{year}: exact 17 raw prediction lock inventory")
    target_event = "historical_2024_targets_opened_after_prediction_lock" if year == 2024 else "historical_test_targets_joined"
    first_lock, latest_lock, latest_join = None, None, None
    previous = timestamp(read_json(freeze)["frozen_at_utc"])
    completed_time = timestamp(completed["completed_at_utc"])
    for index, event in enumerate(events):
        if event["event"] not in ("all_predictions_locked", target_event):
            continue
        when = timestamp(event["utc"])
        audit.require(previous <= when <= completed_time, f"{year}: chronological freeze/lock/join/completion")
        previous = when
        if event["event"] == "all_predictions_locked":
            audit.require(event["count"] == 17 and event["prediction_lock_sha256"] == audit.hash(lock_path), f"{year}: lock event hash/count")
            first_lock = index if first_lock is None else first_lock
            latest_lock, latest_join = index, None
        else:
            audit.require(latest_lock is not None, f"{year}: every target join follows its complete lock")
            latest_join = when
    audit.require(first_lock is not None and latest_join is not None, f"{year}: completed prediction-before-scoring sequence")
    before = {(e["model"], e["window"]) for e in events[:first_lock]
              if e["event"] in ("fit_complete", "source_unit_reused_no_fit", "resume_completed_unit_no_fit")}
    audit.require(before == CONFIGS, f"{year}: all units available before first target join")
    for path in [lock_path, *map(Path, locked)]:
        audit.require(path.stat().st_mtime <= latest_join + .001, f"{year}: locked artifact timestamp {path.name}")


def audit_selection(audit, root, cumulative_root, year, samples, run, units, labels):
    out = root / f"results/{year}/development"
    _, source = source_dirs(cumulative_root, year, "development")
    frozen_path = out / "frozen_selection.json"
    frozen = read_json(frozen_path)
    audit.require(frozen["test_year"] == year and frozen["validation_year"] == year-1 and frozen["seed"] == SEED
                  and frozen["windows"] == list(HORIZONS), f"{year}: selection time/budget")
    audit.require(len(frozen["configs"]) == 15 and {(c["model"], c["window"]) for c in frozen["configs"]} == PAIRS,
                  f"{year}: exact 15 frozen choices")
    audit.require(frozen["development_run_manifest_sha256"] == audit.hash(out / "run_manifest.json")
                  and frozen["development_input_hashes"] == run["inputs"], f"{year}: freeze binds all development inputs")
    candidates, selected = [], []
    for model, horizon in sorted(PAIRS):
        unit = units[(model, horizon)]
        score_rows = []
        candidate_dir = source if horizon in (0, 30) else out
        audit.require(len(unit["candidates"]) == 2 and {c["candidate"] for c in unit["candidates"]} == {0, 1},
                      f"{year} {model} W{horizon}: exactly two validation candidates")
        for candidate in (0, 1):
            key = f"{model}_W{horizon}_c{candidate}"
            path = candidate_dir / "candidate_predictions" / f"{key}.csv"
            if horizon in (0, 30):
                audit.require(unit["source_artifacts"].get(str(path.resolve())) == audit.hash(path), f"{year} {key}: candidate reference exact reuse")
            prediction = read_predictions(audit, path, samples["validation"], labels, f"{year} {key}")
            audit.require(prediction.y_true.notna().all(), f"{year} {key}: complete preceding-year outcomes")
            summary, environment = numerical_metrics(prediction, True)
            recorded = next(c for c in unit["candidates"] if c["candidate"] == candidate)
            audit.require(recorded["params"] == PRESET[model][candidate], f"{year} {key}: declared parameters")
            compare_record(audit, recorded, summary, f"{year} {key} numeric score")
            audit.table(pd.read_csv(candidate_dir / "candidate_predictions" / f"{key}_environments.csv"), environment, ["Env"], f"{year} {key} environment score")
            candidates.append({"model": model, "window": horizon, "candidate": candidate, **summary})
            score_rows.append(summary)
        best = min(range(2), key=lambda c: score_rows[c]["macro_rmse"])
        config = next(c for c in frozen["configs"] if (c["model"], c["window"]) == (model, horizon))
        audit.require(config["candidate"] == best == unit["selected"]["candidate"] and config["params"] == PRESET[model][best]
                      and config["reused"] == (horizon in (0, 30)), f"{year} {model} W{horizon}: independent validation minimum")
        audit.close(config["validation_macro_rmse"], score_rows[best]["macro_rmse"], f"{year}: frozen validation score")
        chosen = read_predictions(audit, out / "predictions" / raw_name(model, horizon), samples["validation"], labels, f"{year}: selected validation")
        candidate = pd.read_csv(candidate_dir / "candidate_predictions" / f"{model}_W{horizon}_c{best}.csv")
        audit.close(chosen.prediction, candidate.prediction, f"{year}: selected prediction matches actual best candidate")
        selected.append({"model": model, "window": horizon, **score_rows[best]})
    for model in BASELINES:
        frame = read_predictions(audit, out / "predictions" / raw_name(model, 0), samples["validation"], labels, f"{year}: baseline validation")
        audit.close(frame.prediction, baseline(model, samples["train"], labels, samples["validation"]), f"{year}: training-only baseline")
        summary, _ = numerical_metrics(frame, True)
        selected.append({"model": model, "window": 0, **summary})
    audit.table(pd.read_csv(out / "candidate_metrics.csv"), pd.DataFrame(candidates), ["model", "window", "candidate"], f"{year}: all 30 candidate metrics")
    audit.table(pd.read_csv(out / "selected_metrics.csv"), pd.DataFrame(selected), ["model", "window"], f"{year}: all 17 validation selections")
    audit.require(frozen["recommended_by_validation"] == min(frozen["configs"], key=lambda c: c["validation_macro_rmse"]), f"{year}: validation-only recommendation")
    audit.read_receipts[str(frozen_path)] = audit.hash(frozen_path)
    return frozen


def replay(audit, root, year, samples, units, frames):
    """Replay only new final models, never loading an original source model."""
    from threadpoolctl import threadpool_limits
    rows = []
    for horizon in (60, 90, 120):
        with np.load(root / f"features/{year}/final/W{horizon}.npz", allow_pickle=False) as saved:
            x = saved["X_test"]
        for model in MODELS:
            path = root / f"results/{year}/final/models/{model}_W{horizon}{'.cbm' if model == 'CatBoost' else '.joblib'}"
            audit.bound_paths(units[(model, horizon)], [path], root, "Replay uses hash-bound new model")
            if model == "CatBoost":
                from catboost import CatBoostRegressor
                fitted = CatBoostRegressor()
                fitted.load_model(str(path))
            else:
                fitted = joblib.load(path)
            with threadpool_limits(limits=4):
                predicted = np.asarray(fitted.predict(x, thread_count=4) if model == "CatBoost" else fitted.predict(x), float)
            expected = frames[(model, horizon)].prediction.to_numpy(float)
            audit.require(predicted.shape == (len(samples["test"]),) and np.isfinite(predicted).all(),
                          f"{year} {model} W{horizon}: replay complete finite predictions")
            audit.close(predicted, expected, f"{year} {model} W{horizon}: new saved-model prediction replay")
            rows.append({"year": year, "model": model, "window": horizon, "n_predictions": len(predicted),
                         "max_absolute_difference": float(np.max(np.abs(predicted - expected))),
                         "model_sha256": audit.hash(path)})
            del fitted
    return rows


def comparison_tables(differences, group_differences):
    """All inferential arithmetic comes from the independent residual audit."""
    delta = pd.concat(differences, ignore_index=True)
    groups = pd.concat(group_differences, ignore_index=True)
    annual = [{"Year": y, "model": m, "window": h, "reference": r, **annual_stability(g)}
              for (y, m, h, r), g in delta.groupby(["Year", "model", "window", "reference"])]
    multi = [{"model": m, "window": h, "reference": r, **cross_year_stability(g)}
             for (m, h, r), g in delta.groupby(["model", "window", "reference"])]
    subgroup_annual = [{"Year": y, "model": m, "window": h, "reference": r, "group_type": k, "group": s,
                       **annual_stability(g)} for (y, m, h, r, k, s), g in groups.groupby(
                           ["Year", "model", "window", "reference", "group_type", "group"])]
    subgroup_multi = [{"model": m, "window": h, "reference": r, "group_type": k, "group": s,
                      **cross_year_stability(g)} for (m, h, r, k, s), g in groups.groupby(
                          ["model", "window", "reference", "group_type", "group"])]
    primary = [{**r, **screen(r)} for r in multi if r["window"] == 120]
    assessments = []
    for model in MODELS:
        by_reference = {r["reference"]: r["exploratory_engineering_signal"] for r in primary if r["model"] == model}
        assessments.append({"model": model, "signal_vs_cumulative": by_reference["cumulative"],
                            "signal_vs_no_weather": by_reference["no_weather"],
                            "both_reference_screens_pass": all(by_reference.values()),
                            "interpretation": "exploratory_historical_signal_only_not_new_blind_confirmation"})
    return {"environment_comparisons": renamed(delta), "yearly_comparisons": renamed(pd.DataFrame(annual)),
            "cross_year_comparisons": renamed(pd.DataFrame(multi)), "primary_assessment": renamed(pd.DataFrame(primary)),
            "model_upgrade_assessment": pd.DataFrame(assessments),
            "subgroup_environment_comparisons": renamed(groups),
            "yearly_subgroup_comparisons": renamed(complete_groups(subgroup_annual, False)),
            "cross_year_subgroup_comparisons": renamed(complete_groups(subgroup_multi, True))}


TABLE_KEYS = {
    "coverage": ["Year"], "yearly_metrics": ["Year", "representation", "model", "window"],
    "environment_metrics": ["Year", "representation", "model", "window", "Env"],
    "yearly_subgroup_metrics": ["Year", "representation", "model", "window", "group_type", "group"],
    "environment_comparisons": ["Year", "model", "window", "reference", "Env"],
    "yearly_comparisons": ["Year", "model", "window", "reference"],
    "cross_year_comparisons": ["model", "window", "reference"],
    "primary_assessment": ["model", "window", "reference"], "model_upgrade_assessment": ["model"],
    "subgroup_environment_comparisons": ["Year", "model", "window", "reference", "group_type", "group", "Env"],
    "yearly_subgroup_comparisons": ["Year", "model", "window", "reference", "group_type", "group"],
    "cross_year_subgroup_comparisons": ["model", "window", "reference", "group_type", "group"],
    "validation_selections": ["Year", "representation"],
    "yearly_sufficient_windows": ["Year", "representation", "model"], "reuse_checks": ["Year", "model", "window"],
}
EXPECTED_TABLE_ROWS = {"coverage": 5, "yearly_metrics": 170, "environment_metrics": 4386,
    "yearly_subgroup_metrics": 680, "environment_comparisons": 3096, "yearly_comparisons": 120,
    "cross_year_comparisons": 24, "primary_assessment": 6, "model_upgrade_assessment": 3,
    "yearly_subgroup_comparisons": 480, "cross_year_subgroup_comparisons": 96,
    "validation_selections": 10, "yearly_sufficient_windows": 30, "reuse_checks": 40}


def validate(root=ROOT, cumulative_root=CUMULATIVE_ROOT, output=None):
    root, cumulative_root = Path(root).resolve(), Path(cumulative_root).resolve()
    output = Path(output).resolve() if output else root / "validation_checks.json"
    if output.is_relative_to(cumulative_root) or output.is_relative_to(PROJECT / "modeling/results") or output.is_relative_to(PROJECT / "modeling/features"):
        raise AuditError("Validation output cannot overwrite cumulative/original evidence")
    audit, failure, physical_samples, replays, fit_accounting = Audit(), None, [], [], []
    try:
        old_analysis = readiness(audit, root, cumulative_root)
        # Establish every completed prediction lock before the first target read.
        state = {}
        for year in YEARS:
            state[year] = {}
            for stage in ("development", "final"):
                feature_dir = root / f"features/{year}/{stage}"
                samples = {s: pd.read_csv(feature_dir / f"{s}_samples.csv") for s in SPLITS}
                state[year][stage] = audit_stage(audit, root, cumulative_root, year, stage, samples)
                events = state[year][stage][3]
                fit_accounting.append({"year": year, "stage": stage,
                    "planned_unique_fits": 18 if stage == "development" else 9,
                    "fit_start_events": sum(e["event"] == "fit_start" for e in events),
                    "fit_complete_events": sum(e["event"] == "fit_complete" for e in events)})
            audit_lock(audit, root, year, state[year]["final"][0], state[year]["final"][3])
        weather = pd.read_csv(PREP / "dimensions/weather_daily.csv.gz", parse_dates=["Date"])
        audit.require(not weather.duplicated(["Env", "Date"]).any(), "Unique daily weather keys")
        daily = {env: g.set_index("Date")[WEATHER].sort_index() for env, g in weather.groupby("Env")}
        bridge = pd.read_csv(PREP / "links/sample_plantings.csv.gz", parse_dates=["Planting_Date"])
        audit.require(not bridge.duplicated(["sample_id", "Env", "Planting_Date"]).any(), "Unique planting-mixture keys")
        labels_frame = pd.concat([pd.read_csv(PREP / f"targets/{s}.csv") for s in ("train", "validation")] +
                                 [pd.read_csv(PREP / "holdout/test_targets.csv")], ignore_index=True)
        audit.require(labels_frame.sample_id.is_unique, "Unique prepared historical/2024 targets")
        labels = labels_frame.set_index("sample_id").Yield_Mg_ha
        audit.require(np.isfinite(labels).all() and (labels > 0).all(), "Only finite positive scored targets")
        collected = {name: [] for name in ("coverage", "yearly_metrics", "environment_metrics",
                    "yearly_subgroup_metrics", "validation_selections", "yearly_sufficient_windows", "reuse_checks")}
        differences, group_differences, analysis_required_sources = [], [], []
        for year in YEARS:
            print(f"Independent segmented audit: {year}", flush=True)
            dev_samples = audit_feature(audit, root, cumulative_root, year, "development", daily, bridge, physical_samples)
            samples = audit_feature(audit, root, cumulative_root, year, "final", daily, bridge, physical_samples)
            dev_completed, dev_run, dev_units, _ = state[year]["development"]
            completed, run, units, _ = state[year]["final"]
            frozen = audit_selection(audit, root, cumulative_root, year, dev_samples, dev_run, dev_units, labels)
            frozen_path = root / f"results/{year}/development/frozen_selection.json"
            audit.require(dev_completed["frozen_selection_sha256"] == run["frozen_selection_sha256"] == audit.hash(frozen_path),
                          f"{year}: development/final both bind independently checked selection")
            audit.require(completed["recommended_by_validation"] == frozen["recommended_by_validation"],
                          f"{year}: completion does not select on final outcomes")
            old_features, old_out = source_dirs(cumulative_root, year, "final")
            _, old_dev = source_dirs(cumulative_root, year, "development")
            source_completion = old_out / "final_evaluation_manifest.json"
            source_record = audit.artifacts(source_completion, PROJECT if year == 2024 else cumulative_root)
            audit.require(old_analysis["source_hashes"].get(str(source_completion)) == audit.hash(source_completion),
                          f"{year}: cumulative completion bound to controlling source analysis")
            audit.bound_paths(source_record, [old_out / "predictions" / f"{m}_W{h}_evaluated.csv" for m, h in CONFIGS],
                              PROJECT if year == 2024 else cumulative_root, f"{year}: cumulative evaluated inputs bound")
            frames, raw_new = {}, {}
            for representation, directory, selection_path in (
                    ("cumulative", old_out, old_dev / "frozen_selection.json"),
                    ("segmented", root / f"results/{year}/final", frozen_path)):
                source_frozen = read_json(selection_path)
                chosen = source_frozen["recommended_by_validation"]
                collected["validation_selections"].append({"Year": year, "representation": representation,
                    **{k: chosen[k] for k in ("model", "window", "candidate", "validation_macro_rmse")}})
                local_frames, local_summary, local_environments, local_groups = {}, [], [], []
                for model, horizon in sorted(CONFIGS):
                    context = f"{year} {representation} {model} W{horizon}"
                    raw = read_predictions(audit, directory / "predictions" / raw_name(model, horizon), samples["test"], labels, context)
                    evaluated = read_predictions(audit, directory / "predictions" / f"{model}_W{horizon}_evaluated.csv",
                                                 samples["test"], labels, context + " evaluated", True)
                    audit.close(evaluated.prediction, raw.loc[raw.y_true.notna()].prediction, context + ": exact scored subset")
                    if model in BASELINES:
                        audit.close(raw.prediction, baseline(model, samples["train"], labels, samples["test"]), context + ": training-only baseline")
                    elif representation == "segmented":
                        unit = units[(model, horizon)]
                        config = next(c for c in frozen["configs"] if (c["model"], c["window"]) == (model, horizon))
                        audit.require(unit["frozen_selection_sha256"] == audit.hash(frozen_path) and
                                      unit["selected"]["candidate"] == config["candidate"] and
                                      unit["selected"]["params"] == config["params"], context + ": exactly frozen refit parameters")
                    summary, envs = numerical_metrics(evaluated, True)
                    entry = {"Year": year, "representation": representation, "model": model, "window": horizon,
                             "n_prediction_inputs": len(raw), "n_without_observed_target": len(raw)-len(evaluated), **summary}
                    collected["yearly_metrics"].append(entry)
                    local_summary.append(entry)
                    local_environments.append(envs.assign(Year=year, representation=representation, model=model, window=horizon))
                    frames[(representation, model, horizon)] = evaluated
                    local_frames[(model, horizon)] = evaluated
                    if representation == "segmented":
                        raw_new[(model, horizon)] = raw
                    for kind, column in (("hybrid", "Hybrid"), ("site_code", "Experiment_Code")):
                        seen = evaluated[column].isin(set(samples["train"][column]))
                        for flag, group in ((True, "seen"), (False, "unseen")):
                            subset = evaluated.loc[seen == flag]
                            stats, _ = numerical_metrics(subset)
                            row = {"Year": year, "representation": representation, "model": model, "window": horizon,
                                   "group_type": kind, "group": group, **stats}
                            collected["yearly_subgroup_metrics"].append(row)
                            if len(subset):
                                local_groups.append({**row, "group_type": "site" if kind == "site_code" else kind})
                summary_expected = pd.DataFrame(local_summary)
                environment_expected = pd.concat(local_environments, ignore_index=True)
                audit.table(pd.read_csv(directory / "summary_metrics.csv").assign(Year=year, representation=representation),
                            summary_expected, TABLE_KEYS["yearly_metrics"], f"{year} {representation}: all numerical summaries")
                audit.table(pd.read_csv(directory / "environment_metrics.csv").assign(Year=year, representation=representation),
                            environment_expected, TABLE_KEYS["environment_metrics"], f"{year} {representation}: environment metrics")
                audit.table(pd.read_csv(directory / "subgroup_metrics.csv").assign(Year=year, representation=representation),
                            pd.DataFrame(local_groups), TABLE_KEYS["yearly_subgroup_metrics"], f"{year} {representation}: subgroup metrics")
                collected["environment_metrics"].append(environment_expected)
                comparisons, rules = within_year_window_rules(local_frames, year)
                audit.table(pd.read_csv(directory / "weather_increment_comparisons.csv").assign(Year=year),
                            pd.DataFrame(comparisons), ["Year", "model", "window"], f"{year} {representation}: sufficient-window arithmetic")
                decisions = read_json(directory / "earliest_sufficient_windows.json")
                audit.require(decisions["test_result_selection_permitted"] is False, f"{year} {representation}: no test-based recommendation")
                for rule in rules:
                    compare_record(audit, decisions["models"][rule["model"]],
                                   {k: rule[k] for k in ("earliest_sufficient_window", "W120_gain_supported", "W120_relative_gain")},
                                   f"{year} {representation}: sufficient-window rule")
                    collected["yearly_sufficient_windows"].append({**rule, "representation": representation})
            canonical = frames[("cumulative", "CatBoost", 0)]
            for frame in frames.values():
                audit.require(frame[IDENTITY].equals(canonical[IDENTITY]), f"{year}: all 34 configurations same identity/order")
                audit.close(frame.y_true, canonical.y_true, f"{year}: all 34 configurations same observed targets")
            audit.require((len(canonical), canonical.Env.nunique()) == EXPECTED_COVERAGE[year], f"{year}: fixed cohort")
            missing = len(samples["test"]) - len(canonical)
            audit.require(missing == (571 if year == 2024 else 0), f"{year}: fixed missing-target exclusions")
            audit.require(completed["n_scored"] == len(canonical) and completed["n_prediction_inputs"] == len(samples["test"])
                          and completed["n_without_observed_target"] == missing and completed["preset_models_scored"] == 17,
                          f"{year}: completion sample accounting")
            collected["coverage"].append({"Year": year, "n_scored": len(canonical),
                "n_environments": canonical.Env.nunique(), "n_site_codes": canonical.Experiment_Code.nunique(),
                "n_hybrids": canonical.Hybrid.nunique(), "n_training_samples": len(samples["train"]),
                "training_last_year": year-1, "representation_pairs": 17, "new_final_fits": 9})
            for model, horizon in sorted(SHARED):
                path = old_out / "predictions" / raw_name(model, horizon)
                audit.require(audit.hash(path) == audit.hash(root / f"results/{year}/final/predictions" / path.name),
                              f"{year}: shared final file byte identity")
                collected["reuse_checks"].append({"Year": year, "model": model, "window": horizon,
                    "raw_prediction_sha256": audit.hash(path), "byte_identical": True, "new_fit_required": False})
            for model in MODELS:
                for horizon in HORIZONS[1:]:
                    candidate = frames[("segmented", model, horizon)]
                    for reference in REFERENCES:
                        ref = frames[("cumulative", model, horizon if reference == "cumulative" else 0)]
                        differences.append(paired_rows(ref, candidate, model, horizon).assign(reference=reference))
                        for kind, column in (("hybrid", "Hybrid"), ("site_code", "Experiment_Code")):
                            seen = ref[column].isin(set(samples["train"][column]))
                            for flag, group in ((True, "seen"), (False, "unseen")):
                                mask = seen == flag
                                if mask.any():
                                    group_differences.append(paired_rows(ref.loc[mask], candidate.loc[mask], model, horizon).assign(
                                        reference=reference, group_type=kind, group=group))
            replays.extend(replay(audit, root, year, samples, units, raw_new))
            analysis_required_sources += [source_completion, old_dev / "frozen_selection.json",
                root / f"results/{year}/final/final_evaluation_manifest.json", frozen_path]
            analysis_required_sources += [directory / f"{s}_samples.csv" for directory in (old_features, root / f"features/{year}/final") for s in SPLITS]
        expected = {name: pd.concat(values, ignore_index=True) if name == "environment_metrics" else pd.DataFrame(values)
                    for name, values in collected.items()}
        expected.update(comparison_tables(differences, group_differences))
        manifest_path = root / "analysis/analysis_manifest.json"
        manifest = audit.artifacts(manifest_path, root)
        sources = audit.hash_map(manifest["source_hashes"], PROJECT, "Segmented analysis source")
        analysis_required_sources += [root / "PROTOCOL.md", cumulative_root / "analysis/analysis_manifest.json",
            cumulative_root / "validation_checks.json", *[PROJECT / "modeling" / n for n in (
                "analyze_segmented_results.py", "analyze_rolling_stability.py", "metrics.py", "run_experiments.py")]]
        for path in analysis_required_sources:
            audit.require(sources.get(str(path.resolve())) == audit.hash(path), f"Analysis binds consumed source {path}")
        audit.require(manifest["years"] == list(YEARS) and manifest["n_scored_observations"] == 57673
                      and manifest["n_environment_years"] == 129 and manifest["bootstrap_repeats"] == REPEATS
                      and manifest["seed"] == SEED, "Analysis exact coverage/seed/repeats")
        audit.require(set(manifest["table_rows"]) == set(TABLE_KEYS) == set(expected), "All 15 analysis tables present")
        for name, table in expected.items():
            audit.require(manifest["table_rows"][name] == len(table), f"{name}: declared row count")
            if name in EXPECTED_TABLE_ROWS:
                audit.require(len(table) == EXPECTED_TABLE_ROWS[name], f"{name}: prespecified row count")
            audit.bound_paths(manifest, [root / f"analysis/{name}.csv"], root, f"{name}: analysis artifact bound")
            audit.table(pd.read_csv(root / f"analysis/{name}.csv"), table, TABLE_KEYS[name], f"Analysis {name}")
        audit.require(len(replays) == 45 and sum(r["n_predictions"] for r in replays) == 524196,
                      "Exactly 45 new final model replays and 524,196 predictions including unscored 2024 inputs")
        audit.require(digest(PROJECT / "modeling/results/final/final_evaluation_manifest.json") == ORIGINAL_2024_SHA256,
                      "Original 2024 unchanged at end of audit")
        for path, sha in audit.read_receipts.items():
            audit.require(digest(path) == sha, f"Audited manifest/unit unchanged during validation: {path}")
        for path, stats in audit.hash_stats.items():
            current = Path(path).stat()
            audit.require((current.st_size, current.st_mtime_ns) == stats, f"Hashed source/artifact unchanged during validation: {path}")
        audit.notes.extend([
            "Only the 45 new final models were replayed; no original/cumulative models were replayed or refitted.",
            "All base/quality matrices and shared byte copies checked in full. Physical weather summaries independently recomputed only for deterministic sampled Env-planting pairs, not all rows.",
            "135 is the planned logical fit count (90 candidates + 45 final models). Progress records retain any retry attempts.",
            "Five years, overlapping training histories and Experiment_Code proxies limit interval interpretation. No row-level bootstrap or new blind-test claim.",
            "Hashes and log/timestamp ordering are computational provenance, not proof that historical labels were never previously seen.",
            "No production scorer, weather builder, stability implementation or model fitting function is imported by this validator.",
        ])
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
    result = {"status": "FAIL" if failure else "PASS", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": digest(Path(__file__)), "independent_arithmetic_sha256": digest(Path(__file__).with_name("validate_rolling_results.py")),
              "original_2024_manifest_sha256": ORIGINAL_2024_SHA256, "cumulative_analysis_sha256": CUMULATIVE_ANALYSIS_SHA256,
              "years": list(YEARS), "checks_passed": sum(c["passed"] for c in audit.checks), "checks": audit.checks,
              "notes": audit.notes, "failure": failure, "physical_feature_sampling": physical_samples,
              "fit_event_accounting": fit_accounting,
              "actual_fit_start_events": sum(r["fit_start_events"] for r in fit_accounting),
              "actual_fit_complete_events": sum(r["fit_complete_events"] for r in fit_accounting),
              "new_final_model_replays": replays, "physical_feature_audit_scope": "deterministic_sample_not_full_independent_recomputation",
              "arithmetic_source": "Independent rolling validator residual/tie/bootstrap arithmetic plus independently implemented clipped calendar-block weather arithmetic."}
    if failure is None:
        result.update(audited_manifest_path=str(root / "analysis/analysis_manifest.json"),
                      audited_manifest_sha256=digest(root / "analysis/analysis_manifest.json"),
                      n_representation_year_configuration_rows=170, n_new_candidate_fits=90, n_new_final_fits=45,
                      n_observations_per_configuration=57673, n_environment_years_per_configuration=129)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"{result['status']}: {result['checks_passed']} checks; {output}", flush=True)
    return result


class SyntheticTests(unittest.TestCase):
    """Synthetic numerical/boundary/tamper tests; no real data or model fitting."""

    def weather(self, n=125):
        dates = pd.date_range("2020-01-01", periods=n)
        frame = pd.DataFrame(np.tile([20., 32., 12., 2., 10., 50., 3.], (n, 1)), index=dates, columns=WEATHER)
        return frame, dict(zip(WEATHER, [18., 25., 10., 0., 8., 40., 2.]))

    def test_constant_weather_hand_calculation(self):
        daily, medians = self.weather()
        actual = independent_weather(daily, daily.index[0], 60, medians)
        expected = [20., 0., 32., 32., 12., 12., 30., 0., 330., 60., 30., 0., 0., 0., 2., 300., 10., 50., 0., 3., 20.]
        np.testing.assert_allclose(actual, expected + expected + [0.]*8 + [60.])
        self.assertEqual(len(weather_names(60)), 51)
        self.assertEqual(len(weather_names(120)), 93)

    def test_crop_before_interpolation(self):
        daily, medians = self.weather()
        start = daily.index[0]
        daily.loc[daily.index[29], "T2M"] = np.nan
        daily.loc[daily.index[30], "T2M"] = 1000.
        short = independent_weather(daily, start, 30, medians)
        expected_mean = (29*20 + 18) / 30
        self.assertAlmostEqual(short[0], expected_mean)
        self.assertEqual(short[-2], 0.)
        long = independent_weather(daily, start, 60, medians)
        self.assertAlmostEqual(long[0], (29*20 + 510) / 30)
        self.assertEqual(long[-2], 1.)
        mutated = daily.copy()
        mutated.loc[mutated.index[30:], WEATHER] = 1e8
        np.testing.assert_array_equal(short, independent_weather(mutated, start, 30, medians))

    def test_quality_whole_horizon_and_dry_boundary(self):
        daily, medians = self.weather()
        daily.loc[daily.index[20:40], "PRECTOTCORR"] = 0.
        daily.loc[daily.index[5], "T2M"] = np.nan
        actual = independent_weather(daily, daily.index[0], 60, medians)
        self.assertEqual(actual[13], 10.)
        self.assertEqual(actual[21+13], 10.)
        self.assertAlmostEqual(actual[-9], 1/60)
        self.assertEqual(actual[-2], 1.)
        self.assertEqual(actual[-1], 59.)

    def test_weight_after_nonlinear_summary(self):
        daily, medians = self.weather()
        daily.loc[daily.index[:20], "PRECTOTCORR"] = 0.
        a = independent_weather(daily, daily.index[0], 30, medians)
        b = independent_weather(daily, daily.index[15], 30, medians)
        self.assertEqual(a[13], 20.)
        self.assertEqual(b[13], 5.)
        self.assertEqual((.25*a + .75*b)[13], 8.75)
        self.assertNotEqual((.25*a + .75*b)[13], independent_weather(daily, daily.index[11], 30, medians)[13])

    def test_engineering_screen_conjunction(self):
        row = {"n_years": 5, "relative_gain_of_year_equal_rmse_percent": 2.,
               "years_improved": 4, "crossed_year_site_ci95_high": -.001}
        self.assertTrue(screen(row)["exploratory_engineering_signal"])
        for field, value in (("n_years", 4), ("relative_gain_of_year_equal_rmse_percent", 1.999),
                             ("years_improved", 3), ("crossed_year_site_ci95_high", 0.),
                             ("crossed_year_site_ci95_high", np.nan)):
            self.assertFalse(screen({**row, field: value})["exploratory_engineering_signal"])

    def test_crossed_year_site_not_row_or_environment_weighting(self):
        frame = pd.DataFrame({"Year": [2020, 2020, 2020, 2021],
            "Experiment_Code": ["A", "A", "B", "A"], "Env": ["A1", "A2", "B1", "A3"],
            "delta_rmse": [-1., -1., 2., 2.], "rmse_W0": [3., 3., 3., 3.], "rmse_weather": [2., 2., 5., 5.]})
        result = cross_year_stability(frame)
        self.assertEqual(result["n_site_codes"], 2)
        self.assertEqual(result["n_environment_years"], 4)
        self.assertEqual(result["year_equal_delta_rmse"], 1.)
        self.assertEqual(result["environment_year_equal_delta_rmse"], .5)
        self.assertEqual(result["evaluated_years"], "2020;2021")
        self.assertEqual(result["missing_prespecified_years"], "2022;2023;2024")

    def test_identity_comparison_and_single_environment(self):
        frame = pd.DataFrame({"Env": ["A", "A"], "Year": [2020, 2020],
            "Experiment_Code": ["S", "S"], "y_true": [1., 3.], "prediction": [2., 2.]})
        paired = paired_rows(frame, frame.copy(), "Ridge", 30)
        result = annual_stability(paired)
        self.assertEqual(result["delta_rmse"], 0.)
        self.assertTrue(np.isnan(result["environment_ci95_low"]))
        self.assertEqual(result["interval_status"], "not_informative_single_environment")
        self.assertEqual(renamed(result)["macro_rmse_reference"], 1.)
        self.assertEqual(renamed(result)["macro_rmse_segmented"], 1.)

    def test_complete_empty_subgroup_grid(self):
        annual = [{"Year": 2020, "model": "Ridge", "window": 120, "reference": "cumulative",
                   "group_type": "hybrid", "group": "seen", "n_samples": 2, "n_environments": 1,
                   "n_site_codes": 1, "interval_status": "not_informative_single_environment"}]
        result = complete_groups(annual, False)
        self.assertEqual(len(result), 480)
        self.assertEqual((result.interval_status == "empty_subgroup").sum(), 479)
        multi = [{"model": "Ridge", "window": 120, "reference": "cumulative", "group_type": "hybrid",
                  "group": "seen", "n_years": 1, "n_environment_years": 1, "n_site_codes": 1,
                  "evaluated_years": "2020", "missing_prespecified_years": "2021;2022;2023;2024",
                  "bootstrap_status": "insufficient_years", "interpretation": "conditional"}]
        result = complete_groups(multi, True)
        self.assertEqual(len(result), 96)
        self.assertEqual((result.bootstrap_status == "empty_subgroup").sum(), 95)
        self.assertTrue((result.loc[result.n_years == 0, "missing_prespecified_years"] == "2020;2021;2022;2023;2024").all())

    def test_table_numeric_and_key_tampering(self):
        expected = pd.DataFrame({"Year": [2020, 2021], "delta_rmse": [-.1, .2]})
        Audit().table(expected.copy(), expected, ["Year"], "synthetic unchanged")
        for changed in (expected.assign(delta_rmse=[-.1, .21]), expected.iloc[:1], pd.concat([expected, expected.iloc[:1]])):
            with self.assertRaises(AuditError):
                Audit().table(changed, expected, ["Year"], "synthetic tamper")

    def test_prediction_lock_order_and_hash_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            year = 2024
            out = root / f"results/{year}/final"
            development = root / f"results/{year}/development"
            features = root / f"features/{year}/final"
            for folder in (out / "predictions", development, features):
                folder.mkdir(parents=True, exist_ok=True)
            freeze = development / "frozen_selection.json"
            freeze.write_text(json.dumps({"frozen_at_utc": "2000-01-01T00:00:00+00:00"}))
            (out / "run_manifest.json").write_text("{}")
            (features / "test_samples.csv").write_text("synthetic")
            hashes = {}
            for model, horizon in CONFIGS:
                path = out / "predictions" / raw_name(model, horizon)
                path.write_text("synthetic prediction, not real data")
                hashes[str(path.relative_to(root))] = digest(path)
            lock = {"test_year": year, "frozen_selection_sha256": digest(freeze),
                    "run_manifest_sha256": digest(out / "run_manifest.json"),
                    "test_sample_ids_sha256": digest(features / "test_samples.csv"), "prediction_hashes": hashes}
            lock_path = out / "prediction_lock_before_holdout.json"
            lock_path.write_text(json.dumps(lock))
            completed = {"frozen_selection_sha256": digest(freeze), "completed_at_utc": "2100-01-02T00:00:00+00:00"}
            before = [{"event": "source_unit_reused_no_fit" if (m, h) in SHARED else "fit_complete",
                       "model": m, "window": h} for m, h in CONFIGS]
            locking = {"event": "all_predictions_locked", "utc": "2100-01-01T00:00:00+00:00",
                       "count": 17, "prediction_lock_sha256": digest(lock_path)}
            joining = {"event": "historical_2024_targets_opened_after_prediction_lock", "utc": "2100-01-01T00:00:01+00:00"}
            audit_lock(Audit(), root, year, completed, [*before, locking, joining])
            # Scoring-only retries preserve the same locked predictions.
            audit_lock(Audit(), root, year, completed, [*before, locking, joining,
                {**locking, "utc": "2100-01-01T00:00:02+00:00"},
                {**joining, "utc": "2100-01-01T00:00:03+00:00"}])
            with self.assertRaises(AuditError):
                audit_lock(Audit(), root, year, completed, [*before, joining, locking])
            with self.assertRaises(AuditError):
                audit_lock(Audit(), root, year, completed, [*before, {**locking, "count": 16}, joining])
            (out / "predictions" / raw_name("Ridge", 60)).write_text("changed")
            with self.assertRaises(AuditError):
                audit_lock(Audit(), root, year, completed, [*before, locking, joining])

    def test_hash_inventory_requires_binding_not_only_valid_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first, missing = base / "first.csv", base / "required.csv"
            first.write_text("1")
            missing.write_text("2")
            record = {"artifacts": {"first.csv": digest(first)}}
            Audit().hash_map(record["artifacts"], base)
            with self.assertRaises(AuditError):
                Audit().bound_paths(record, [first, missing], base, "Required artifact cannot be silently omitted")

    def test_full_synthetic_comparison_inventory(self):
        deltas, groups = [], []
        for year in YEARS:
            for model in MODELS:
                for horizon in HORIZONS[1:]:
                    for reference in REFERENCES:
                        # Two repeated site codes across all years, no unseen subjects.
                        frame = pd.DataFrame({"Env": [f"A_{year}", f"B_{year}"], "Year": [year, year],
                            "Experiment_Code": ["A", "B"], "n_W0": [3, 100], "n_weather": [3, 100],
                            "rmse_W0": [2., 4.], "rmse_weather": [1.9, 3.8], "delta_rmse": [-.1, -.2]})
                        if horizon == 30 and reference == "cumulative":
                            frame["rmse_weather"] = frame.rmse_W0
                            frame["delta_rmse"] = 0.
                        frame = frame.assign(model=model, window=horizon, reference=reference)
                        deltas.append(frame)
                        for kind in ("hybrid", "site_code"):
                            groups.append(frame.assign(group_type=kind, group="seen"))
        tables = comparison_tables(deltas, groups)
        self.assertEqual(len(tables["yearly_comparisons"]), 120)
        self.assertEqual(len(tables["cross_year_comparisons"]), 24)
        self.assertEqual(len(tables["primary_assessment"]), 6)
        self.assertEqual(len(tables["yearly_subgroup_comparisons"]), 480)
        self.assertEqual(len(tables["cross_year_subgroup_comparisons"]), 96)
        identity = tables["cross_year_comparisons"].query("window == 30 and reference == 'cumulative'")
        np.testing.assert_array_equal(identity.year_equal_delta_rmse, 0.)
        np.testing.assert_array_equal(identity.crossed_year_site_ci95_high, 0.)
        self.assertTrue(tables["model_upgrade_assessment"].both_reference_screens_pass.all())
        # 3 versus 100 rows must not affect the environment-equal/year-equal mean.
        self.assertAlmostEqual(tables["primary_assessment"].year_equal_delta_rmse.iloc[0], -.15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cumulative-root", type=Path, default=CUMULATIVE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true", help="Synthetic-only tests; no actual study outcomes/models read")
    arguments = parser.parse_args()
    if arguments.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests)
        raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    raise SystemExit(0 if validate(arguments.root, arguments.cumulative_root, arguments.output)["status"] == "PASS" else 1)
