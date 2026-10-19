"""Train-only rolling-origin features on the immutable prepared cohort.

Only earlier-year supplementary folds are generated. No target file is opened.
Completed feature directories are reused only when sources, code, protocol,
configuration and every output hash match; incomplete or changed outputs are
never overwritten. This module does not fit a predictive model.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import tempfile
import uuid

import joblib
import numpy as np
import pandas as pd

import build_features as original

PROJECT = Path(__file__).resolve().parents[1]
PREP = PROJECT / "preparation/outputs"
DEFAULT_ROOT = PROJECT / "modeling/rolling_v1"
SCRIPT = Path(__file__).resolve()
HORIZONS = original.HORIZONS
TEST_YEARS = (2020, 2021, 2022, 2023)
SEED = original.SEED


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_fold_samples(pool, test_year, mode):
    """Filter future rows before splitting; keep the Stage 1 source row order."""
    require(test_year in TEST_YEARS, f"test_year must be one of {TEST_YEARS}")
    require(mode in {"development", "final"}, "mode must be development or final")
    require(not {"Yield_Mg_ha", "y_true", "prediction"}.intersection(pool.columns),
            "Target/prediction columns are forbidden in feature inputs")
    eligible = pool.loc[pool.Year.between(2014, test_year)].copy()
    require(eligible.sample_id.notna().all() and eligible.sample_id.is_unique,
            "Fold sample IDs must be nonmissing and unique")
    require(eligible[["Env", "Hybrid", "Experiment_Code"]].notna().all().all(),
            "Missing fold entity identifiers")
    train_end = test_year - (2 if mode == "development" else 1)
    samples = {
        "train": eligible.loc[eligible.Year <= train_end].copy(),
        "validation": eligible.loc[eligible.Year == test_year - 1].copy()
        if mode == "development" else eligible.iloc[:0].copy(),
        "test": eligible.loc[eligible.Year == test_year].copy(),
    }
    require(set(samples["train"].Year) == set(range(2014, train_end + 1)),
            "Training years do not cover the prescribed fold")
    require(len(samples["test"]) > 0, "Empty fold test cohort")
    require(mode == "final" or len(samples["validation"]) > 0, "Empty fold validation cohort")
    require(sum(len(frame) for frame in samples.values()) == len(eligible),
            "Fold partitions do not preserve the fixed eligible cohort")
    require(eligible.groupby("Env").Year.nunique().max() == 1,
            "An environment must belong to one year")
    return samples


def source_paths(root, test_year, mode):
    # 2024 sample/target files are deliberately absent. The two label-free sample
    # tables below contain the fixed eligible 2014-2023 source population.
    paths = [PREP / "samples/train.csv", PREP / "samples/validation.csv",
             PREP / "dimensions/genotypes.csv.gz", PREP / "dimensions/weather_daily.csv.gz",
             PREP / "links/sample_plantings.csv.gz", PREP / "links/weather_windows.csv",
             PREP / "manifest.json", SCRIPT, Path(original.__file__).resolve(),
             root / "PROTOCOL.md"]
    if mode == "final":
        freeze = root / "results" / str(test_year) / "development/frozen_selection.json"
        require(freeze.is_file(), "Freeze this fold's development selection before final feature fitting")
        paths.append(freeze)
    for path in paths:
        require(path.is_file(), f"Required feature input is missing: {path}")
    return paths


def hashes(paths):
    return {str(path.resolve()): original.digest(path) for path in paths}


def runtime_versions():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")}}


def expected_outputs():
    files = {f"{split}_samples.csv" for split in ("train", "validation", "test")}
    files.update({"preprocessing.joblib", "weather_imputation_audit.csv"})
    for horizon in HORIZONS:
        files.update({f"W{horizon}.npz", f"W{horizon}_columns.json"})
        if horizon:
            files.add(f"W{horizon}_environment_planting_features.csv")
    return files


def verify_complete_cache(out, signature):
    manifest_path = out / "feature_manifest.json"
    require(manifest_path.is_file(),
            f"Incomplete feature directory exists; preserving without overwrite: {out}")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("build_signature") == signature,
            "Feature cache source/code/protocol/configuration changed; preserve the old run and choose a new root")
    require(set(manifest.get("outputs", {})) == expected_outputs(), "Cached feature output inventory differs")
    require({p.name for p in out.iterdir()} == expected_outputs() | {"feature_manifest.json"},
            "Cached feature directory contains missing or unexpected artifacts")
    for filename, sha in manifest["outputs"].items():
        require((out / filename).is_file() and original.digest(out / filename) == sha,
                f"Cached feature artifact changed: {filename}")
    require(manifest.get("mode") == signature["mode"] and manifest.get("test_year") == signature["test_year"],
            "Cached feature fold identity differs")
    require(all(manifest.get("checks", {}).values()) and manifest.get("checks"),
            "Cached feature checks are incomplete")
    return manifest


def scoped_dimensions(samples, test_year):
    cohort = pd.concat(samples.values(), ignore_index=True)
    ids, envs, hybrids = set(cohort.sample_id), set(cohort.Env), set(cohort.Hybrid)

    genotype = pd.read_csv(PREP / "dimensions/genotypes.csv.gz")
    genotype = genotype.loc[genotype.Hybrid.isin(hybrids)].copy()
    require(genotype.Hybrid.is_unique and set(genotype.Hybrid) == hybrids,
            "Fold genotype coverage or key uniqueness failed")

    bridge = pd.read_csv(PREP / "links/sample_plantings.csv.gz")
    bridge = bridge.loc[bridge.sample_id.isin(ids)].copy()
    bridge["Planting_Date"] = pd.to_datetime(bridge.Planting_Date, errors="raise")
    require(set(bridge.sample_id) == ids, "Not every fold sample has planting metadata")
    require(not bridge.duplicated(["sample_id", "Env", "Planting_Date"]).any(), "Duplicate fold planting bridge keys")
    lookup = cohort.set_index("sample_id")
    require((bridge.Env.to_numpy() == bridge.sample_id.map(lookup.Env).to_numpy()).all(),
            "Fold planting bridge environment mismatch")
    require((bridge.Planting_Date.dt.year.to_numpy() == bridge.sample_id.map(lookup.Year).to_numpy()).all(),
            "Planting year differs from its fold sample year")
    require(np.isfinite(bridge.planting_weight.to_numpy(float)).all()
            and (bridge.planting_weight > 0).all(), "Invalid planting mixture weights")
    require(np.allclose(bridge.groupby("sample_id").planting_weight.sum(), 1.0, atol=1e-10, rtol=0),
            "Fold planting weights do not sum to one")
    pairs = bridge[["Env", "Planting_Date"]].drop_duplicates().sort_values(["Env", "Planting_Date"])

    weather = pd.read_csv(PREP / "dimensions/weather_daily.csv.gz")
    weather = weather.loc[weather.Env.isin(envs)].copy()
    weather["Date"] = pd.to_datetime(weather.Date, errors="raise")
    require((weather.Date.dt.year <= test_year).all(), "Future-year weather entered the fold")
    require(not weather.duplicated(["Env", "Date"]).any(), "Duplicate fold weather calendar keys")
    require(set(weather.Env) == envs, "Fold weather environment coverage failed")
    daily = {env: frame.set_index("Date")[original.WEATHER].sort_index()
             for env, frame in weather.groupby("Env")}

    declared = pd.read_csv(PREP / "links/weather_windows.csv")
    declared = declared.loc[declared.Env.isin(envs)].copy()
    declared["Planting_Date"] = pd.to_datetime(declared.Planting_Date, errors="raise")
    declared = declared.merge(pairs, on=["Env", "Planting_Date"], how="inner", validate="many_to_one")
    require(not declared.duplicated(["Env", "Planting_Date", "horizon_days"]).any(), "Duplicate declared fold weather windows")
    expected = {(env, date, h) for env, date in pairs.itertuples(index=False, name=None) for h in HORIZONS[1:]}
    require(set(zip(declared.Env, declared.Planting_Date, declared.horizon_days)) == expected,
            "Declared weather windows differ from fold planting pairs")
    return genotype, bridge, pairs, daily


def generate_fold(stage, samples, test_year, mode, n_components, signature):
    for split, frame in samples.items():
        frame.to_csv(stage / f"{split}_samples.csv", index=False)
    genotype, bridge, pairs, daily = scoped_dimensions(samples, test_year)
    print(f"rolling {test_year} {mode}: unique-training-material PCA", flush=True)
    genomic, genome_estimators, genome_meta = original.genotype_features(samples, genotype, n_components)
    static, static_estimators, static_names = original.static_features(samples)
    base = {split: np.column_stack([genomic[split], static[split]]).astype(np.float32) for split in samples}
    base_names = [f"PC{i+1:03}" for i in range(n_components)] + static_names
    require(all(np.isfinite(array).all() for array in base.values()), "Non-finite base features")
    joblib.dump({"genotype": genome_estimators, "static": static_estimators}, stage / "preprocessing.joblib")
    np.savez_compressed(stage / "W0.npz", **{f"X_{s}": x for s, x in base.items()})
    original.write_json({"columns": base_names, "weather_columns": [], "horizon_days": 0}, stage / "W0_columns.json")

    train_ids = set(samples["train"].sample_id)
    weather_fit, diagnostics = {}, []
    for horizon in HORIZONS[1:]:
        print(f"rolling {test_year} {mode}: W{horizon}", flush=True)
        medians, fit_count = original.training_weather_medians(bridge, daily, train_ids, horizon)
        rows = []
        for env, planting in pairs.itertuples(index=False, name=None):
            dates = pd.date_range(planting, periods=horizon)
            require(dates.isin(daily[env].index).all(), f"Incomplete fold weather calendar: {env}, {planting}")
            values, diag = original.aggregate_window(daily[env], planting, horizon, medians)
            rows.append([env, planting, *values])
            diagnostics.append({"Env": env, "Planting_Date": str(planting.date()), "horizon": horizon, **diag})
        per_pair = pd.DataFrame(rows, columns=["Env", "Planting_Date"] + original.WEATHER_NAMES)
        linked = bridge.merge(per_pair, on=["Env", "Planting_Date"], validate="many_to_one")
        require(linked[original.WEATHER_NAMES].notna().all().all(), "Missing fold weather features")
        weighted = linked[original.WEATHER_NAMES].multiply(linked.planting_weight, axis=0)
        weighted["sample_id"] = linked.sample_id
        per_sample = weighted.groupby("sample_id")[original.WEATHER_NAMES].sum()
        arrays = {f"X_{split}": np.column_stack([base[split], per_sample.loc[frame.sample_id].to_numpy()]).astype(np.float32)
                  for split, frame in samples.items()}
        require(all(np.isfinite(array).all() for array in arrays.values()), "Non-finite weather feature matrix")
        np.savez_compressed(stage / f"W{horizon}.npz", **arrays)
        original.write_json({"columns": base_names + original.WEATHER_NAMES,
                             "weather_columns": original.WEATHER_NAMES, "horizon_days": horizon},
                            stage / f"W{horizon}_columns.json")
        per_pair.to_csv(stage / f"W{horizon}_environment_planting_features.csv", index=False)
        weather_fit[str(horizon)] = {"training_distinct_env_dates": fit_count, "medians": medians.to_dict()}
    pd.DataFrame(diagnostics).to_csv(stage / "weather_imputation_audit.csv", index=False)
    metadata = {
        "schema_version": 1, "stage": "features", "experiment": "rolling_v1",
        "mode": mode, "test_year": test_year,
        "fit_years": [2014, test_year - (2 if mode == "development" else 1)],
        "validation_year": test_year - 1 if mode == "development" else None,
        "seed": SEED, "rows": {split: len(frame) for split, frame in samples.items()},
        "maximum_processed_sample_year": test_year,
        "base_feature_count": len(base_names), "weather_feature_count": len(original.WEATHER_NAMES),
        "physical_weather_features": len(original.PHYSICAL), "genotype": genome_meta,
        "weather_fitting": weather_fit,
        "input_hashes": signature["input_hashes"], "input_path_base": "absolute",
        "script_sha256": signature["input_hashes"][str(SCRIPT.resolve())],
        "helper_sha256": signature["input_hashes"][str(Path(original.__file__).resolve())],
        "protocol_sha256": signature["protocol_sha256"],
        "build_signature": signature, "python": platform.python_version(),
        "checks": {"no_targets_read": True, "fit_training_years_only": True,
                   "genotype_fit_unique_training_hybrids": True,
                   "weather_clipped_before_imputation": True, "no_future_sample_year_processed": True,
                   "paired_sample_ids_all_windows": True, "finite_matrices": True},
        "radiation_units": "Released ALLSKY_SFC_SW_DWN units; no unverified physical-unit conversion.",
        "gdd_definition": "sum(max((min(Tmax,30)+max(Tmin,10))/2-10,0)), Celsius degree-days",
        "rain_thresholds_mm": {"wet_at_least": 1, "heavy_at_least": 10},
        "mixed_dates": "Per-date nonlinear summaries, then original-plot-count-weighted means; not one weather trajectory.",
        "cohort_rule": "Immutable Stage 1 eligible environment-hybrid cohort restricted through fold test year; historical labels were used in the prior original experiment.",
        "artifact_path_base": "feature_directory",
    }
    metadata["outputs"] = {name: original.digest(stage / name) for name in sorted(expected_outputs())}
    return metadata


def build_fold(test_year: int, mode: str, n_components=100, root: Path = DEFAULT_ROOT):
    require(type(test_year) is int and test_year in TEST_YEARS, f"test_year must be one of {TEST_YEARS}")
    require(mode in {"development", "final"}, "mode must be development or final")
    require(type(n_components) is int and n_components > 0, "n_components must be a positive integer")
    root = Path(root).resolve()
    paths = source_paths(root, test_year, mode)
    input_hashes = hashes(paths)
    signature = {"schema_version": 1, "test_year": test_year, "mode": mode,
                 "n_components": n_components, "horizons": list(HORIZONS), "seed": SEED,
                 "input_hashes": input_hashes, "versions": runtime_versions(),
                 "protocol_sha256": input_hashes[str((root / "PROTOCOL.md").resolve())]}
    out = root / "features" / str(test_year) / mode
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.parent / f".{mode}.lock"
    token = json.dumps({"pid": os.getpid(), "token": uuid.uuid4().hex, "destination": str(out)})
    try:
        with lock.open("x") as stream:
            stream.write(token)
    except FileExistsError as error:
        raise ValueError(f"Another build or unresolved lock exists; preserving outputs: {lock}") from error
    try:
        if out.exists():
            metadata = verify_complete_cache(out, signature)
            require(hashes(paths) == input_hashes, "Input changed during cache validation")
            print(f"Verified completed feature cache; no recomputation: {out}", flush=True)
            return metadata
        pool = pd.concat([pd.read_csv(PREP / "samples/train.csv"),
                          pd.read_csv(PREP / "samples/validation.csv")], ignore_index=True)
        samples = select_fold_samples(pool, test_year, mode)
        stage = Path(tempfile.mkdtemp(prefix=f".{mode}-building-", dir=out.parent))
        metadata = generate_fold(stage, samples, test_year, mode, n_components, signature)
        require(hashes(paths) == input_hashes, f"Inputs changed during build; preserving partial staging folder: {stage}")
        metadata["checks"]["all_source_code_protocol_hashes_unchanged"] = True
        original.write_json(metadata, stage / "feature_manifest.json")
        verify_complete_cache(stage, signature)
        require(not out.exists(), f"Destination appeared during generation; staging preserved: {stage}")
        stage.rename(out)
        print(f"Complete: {out}; rows={metadata['rows']}", flush=True)
        return metadata
    finally:
        # Remove only the lock this invocation created; never another owner's.
        if lock.is_file() and lock.read_text() == token:
            lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-year", type=int, choices=TEST_YEARS, required=True)
    parser.add_argument("--mode", choices=("development", "final"), required=True)
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    build_fold(args.test_year, args.mode, args.n_components, args.root)
