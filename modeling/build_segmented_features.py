"""Thirty-day physical weather blocks on immutable cumulative-feature cohorts.

No target is read and no estimator is fitted. The original horizon-specific
cleaning, medians, nine quality fields and planting-date mixture are retained.
W0/W30, sample tables and preprocessing are byte-copied only after verification.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import uuid

import numpy as np
import pandas as pd

import build_features as original

PROJECT = Path(__file__).resolve().parents[1]
PREP = PROJECT / "preparation/outputs"
DEFAULT_ROOT = PROJECT / "modeling/segmented_v1"
CUMULATIVE_ROOT = PROJECT / "modeling/rolling_v1"
SCRIPT = Path(__file__).resolve()
YEARS = (2020, 2021, 2022, 2023, 2024)
HORIZONS = (0, 30, 60, 90, 120)
SEGMENT_DAYS = 30
QUALITY = original.WEATHER_NAMES[len(original.PHYSICAL):]
SPLITS = ("train", "validation", "test")
COPIED = {"preprocessing.joblib", "W0.npz", "W0_columns.json", "W30.npz", "W30_columns.json",
          *[f"{split}_samples.csv" for split in SPLITS]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_feature_dir(year, mode, cumulative_root=CUMULATIVE_ROOT):
    require(type(year) is int and year in YEARS, f"test_year must be one of {YEARS}")
    require(mode in ("development", "final"), "mode must be development or final")
    return (PROJECT / "modeling/features" / mode if year == 2024
            else Path(cumulative_root).resolve() / "features" / str(year) / mode)


def weather_names(horizon):
    require(horizon in HORIZONS[1:], "Only 30/60/90/120-day weather windows are supported")
    if horizon == 30:
        return list(original.WEATHER_NAMES)
    return [f"D{start:03}_{start + SEGMENT_DAYS:03}_{name}"
            for start in range(0, horizon, SEGMENT_DAYS) for name in original.PHYSICAL] + QUALITY


def aggregate_segmented_window(daily, planting, horizon, medians):
    """Clean the entire available horizon first, then summarize physical blocks.

Block endpoints are exclusive; a block may use another block's observations for
interior interpolation only within this prediction horizon. Quality fields retain
the original whole-horizon definition. These are calendar bins, not growth stages.
"""
    require(horizon in HORIZONS[1:], "Only 30/60/90/120-day weather windows are supported")
    dates = pd.date_range(planting, periods=horizon, freq="D")
    raw = daily.reindex(dates)[original.WEATHER].copy()
    clean = raw.interpolate(method="linear", limit_area="inside").fillna(medians)
    require(np.isfinite(clean.to_numpy()).all(), "Non-finite weather after inherited train-only filling")
    cumulative, diagnostics = original.aggregate_window(daily, planting, horizon, medians)
    clean_cumulative, _ = original.aggregate_window(clean, planting, horizon, medians)
    require(np.array_equal(clean_cumulative[:len(original.PHYSICAL)], cumulative[:len(original.PHYSICAL)]),
            "Full-horizon cleaning differs from the original physical summaries")
    blocks = []
    for start in range(0, horizon, SEGMENT_DAYS):
        physical, _ = original.aggregate_window(clean, planting + pd.Timedelta(days=start), SEGMENT_DAYS, medians)
        blocks.append(physical[:len(original.PHYSICAL)])
    values = np.concatenate([*blocks, cumulative[len(original.PHYSICAL):]])
    require(len(values) == len(weather_names(horizon)) and np.isfinite(values).all(), "Invalid segmented feature values")
    if horizon == 30:
        require(np.array_equal(values, cumulative), "Single-block W30 is not exactly equivalent")
    return values, {**diagnostics, "segment_days": SEGMENT_DAYS, "n_segments": horizon // SEGMENT_DAYS}


def fingerprints(paths):
    return {str(Path(path).resolve()): original.digest(Path(path)) for path in paths}


def verify_source(source, year, mode):
    manifest_path = source / "feature_manifest.json"
    require(manifest_path.is_file(), f"Missing immutable cumulative feature manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["mode"] == mode and manifest["fit_years"] == [2014, year - (2 if mode == "development" else 1)],
            "Cumulative preprocessing period differs from requested fold")
    require(year == 2024 or manifest.get("test_year") == year, "Wrong cumulative feature fold")
    require(manifest.get("checks") and all(manifest["checks"].values()), "Cumulative feature checks are absent or failed")
    paths = [manifest_path]
    source_inputs = {}
    for key, expected in manifest["input_hashes"].items():
        path = Path(key) if Path(key).is_absolute() else PROJECT / key
        require(path.is_file() and original.digest(path) == expected, f"Cumulative source input changed: {path}")
        paths.append(path)
        source_inputs[str(path.resolve())] = expected
    for path in (PREP / "dimensions/weather_daily.csv.gz", PREP / "links/sample_plantings.csv.gz"):
        require(source_inputs.get(str(path.resolve())) == original.digest(path),
                f"Weather/planting input is not bound to cumulative feature source: {path}")
    required = COPIED | {f"W{h}{suffix}" for h in HORIZONS for suffix in (".npz", "_columns.json")}
    require(required.issubset(manifest["outputs"]), "Cumulative feature output inventory is incomplete")
    for name, expected in manifest["outputs"].items():
        path = source / name
        require(not Path(name).is_absolute() and path.resolve().is_relative_to(source.resolve()), "Cumulative output escapes feature directory")
        require(path.is_file() and original.digest(path) == expected, f"Cumulative feature output changed: {name}")
        paths.append(path)
    code = PROJECT / "modeling" / ("build_features.py" if year == 2024 else "build_rolling_features.py")
    require(original.digest(code) == manifest["script_sha256"], "Cumulative feature builder code changed")
    paths.append(code)
    return manifest, paths


def load_samples(source, year, mode):
    samples = {split: pd.read_csv(source / f"{split}_samples.csv") for split in SPLITS}
    end = year - (2 if mode == "development" else 1)
    expected_years = {"train": set(range(2014, end + 1)), "validation": {year - 1} if mode == "development" else set(), "test": {year}}
    for split, frame in samples.items():
        require(not {"Yield_Mg_ha", "y_true", "prediction", "target"}.intersection(frame.columns), "Target columns forbidden in feature samples")
        require(frame.sample_id.notna().all() and frame.sample_id.is_unique, "Invalid source sample IDs")
        require(set(frame.Year.unique()) == expected_years[split], f"Wrong source years in {split}")
        require(frame[["Env", "Hybrid", "Experiment_Code"]].notna().all().all(), "Missing sample entity identifier")
    pool = pd.concat([frame for frame in samples.values() if len(frame)], ignore_index=True)
    require(pool.sample_id.is_unique and pool.groupby("Env").Year.nunique().max() == 1, "Fold sample overlap or conflicting environment year")
    return samples


def scoped_weather(samples, year):
    pool = pd.concat([frame for frame in samples.values() if len(frame)], ignore_index=True)
    ids, envs = set(pool.sample_id), set(pool.Env)
    bridge = pd.read_csv(PREP / "links/sample_plantings.csv.gz")
    bridge = bridge.loc[bridge.sample_id.isin(ids)].copy()
    bridge["Planting_Date"] = pd.to_datetime(bridge.Planting_Date, errors="raise")
    require(set(bridge.sample_id) == ids, "Missing sample planting records")
    require(not bridge.duplicated(["sample_id", "Env", "Planting_Date"]).any(), "Duplicate planting bridge keys")
    lookup = pool.set_index("sample_id")
    require((bridge.Env.to_numpy() == bridge.sample_id.map(lookup.Env).to_numpy()).all(), "Planting environment mismatch")
    require((bridge.Planting_Date.dt.year.to_numpy() == bridge.sample_id.map(lookup.Year).to_numpy()).all(), "Planting year mismatch")
    require(np.isfinite(bridge.planting_weight.to_numpy()).all() and (bridge.planting_weight > 0).all(), "Invalid planting weights")
    require(np.allclose(bridge.groupby("sample_id").planting_weight.sum(), 1., atol=1e-10, rtol=0), "Planting weights must sum to one")
    pairs = bridge[["Env", "Planting_Date"]].drop_duplicates().sort_values(["Env", "Planting_Date"])
    weather = pd.read_csv(PREP / "dimensions/weather_daily.csv.gz")
    weather = weather.loc[weather.Env.isin(envs)].copy()
    weather["Date"] = pd.to_datetime(weather.Date, errors="raise")
    require(set(weather.Env) == envs and (weather.Date.dt.year <= year).all(), "Weather environment coverage or year mismatch")
    require(not weather.duplicated(["Env", "Date"]).any(), "Duplicate weather keys")
    daily = {env: frame.set_index("Date")[original.WEATHER].sort_index() for env, frame in weather.groupby("Env")}
    return bridge, pairs, daily


def expected_outputs():
    return COPIED | {"weather_imputation_audit.csv"} | {
        f"W{h}{suffix}" for h in HORIZONS[2:] for suffix in (".npz", "_columns.json", "_environment_planting_features.csv")}


def verify_complete_cache(out, signature):
    path = out / "feature_manifest.json"
    require(path.is_file(), f"Incomplete feature directory exists; preserving without overwrite: {out}")
    manifest = json.loads(path.read_text())
    require(manifest.get("build_signature") == signature, "Feature cache source/code/protocol/configuration changed; use a new root")
    require(set(manifest.get("outputs", {})) == expected_outputs(), "Cached output inventory differs")
    require({p.name for p in out.iterdir()} == expected_outputs() | {"feature_manifest.json"}, "Missing or unexpected cached artifacts")
    for name, sha in manifest["outputs"].items():
        require((out / name).is_file() and original.digest(out / name) == sha, f"Cached feature artifact changed: {name}")
    require(manifest.get("checks") and all(manifest["checks"].values()), "Incomplete cached checks")
    return manifest


def generate_fold(stage, source, source_manifest, samples, year, mode, signature):
    for name in sorted(COPIED):
        shutil.copy2(source / name, stage / name)
        require(original.digest(stage / name) == source_manifest["outputs"][name], f"Byte copy differs: {name}")
    base_names = json.loads((source / "W0_columns.json").read_text())["columns"]
    with np.load(source / "W0.npz", allow_pickle=False) as saved:
        base = {split: saved[f"X_{split}"] for split in SPLITS}
    for split in SPLITS:
        require(base[split].dtype == np.float32 and base[split].shape == (len(samples[split]), len(base_names))
                and np.isfinite(base[split]).all(), "Invalid copied base matrix")
    bridge, pairs, daily = scoped_weather(samples, year)
    diagnostics = []
    for horizon in HORIZONS[1:]:
        print(f"segmented {year} {mode}: W{horizon} (inherited preprocessing)", flush=True)
        medians = pd.Series(source_manifest["weather_fitting"][str(horizon)]["medians"]).reindex(original.WEATHER)
        require(np.isfinite(medians.to_numpy()).all(), "Invalid inherited training-window medians")
        names, rows = weather_names(horizon), []
        for env, planting in pairs.itertuples(index=False, name=None):
            require(pd.date_range(planting, periods=horizon).isin(daily[env].index).all(), "Incomplete inherited weather calendar")
            values, diagnostic = aggregate_segmented_window(daily[env], planting, horizon, medians)
            rows.append([env, planting, *values])
            diagnostics.append({"Env": env, "Planting_Date": str(planting.date()), "horizon": horizon, **diagnostic})
        pair_features = pd.DataFrame(rows, columns=["Env", "Planting_Date", *names])
        linked = bridge.merge(pair_features, on=["Env", "Planting_Date"], validate="many_to_one")
        require(linked[names].notna().all().all(), "Missing segmented pair features")
        weighted = linked[names].multiply(linked.planting_weight, axis=0)
        weighted["sample_id"] = linked.sample_id
        by_sample = weighted.groupby("sample_id")[names].sum()
        arrays = {f"X_{split}": np.column_stack([base[split], by_sample.loc[frame.sample_id].to_numpy()]).astype(np.float32)
                  for split, frame in samples.items()}
        require(all(np.isfinite(x).all() for x in arrays.values()), "Non-finite segmented matrices")
        for split in SPLITS:
            require(np.array_equal(arrays[f"X_{split}"][:, :len(base_names)], base[split]),
                    f"Base columns changed: W{horizon} {split}")
        if horizon == 30:
            with np.load(source / "W30.npz", allow_pickle=False) as saved:
                require(set(saved.files) == set(arrays), "Source W30 split inventory differs")
                for key, values in arrays.items():
                    require(saved[key].dtype == np.float32 and np.array_equal(values, saved[key]), f"W30 exact float32 equivalence failed: {key}")
            require(json.loads((source / "W30_columns.json").read_text())["columns"] == base_names + names, "W30 source feature names differ")
            continue
        source_columns = json.loads((source / f"W{horizon}_columns.json").read_text())["columns"]
        require(source_columns == base_names + original.WEATHER_NAMES, "Unexpected cumulative weather layout")
        with np.load(source / f"W{horizon}.npz", allow_pickle=False) as cumulative_arrays:
            for split in SPLITS:
                key = f"X_{split}"
                require(np.array_equal(cumulative_arrays[key][:, :len(base_names)], base[split]),
                        f"Cumulative source base differs from W0: W{horizon} {split}")
                require(np.array_equal(arrays[key][:, -len(QUALITY):], cumulative_arrays[key][:, -len(QUALITY):]),
                        f"Whole-horizon quality fields changed: W{horizon} {split}")
        np.savez_compressed(stage / f"W{horizon}.npz", **arrays)
        original.write_json({"columns": base_names + names, "weather_columns": names, "horizon_days": horizon,
                             "segment_days": SEGMENT_DAYS, "segment_end_exclusive": True,
                             "segments": [[start, start + SEGMENT_DAYS] for start in range(0, horizon, SEGMENT_DAYS)],
                             "quality_scope": "whole_current_horizon"}, stage / f"W{horizon}_columns.json")
        pair_features.to_csv(stage / f"W{horizon}_environment_planting_features.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(stage / "weather_imputation_audit.csv", index=False)
    return {"schema_version": 1, "stage": "features", "experiment": "segmented_v1", "representation": "segmented_30d", "mode": mode, "test_year": year,
            "fit_years": source_manifest["fit_years"], "rows": {s: len(f) for s, f in samples.items()},
            "base_feature_count": len(base_names), "weather_feature_count_by_horizon": {str(h): len(weather_names(h)) for h in HORIZONS[1:]},
            "physical_weather_features_by_horizon": {str(h): len(original.PHYSICAL) * (h // SEGMENT_DAYS) for h in HORIZONS[1:]},
            "physical_weather_features": len(original.PHYSICAL), "quality_feature_count": len(QUALITY),
            "quality_columns": QUALITY, "segment_days": SEGMENT_DAYS,
            "genotype": source_manifest["genotype"], "weather_fitting": source_manifest["weather_fitting"],
            "maximum_processed_sample_year": year, "reused_horizons": [0, 30], "new_horizons": [60, 90, 120],
            "seed": original.SEED, "input_hashes": signature["input_hashes"], "input_path_base": "absolute",
            "build_signature": signature, "script_sha256": signature["input_hashes"][str(SCRIPT.resolve())],
            "protocol_sha256": signature["protocol_sha256"], "artifact_path_base": "feature_directory",
            "cumulative_source": {"feature_directory": str(source), "feature_manifest_path": str(source / "feature_manifest.json"),
                "feature_manifest_sha256": original.digest(source / "feature_manifest.json"),
                "copied_artifact_hashes": {name: source_manifest["outputs"][name] for name in sorted(COPIED)}},
            "checks": {"no_targets_read": True, "no_estimators_refitted": True, "same_source_sample_order": True,
                "all_copied_files_byte_identical": True, "W30_all_splits_float32_exact_equivalence": True,
                "all_horizon_base_columns_exact_source_W0": True, "all_horizon_quality_columns_exact_cumulative": True,
                "weather_clipped_before_imputation_and_segmentation": True, "inherited_horizon_training_medians_only": True,
                "full_horizon_cleaning_matches_original": True, "original_whole_horizon_quality_features": True,
                "original_planting_weights_after_nonlinear_summary": True, "finite_matrices": True},
            "interpretation": "Thirty-day calendar blocks, not observed phenological stages. All physical blocks share only their current prediction horizon's cleaning; quality is whole-horizon. Planting-date mixture features are not a single trajectory.",
            "outputs": {name: original.digest(stage / name) for name in sorted(expected_outputs())}}


def build_fold(test_year, mode, root=DEFAULT_ROOT, cumulative_root=CUMULATIVE_ROOT):
    source = source_feature_dir(test_year, mode, cumulative_root).resolve()
    root = Path(root).resolve()
    require(root != Path(cumulative_root).resolve() and not root.is_relative_to(PROJECT / "modeling/features"), "Use an isolated segmented feature root")
    out = root / "features" / str(test_year) / mode
    require(out != source, "Cannot overwrite cumulative source features")
    protocol = root / "PROTOCOL.md"
    require(protocol.is_file(), "Segmented protocol must exist before feature generation")
    source_manifest, paths = verify_source(source, test_year, mode)
    paths += [SCRIPT, Path(original.__file__).resolve(), protocol, PREP / "dimensions/weather_daily.csv.gz", PREP / "links/sample_plantings.csv.gz"]
    if mode == "final":
        freeze = root / "results" / str(test_year) / "development/frozen_selection.json"
        require(freeze.is_file(), "Freeze this segmented fold's development selection before final features")
        paths.append(freeze)
    input_hashes = fingerprints(paths)
    signature = {"schema_version": 1, "test_year": test_year, "mode": mode, "segment_days": SEGMENT_DAYS,
                 "horizons": list(HORIZONS), "quality_scope": "whole_current_horizon", "input_hashes": input_hashes,
                 "protocol_sha256": input_hashes[str(protocol)], "source_feature_directory": str(source),
                 "versions": {"python": platform.python_version(), **{n: importlib.metadata.version(n) for n in ("numpy", "pandas")}}}
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
            result = verify_complete_cache(out, signature)
            require(fingerprints(paths) == input_hashes, "Input changed during cache validation")
            print(f"Verified segmented cache; no recomputation: {out}", flush=True)
            return result
        samples = load_samples(source, test_year, mode)
        require(source_manifest["rows"] == {s: len(f) for s, f in samples.items()}, "Source manifest row counts differ")
        stage = Path(tempfile.mkdtemp(prefix=f".{mode}-building-", dir=out.parent))
        result = generate_fold(stage, source, source_manifest, samples, test_year, mode, signature)
        require(fingerprints(paths) == input_hashes, f"Inputs changed during build; preserving staging directory: {stage}")
        result["checks"]["all_source_code_protocol_hashes_unchanged"] = True
        original.write_json(result, stage / "feature_manifest.json")
        verify_complete_cache(stage, signature)
        require(not out.exists(), "Destination appeared during generation; preserving staging directory")
        stage.rename(out)
        print(f"Complete segmented features: {out}", flush=True)
        return result
    finally:
        if lock.is_file() and lock.read_text() == token:
            lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-year", type=int, choices=YEARS, required=True)
    parser.add_argument("--mode", choices=("development", "final"), required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cumulative-root", type=Path, default=CUMULATIVE_ROOT)
    args = parser.parse_args()
    build_fold(args.test_year, args.mode, args.root, args.cumulative_root)
