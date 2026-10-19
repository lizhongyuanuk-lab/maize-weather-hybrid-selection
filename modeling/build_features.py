"""Train-only feature fitting on existing Stage 1 exports; never reads targets.

Development fits 2014–2022; final fits 2014–2023 only after selection freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
PREP = PROJECT / "preparation/outputs"
SEED = 20261018
HORIZONS = (0, 30, 60, 90, 120)
WEATHER = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "RH2M", "WS2M"]
PHYSICAL = ["temperature_mean", "temperature_sd", "tmax_mean", "tmax_max",
            "tmin_mean", "tmin_min", "hot30_days", "hot35_days", "gdd_celsius_sum",
            "precipitation_sum", "wet_days", "heavy_rain_days", "dry_days",
            "longest_dry_spell", "wet_day_intensity", "solar_sum", "solar_mean",
            "humidity_mean", "humidity_sd", "wind_mean", "diurnal_range_mean"]
WEATHER_NAMES = PHYSICAL + [f"missing_fraction_{c}" for c in WEATHER] + [
    "any_interpolation", "available_weather_days"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_json(value, path):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def longest_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def aggregate_window(daily: pd.DataFrame, planting: pd.Timestamp, horizon: int,
                     medians: pd.Series) -> tuple[np.ndarray, dict]:
    """Crop first; interpolate only bounded interior gaps; then training medians."""
    dates = pd.date_range(planting, periods=horizon, freq="D")
    raw = daily.reindex(dates)[WEATHER].copy()
    missing = raw.isna()
    interior = raw.interpolate(method="linear", limit_area="inside")
    interpolation_count = int((missing & interior.notna()).sum().sum())
    fallback_count = int(interior.isna().sum().sum())
    clean = interior.fillna(medians)
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("Non-finite weather after train-only filling")
    t, tx, tn, rain, solar, rh, wind = [clean[c].to_numpy(float) for c in WEATHER]
    if (rain < 0).any():
        raise ValueError("Negative precipitation needs adjudication")
    wet, dry = rain >= 1.0, rain < 1.0
    # Modified corn GDD, Celsius equivalent of 86/50 F; daily accumulation >= 0.
    gdd = np.maximum((np.minimum(tx, 30.0) + np.maximum(tn, 10.0))/2 - 10.0, 0.0)
    physical = [t.mean(), t.std(ddof=0), tx.mean(), tx.max(), tn.mean(), tn.min(),
                (tx >= 30).sum(), (tx >= 35).sum(), gdd.sum(), rain.sum(), wet.sum(),
                (rain >= 10).sum(), dry.sum(), longest_run(dry),
                rain[wet].mean() if wet.any() else 0.0, solar.sum(), solar.mean(),
                rh.mean(), rh.std(ddof=0), wind.mean(), (tx-tn).mean()]
    values = np.asarray(physical + missing.mean().tolist() + [
        float(interpolation_count > 0), float((~missing.any(axis=1)).sum())], dtype=np.float64)
    return values, {"missing_cells": int(missing.sum().sum()),
                    "interpolated_cells": interpolation_count, "median_filled_cells": fallback_count,
                    "start_inclusive": str(planting.date()),
                    "end_exclusive": str((planting + pd.Timedelta(days=horizon)).date())}


def training_weather_medians(bridge, daily_by_env, train_ids, horizon):
    """Fit on distinct training Env-Date rows inside this horizon's windows."""
    selected = bridge.loc[bridge.sample_id.isin(train_ids), ["Env", "Planting_Date"]].drop_duplicates()
    indices: dict[str, set] = {}
    for env, date in selected.itertuples(index=False, name=None):
        indices.setdefault(env, set()).update(pd.date_range(date, periods=horizon))
    data = pd.concat([daily_by_env[e].reindex(sorted(d))[WEATHER] for e, d in sorted(indices.items())])
    medians = data.median()
    if medians.isna().any():
        raise ValueError("Training window has entirely missing weather variable")
    return medians, len(data)


def genotype_features(samples, genotype, n_components):
    training_hybrids = sorted(samples["train"].Hybrid.unique())
    matrix = genotype.set_index("Hybrid")
    if matrix.index.has_duplicates:
        raise ValueError("Duplicate genotype keys")
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    train = imputer.fit_transform(matrix.loc[training_hybrids])
    pca = PCA(n_components=n_components, svd_solver="randomized", iterated_power=4, random_state=SEED)
    with threadpool_limits(limits=4):
        pca.fit(train)
        all_pcs = pca.transform(imputer.transform(matrix))
    projection = pd.DataFrame(all_pcs, index=matrix.index)
    transformed = {split: projection.loc[part.Hybrid].to_numpy(np.float32)
                   for split, part in samples.items()}
    return transformed, {"imputer": imputer, "pca": pca}, {
        "fit_unique_hybrids": len(training_hybrids), "fit_hybrids": training_hybrids,
        "components": n_components, "variance_explained": float(pca.explained_variance_ratio_.sum()),
        "marker_scaling": "Mean imputation and centering; no per-marker variance scaling.",
        "fit_grain": "Unique training hybrids, not repeated environment-hybrid rows."}


def static_features(samples):
    columns = ["Latitude", "Longitude", "Planting_DOY"]
    train = samples["train"]
    medians = train[columns].median()
    if medians.isna().any():
        raise ValueError("Entirely missing training static feature")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    encoder.fit(train[["Experiment_Code"]].astype(str))
    result = {}
    for split, frame in samples.items():
        if len(frame) == 0:
            result[split] = np.empty((0, 5 + len(encoder.categories_[0])), np.float32)
            continue
        result[split] = np.column_stack([
            frame[columns].fillna(medians).to_numpy(float),
            frame[["Latitude", "Longitude"]].isna().to_numpy(float),
            encoder.transform(frame[["Experiment_Code"]].astype(str)),
        ]).astype(np.float32)
    names = columns + ["Latitude_missing", "Longitude_missing"] + encoder.get_feature_names_out().tolist()
    return result, {"medians": medians, "encoder": encoder}, names


def build(mode="development", n_components=100):
    if mode not in {"development", "final"}:
        raise ValueError(mode)
    if mode == "final" and not (PROJECT / "modeling/results/development/frozen_selection.json").exists():
        raise ValueError("Freeze development model selection before final feature fitting")
    out = PROJECT / "modeling/features" / mode
    out.mkdir(parents=True, exist_ok=True)
    inputs = [PREP / f"samples/{s}.csv" for s in ("train", "validation", "test")]
    inputs += [PREP / "dimensions/genotypes.csv.gz", PREP / "dimensions/weather_daily.csv.gz",
               PREP / "links/sample_plantings.csv.gz", PREP / "links/weather_windows.csv", PREP / "manifest.json"]
    hashes = {str(p.relative_to(PROJECT)): digest(p) for p in inputs}
    samples = {s: pd.read_csv(PREP / f"samples/{s}.csv") for s in ("train", "validation", "test")}
    if mode == "final":
        samples["train"] = pd.concat([samples["train"], samples["validation"]], ignore_index=True)
        samples["validation"] = samples["validation"].iloc[:0].copy()
    assert samples["train"].Year.max() == (2022 if mode == "development" else 2023)
    for split, part in samples.items():
        assert part.sample_id.is_unique
        part.to_csv(out / f"{split}_samples.csv", index=False)
    print(f"{mode}: fitting genotype PCA on unique training hybrids", flush=True)
    genotype = pd.read_csv(PREP / "dimensions/genotypes.csv.gz")
    genomic, genome_estimators, genome_meta = genotype_features(samples, genotype, n_components)
    static, static_estimators, static_names = static_features(samples)
    base = {s: np.column_stack([genomic[s], static[s]]).astype(np.float32) for s in samples}
    base_names = [f"PC{i+1:03}" for i in range(n_components)] + static_names
    joblib.dump({"genotype": genome_estimators, "static": static_estimators}, out / "preprocessing.joblib")
    np.savez_compressed(out / "W0.npz", **{f"X_{s}": x for s, x in base.items()})
    write_json({"columns": base_names, "weather_columns": [], "horizon_days": 0}, out / "W0_columns.json")

    weather = pd.read_csv(PREP / "dimensions/weather_daily.csv.gz", parse_dates=["Date"])
    assert not weather.duplicated(["Env", "Date"]).any()
    daily = {env: group.set_index("Date")[WEATHER].sort_index() for env, group in weather.groupby("Env")}
    bridge = pd.read_csv(PREP / "links/sample_plantings.csv.gz", parse_dates=["Planting_Date"])
    declared = pd.read_csv(PREP / "links/weather_windows.csv", parse_dates=["Planting_Date"])
    pairs = bridge[["Env", "Planting_Date"]].drop_duplicates().sort_values(["Env", "Planting_Date"])
    assert np.allclose(bridge.groupby("sample_id").planting_weight.sum(), 1.0)
    train_ids = set(samples["train"].sample_id)
    weather_fit = {}
    diagnostics = []
    for horizon in HORIZONS[1:]:
        print(f"{mode}: W{horizon} weather windows", flush=True)
        medians, fit_count = training_weather_medians(bridge, daily, train_ids, horizon)
        rows = []
        for env, planting in pairs.itertuples(index=False, name=None):
            expected = pd.date_range(planting, periods=horizon)
            assert expected.isin(daily[env].index).all(), f"Calendar incomplete: {env} {planting}"
            values, diag = aggregate_window(daily[env], planting, horizon, medians)
            rows.append([env, planting, *values])
            diagnostics.append({"Env": env, "Planting_Date": str(planting.date()), "horizon": horizon, **diag})
        pair_features = pd.DataFrame(rows, columns=["Env", "Planting_Date"] + WEATHER_NAMES)
        linked = bridge.merge(pair_features, on=["Env", "Planting_Date"], validate="many_to_one")
        assert linked[WEATHER_NAMES].notna().all().all()
        weighted = linked[WEATHER_NAMES].multiply(linked.planting_weight, axis=0)
        weighted["sample_id"] = linked.sample_id
        by_sample = weighted.groupby("sample_id")[WEATHER_NAMES].sum()
        arrays = {f"X_{s}": np.column_stack([base[s], by_sample.loc[f.sample_id].to_numpy()]).astype(np.float32)
                  for s, f in samples.items()}
        assert all(np.isfinite(x).all() for x in arrays.values())
        np.savez_compressed(out / f"W{horizon}.npz", **arrays)
        write_json({"columns": base_names + WEATHER_NAMES, "weather_columns": WEATHER_NAMES,
                    "horizon_days": horizon}, out / f"W{horizon}_columns.json")
        pair_features.to_csv(out / f"W{horizon}_environment_planting_features.csv", index=False)
        weather_fit[str(horizon)] = {"training_distinct_env_dates": fit_count, "medians": medians.to_dict()}
    expected_pairs = set(zip(declared.Env, declared.Planting_Date, declared.horizon_days))
    actual_pairs = {(env, date, h) for env, date in pairs.itertuples(index=False, name=None) for h in HORIZONS[1:]}
    assert actual_pairs == expected_pairs
    pd.DataFrame(diagnostics).to_csv(out / "weather_imputation_audit.csv", index=False)
    checks = {"no_targets_read": True, "fit_training_years_only": True,
              "genotype_fit_unique_training_hybrids": True, "weather_clipped_before_imputation": True,
              "paired_sample_ids_all_windows": True, "finite_matrices": True,
              "all_preparation_input_hashes_unchanged": all(digest(PROJECT/p) == h for p, h in hashes.items())}
    assert all(checks.values())
    metadata = {"stage": "features", "mode": mode, "fit_years": [2014, int(samples["train"].Year.max())],
                "seed": SEED, "rows": {s: len(f) for s, f in samples.items()},
                "base_feature_count": len(base_names), "weather_feature_count": len(WEATHER_NAMES),
                "physical_weather_features": len(PHYSICAL), "genotype": genome_meta,
                "weather_fitting": weather_fit, "checks": checks,
                "input_hashes": hashes, "script_sha256": digest(Path(__file__)),
                "python": platform.python_version(),
                "radiation_units": "ALLSKY_SFC_SW_DWN retained in released data units; no unverified physical-unit conversion.",
                "gdd_definition": "sum(max((min(Tmax,30)+max(Tmin,10))/2-10,0)), Celsius degree-days",
                "rain_thresholds_mm": {"wet_at_least": 1, "heavy_at_least": 10},
                "mixed_dates": "Compute nonlinear per-date features first, then original-plot-count-weighted mean; these are mixture features, not a single weather trajectory."}
    metadata["outputs"] = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "feature_manifest.json"}
    write_json(metadata, out / "feature_manifest.json")
    print(f"Complete: {out}; {len(base_names)} base + {len(WEATHER_NAMES)} weather/quality features", flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["development", "final"], default="development")
    args = parser.parse_args()
    build(args.mode)
