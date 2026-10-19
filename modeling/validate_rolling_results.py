"""Independent numerical and provenance audit of the completed rolling study.

Never fits models or changes experiments. All four new folds and the stability
analysis must be complete before evaluated outcomes are inspected. Historical
outcomes are not newly blinded data. Experiment_Code is only a site-code proxy.
This module does not import the production metrics, runner or stability helpers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
import traceback
import unittest

import joblib
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
PREP = PROJECT / "preparation/outputs"
ROOT = PROJECT / "modeling/rolling_v1"
YEARS = (2020, 2021, 2022, 2023, 2024)
HORIZONS = (0, 30, 60, 90, 120)
MODELS = ("Ridge", "ExtraTrees", "CatBoost")
BASELINES = ("GlobalMean", "HistoricalSiteMean")
SEED, REPEATS = 20261018, 2000
ORIGINAL_2024_SHA256 = "c9cce89304d08087b48fc0c2f0bfdb52794eaee4b89aeb97a1b7cee51d38581a"
METRICS = ("rmse", "mae", "bias", "pearson", "spearman", "top20_recall")
IDENTITY = ["sample_id", "Env", "Hybrid", "Year", "Experiment_Code"]
PAIRS = {(m, w) for m in MODELS for w in HORIZONS}
ALL_PAIRS = PAIRS | {(m, 0) for m in BASELINES}
EXPECTED_COVERAGE = {2020: (11298, 25), 2021: (13615, 27), 2022: (11783, 28),
                     2023: (11491, 27), 2024: (9486, 22)}
PRESET = {
    "Ridge": [{"alpha": 100.0}, {"alpha": 10000.0}],
    "ExtraTrees": [{"n_estimators": 160, "max_depth": 18, "min_samples_leaf": leaf,
                    "max_features": 0.8, "n_jobs": 4, "random_state": SEED} for leaf in (5, 20)],
    "CatBoost": [{"iterations": 600, "depth": depth, "learning_rate": 0.05,
                  "l2_leaf_reg": 5, "thread_count": 4, "random_seed": SEED,
                  "loss_function": "RMSE", "verbose": False, "allow_writing_files": False}
                 for depth in (6, 8)],
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


class AuditError(ValueError):
    pass


class Audit:
    def __init__(self):
        self.checks, self.notes, self.hash_cache, self.read_receipts, self.hash_stats = [], [], {}, {}, {}

    def require(self, condition, name):
        passed = bool(condition)
        self.checks.append({"name": name, "passed": passed})
        if not passed:
            raise AuditError(name)

    def close(self, actual, expected, name):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        self.require(a.shape == b.shape and np.allclose(a, b, atol=1e-9, rtol=1e-9, equal_nan=True), name)

    def hash(self, path):
        path = Path(path).resolve()
        self.require(path.is_file(), f"File exists: {path}")
        if str(path) not in self.hash_cache:
            self.hash_cache[str(path)] = digest(path)
            info = path.stat()
            self.hash_stats[str(path)] = (info.st_size, info.st_mtime_ns)
        return self.hash_cache[str(path)]

    def hash_map(self, mapping, base=PROJECT, name="Fingerprint"):
        normalized = {}
        for key, expected in mapping.items():
            p = Path(key)
            p = p if p.is_absolute() else base / p
            self.require(self.hash(p) == expected, f"{name}: {p}")
            normalized[str(p.resolve())] = expected
        return normalized

    def artifacts(self, path, base):
        record = read_json(path)
        actual_base = Path(record.get("artifact_path_base", base)).resolve()
        self.require(actual_base == base.resolve(), f"Artifact base: {path}")
        for name in record["artifacts"]:
            p = Path(name)
            self.require(not p.is_absolute() and (actual_base / p).resolve().is_relative_to(actual_base),
                         f"Contained artifact path: {name}")
        self.hash_map(record["artifacts"], actual_base, f"Artifact of {path.name}")
        self.read_receipts[str(Path(path).resolve())] = self.hash(path)
        return record

    def bound_paths(self, record, paths, base, name):
        recorded = {(base / key).resolve() for key in record["artifacts"]}
        self.require({Path(p).resolve() for p in paths}.issubset(recorded),
                     f"{name}: all required artifacts bound, not merely listed hashes valid")

    def table(self, actual, expected, keys, name):
        self.require(set(expected.columns).issubset(actual.columns), f"{name}: required columns")
        self.require(not actual.duplicated(keys).any(), f"{name}: unique keys")
        self.require(not expected.duplicated(keys).any(), f"{name}: independent keys unique")
        a, e = actual.set_index(keys), expected.set_index(keys)
        self.require(set(a.index) == set(e.index), f"{name}: exact key coverage")
        a = a.reindex(e.index)
        for column in e.columns:
            if pd.api.types.is_numeric_dtype(e[column]) or pd.api.types.is_bool_dtype(e[column]):
                self.close(pd.to_numeric(a[column], errors="raise").to_numpy(), e[column].to_numpy(),
                           f"{name}: independent {column}")
            else:
                self.require(a[column].astype("string").fillna("").tolist() == e[column].astype("string").fillna("").tolist(),
                             f"{name}: {column}")


def correlation(y, p, ranked=False):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 2 or y.max() == y.min() or p.max() == p.min():
        return np.nan
    if ranked:
        y, p = (pd.Series(v).rank(method="average").to_numpy() for v in (y, p))
    y, p = y - y.mean(), p - p.mean()
    return float(np.sum(y * p) / math.sqrt(np.sum(y * y) * np.sum(p * p)))


def top20(y, p):
    k = math.ceil(len(y) / 5)

    def inclusion(values):
        values = np.asarray(values)
        w, remaining = np.zeros(len(values)), k
        for value in sorted(set(values), reverse=True):
            mask = values == value
            n = int(mask.sum())
            w[mask] = min(remaining, n) / n
            remaining -= min(remaining, n)
            if not remaining:
                break
        return w

    return float(np.sum(inclusion(y) * inclusion(p)) / k)


def mean_interval(values):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    # Explicit independent arithmetic; identical declared seed/sampling scheme.
    draws = np.empty(REPEATS)
    for i in range(REPEATS):
        sampled = values[rng.integers(0, len(values), size=len(values))]
        draws[i] = np.sum(sampled) / len(sampled)
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def numerical_metrics(frame, interval=False):
    if not len(frame):
        return {"n": 0, "n_environments": 0,
                **{f"macro_{m}": np.nan for m in METRICS}}, pd.DataFrame()
    rows = []
    for env in sorted(frame.Env.unique()):
        g = frame.loc[frame.Env == env]
        y, p = g.y_true.to_numpy(float), g.prediction.to_numpy(float)
        error = p - y
        rows.append({"Env": env, "n": len(g), "rmse": math.sqrt(np.sum(error * error) / len(g)),
                     "mae": np.sum(np.abs(error)) / len(g), "bias": np.sum(error) / len(g),
                     "pearson": correlation(y, p), "spearman": correlation(y, p, True),
                     "top20_recall": top20(y, p)})
    environments = pd.DataFrame(rows)
    error = frame.prediction.to_numpy(float) - frame.y_true.to_numpy(float)
    summary = {"n": len(frame), "n_environments": len(environments),
               "pooled_rmse": math.sqrt(np.sum(error * error) / len(frame)),
               "pooled_mae": np.sum(np.abs(error)) / len(frame), "pooled_bias": np.sum(error) / len(frame)}
    for metric in METRICS:
        valid = environments[metric].dropna().to_numpy()
        summary[f"macro_{metric}"] = np.sum(valid) / len(valid) if len(valid) else np.nan
    for metric in ("pearson", "spearman"):
        summary[f"{metric}_defined_environments"] = int(environments[metric].notna().sum())
    if interval:
        summary["macro_rmse_ci95_low"], summary["macro_rmse_ci95_high"] = mean_interval(environments.rmse)
    return summary, environments


def compare_record(audit, actual, expected, context):
    for key, expected_value in expected.items():
        audit.require(key in actual, f"{context}: contains {key}")
        audit.close(np.nan if actual[key] is None else actual[key], expected_value, f"{context}: {key}")


def paired_rows(reference, candidate, model, horizon):
    # Both scores and their pairing are independently derived from prediction rows.
    _, first = numerical_metrics(reference)
    _, second = numerical_metrics(candidate)
    rows = first.merge(second, on="Env", suffixes=("_W0", "_weather"), validate="one_to_one")
    meta = reference[["Env", "Year", "Experiment_Code"]].drop_duplicates()
    rows = rows.merge(meta, on="Env", validate="one_to_one")
    for metric in METRICS:
        rows[f"delta_{metric}"] = rows[f"{metric}_weather"] - rows[f"{metric}_W0"]
    return rows.assign(model=model, window=horizon)


def annual_stability(frame):
    d = frame.delta_rmse.to_numpy(float)
    e0, ew = float(frame.rmse_W0.mean()), float(frame.rmse_weather.mean())
    improved, worsened = int(np.sum(d < -1e-12)), int(np.sum(d > 1e-12))
    lower, upper = mean_interval(d) if len(d) > 1 else (np.nan, np.nan)
    return {"n_environments": len(d), "n_site_codes": frame.Experiment_Code.nunique(),
            "n_samples": int(frame.n_W0.sum()), "macro_rmse_W0": e0, "macro_rmse_weather": ew,
            "delta_rmse": float(np.sum(d) / len(d)), "relative_gain_percent": 100 * (e0 - ew) / e0 if e0 > 0 else np.nan,
            "environment_ci95_low": lower, "environment_ci95_high": upper,
            "improved_environments": improved, "worsened_environments": worsened,
            "unchanged_environments": len(d) - improved - worsened,
            "improved_fraction": improved / len(d), "worsened_fraction": worsened / len(d),
            "delta_p10": np.quantile(d, 0.1), "delta_median": np.median(d),
            "delta_p90": np.quantile(d, 0.9), "delta_min": d.min(), "delta_max": d.max(),
            "interval_status": "conditional_environment_bootstrap" if len(d) > 1 else "not_informative_single_environment"}


def cross_year_stability(frame):
    """Independent matrix formulation of the declared crossed cluster bootstrap."""
    years, sites = sorted(frame.Year.unique()), sorted(frame.Experiment_Code.unique())
    annual = frame.groupby("Year", sort=True)[["delta_rmse", "rmse_W0", "rmse_weather"]].mean()
    annual_d = annual.delta_rmse.to_numpy()
    denominator = annual.rmse_W0.where(annual.rmse_W0 > 0)
    overall_denominator = float(annual.rmse_W0.mean())
    result = {"n_years": len(years), "n_environment_years": len(frame), "n_site_codes": len(sites),
              "evaluated_years": ";".join(str(y) for y in years),
              "missing_prespecified_years": ";".join(str(y) for y in YEARS if y not in years),
              "year_equal_delta_rmse": annual_d.mean(),
              "environment_year_equal_delta_rmse": frame.delta_rmse.mean(),
              "year_equal_rmse_W0": annual.rmse_W0.mean(), "year_equal_rmse_weather": annual.rmse_weather.mean(),
              "relative_gain_of_year_equal_rmse_percent": 100 * (overall_denominator - annual.rmse_weather.mean()) / overall_denominator if overall_denominator > 0 else np.nan,
              "mean_annual_relative_gain_percent": (100 * (annual.rmse_W0 - annual.rmse_weather) / denominator).mean(),
              "years_improved": int(np.sum(annual_d < -1e-12)), "years_worsened": int(np.sum(annual_d > 1e-12)),
              "year_min_delta": annual_d.min(), "year_max_delta": annual_d.max(),
              "year_block_ci95_low": np.nan, "year_block_ci95_high": np.nan,
              "crossed_year_site_ci95_low": np.nan, "crossed_year_site_ci95_high": np.nan,
              "bootstrap_repeats": REPEATS, "seed": SEED,
              "interpretation": "descriptive_conditional_sensitivity_not_future_year_guarantee"}
    if len(years) < 2:
        return {**result, "bootstrap_status": "insufficient_years"}
    rng = np.random.default_rng(SEED)
    year_draws = rng.integers(0, len(years), size=(REPEATS, len(years)))
    year_means = np.sum(annual_d[year_draws], axis=1) / len(years)
    result["year_block_ci95_low"], result["year_block_ci95_high"] = np.quantile(year_means, [0.025, 0.975])
    counts, totals = np.zeros((len(years), len(sites))), np.zeros((len(years), len(sites)))
    yi, si = {v: i for i, v in enumerate(years)}, {v: i for i, v in enumerate(sites)}
    for row in frame.itertuples():
        counts[yi[row.Year], si[row.Experiment_Code]] += 1
        totals[yi[row.Year], si[row.Experiment_Code]] += row.delta_rmse
    draws, attempts = [], 0
    while len(draws) < REPEATS and attempts < REPEATS * 100:
        attempts += 1
        yw = rng.multinomial(len(years), np.full(len(years), 1 / len(years)))
        sw = rng.multinomial(len(sites), np.full(len(sites), 1 / len(sites)))
        denominator = counts @ sw
        if np.any((yw > 0) & (denominator == 0)):
            continue
        weighted_years = np.divide(totals @ sw, denominator, out=np.zeros(len(years)), where=denominator != 0)
        draws.append(float(np.sum(weighted_years * yw) / yw.sum()))
    if len(draws) != REPEATS:
        raise AuditError("Independent crossed bootstrap did not obtain enough valid draws")
    result["crossed_year_site_ci95_low"], result["crossed_year_site_ci95_high"] = np.quantile(draws, [0.025, 0.975])
    return {**result, "bootstrap_status": "complete", "crossed_empty_draws_rejected": attempts - REPEATS}


def complete_cross_subgroups(frame):
    """Retain every prescribed group, including groups with no observed members."""
    index = pd.MultiIndex.from_product([MODELS, HORIZONS[1:], ("hybrid", "site_code"), ("seen", "unseen")],
                                      names=["model", "window", "group_type", "group"])
    result = frame.set_index(index.names).reindex(index).reset_index()
    for name in ("n_years", "n_environment_years", "n_site_codes"):
        result[name] = result[name].fillna(0).astype(int)
    for name, default in (("evaluated_years", ""), ("missing_prespecified_years", ";".join(map(str, YEARS))),
                          ("interpretation", "empty_subgroup"), ("bootstrap_status", "empty_subgroup")):
        result[name] = result[name].fillna(default)
    return result


def directories(root, year, stage):
    if year == 2024:
        return PROJECT / "modeling/features" / stage, PROJECT / "modeling/results" / stage
    return root / "features" / str(year) / stage, root / "results" / str(year) / stage


def readiness(audit, root):
    for year in YEARS[:-1]:
        for stage, filename in (("development", "development_completion_manifest.json"),
                                ("final", "final_evaluation_manifest.json")):
            audit.require((root / "results" / str(year) / stage / filename).is_file(),
                          f"Complete {year} {stage} before reading any evaluated outcomes")
    audit.require((root / "analysis/analysis_manifest.json").is_file(), "Complete stability analysis before validation")
    original = PROJECT / "modeling/results/final/final_evaluation_manifest.json"
    audit.require(audit.hash(original) == ORIGINAL_2024_SHA256, "Original 2024 completion SHA-256 remains exactly unchanged")


def input_chain(audit, root, year, stage, pool, genotype):
    feature_dir, result_dir = directories(root, year, stage)
    run_path = result_dir / "run_manifest.json"
    run = read_json(run_path)
    audit.require(run["stage"] == stage and run["seed"] == SEED and run["candidates"] == PRESET,
                  f"{year} {stage}: run identity and unchanged candidate budget")
    inputs = audit.hash_map(run["inputs"], PROJECT, f"{year} {stage} run input")
    for source in (PREP / "targets/train.csv", PREP / "targets/validation.csv", PREP / "manifest.json",
                   PROJECT / "modeling/run_experiments.py", PROJECT / "modeling/metrics.py"):
        audit.require(inputs.get(str(source.resolve())) == audit.hash(source), f"{year} {stage}: required source/code bound: {source.name}")
    if year != 2024:
        audit.require(run["test_year"] == year and run["retrospective_backtest"]
                      and not run["original_2024_holdout_read"], f"{year} {stage}: historical—not fresh blind—run")
        audit.require(inputs.get(str((root / "PROTOCOL.md").resolve())) == audit.hash(root / "PROTOCOL.md"),
                      f"{year} {stage}: run binds rolling protocol")
        audit.require(not any("/holdout/" in path for path in inputs), f"{year} {stage}: original holdout not a rolling input")
    feature_path = feature_dir / "feature_manifest.json"
    audit.require(inputs.get(str(feature_path.resolve())) == audit.hash(feature_path), f"{year} {stage}: feature manifest bound to run")
    feature = read_json(feature_path)
    end = year - (2 if stage == "development" else 1)
    audit.require(feature["mode"] == stage and feature["fit_years"] == [2014, end], f"{year} {stage}: preprocessing fit years")
    audit.require(all(feature["checks"].values()), f"{year} {stage}: feature checks passed")
    source_hashes = audit.hash_map(feature["input_hashes"], PROJECT, f"{year} {stage} feature source/code")
    if year != 2024:
        audit.require(feature["test_year"] == year, f"{year} {stage}: feature fold identity")
        audit.require(all(inputs.get(path) == sha for path, sha in source_hashes.items()),
                      f"{year} {stage}: complete feature-source chain is included in run inputs")
    feature_outputs = audit.hash_map(feature["outputs"], feature_dir, f"{year} {stage} feature artifact")
    required_outputs = [feature_dir / "preprocessing.joblib"]
    required_outputs += [feature_dir / f"{s}_samples.csv" for s in ("train", "validation", "test")]
    required_outputs += [feature_dir / f"W{w}{suffix}" for w in HORIZONS for suffix in (".npz", "_columns.json")]
    audit.require({str(p.resolve()) for p in required_outputs}.issubset(feature_outputs), f"{year} {stage}: complete feature output fingerprint inventory")
    expected = {"train": pool.loc[pool.Year.between(2014, end)],
                "validation": pool.loc[pool.Year == year - 1] if stage == "development" else pool.iloc[:0],
                "test": pool.loc[pool.Year == year]}
    samples = {}
    for split in ("train", "validation", "test"):
        path = feature_dir / f"{split}_samples.csv"
        audit.require(inputs.get(str(path.resolve())) == audit.hash(path), f"{year} {stage}: bound {split} membership")
        frame = pd.read_csv(path)
        audit.require(frame.sample_id.is_unique and frame.sample_id.notna().all(), f"{year} {stage} {split}: unique IDs")
        audit.require(not {"Yield_Mg_ha", "y_true", "target"}.intersection(frame.columns), f"{year} {stage} {split}: label-free inputs")
        for column in IDENTITY:
            audit.require(frame[column].tolist() == expected[split][column].tolist(), f"{year} {stage} {split}: fixed-cohort order/{column}")
        audit.require(feature["rows"][split] == len(frame), f"{year} {stage}: manifest {split} count")
        samples[split] = frame
    hybrid_ids = sorted(samples["train"].Hybrid.unique())
    audit.require(feature["genotype"]["fit_hybrids"] == hybrid_ids
                  and feature["genotype"]["fit_unique_hybrids"] == len(hybrid_ids)
                  and feature["genotype"]["components"] == 100, f"{year} {stage}: PCA unique training hybrids only")
    preprocessing = joblib.load(feature_dir / "preprocessing.joblib")
    fitted = preprocessing["genotype"]
    audit.require(fitted["pca"].n_samples_ == len(hybrid_ids) and fitted["pca"].n_components_ == 100,
                  f"{year} {stage}: serialized PCA fit sample/component counts")
    raw = genotype.loc[hybrid_ids].to_numpy(float)
    counts = np.sum(np.isfinite(raw), axis=0)
    means = np.divide(np.nansum(raw, axis=0), counts, out=np.zeros(raw.shape[1]), where=counts > 0)
    audit.close(fitted["imputer"].statistics_, means, f"{year} {stage}: independent unique-training genotype means")
    audit.close(fitted["pca"].mean_, means, f"{year} {stage}: PCA center from training-only means")
    static = preprocessing["static"]
    columns = ["Latitude", "Longitude", "Planting_DOY"]
    audit.close(static["medians"].reindex(columns).to_numpy(), np.nanmedian(samples["train"][columns].to_numpy(), axis=0),
                f"{year} {stage}: training-only static medians")
    audit.require(list(static["encoder"].categories_[0]) == sorted(samples["train"].Experiment_Code.astype(str).unique()),
                  f"{year} {stage}: training-only site-code categories")
    audit.read_receipts[str(run_path)] = audit.hash(run_path)
    return run, samples


def read_predictions(audit, path, samples, labels, context, evaluated=False):
    frame = pd.read_csv(path)
    expected = samples.loc[samples.sample_id.isin(labels.index)] if evaluated else samples
    audit.require(frame.sample_id.is_unique, f"{context}: unique prediction IDs")
    for column in IDENTITY:
        audit.require(frame[column].tolist() == expected[column].tolist(), f"{context}: same prediction order/{column}")
    audit.require(np.isfinite(frame.prediction.to_numpy(float)).all(), f"{context}: finite predictions")
    observed = labels.reindex(frame.sample_id).to_numpy(float)
    if evaluated:
        audit.close(frame.y_true.to_numpy(), observed, f"{context}: original observed targets")
        audit.require(np.isfinite(observed).all(), f"{context}: all evaluated observations exist")
    else:
        audit.require("y_true" not in frame and "Yield_Mg_ha" not in frame, f"{context}: unscored predictions are label-free")
        frame = frame.assign(y_true=observed)
    return frame


def verify_units(audit, root, year, stage, run):
    _, out = directories(root, year, stage)
    expected = PAIRS | ({(m, 0) for m in BASELINES} if year != 2024 else set())
    audit.require({p.stem for p in (out / "units").glob("*.json")} == {f"{m}_W{w}" for m, w in expected},
                  f"{year} {stage}: complete unit inventory")
    records = {}
    for model, window in sorted(expected):
        record = audit.artifacts(out / "units" / f"{model}_W{window}.json", PROJECT if year == 2024 else root)
        if year != 2024:
            audit.require(record["run_manifest_sha256"] == audit.hash(out / "run_manifest.json"),
                          f"{year} {stage} {model} W{window}: unit belongs to run")
        records[(model, window)] = record
    return records


def development_selection(audit, root, year, pool, genotype, labels):
    run, samples = input_chain(audit, root, year, "development", pool, genotype)
    _, out = directories(root, year, "development")
    records = verify_units(audit, root, year, "development", run)
    freeze_path = out / "frozen_selection.json"
    frozen = read_json(freeze_path)
    audit.require(frozen["development_run_manifest_sha256"] == audit.hash(out / "run_manifest.json"), f"{year}: freeze binds development run")
    audit.require(frozen["development_input_hashes"] == run["inputs"], f"{year}: freeze source inputs exact")
    audit.require(frozen["windows"] == list(HORIZONS) and len(frozen["configs"]) == 15
                  and {(c["model"], c["window"]) for c in frozen["configs"]} == PAIRS, f"{year}: 15 frozen model/window choices")
    if year != 2024:
        completion = audit.artifacts(out / "development_completion_manifest.json", root)
        required_artifacts = [out / n for n in ("run_manifest.json", "frozen_selection.json", "candidate_metrics.csv", "selected_metrics.csv")]
        required_artifacts += [out / "units" / f"{m}_W{w}.json" for m, w in ALL_PAIRS]
        required_artifacts += [out / "predictions" / (f"{m}.csv" if m in BASELINES else f"{m}_W{w}.csv") for m, w in ALL_PAIRS]
        required_artifacts += [out / "candidate_predictions" / f"{m}_W{w}_c{c}{suffix}.csv"
                               for m, w in PAIRS for c in (0, 1) for suffix in ("", "_environments")]
        audit.bound_paths(completion, required_artifacts, root, f"{year} development completion")
        audit.require(completion["run_manifest_sha256"] == audit.hash(out / "run_manifest.json")
                      and completion["test_year"] == year and completion["stage"] == "development",
                      f"{year}: development completion fold/run identity")
        audit.require(completion["frozen_selection_sha256"] == audit.hash(freeze_path), f"{year}: completed selection freeze hash")
        audit.require(frozen["test_year"] == year and frozen["validation_year"] == year - 1
                      and frozen["fold_test_outcomes_joined_for_selection"] is False, f"{year}: preceding-year-only selection declaration")
    else:
        audit.require(frozen["holdout_labels_read"] is False, "Original selection did not use holdout outcomes")
    chosen, expected_candidates, expected_selected = [], [], []
    for horizon in HORIZONS:
        for model in MODELS:
            unit = records[(model, horizon)]
            scores = []
            for candidate in (0, 1):
                key = f"{model}_W{horizon}_c{candidate}"
                prediction = read_predictions(audit, out / "candidate_predictions" / f"{key}.csv",
                                              samples["validation"], labels, f"{year} {key}")
                audit.require(prediction.y_true.notna().all(), f"{year} {key}: validation target coverage")
                summary, envs = numerical_metrics(prediction, interval=True)
                scores.append(summary)
                candidate_record = [v for v in unit["candidates"] if v["candidate"] == candidate]
                audit.require(len(candidate_record) == 1 and candidate_record[0]["params"] == PRESET[model][candidate],
                              f"{year} {key}: preset candidate identity")
                compare_record(audit, candidate_record[0], summary, f"{year} {key} unit")
                audit.table(pd.read_csv(out / "candidate_predictions" / f"{key}_environments.csv"), envs, ["Env"], f"{year} {key} environments")
                expected_candidates.append({"model": model, "window": horizon, "candidate": candidate, **summary})
            best = min(range(2), key=lambda i: scores[i]["macro_rmse"])
            config = next(c for c in frozen["configs"] if (c["model"], c["window"]) == (model, horizon))
            audit.require(config["candidate"] == best == unit["selected"]["candidate"]
                          and config["params"] == PRESET[model][best], f"{year} {model} W{horizon}: independently minimum validation macro RMSE chosen")
            audit.close(config["validation_macro_rmse"], scores[best]["macro_rmse"], f"{year} {model} W{horizon}: frozen score")
            frame = read_predictions(audit, out / "predictions" / f"{model}_W{horizon}.csv", samples["validation"], labels,
                                     f"{year} selected {model} W{horizon}")
            candidate = pd.read_csv(out / "candidate_predictions" / f"{model}_W{horizon}_c{best}.csv")
            audit.close(frame.prediction.to_numpy(), candidate.prediction.to_numpy(), f"{year} {model} W{horizon}: selected prediction identity")
            expected_selected.append({"model": model, "window": horizon, **scores[best]})
            chosen.append(config)
    for model in BASELINES:
        frame = read_predictions(audit, out / "predictions" / f"{model}.csv", samples["validation"], labels, f"{year} {model}")
        audit.close(frame.prediction.to_numpy(), baseline(model, samples["train"], labels, samples["validation"]), f"{year}: development {model} mapping")
        summary, envs = numerical_metrics(frame, True)
        expected_selected.append({"model": model, "window": 0, **summary})
        audit.table(pd.read_csv(out / f"{model}_environment_metrics.csv"), envs, ["Env"], f"{year} {model} environment scores")
    audit.table(pd.read_csv(out / "candidate_metrics.csv"), pd.DataFrame(expected_candidates), ["model", "window", "candidate"], f"{year}: all 30 validation candidates")
    audit.table(pd.read_csv(out / "selected_metrics.csv"), pd.DataFrame(expected_selected), ["model", "window"], f"{year}: 17 selected validation rows")
    audit.require(frozen["recommended_by_validation"] == min(chosen, key=lambda c: c["validation_macro_rmse"]), f"{year}: unchanged validation-only recommendation")
    audit.read_receipts[str(freeze_path)] = audit.hash(freeze_path)
    return frozen


def baseline(model, training, labels, targets):
    y = labels.reindex(training.sample_id).to_numpy(float)
    if not np.isfinite(y).all():
        raise AuditError("Missing training label while independently computing baseline")
    overall = np.sum(y) / len(y)
    if model == "GlobalMean":
        return np.repeat(overall, len(targets))
    source = pd.DataFrame({"site": training.Experiment_Code.to_numpy(), "y": y})
    grouped = source.groupby("site").y.agg(["sum", "count"])
    mapping = (grouped["sum"] / grouped["count"]).to_dict()
    return np.array([mapping.get(code, overall) for code in targets.Experiment_Code])


def prediction_events(audit, year, events, lock_hash, frozen_at, completed_at):
    """Allow score-only retries, but every outcome join needs the same valid lock."""
    lock_name = "all_predictions_persisted_before_holdout" if year == 2024 else "all_predictions_locked"
    target_name = "holdout_labels_opened_for_preset_evaluation" if year == 2024 else "historical_test_targets_joined"
    first_lock, last_lock, last_lock_join = None, None, None
    previous_time, joins = timestamp(frozen_at), 0
    completed_time = timestamp(completed_at)
    for index, event in enumerate(events):
        if event["event"] not in (lock_name, target_name):
            continue
        current_time = timestamp(event["utc"])
        audit.require(previous_time <= current_time <= completed_time,
                      f"{year}: lock/join event {index} chronological and within freeze/completion")
        previous_time = current_time
        if event["event"] == lock_name:
            audit.require(event["count"] == 17 and event["prediction_lock_sha256"] == lock_hash,
                          f"{year}: lock event {index} preserves complete prediction fingerprint")
            if first_lock is None:
                first_lock = index
            last_lock, last_lock_join = index, None
        else:
            audit.require(last_lock is not None, f"{year}: outcome join {index} preceded by a complete valid lock")
            joins += 1
            if last_lock_join is None:
                last_lock_join = current_time
    audit.require(first_lock is not None and joins > 0 and last_lock_join is not None,
                  f"{year}: complete lock/join sequence including final lock write")
    return first_lock, last_lock_join


def prediction_ordering(audit, root, year, out, feature_dir, frozen_path, completion):
    base = PROJECT if year == 2024 else root
    lock_path = out / "prediction_lock_before_holdout.json"
    lock = read_json(lock_path)
    audit.require(len(lock["prediction_hashes"]) == 17, f"{year}: 17 predictions in lock")
    audit.hash_map(lock["prediction_hashes"], base, f"{year} prediction lock")
    names = {Path(key).name for key in lock["prediction_hashes"]}
    audit.require(names == {f"{m}_W{w}.csv" for m, w in PAIRS} | {f"{m}.csv" for m in BASELINES}, f"{year}: exact locked configuration coverage")
    audit.require(lock["frozen_selection_sha256"] == completion["frozen_selection_sha256"] == audit.hash(frozen_path), f"{year}: lock/final/freeze chain")
    audit.require(lock["test_sample_ids_sha256"] == audit.hash(feature_dir / "test_samples.csv"), f"{year}: locked test sample order")
    events = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines() if line.strip()]
    li, last_lock_join = prediction_events(audit, year, events, audit.hash(lock_path),
                                          read_json(frozen_path)["frozen_at_utc"], completion["completed_at_utc"])
    fit_names = {"final_fit_complete", "resume_completed_final_prediction"} if year == 2024 else {"fit_complete", "resume_completed_unit_no_fit"}
    fits = {(e["model"], e["window"]) for e in events[:li] if e["event"] in fit_names}
    audit.require(fits == PAIRS, f"{year}: all 15 model fits completed before scoring")
    for name in lock["prediction_hashes"]:
        audit.require((base / name).stat().st_mtime <= last_lock_join + 0.001, f"{year}: prediction timestamp before final-lock outcome event: {Path(name).name}")
    audit.require(lock_path.stat().st_mtime <= last_lock_join + 0.001, f"{year}: latest lock write timestamp before its outcome event")


def within_year_window_rules(frames, year):
    comparisons, rules = [], []
    for model in MODELS:
        environments = {w: numerical_metrics(frames[(model, w)])[1].set_index("Env").sort_index() for w in HORIZONS}
        zero, full = environments[0].rmse.to_numpy(), environments[120].rmse.to_numpy()
        gain = zero.mean() - full.mean()
        relative = gain / zero.mean() if zero.mean() > 0 else np.nan
        supported = bool(gain > 0 and relative >= 0.02 and mean_interval(full - zero)[1] < 0)
        earliest = None
        for horizon in HORIZONS:
            candidate = environments[horizon].rmse.to_numpy()
            d0, d120 = candidate - zero, candidate - full
            lo0, hi0 = mean_interval(d0)
            lo120, hi120 = mean_interval(d120)
            retention = (zero.mean() - candidate.mean()) / gain if gain > 0 else np.nan
            enough = bool(supported and horizon > 0 and retention >= 0.9)
            if enough and earliest is None:
                earliest = horizon
            comparisons.append({"Year": year, "model": model, "window": horizon,
                                "macro_rmse": candidate.mean(), "gain_retention": retention,
                                "full_window_relative_gain": relative, "vs_W0_delta_rmse": d0.mean(),
                                "vs_W0_ci95_low": lo0, "vs_W0_ci95_high": hi0,
                                "vs_W120_delta_rmse": d120.mean(), "vs_W120_ci95_low": lo120,
                                "vs_W120_ci95_high": hi120, "sufficient_by_operational_rule": enough})
        rules.append({"Year": year, "model": model, "earliest_sufficient_window": np.nan if earliest is None else earliest,
                      "W120_gain_supported": supported, "W120_relative_gain": relative})
    return comparisons, rules


def validate(root=ROOT, output=None):
    root = Path(root).resolve()
    output = Path(output) if output else root / "validation_checks.json"
    audit, failure = Audit(), None
    try:
        readiness(audit, root)
        pool = pd.concat([pd.read_csv(PREP / f"samples/{split}.csv") for split in ("train", "validation", "test")], ignore_index=True)
        audit.require(pool.sample_id.is_unique, "Prepared sample IDs globally unique")
        historical = pd.concat([pd.read_csv(PREP / f"targets/{split}.csv") for split in ("train", "validation")], ignore_index=True)
        original_labels = pd.read_csv(PREP / "holdout/test_targets.csv")
        labels_frame = pd.concat([historical, original_labels], ignore_index=True)
        audit.require(labels_frame.sample_id.is_unique, "Prepared historical and original holdout targets unique")
        labels = labels_frame.set_index("sample_id").Yield_Mg_ha
        audit.require(np.isfinite(labels.to_numpy()).all() and (labels > 0).all(), "Finite positive eligible targets")
        genotype = pd.read_csv(PREP / "dimensions/genotypes.csv.gz").set_index("Hybrid")
        audit.require(genotype.index.is_unique, "Genotype source keys unique")
        collected = {key: [] for key in ("coverage", "yearly_metrics", "environment_metrics", "yearly_weather_comparisons",
                     "yearly_subgroup_metrics", "environment_weather_deltas", "subgroup_environment_weather_deltas",
                     "validation_selections", "yearly_sufficient_windows")}
        completion_hashes = {}
        for year in YEARS:
            print(f"Independent rolling audit: {year}", flush=True)
            frozen = development_selection(audit, root, year, pool, genotype, labels)
            run, samples = input_chain(audit, root, year, "final", pool, genotype)
            feature_dir, out = directories(root, year, "final")
            _, development_out = directories(root, year, "development")
            frozen_path = development_out / "frozen_selection.json"
            completion_path = out / "final_evaluation_manifest.json"
            completed = audit.artifacts(completion_path, PROJECT if year == 2024 else root)
            required_artifacts = [out / n for n in ("run_manifest.json", "summary_metrics.csv", "environment_metrics.csv",
                "subgroup_metrics.csv", "weather_increment_comparisons.csv", "earliest_sufficient_windows.json", "prediction_lock_before_holdout.json")]
            required_artifacts += [out / "predictions" / (f"{m}.csv" if m in BASELINES else f"{m}_W{w}.csv") for m, w in ALL_PAIRS]
            required_artifacts += [out / "predictions" / f"{m}_W{w}_evaluated.csv" for m, w in ALL_PAIRS]
            if year != 2024:
                required_artifacts += [out / "units" / f"{m}_W{w}.json" for m, w in ALL_PAIRS]
                audit.require(completed["test_year"] == year and completed["stage"] == "final", f"{year}: final completion fold identity")
            else:
                audit.require(completed["holdout_sha256"] == audit.hash(PREP / "holdout/test_targets.csv"), "Original observed targets bound to immutable 2024 completion")
            audit.bound_paths(completed, required_artifacts, PROJECT if year == 2024 else root, f"{year} final completion")
            completion_hashes[str(completion_path)] = audit.hash(completion_path)
            audit.require(run["frozen_selection_sha256"] == audit.hash(frozen_path), f"{year}: final run uses frozen development choice")
            audit.require(completed.get("run_manifest_sha256", audit.hash(out / "run_manifest.json")) == audit.hash(out / "run_manifest.json"), f"{year}: completion binds run")
            audit.require(completed["recommended_by_validation"] == frozen["recommended_by_validation"], f"{year}: final recommendation unchanged")
            records = verify_units(audit, root, year, "final", run)
            prediction_ordering(audit, root, year, out, feature_dir, frozen_path, completed)
            frames, summary_rows, environment_rows, local_subgroups = {}, [], [], []
            for model, horizon in sorted(ALL_PAIRS):
                context = f"{year} final {model} W{horizon}"
                filename = f"{model}.csv" if model in BASELINES else f"{model}_W{horizon}.csv"
                raw = read_predictions(audit, out / "predictions" / filename, samples["test"], labels, context)
                frame = read_predictions(audit, out / "predictions" / f"{model}_W{horizon}_evaluated.csv", samples["test"], labels, context + " evaluated", True)
                audit.close(frame.prediction.to_numpy(), raw.loc[raw.y_true.notna()].prediction.to_numpy(), context + ": scored/locked predictions identical")
                if model in BASELINES:
                    audit.close(raw.prediction.to_numpy(), baseline(model, samples["train"], labels, samples["test"]), context + ": train-only baseline")
                else:
                    unit = records[(model, horizon)]
                    audit.require(unit["frozen_selection_sha256"] == audit.hash(frozen_path), context + ": unit freeze hash")
                    if year != 2024:
                        config = next(c for c in frozen["configs"] if (c["model"], c["window"]) == (model, horizon))
                        audit.require(unit["selected"]["candidate"] == config["candidate"] and unit["selected"]["params"] == config["params"], context + ": exact frozen refit parameters")
                summary, envs = numerical_metrics(frame, True)
                summary_rows.append({"Year": year, "model": model, "window": horizon, "n_prediction_inputs": len(raw),
                                     "n_without_observed_target": len(raw) - len(frame), **summary})
                environment_rows.append(envs.assign(Year=year, model=model, window=horizon))
                frames[(model, horizon)] = frame
                for kind, column in (("hybrid", "Hybrid"), ("site_code", "Experiment_Code")):
                    seen = frame[column].isin(set(samples["train"][column]))
                    audit.require(int(seen.sum()) + int((~seen).sum()) == len(frame), context + f": {kind} support partitions rows")
                    for flag, label in ((True, "seen"), (False, "unseen")):
                        subset = frame.loc[seen == flag]
                        statistics, _ = numerical_metrics(subset)
                        row = {"Year": year, "model": model, "window": horizon, "group_type": kind, "group": label, **statistics}
                        collected["yearly_subgroup_metrics"].append(row)
                        if len(subset):
                            local_subgroups.append({**row, "group_type": "site" if kind == "site_code" else kind})
            canonical = frames[("CatBoost", 0)]
            for frame in frames.values():
                audit.require(frame[IDENTITY].equals(canonical[IDENTITY]), f"{year}: all 17 evaluated configurations share sample order/metadata")
                audit.close(frame.y_true.to_numpy(), canonical.y_true.to_numpy(), f"{year}: all 17 configurations share observed values")
            audit.require((len(canonical), canonical.Env.nunique()) == EXPECTED_COVERAGE[year], f"{year}: prescribed fixed-cohort sample/environment counts")
            audit.require(completed["n_scored"] == len(canonical) and completed["n_prediction_inputs"] == len(samples["test"])
                          and completed["preset_models_scored"] == 17, f"{year}: completion sample/configuration accounting")
            expected_summary = pd.DataFrame(summary_rows)
            actual_summary = pd.read_csv(out / "summary_metrics.csv").assign(Year=year)
            audit.table(actual_summary, expected_summary, ["Year", "model", "window"], f"{year}: all 17 final numerical summaries")
            expected_environments = pd.concat(environment_rows, ignore_index=True)
            audit.table(pd.read_csv(out / "environment_metrics.csv").assign(Year=year), expected_environments,
                        ["Year", "model", "window", "Env"], f"{year}: final environment arithmetic")
            audit.table(pd.read_csv(out / "subgroup_metrics.csv").assign(Year=year), pd.DataFrame(local_subgroups),
                        ["Year", "model", "window", "group_type", "group"], f"{year}: seen/unseen metrics and support")
            collected["yearly_metrics"].extend(summary_rows)
            collected["environment_metrics"].append(expected_environments)
            collected["coverage"].append({"Year": year, "n_scored": len(canonical), "n_environments": canonical.Env.nunique(),
                "n_site_codes": canonical.Experiment_Code.nunique(), "n_hybrids": canonical.Hybrid.nunique(),
                "n_training_samples": len(samples["train"]), "training_last_year": year - 1,
                "evaluation_role": "original_frozen_2024" if year == 2024 else "supplementary_retrospective"})
            collected["validation_selections"].append({"Year": year, **{k: frozen["recommended_by_validation"][k]
                for k in ("model", "window", "candidate", "validation_macro_rmse")}})
            comparisons, rules = within_year_window_rules(frames, year)
            audit.table(pd.read_csv(out / "weather_increment_comparisons.csv").assign(Year=year), pd.DataFrame(comparisons),
                        ["Year", "model", "window"], f"{year}: independent paired intervals and window rule")
            decisions = read_json(out / "earliest_sufficient_windows.json")
            audit.require(decisions["recommended_by_2023_validation"] == frozen["recommended_by_validation"] if year == 2024
                          else decisions["recommended_by_validation"] == frozen["recommended_by_validation"], f"{year}: no test-based replacement of recommendation")
            audit.require(decisions["test_result_selection_permitted"] is False, f"{year}: test result selection disabled")
            for rule in rules:
                stored = decisions["models"][rule["model"]]
                compare_record(audit, stored, {k: rule[k] for k in ("earliest_sufficient_window", "W120_gain_supported", "W120_relative_gain")}, f"{year} {rule['model']}: sufficient-window decision")
            collected["yearly_weather_comparisons"].extend(comparisons)
            collected["yearly_sufficient_windows"].extend(rules)
            for model in MODELS:
                reference = frames[(model, 0)]
                for horizon in HORIZONS[1:]:
                    candidate = frames[(model, horizon)]
                    collected["environment_weather_deltas"].append(paired_rows(reference, candidate, model, horizon))
                    for kind, column in (("hybrid", "Hybrid"), ("site_code", "Experiment_Code")):
                        seen = reference[column].isin(set(samples["train"][column]))
                        for flag, label in ((True, "seen"), (False, "unseen")):
                            mask = seen == flag
                            if mask.any():
                                collected["subgroup_environment_weather_deltas"].append(paired_rows(reference.loc[mask], candidate.loc[mask], model, horizon).assign(group_type=kind, group=label))
        expected = {name: pd.concat(values, ignore_index=True) if name in {"environment_metrics", "environment_weather_deltas", "subgroup_environment_weather_deltas"}
                    else pd.DataFrame(values) for name, values in collected.items()}
        differences, subgroup = expected["environment_weather_deltas"], expected["subgroup_environment_weather_deltas"]
        expected["yearly_stability"] = pd.DataFrame([{"Year": y, "model": m, "window": w, **annual_stability(g)}
            for (y, m, w), g in differences.groupby(["Year", "model", "window"])])
        expected["cross_year_stability"] = pd.DataFrame([{"model": m, "window": w, **cross_year_stability(g)}
            for (m, w), g in differences.groupby(["model", "window"])])
        annual_groups = pd.DataFrame([{"Year": y, "model": m, "window": w, "group_type": k, "group": s, **annual_stability(g)}
            for (y, m, w, k, s), g in subgroup.groupby(["Year", "model", "window", "group_type", "group"])])
        index = pd.MultiIndex.from_product([YEARS, MODELS, HORIZONS[1:], ("hybrid", "site_code"), ("seen", "unseen")],
                                          names=["Year", "model", "window", "group_type", "group"])
        annual_groups = annual_groups.set_index(index.names).reindex(index).reset_index()
        for name in ("n_samples", "n_environments", "n_site_codes"):
            annual_groups[name] = annual_groups[name].fillna(0).astype(int)
        annual_groups["interval_status"] = annual_groups.interval_status.fillna("empty_subgroup")
        expected["yearly_subgroup_stability"] = annual_groups
        expected["cross_year_subgroup_stability"] = complete_cross_subgroups(pd.DataFrame([
            {"model": m, "window": w, "group_type": k, "group": s, **cross_year_stability(g)}
            for (m, w, k, s), g in subgroup.groupby(["model", "window", "group_type", "group"])]))
        audit.require(len(expected["yearly_subgroup_stability"]) == 240 and len(expected["cross_year_subgroup_stability"]) == 48,
                      "Complete prespecified annual and cross-year subgroup grids, including empty groups")
        audit.require(len(expected["yearly_metrics"]) == 85, "Exactly 85 year/configuration summary rows")
        audit.require(int(expected["coverage"].n_scored.sum()) == 57673 and int(expected["coverage"].n_environments.sum()) == 129,
                      "Exactly 57,673 scored combinations and 129 environment-years per configuration")
        totals = expected["yearly_metrics"].groupby(["model", "window"])[["n", "n_environments"]].sum()
        audit.require(len(totals) == 17 and (totals.n == 57673).all() and (totals.n_environments == 129).all(), "All 17 configurations have identical five-year support")
        manifest_path = root / "analysis/analysis_manifest.json"
        manifest = audit.artifacts(manifest_path, root)
        sources = audit.hash_map(manifest["source_hashes"], PROJECT, "Analysis source")
        audit.require(all(sources.get(path) == sha for path, sha in completion_hashes.items()), "Analysis source chain binds all five final completions")
        audit.require(sources.get(str(root / "PROTOCOL.md")) == audit.hash(root / "PROTOCOL.md"), "Analysis binds unchanged rolling protocol")
        required_sources = [PROJECT / "modeling" / name for name in ("analyze_rolling_stability.py", "metrics.py", "run_experiments.py")]
        for year in YEARS:
            feature_dir, _ = directories(root, year, "final")
            _, development_out = directories(root, year, "development")
            required_sources += [feature_dir / "train_samples.csv", development_out / "frozen_selection.json"]
        for path in required_sources:
            audit.require(sources.get(str(path.resolve())) == audit.hash(path), f"Analysis binds authoritative membership/selection/code: {path}")
        audit.require(manifest["years"] == list(YEARS) and manifest["n_scored_observations"] == 57673
                      and manifest["n_environment_years"] == 129 and manifest["bootstrap_repeats"] == REPEATS,
                      "Analysis manifest counts, all-year inclusion and bootstrap budget")
        keys = {
            "coverage": ["Year"], "yearly_metrics": ["Year", "model", "window"],
            "environment_metrics": ["Year", "model", "window", "Env"],
            "yearly_weather_comparisons": ["Year", "model", "window"],
            "yearly_subgroup_metrics": ["Year", "model", "window", "group_type", "group"],
            "environment_weather_deltas": ["Year", "model", "window", "Env"],
            "subgroup_environment_weather_deltas": ["Year", "model", "window", "group_type", "group", "Env"],
            "yearly_stability": ["Year", "model", "window"], "cross_year_stability": ["model", "window"],
            "yearly_subgroup_stability": ["Year", "model", "window", "group_type", "group"],
            "cross_year_subgroup_stability": ["model", "window", "group_type", "group"],
            "validation_selections": ["Year"], "yearly_sufficient_windows": ["Year", "model"],
        }
        audit.require(set(manifest["table_rows"]) == set(expected), "All 13 prescribed stability tables present")
        for name, table in expected.items():
            audit.require(manifest["table_rows"][name] == len(table), f"Analysis {name}: row count")
            audit.require(f"analysis/{name}.csv" in manifest["artifacts"], f"Analysis {name}: output fingerprint bound")
            audit.table(pd.read_csv(root / "analysis" / f"{name}.csv"), table, keys[name], f"Analysis {name}")
        pipeline = root / "pipeline_completion.json"
        if pipeline.exists():
            record = read_json(pipeline)
            audit.require(record["pipeline_manifest_sha256"] == audit.hash(root / "pipeline_manifest.json")
                          and record["analysis_manifest_sha256"] == audit.hash(manifest_path), "Pipeline completion binds analysis and execution identity")
            audit.hash_map(read_json(root / "pipeline_manifest.json")["input_hashes"], PROJECT, "Pipeline original/code/protocol preservation")
            audit.require(record["final_report_created"] is False, "Pipeline did not create an unauthorized final report")
        audit.require(digest(PROJECT / "modeling/results/final/final_evaluation_manifest.json") == ORIGINAL_2024_SHA256,
                      "Original 2024 fingerprint unchanged at end of audit")
        for path, sha in audit.read_receipts.items():
            audit.require(digest(path) == sha, f"Audited manifest/unit did not change during review: {path}")
        for path, expected_stat in audit.hash_stats.items():
            info = Path(path).stat()
            audit.require((info.st_size, info.st_mtime_ns) == expected_stat, f"Hashed file size/mtime unchanged during review: {path}")
        audit.notes.extend([
            "Historical backtests are supplementary retrospective evidence; outcomes were already available in the original project.",
            "Seen/unseen site_code means Experiment_Code membership, not verified physical-location or independent spatial transfer.",
            "Independent intervals preserve the declared seed and sampling design. Five years, overlapping training sets and site proxies limit interpretation.",
            "Hash/log/timestamp order supports computational provenance, not secure proof that nobody previously accessed labels.",
            "No model was fitted or predictions regenerated by this audit; serialized preprocessing statistics were inspected only.",
        ])
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
    result = {"status": "FAIL" if failure else "PASS", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": digest(Path(__file__)), "original_2024_manifest_sha256": ORIGINAL_2024_SHA256,
              "years": list(YEARS), "checks_passed": sum(c["passed"] for c in audit.checks),
              "checks": audit.checks, "notes": audit.notes, "failure": failure,
              "arithmetic_source": "Independent NumPy residual arithmetic, tie-group ranking, explicit environment bootstrap and matrix-form year/site cluster bootstrap; no production scorer imports."}
    if failure is None:
        result.update(audited_manifest_path=str(root / "analysis/analysis_manifest.json"),
                      audited_manifest_sha256=digest(root / "analysis/analysis_manifest.json"),
                      n_year_configuration_rows=85, n_observations_per_configuration=57673,
                      n_environment_years_per_configuration=129)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"{result['status']}: {result['checks_passed']} checks; {output}", flush=True)
    return result


class SyntheticTests(unittest.TestCase):
    def test_numerical_metrics_and_ties(self):
        frame = pd.DataFrame({"Env": ["A", "A", "B", "B"], "y_true": [0., 2., 10., 10.], "prediction": [1., 1., 8., 12.]})
        summary, envs = numerical_metrics(frame)
        self.assertEqual(summary["macro_rmse"], 1.5)
        self.assertAlmostEqual(summary["pooled_rmse"], math.sqrt(2.5))
        self.assertEqual(summary["macro_mae"], 1.5)
        self.assertEqual(summary["macro_bias"], 0.)
        self.assertTrue(np.isnan(summary["macro_pearson"]))
        self.assertEqual(top20([1, 2, 3, 4, 5], [8, 8, 8, 8, 8]), 0.2)

    def test_year_equal_not_environment_weighted(self):
        frame = pd.DataFrame({"Year": [2020, 2021, 2021, 2021], "Env": ["a", "b", "c", "d"],
                              "Experiment_Code": ["A", "A", "B", "C"], "delta_rmse": [3., -1., -1., -1.],
                              "rmse_W0": [4., 4., 4., 4.], "rmse_weather": [7., 3., 3., 3.]})
        result = cross_year_stability(frame)
        self.assertEqual(result["year_equal_delta_rmse"], 1.)
        self.assertEqual(result["environment_year_equal_delta_rmse"], 0.)
        self.assertEqual(result["years_improved"], 1)
        self.assertEqual(result["years_worsened"], 1)
        self.assertEqual(result["bootstrap_status"], "complete")

    def test_annual_support_and_single_environment_interval(self):
        frame = pd.DataFrame({"Year": [2020], "Env": ["a"], "Experiment_Code": ["proxy"],
                              "n_W0": [20], "delta_rmse": [-1.], "rmse_W0": [4.], "rmse_weather": [3.]})
        result = annual_stability(frame)
        self.assertEqual(result["n_samples"], 20)
        self.assertEqual(result["improved_fraction"], 1.)
        self.assertTrue(np.isnan(result["environment_ci95_low"]))
        self.assertEqual(result["interval_status"], "not_informative_single_environment")
        self.assertEqual(cross_year_stability(frame)["bootstrap_status"], "insufficient_years")

    def test_zero_denominators_are_missing_not_infinite(self):
        frame = pd.DataFrame({"Year": [2020, 2021], "Env": ["a", "b"], "Experiment_Code": ["proxy", "proxy"],
                              "n_W0": [20, 20], "delta_rmse": [1., 2.], "rmse_W0": [0., 0.], "rmse_weather": [1., 2.]})
        self.assertTrue(np.isnan(annual_stability(frame)["relative_gain_percent"]))
        cross = cross_year_stability(frame)
        self.assertTrue(np.isnan(cross["relative_gain_of_year_equal_rmse_percent"]))
        self.assertTrue(np.isnan(cross["mean_annual_relative_gain_percent"]))

    def test_cross_subgroups_preserve_empty_cells_and_support(self):
        frame = pd.DataFrame({"Year": [2020], "Env": ["a"], "Experiment_Code": ["proxy"],
                              "n_W0": [20], "delta_rmse": [-1.], "rmse_W0": [4.], "rmse_weather": [3.]})
        row = {"model": "Ridge", "window": 30, "group_type": "hybrid", "group": "seen", **cross_year_stability(frame)}
        full = complete_cross_subgroups(pd.DataFrame([row]))
        self.assertEqual(len(full), 48)
        populated = full.loc[full.n_years > 0].iloc[0]
        self.assertEqual(populated.evaluated_years, "2020")
        self.assertEqual(populated.missing_prespecified_years, "2021;2022;2023;2024")
        empty = full.loc[full.n_years == 0]
        self.assertEqual(len(empty), 47)
        self.assertTrue((empty.n_environment_years == 0).all())
        self.assertTrue(empty.year_equal_delta_rmse.isna().all())
        self.assertTrue((empty.evaluated_years == "").all())
        self.assertTrue((empty.bootstrap_status == "empty_subgroup").all())
        self.assertTrue((empty.missing_prespecified_years == "2020;2021;2022;2023;2024").all())

    def test_required_artifact_cannot_be_silently_omitted(self):
        with tempfile.TemporaryDirectory(prefix="rolling_audit_binding_") as directory:
            base = Path(directory)
            with self.assertRaises(AuditError):
                Audit().bound_paths({"artifacts": {"other.csv": "hash"}}, [base / "required.csv"], base, "synthetic")
            Audit().bound_paths({"artifacts": {"required.csv": "hash"}}, [base / "required.csv"], base, "synthetic")

    def test_five_year_constant_effect_has_exact_bootstrap_bounds(self):
        frame = pd.DataFrame({"Year": np.repeat(YEARS, 2), "Env": [f"env{i}" for i in range(10)],
                              "Experiment_Code": ["A", "B"] * 5, "delta_rmse": [-0.5] * 10,
                              "rmse_W0": [2.] * 10, "rmse_weather": [1.5] * 10})
        cross = cross_year_stability(frame)
        self.assertEqual(cross["n_years"], 5)
        self.assertEqual(cross["n_environment_years"], 10)
        self.assertEqual(cross["years_improved"], 5)
        self.assertEqual(cross["year_equal_delta_rmse"], -0.5)
        self.assertEqual(cross["relative_gain_of_year_equal_rmse_percent"], 25.)
        for name in ("year_block_ci95_low", "year_block_ci95_high", "crossed_year_site_ci95_low", "crossed_year_site_ci95_high"):
            self.assertEqual(cross[name], -0.5)

    def test_completion_gate_before_source_reads(self):
        with tempfile.TemporaryDirectory(prefix="rolling_audit_gate_") as directory:
            audit = Audit()
            with self.assertRaises(AuditError):
                readiness(audit, Path(directory))
            self.assertEqual(len(audit.checks), 1)
            self.assertFalse(audit.hash_cache)

    def test_repeated_identical_prediction_lock_and_join_are_valid(self):
        for year, lock_name, join_name in (
                (2020, "all_predictions_locked", "historical_test_targets_joined"),
                (2024, "all_predictions_persisted_before_holdout", "holdout_labels_opened_for_preset_evaluation")):
            with self.subTest(year=year):
                events = [{"event": lock_name, "utc": "2026-10-18T01:00:01+00:00", "count": 17, "prediction_lock_sha256": "same"},
                          {"event": join_name, "utc": "2026-10-18T01:00:02+00:00"},
                          {"event": lock_name, "utc": "2026-10-18T01:00:03+00:00", "count": 17, "prediction_lock_sha256": "same"},
                          {"event": join_name, "utc": "2026-10-18T01:00:04+00:00"}]
                first, last_join = prediction_events(Audit(), year, events, "same",
                    "2026-10-18T01:00:00+00:00", "2026-10-18T01:00:05+00:00")
                self.assertEqual(first, 0)
                self.assertEqual(last_join, timestamp(events[-1]["utc"]))

    def test_outcome_join_before_prediction_lock_is_invalid(self):
        events = [{"event": "historical_test_targets_joined", "utc": "2026-10-18T01:00:01+00:00"},
                  {"event": "all_predictions_locked", "utc": "2026-10-18T01:00:02+00:00", "count": 17, "prediction_lock_sha256": "same"},
                  {"event": "historical_test_targets_joined", "utc": "2026-10-18T01:00:03+00:00"}]
        with self.assertRaises(AuditError):
            prediction_events(Audit(), 2020, events, "same", "2026-10-18T01:00:00+00:00", "2026-10-18T01:00:05+00:00")

    def test_score_retry_cannot_change_lock_or_bypass_completion_time(self):
        events = [{"event": "all_predictions_locked", "utc": "2026-10-18T01:00:01+00:00", "count": 17, "prediction_lock_sha256": "same"},
                  {"event": "historical_test_targets_joined", "utc": "2026-10-18T01:00:02+00:00"},
                  {"event": "all_predictions_locked", "utc": "2026-10-18T01:00:03+00:00", "count": 17, "prediction_lock_sha256": "changed"},
                  {"event": "historical_test_targets_joined", "utc": "2026-10-18T01:00:04+00:00"}]
        with self.assertRaises(AuditError):
            prediction_events(Audit(), 2020, events, "same", "2026-10-18T01:00:00+00:00", "2026-10-18T01:00:05+00:00")
        events[2]["prediction_lock_sha256"] = "same"
        with self.assertRaises(AuditError):
            prediction_events(Audit(), 2020, events, "same", "2026-10-18T01:00:00+00:00", "2026-10-18T01:00:03+00:00")

    def test_table_rejects_wrong_arithmetic(self):
        with self.assertRaises(AuditError):
            Audit().table(pd.DataFrame({"key": [1], "value": [2.]}), pd.DataFrame({"key": [1], "value": [3.]}), ["key"], "synthetic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true", help="Only run synthetic arithmetic/gate tests; do not inspect research results")
    args = parser.parse_args()
    if args.self_test:
        outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests))
        raise SystemExit(0 if outcome.wasSuccessful() else 1)
    raise SystemExit(0 if validate(args.root, args.output)["status"] == "PASS" else 1)
