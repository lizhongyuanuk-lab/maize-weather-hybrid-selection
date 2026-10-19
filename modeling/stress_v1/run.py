"""Resumable, post-hoc five-year CatBoost stress-feature comparison.

The existing W120 data, predictions and models are read-only. Candidate test
predictions are saved and SHA-256 locked before any test-year labels are loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
MODELING = HERE.parent
PROJECT = MODELING.parent
sys.path.insert(0, str(MODELING))

from features import build_extra, NAMES
from metrics import bootstrap_mean, evaluate_predictions
from run_experiments import CANDIDATES, arrays_for, digest, estimator, persist_model, write_json
from run_rolling_experiments import targets_for_ids

PREP = PROJECT / "preparation/outputs"
YEARS = (2020, 2021, 2022, 2023, 2024)
SEED = 20261018


def feature_directory(year: int, stage: str) -> Path:
    return (MODELING / "features" / stage if year == 2024
            else MODELING / "rolling_v1/features" / str(year) / stage)


def reference_prediction(year: int) -> Path:
    return (MODELING / "results/final/predictions/CatBoost_W120.csv" if year == 2024
            else MODELING / "rolling_v1/results" / str(year) / "final/predictions/CatBoost_W120.csv")


def read_samples(directory: Path, year: int, stage: str) -> dict[str, pd.DataFrame]:
    samples = {s: pd.read_csv(directory / f"{s}_samples.csv") for s in ("train", "validation", "test")}
    expected = {"train": set(range(2014, year - (1 if stage == "development" else 0))),
                "validation": {year - 1} if stage == "development" else set(), "test": {year}}
    for split, frame in samples.items():
        if not frame.sample_id.is_unique or frame.sample_id.isna().any():
            raise ValueError(f"Bad {split} sample IDs")
        if set(frame.Year) != expected[split]:
            raise ValueError(f"Wrong year partition: {year} {stage} {split}")
        if {"Yield_Mg_ha", "y_true", "prediction"} & set(frame.columns):
            raise ValueError("Target or prediction in model-facing sample metadata")
    if any(set(samples[a].sample_id) & set(samples[b].sample_id)
           for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Overlapping split IDs")
    return samples


def signature(directory: Path, selection: Path | None = None) -> dict[str, str]:
    paths = [HERE / "PROTOCOL.md", HERE / "features.py", HERE / "run.py",
             MODELING / "build_features.py", directory / "feature_manifest.json",
             PREP / "dimensions/weather_daily.csv.gz", PREP / "links/sample_plantings.csv.gz"]
    if selection is not None:
        paths.append(selection)
    return {str(p.resolve()): digest(p) for p in paths}


def load_weather_and_bridge() -> tuple[pd.DataFrame, pd.DataFrame]:
    weather = pd.read_csv(PREP / "dimensions/weather_daily.csv.gz",
                          usecols=["Env", "Date", "T2M", "RH2M", "T2M_MAX", *[
                              n for n in ("T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "WS2M")]])
    bridge = pd.read_csv(PREP / "links/sample_plantings.csv.gz",
                         usecols=["sample_id", "Env", "Planting_Date", "planting_weight"])
    return weather, bridge


def combined_arrays(directory: Path, samples: dict[str, pd.DataFrame], year: int,
                    bridge: pd.DataFrame, weather: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict]:
    base = arrays_for(directory, 120, samples)
    extra, audit = build_extra(samples, year, bridge, weather)
    matrices = {s: np.column_stack([base[s], extra[s]]).astype(np.float32)
                for s in samples}
    for split, x in matrices.items():
        if x.shape != (len(samples[split]), base[split].shape[1] + len(NAMES)) or not np.isfinite(x).all():
            raise ValueError(f"Combined matrix invalid: {year} {split}")
    audit["base_feature_count"] = base["train"].shape[1]
    audit["combined_feature_count"] = matrices["train"].shape[1]
    return matrices, audit


def training_targets(samples: pd.DataFrame) -> np.ndarray:
    return targets_for_ids(samples.sample_id, PREP)


def development(year: int, bridge: pd.DataFrame, weather: pd.DataFrame) -> dict:
    directory = feature_directory(year, "development")
    output = HERE / "results" / str(year) / "development"
    output.mkdir(parents=True, exist_ok=True)
    identity = signature(directory)
    frozen = output / "selection.json"
    if frozen.exists():
        selected = json.loads(frozen.read_text())
        if selected["input_signature"] != identity:
            raise ValueError("Development inputs changed; preserve results and use a new version")
        print(f"{year} development: verified existing selection, no refit", flush=True)
        return selected
    samples = read_samples(directory, year, "development")
    x, audit = combined_arrays(directory, samples, year, bridge, weather)
    y_train, y_val = training_targets(samples["train"]), training_targets(samples["validation"])
    if len(y_train) != len(x["train"]) or len(y_val) != len(x["validation"]):
        raise ValueError("Development target/feature alignment failed")
    rows = []
    for idx, params in enumerate(CANDIDATES["CatBoost"]):
        start = time.monotonic()
        model = estimator("CatBoost", params)
        with threadpool_limits(limits=4):
            model.fit(x["train"], y_train)
            pred = model.predict(x["validation"])
        scored = samples["validation"][["sample_id", "Env"]].copy()
        scored["y_true"], scored["prediction"] = y_val, pred
        metrics, _ = evaluate_predictions(scored)
        rows.append({"candidate": idx, "fit_seconds": time.monotonic() - start,
                     "validation_macro_rmse": metrics["macro_rmse"],
                     "validation_macro_top20_recall": metrics["macro_top20_recall"],
                     "validation_n_environments": metrics["n_environments"]})
        print(f"{year} development candidate {idx}: RMSE {metrics['macro_rmse']:.4f}", flush=True)
    winner = min(rows, key=lambda r: r["validation_macro_rmse"])
    record = {"year": year, "stage": "development", "input_signature": identity,
              "candidate_metrics": rows, "chosen_candidate": winner["candidate"],
              "chosen_params": CANDIDATES["CatBoost"][winner["candidate"]],
              "feature_audit": audit,
              "historical_outcomes_previously_available": True}
    write_json(record, frozen)
    return record


def target_frame(year: int, ids: pd.Series) -> pd.DataFrame:
    if year == 2024:
        frame = pd.read_csv(PREP / "holdout/test_targets.csv",
                            usecols=["sample_id", "Yield_Mg_ha"])
        if not frame.sample_id.is_unique or not set(frame.sample_id).issubset(set(ids)):
            raise ValueError("2024 holdout key mismatch")
        return frame
    y = targets_for_ids(ids, PREP)
    return pd.DataFrame({"sample_id": ids.to_numpy(), "Yield_Mg_ha": y})


def score_final(year: int, samples: pd.DataFrame, candidate_path: Path,
                baseline_path: Path, output: Path) -> dict:
    candidate = pd.read_csv(candidate_path)
    baseline = pd.read_csv(baseline_path)
    ids = samples.sample_id
    if (not candidate.sample_id.is_unique or not baseline.sample_id.is_unique
            or set(candidate.sample_id) != set(ids) or set(baseline.sample_id) != set(ids)):
        raise ValueError("Candidate/reference sample cohorts differ")
    aligned = samples[["sample_id", "Env", "Year", "Experiment_Code"]].merge(
        candidate[["sample_id", "prediction"]], on="sample_id", validate="one_to_one")
    aligned = aligned.merge(baseline[["sample_id", "prediction"]], on="sample_id",
                            validate="one_to_one", suffixes=("_candidate", "_reference"))
    targets = target_frame(year, ids)
    scored = aligned.merge(targets, on="sample_id", how="inner", validate="one_to_one")
    if scored.empty or scored.sample_id.duplicated().any() or scored.Env.isna().any():
        raise ValueError("Empty or invalid scored cohort")
    if not np.isfinite(scored[["Yield_Mg_ha", "prediction_candidate", "prediction_reference"]].to_numpy(float)).all():
        raise ValueError("Nonfinite scored predictions or yields")
    environments = {}
    annual = {"year": year, "n_input_rows": len(samples), "n_scored_rows": len(scored),
              "n_unscored_rows": len(samples) - len(scored)}
    for role in ("reference", "candidate"):
        frame = scored[["Env", "Yield_Mg_ha", f"prediction_{role}"]].rename(
            columns={"Yield_Mg_ha": "y_true", f"prediction_{role}": "prediction"})
        metrics, env = evaluate_predictions(frame)
        annual.update({f"{role}_{k}": v for k, v in metrics.items()})
        environments[role] = env
    paired = environments["reference"].merge(environments["candidate"], on="Env",
                                              validate="one_to_one", suffixes=("_reference", "_candidate"))
    site = scored.groupby("Env").Experiment_Code.nunique()
    if site.max() != 1:
        raise ValueError("Environment has multiple site codes")
    paired["Experiment_Code"] = paired.Env.map(scored.groupby("Env").Experiment_Code.first())
    paired["Year"] = year
    if not (paired.n_reference == paired.n_candidate).all():
        raise ValueError("Reference and candidate denominators differ")
    for metric in ("rmse", "mae", "spearman", "top20_recall"):
        paired[f"delta_{metric}"] = paired[f"{metric}_candidate"] - paired[f"{metric}_reference"]
    for metric in ("rmse", "top20_recall"):
        vals = paired[f"delta_{metric}"].to_numpy(float)
        ci = bootstrap_mean(vals, repeats=2000, seed=SEED)
        annual[f"delta_{metric}"] = float(vals.mean())
        annual[f"delta_{metric}_env_ci_low"] = ci["ci95_low"]
        annual[f"delta_{metric}_env_ci_high"] = ci["ci95_high"]
    annual["relative_rmse_gain_percent"] = 100 * (
        annual["reference_macro_rmse"] - annual["candidate_macro_rmse"]) / annual["reference_macro_rmse"]
    paired.to_csv(output / "paired_environments.csv", index=False)
    write_json(annual, output / "annual_metrics.json")
    return annual


def final(year: int, bridge: pd.DataFrame, weather: pd.DataFrame) -> dict:
    directory = feature_directory(year, "final")
    output = HERE / "results" / str(year) / "final"
    output.mkdir(parents=True, exist_ok=True)
    selection = HERE / "results" / str(year) / "development/selection.json"
    if not selection.exists():
        raise ValueError("Development selection must exist before final refit")
    chosen = json.loads(selection.read_text())
    identity = signature(directory, selection)
    baseline_path = reference_prediction(year)
    baseline_hash = digest(baseline_path)
    samples = read_samples(directory, year, "final")
    predictions = output / "predictions.csv"
    lock = output / "prediction_lock.json"
    if lock.exists():
        record = json.loads(lock.read_text())
        if (record["input_signature"] != identity or record["prediction_sha256"] != digest(predictions)
                or record["reference_prediction_sha256"] != baseline_hash):
            raise ValueError("Locked prediction/input/reference hash mismatch")
        print(f"{year} final: verified locked predictions, no refit", flush=True)
    else:
        if predictions.exists():
            raise ValueError("Unsealed candidate prediction file exists; inspect before resuming")
        x, audit = combined_arrays(directory, samples, year, bridge, weather)
        y_train = training_targets(samples["train"])
        model = estimator("CatBoost", chosen["chosen_params"])
        start = time.monotonic()
        with threadpool_limits(limits=4):
            model.fit(x["train"], y_train)
            pred = model.predict(x["test"])
        model_path = persist_model(model, "CatBoost", output / "candidate_model")
        frame = samples["test"][["sample_id", "Env", "Hybrid", "Year", "Experiment_Code"]].copy()
        frame["prediction"] = pred
        if not np.isfinite(frame.prediction.to_numpy(float)).all():
            raise ValueError("Nonfinite candidate test predictions")
        frame.to_csv(predictions, index=False)
        record = {"year": year, "stage": "final", "input_signature": identity,
                  "reference_prediction_sha256": baseline_hash,
                  "prediction_sha256": digest(predictions), "model_sha256": digest(model_path),
                  "chosen_candidate": chosen["chosen_candidate"],
                  "fit_seconds": time.monotonic() - start, "feature_audit": audit,
                  "test_targets_opened_before_lock": False}
        write_json(record, lock)
        print(f"{year} final: predictions SHA-256 locked before scoring", flush=True)
    annual = score_final(year, samples["test"], predictions, baseline_path, output)
    print(f"{year} final: RMSE delta {annual['delta_rmse']:+.4f}; top20 delta {annual['delta_top20_recall']:+.4f}", flush=True)
    return annual


def summarize() -> dict:
    rows = []
    paired = []
    for year in YEARS:
        output = HERE / "results" / str(year) / "final"
        rows.append(json.loads((output / "annual_metrics.json").read_text()))
        paired.append(pd.read_csv(output / "paired_environments.csv"))
    years = pd.DataFrame(rows).sort_values("year")
    environments = pd.concat(paired, ignore_index=True)
    if environments.duplicated(["Year", "Env"]).any() or len(environments) != 129:
        raise ValueError("Expected 129 distinct scored environment-years")
    result = {"years": list(YEARS), "n_scored_rows": int(years.n_scored_rows.sum()),
              "n_environment_years": len(environments), "n_site_code_proxies": int(environments.Experiment_Code.nunique()),
              "retrospective_post_hoc": True, "not_new_blind_confirmation": True}
    rng = np.random.default_rng(SEED)
    site_codes = sorted(environments.Experiment_Code.unique())
    site_index = {s: i for i, s in enumerate(site_codes)}
    for metric in ("rmse", "top20_recall"):
        annual_deltas = years[f"delta_{metric}"].to_numpy(float)
        point = float(annual_deltas.mean())
        year_draws = annual_deltas[rng.integers(0, len(YEARS), (2000, len(YEARS)))].mean(axis=1)
        groups = [(group[f"delta_{metric}"].to_numpy(float),
                   np.array([site_index[s] for s in group.Experiment_Code]))
                  for _, group in environments.groupby("Year", sort=True)]
        crossed = []
        while len(crossed) < 2000:
            yw = rng.multinomial(len(YEARS), np.full(len(YEARS), 1/len(YEARS)))
            sw = rng.multinomial(len(site_codes), np.full(len(site_codes), 1/len(site_codes)))
            means = []
            valid = True
            for j, (values, codes) in enumerate(groups):
                weights = sw[codes]
                if yw[j] and not weights.sum():
                    valid = False
                    break
                means.append(float(np.dot(weights, values) / weights.sum()) if weights.sum() else 0.0)
            if valid:
                crossed.append(float(np.dot(yw, means) / yw.sum()))
        result[metric] = {"year_equal_reference": float(years[f"reference_macro_{metric}"].mean()),
                          "year_equal_candidate": float(years[f"candidate_macro_{metric}"].mean()),
                          "year_equal_delta": point,
                          "years_favoring_candidate": int((annual_deltas < 0).sum() if metric == "rmse" else (annual_deltas > 0).sum()),
                          "year_block_ci95": list(map(float, np.quantile(year_draws, [.025, .975]))),
                          "crossed_year_site_ci95": list(map(float, np.quantile(crossed, [.025, .975])))}
    result["rmse"]["relative_gain_percent"] = 100 * (
        result["rmse"]["year_equal_reference"] - result["rmse"]["year_equal_candidate"]) / result["rmse"]["year_equal_reference"]
    years.to_csv(HERE / "results/yearly_metrics.csv", index=False)
    environments.to_csv(HERE / "results/paired_environments.csv", index=False)
    write_json(result, HERE / "results/summary.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if not set(args.years).issubset(YEARS):
        raise ValueError("Unsupported year")
    if args.summarize:
        print(json.dumps(summarize(), indent=2))
        return
    weather, bridge = load_weather_and_bridge()
    for year in args.years:
        development(year, bridge, weather)
        final(year, bridge, weather)
    if set(args.years) == set(YEARS):
        print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
