"""Post-hoc additive versus explicit PC x stress Ridge comparison."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run as stress
from metrics import evaluate_predictions
from run_experiments import digest, estimator, write_json

HERE = Path(__file__).resolve().parent
YEARS = stress.YEARS
ALPHAS = (100.0, 10000.0)


def add_cross_terms(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    means = arrays["train"][:, [-3, -1]].mean(axis=0, dtype=np.float64)
    standard_deviations = arrays["train"][:, [-3, -1]].std(axis=0, dtype=np.float64)
    scales = np.where(standard_deviations > 0, standard_deviations, 1.0)
    out = {}
    for split, base in arrays.items():
        pcs = base[:, :100].astype(np.float32, copy=False)
        stress_z = ((base[:, [-3, -1]] - means) / scales).astype(np.float32)
        cross = np.concatenate([pcs * stress_z[:, [0]], pcs * stress_z[:, [1]]], axis=1)
        out[split] = np.column_stack([base, cross]).astype(np.float32)
        if not np.isfinite(out[split]).all():
            raise ValueError("Nonfinite genetic-stress interactions")
    return out, {"stress_train_mean": means.tolist(), "stress_train_sd": standard_deviations.tolist(),
                 "cross_terms": "PC001-PC100 x vpd_proxy_mean_kpa and heat_excess35_cday"}


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray,
                alpha: float) -> tuple[object, np.ndarray, float]:
    model = estimator("Ridge", {"alpha": alpha})
    started = time.monotonic()
    with threadpool_limits(limits=4):
        model.fit(x_train, y_train)
        pred = model.predict(x_eval)
    return model, np.asarray(pred, float), time.monotonic() - started


def choose(year: int, bridge: pd.DataFrame, weather: pd.DataFrame) -> dict:
    output = HERE / "gxe_results" / str(year) / "development"
    output.mkdir(parents=True, exist_ok=True)
    directory = stress.feature_directory(year, "development")
    signature = {**stress.signature(directory), str((HERE / "gxe.py").resolve()): digest(HERE / "gxe.py"),
                 str((HERE / "GXE_EXPLORATORY.md").resolve()): digest(HERE / "GXE_EXPLORATORY.md")}
    freeze = output / "selection.json"
    if freeze.exists():
        result = json.loads(freeze.read_text())
        if result["input_signature"] != signature:
            raise ValueError("GXE selection input changed")
        print(f"GXE {year} development: verified selection, no refit", flush=True)
        return result
    samples = stress.read_samples(directory, year, "development")
    additive, audit = stress.combined_arrays(directory, samples, year, bridge, weather)
    interaction, cross_audit = add_cross_terms(additive)
    y_train = stress.training_targets(samples["train"])
    y_validation = stress.training_targets(samples["validation"])
    rows = []
    for name, matrix in (("additive", additive), ("gxe", interaction)):
        for alpha in ALPHAS:
            _, pred, seconds = fit_predict(matrix["train"], y_train, matrix["validation"], alpha)
            score = pd.DataFrame({"Env": samples["validation"].Env, "y_true": y_validation,
                                  "prediction": pred})
            metrics, _ = evaluate_predictions(score)
            rows.append({"model": name, "alpha": alpha,
                         "validation_macro_rmse": metrics["macro_rmse"],
                         "validation_macro_top20_recall": metrics["macro_top20_recall"],
                         "fit_seconds": seconds})
            print(f"GXE {year} development {name} alpha={alpha:g}: RMSE {metrics['macro_rmse']:.4f}", flush=True)
    selected = {name: min((r for r in rows if r["model"] == name),
                          key=lambda r: r["validation_macro_rmse"])["alpha"]
                for name in ("additive", "gxe")}
    result = {"year": year, "input_signature": signature, "selected_alpha": selected,
              "candidate_metrics": rows, "feature_audit": audit, "cross_audit": cross_audit,
              "post_hoc_exploratory": True}
    write_json(result, freeze)
    return result


def final(year: int, bridge: pd.DataFrame, weather: pd.DataFrame, selected: dict) -> dict:
    output = HERE / "gxe_results" / str(year) / "final"
    output.mkdir(parents=True, exist_ok=True)
    directory = stress.feature_directory(year, "final")
    selection_path = HERE / "gxe_results" / str(year) / "development/selection.json"
    signature = {**stress.signature(directory, selection_path),
                 str((HERE / "gxe.py").resolve()): digest(HERE / "gxe.py"),
                 str((HERE / "GXE_EXPLORATORY.md").resolve()): digest(HERE / "GXE_EXPLORATORY.md")}
    samples = stress.read_samples(directory, year, "final")
    lock = output / "prediction_lock.json"
    if lock.exists():
        record = json.loads(lock.read_text())
        if record["input_signature"] != signature or any(
                record["prediction_hashes"][name] != digest(output / f"{name}_predictions.csv")
                for name in ("additive", "gxe")):
            raise ValueError("GXE prediction lock mismatch")
        print(f"GXE {year} final: verified locked predictions, no refit", flush=True)
    else:
        if any((output / f"{name}_predictions.csv").exists() for name in ("additive", "gxe")):
            raise ValueError("Unsealed GXE prediction exists")
        additive, audit = stress.combined_arrays(directory, samples, year, bridge, weather)
        interaction, cross_audit = add_cross_terms(additive)
        y_train = stress.training_targets(samples["train"])
        hashes, times = {}, {}
        for name, matrix in (("additive", additive), ("gxe", interaction)):
            _, pred, seconds = fit_predict(matrix["train"], y_train, matrix["test"],
                                           selected["selected_alpha"][name])
            frame = samples["test"][["sample_id", "Env", "Hybrid", "Year", "Experiment_Code"]].copy()
            frame["prediction"] = pred
            if not np.isfinite(pred).all():
                raise ValueError("Nonfinite GXE prediction")
            path = output / f"{name}_predictions.csv"
            frame.to_csv(path, index=False)
            hashes[name], times[name] = digest(path), seconds
        record = {"year": year, "input_signature": signature,
                  "prediction_hashes": hashes, "fit_seconds": times,
                  "feature_audit": audit, "cross_audit": cross_audit,
                  "test_targets_opened_before_lock": False,
                  "post_hoc_exploratory": True}
        write_json(record, lock)
        print(f"GXE {year} final: both prediction files locked before scoring", flush=True)
    frames = {name: pd.read_csv(output / f"{name}_predictions.csv")
              for name in ("additive", "gxe")}
    ids = samples["test"].sample_id
    if any(not f.sample_id.is_unique or set(f.sample_id) != set(ids) for f in frames.values()):
        raise ValueError("GXE comparison sample cohort mismatch")
    scored = samples["test"][["sample_id", "Env", "Year", "Experiment_Code"]].copy()
    for name, frame in frames.items():
        scored = scored.merge(frame[["sample_id", "prediction"]], on="sample_id",
                              validate="one_to_one").rename(columns={"prediction": f"prediction_{name}"})
    targets = stress.target_frame(year, ids)
    scored = scored.merge(targets, on="sample_id", how="inner", validate="one_to_one")
    envs, row = {}, {"year": year, "n_scored_rows": len(scored)}
    for name in ("additive", "gxe"):
        frame = scored[["Env", "Yield_Mg_ha", f"prediction_{name}"]].rename(
            columns={"Yield_Mg_ha": "y_true", f"prediction_{name}": "prediction"})
        metrics, env = evaluate_predictions(frame)
        row.update({f"{name}_{k}": v for k, v in metrics.items()})
        envs[name] = env
    paired = envs["additive"].merge(envs["gxe"], on="Env", validate="one_to_one",
                                     suffixes=("_additive", "_gxe"))
    paired["Year"] = year
    paired["Experiment_Code"] = paired.Env.map(scored.groupby("Env").Experiment_Code.first())
    if not (paired.n_additive == paired.n_gxe).all():
        raise ValueError("GXE comparison denominators differ")
    for metric in ("rmse", "top20_recall", "spearman"):
        paired[f"delta_{metric}"] = paired[f"{metric}_gxe"] - paired[f"{metric}_additive"]
        row[f"delta_{metric}"] = float(paired[f"delta_{metric}"].mean())
    paired.to_csv(output / "paired_environments.csv", index=False)
    write_json(row, output / "annual_metrics.json")
    print(f"GXE {year} final: RMSE delta {row['delta_rmse']:+.4f}; top20 delta {row['delta_top20_recall']:+.4f}", flush=True)
    return row


def summarize() -> dict:
    years = pd.DataFrame([json.loads((HERE / "gxe_results" / str(year) / "final/annual_metrics.json").read_text())
                          for year in YEARS])
    paired = pd.concat([pd.read_csv(HERE / "gxe_results" / str(year) / "final/paired_environments.csv")
                        for year in YEARS], ignore_index=True)
    if len(paired) != 129 or paired.duplicated(["Year", "Env"]).any():
        raise ValueError("GXE expected 129 unique environment-years")
    result = {"years": list(YEARS), "n_environment_years": len(paired),
              "post_hoc_exploratory": True, "not_new_blind_confirmation": True}
    for metric in ("rmse", "top20_recall"):
        d = years[f"delta_{metric}"].to_numpy(float)
        rng = np.random.default_rng(20261018 + (0 if metric == "rmse" else 1))
        draws = d[rng.integers(0, len(d), (2000, len(d)))].mean(axis=1)
        result[metric] = {"additive_year_equal": float(years[f"additive_macro_{metric}"].mean()),
                          "gxe_year_equal": float(years[f"gxe_macro_{metric}"].mean()),
                          "delta_year_equal": float(d.mean()),
                          "years_favoring_gxe": int((d < 0).sum() if metric == "rmse" else (d > 0).sum()),
                          "year_block_ci95": list(map(float, np.quantile(draws, [.025, .975])))}
    years.to_csv(HERE / "gxe_results/yearly_metrics.csv", index=False)
    paired.to_csv(HERE / "gxe_results/paired_environments.csv", index=False)
    write_json(result, HERE / "gxe_results/summary.json")
    return result


def main() -> None:
    weather, bridge = stress.load_weather_and_bridge()
    for year in YEARS:
        selected = choose(year, bridge, weather)
        final(year, bridge, weather, selected)
    print(json.dumps(summarize(), indent=2), flush=True)


if __name__ == "__main__":
    main()
