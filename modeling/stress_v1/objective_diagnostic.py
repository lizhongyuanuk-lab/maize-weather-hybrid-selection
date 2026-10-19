"""Post-hoc audit: absolute yield error versus within-environment differentiation.

Uses only already saved held-out predictions and observed targets. This is an
exploratory reframing, not another model fit or a new blinded evaluation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
from run_rolling_experiments import targets_for_ids
from metrics import top20_recall

OUT = HERE / "objective_analysis"
YEARS = range(2020, 2025)
MODELS = ("W0", "W120", "stress")


def load_scored(year: int) -> pd.DataFrame:
    frozen = (PROJECT / "modeling/results/final/predictions" if year == 2024 else
              PROJECT / f"modeling/rolling_v1/results/{year}/final/predictions")
    paths = {"W0": frozen / "CatBoost_W0.csv",
             "W120": frozen / "CatBoost_W120.csv",
             "stress": HERE / f"results/{year}/final/predictions.csv"}
    tables = {key: pd.read_csv(path) for key, path in paths.items()}
    if any(not table.sample_id.is_unique for table in tables.values()):
        raise ValueError("Duplicate prediction sample ID")
    cohorts = [set(table.sample_id) for table in tables.values()]
    if not all(ids == cohorts[0] for ids in cohorts[1:]):
        raise ValueError("Prediction cohorts differ")
    frame = tables["W0"][["sample_id", "Env", "Experiment_Code", "Year", "prediction"]].rename(
        columns={"prediction": "W0"})
    for role in ("W120", "stress"):
        frame = frame.merge(tables[role][["sample_id", "prediction"]].rename(
            columns={"prediction": role}), on="sample_id", validate="one_to_one")
    if len(frame) != len(tables["W0"]) or not (frame.Year == year).all():
        raise ValueError("Lost rows or mixed test years")
    if year == 2024:
        labels = pd.read_csv(PROJECT / "preparation/outputs/holdout/test_targets.csv",
                             usecols=["sample_id", "Yield_Mg_ha"])
        frame = frame.merge(labels, on="sample_id", validate="one_to_one")
    else:
        frame["Yield_Mg_ha"] = targets_for_ids(frame.sample_id, PROJECT / "preparation/outputs")
    if (frame.empty or frame.sample_id.duplicated().any() or
            not np.isfinite(frame[["Yield_Mg_ha", *MODELS]].to_numpy(float)).all()):
        raise ValueError("Invalid scored frame")
    return frame


def environment_components(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for env, group in frame.groupby("Env", sort=True):
        if group.Experiment_Code.nunique() != 1:
            raise ValueError("Multiple site codes in one environment")
        y = group.Yield_Mg_ha.to_numpy(float)
        row = {"Year": int(group.Year.iloc[0]), "Env": env,
               "Experiment_Code": group.Experiment_Code.iloc[0], "n": len(group)}
        for role in MODELS:
            pred = group[role].to_numpy(float)
            bias2 = float((pred-y).mean() ** 2)
            centered = float(np.mean(((pred-pred.mean())-(y-y.mean())) ** 2))
            mse = float(np.mean((pred-y) ** 2))
            if not math.isclose(mse, bias2+centered, abs_tol=1e-10):
                raise ValueError("MSE decomposition failed")
            row.update({f"{role}_bias2": bias2,
                        f"{role}_within_mse": centered, f"{role}_mse": mse,
                        f"{role}_top20_recall": top20_recall(y, pred)})
        rows.append(row)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, n_scored = [], 0
    for year in YEARS:
        frame = load_scored(year)
        n_scored += len(frame)
        rows.extend(environment_components(frame))
    env = pd.DataFrame(rows)
    if len(env) != 129 or env.duplicated(["Year", "Env"]).any() or n_scored != 57673:
        raise ValueError("Unexpected scored cohort")
    env.to_csv(OUT / "environment_components.csv", index=False)
    annual = env.groupby("Year", sort=True).mean(numeric_only=True)
    annual.to_csv(OUT / "annual_components.csv")
    frozen_metrics = pd.read_csv(PROJECT / "modeling/rolling_v1/analysis/yearly_metrics.csv")
    for year in YEARS:
        for role, window in (("W0", 0), ("W120", 120)):
            observed = annual.loc[year, f"{role}_top20_recall"]
            reported = frozen_metrics.loc[(frozen_metrics.Year == year) &
                                          (frozen_metrics.model == "CatBoost") &
                                          (frozen_metrics.window == window), "macro_top20_recall"]
            if len(reported) != 1 or not math.isclose(observed, reported.iloc[0], abs_tol=1e-12):
                raise ValueError("Top-20% recalculation disagrees with frozen report")
    rng = np.random.default_rng(20261019)
    contrasts = {}
    for base, candidate in (("W0", "W120"), ("W120", "stress")):
        record = {}
        for metric in ("bias2", "within_mse", "mse", "top20_recall"):
            differences = (annual[f"{candidate}_{metric}"] - annual[f"{base}_{metric}"]).to_numpy(float)
            draws = differences[rng.integers(0, len(YEARS), (10000, len(YEARS)))].mean(axis=1)
            record[metric] = {"candidate_minus_base": float(differences.mean()),
                              "annual_differences": list(map(float, differences)),
                              "years_favoring_candidate": int(((differences > 0) if metric == "top20_recall" else
                                                               (differences < 0)).sum()),
                              "year_block_ci95": list(map(float, np.quantile(draws, [.025, .975])))}
        # Site codes are approximate location identifiers, so this is a
        # sensitivity analysis, not an independence-adjusted confidence claim.
        codes = sorted(env.Experiment_Code.unique())
        code_idx = {code: i for i, code in enumerate(codes)}
        groups = []
        for _, part in env.groupby("Year", sort=True):
            values = (part[f"{candidate}_top20_recall"] - part[f"{base}_top20_recall"]).to_numpy(float)
            groups.append((values, np.array([code_idx[x] for x in part.Experiment_Code])))
        crossed = []
        while len(crossed) < 10000:
            year_weights = rng.multinomial(len(YEARS), np.full(len(YEARS), 1/len(YEARS)))
            site_weights = rng.multinomial(len(codes), np.full(len(codes), 1/len(codes)))
            means, valid = [], True
            for j, (values, indices) in enumerate(groups):
                w = site_weights[indices]
                if year_weights[j] and not w.sum():
                    valid = False
                    break
                means.append(float(np.dot(w, values)/w.sum()) if w.sum() else 0.0)
            if valid:
                crossed.append(float(np.dot(year_weights, means)/year_weights.sum()))
        record["top20_recall"]["crossed_year_site_ci95"] = list(map(float, np.quantile(crossed, [.025, .975])))
        record["bias_component_share_of_point_mse_reduction"] = (
            record["bias2"]["candidate_minus_base"] / record["mse"]["candidate_minus_base"])
        contrasts[f"{candidate}_vs_{base}"] = record
    summary = {"status": "exploratory_post_hoc_diagnostic", "n_scored_rows": n_scored,
               "n_environment_years": len(env), "years": list(YEARS),
               "identity": "MSE = squared environment mean error + centered within-environment MSE",
               "aggregation": "equal environments within year, then equal years",
               "not_primary_metric": "Primary experiment uses mean RMSE, not decomposable MSE",
               "contrasts": contrasts}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
