"""Held-out, environment-balanced SHAP diagnostics for stress_v1.

These plots are diagnostic because the candidate did not demonstrate a stable
predictive gain. They must not be presented as causal/agronomic discoveries.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, EFstrType, Pool

import run as stress
from features import NAMES
from run_experiments import digest, write_json

HERE = Path(__file__).resolve().parent
OUT = HERE / "explain"
SEED = 20261019
NAVY = "#16324f"
BLUE = "#2673a6"
ORANGE = "#da7c38"
GREY = "#65717d"


def scored_sample_ids(year: int, test: pd.DataFrame) -> set[str]:
    if year < 2024:
        return set(test.sample_id)
    labels = pd.read_csv(stress.PREP / "holdout/test_targets.csv", usecols=["sample_id"])
    return set(labels.sample_id)


def choose_balanced(test: pd.DataFrame, year: int, n_per_env: int) -> np.ndarray:
    valid = scored_sample_ids(year, test)
    positions = np.flatnonzero(test.sample_id.isin(valid).to_numpy())
    table = test.iloc[positions].copy()
    table["position"] = positions
    rng = np.random.default_rng(SEED + year)
    chosen = []
    for _, group in table.groupby("Env", sort=True):
        chosen.extend(rng.choice(group.position.to_numpy(), size=min(n_per_env, len(group)),
                                 replace=False).tolist())
    return np.array(sorted(chosen), dtype=int)


def feature_names(directory: Path) -> list[str]:
    names = json.loads((directory / "W120_columns.json").read_text())["columns"] + NAMES
    if names[:100] != [f"PC{i:03d}" for i in range(1, 101)]:
        raise ValueError("Genetic feature order changed")
    return names


def explain_year(year: int, weather: pd.DataFrame, bridge: pd.DataFrame,
                 n_per_env: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = stress.feature_directory(year, "final")
    samples = stress.read_samples(directory, year, "final")
    x, audit = stress.combined_arrays(directory, samples, year, bridge, weather)
    names = feature_names(directory)
    test = samples["test"]
    positions = choose_balanced(test, year, n_per_env)
    model_path = HERE / "results" / str(year) / "final/candidate_model.cbm"
    lock = json.loads((model_path.parent / "prediction_lock.json").read_text())
    if lock["model_sha256"] != digest(model_path):
        raise ValueError("Candidate model differs from locked model")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    selected_x = x["test"][positions]
    shap = np.asarray(model.get_feature_importance(Pool(selected_x), type=EFstrType.ShapValues,
                                                   thread_count=4), float)
    pred = model.predict(selected_x)
    if shap.shape != (len(positions), len(names) + 1) or np.max(np.abs(shap.sum(axis=1) - pred)) > 1e-6:
        raise ValueError("SHAP dimensions or additivity failed")
    target = test.iloc[positions].reset_index(drop=True)
    per_row = target[["sample_id", "Env", "Hybrid", "Year", "Experiment_Code"]].copy()
    per_row["prediction"] = pred
    for name in NAMES + ["hot35_days", "humidity_mean"]:
        j = names.index(name)
        per_row[name] = selected_x[:, j]
        per_row[f"shap_{name}"] = shap[:, j]
    per_row["base_value"] = shap[:, -1]
    per_row.to_csv(OUT / f"{year}_sampled_shap.csv", index=False)
    np.savez_compressed(OUT / f"{year}_full_shap.npz", sample_ids=target.sample_id.to_numpy(str),
                        feature_names=np.array(names), values=shap.astype(np.float32),
                        features=selected_x.astype(np.float32))
    importance = []
    for env, group in per_row.groupby("Env", sort=True):
        idx = group.index.to_numpy()
        importance.append(pd.DataFrame({"Year": year, "Env": env, "feature": names,
                                        "mean_abs_shap_mgha": np.abs(shap[idx, :-1]).mean(axis=0)}))
    importance = pd.concat(importance, ignore_index=True)
    print(f"SHAP {year}: {len(per_row)} held-out hybrids, {per_row.Env.nunique()} environments", flush=True)
    return per_row, importance


def publication_style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": NAVY, "text.color": NAVY,
                         "xtick.color": GREY, "ytick.color": GREY,
                         "axes.edgecolor": "#b8c5cf", "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})


def plot_rank(importance: pd.DataFrame) -> None:
    year_equal = importance.groupby(["Year", "feature"], sort=True).mean_abs_shap_mgha.mean().reset_index()
    # A site indicator absent from one fold's design matrix has zero contribution
    # in that year; do not average it over fewer years than other predictors.
    grid = pd.MultiIndex.from_product([sorted(importance.Year.unique()),
                                      sorted(importance.feature.unique())], names=["Year", "feature"])
    year_equal = year_equal.set_index(["Year", "feature"]).reindex(grid, fill_value=0).reset_index()
    ranked = year_equal.groupby("feature").mean_abs_shap_mgha.mean().sort_values(ascending=False).head(18)
    fig, ax = plt.subplots(figsize=(9.1, 6.2))
    ordered = ranked.iloc[::-1]
    colors = [ORANGE if x in NAMES else BLUE for x in ordered.index]
    ax.barh(ordered.index, ordered.values, color=colors, height=0.68)
    ax.set_xlabel("Mean absolute SHAP contribution (Mg/ha) | equal-year, equal-environment")
    ax.set_title("Which inputs the augmented model used most", loc="left", fontsize=15, pad=16)
    ax.text(0, 1.02, "Held-out sampled hybrids; orange = added stress representation", transform=ax.transAxes,
            fontsize=9, color=GREY)
    ax.grid(axis="x", alpha=0.16)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_style_feature_rank.png", dpi=190)
    plt.close(fig)
    year_equal.to_csv(OUT / "year_equal_feature_importance.csv", index=False)
    ranked.rename("year_equal_mean_abs_shap_mgha").to_csv(OUT / "top_feature_importance.csv")


def plot_dependence(rows: pd.DataFrame) -> None:
    publication_style()
    panels = [("vpd_proxy_mean_kpa", "Mean daily VPD proxy (kPa)"),
              ("vpd_proxy_p90_kpa", "90th percentile VPD proxy (kPa)"),
              ("heat_excess35_cday", "Heat intensity above 35 C (C-days)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for ax, (name, label) in zip(axes, panels):
        for year, part in rows.groupby("Year", sort=True):
            ax.scatter(part[name], part[f"shap_{name}"], s=14, alpha=0.56,
                       color=BLUE if year < 2024 else ORANGE, linewidths=0,
                       label="2020-23" if year == 2020 else ("2024" if year == 2024 else None))
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Feature SHAP value (Mg/ha)")
        ax.set_title(name.replace("_", " "), loc="left", fontsize=10.5)
        ax.grid(alpha=0.13)
    axes[-1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Fig. 6-style dependence: model contribution, not causal response", x=0.01,
                 ha="left", fontsize=14)
    fig.text(0.01, 0.01, "5 random scored hybrids per environment; 129 environment-years; full-season retrospective weather.",
             fontsize=8.5, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(OUT / "fig6_style_dependence.png", dpi=190)
    plt.close(fig)


def interactions_2024(weather: pd.DataFrame, bridge: pd.DataFrame,
                      importance: pd.DataFrame, n_per_env: int = 2) -> dict:
    year = 2024
    directory = stress.feature_directory(year, "final")
    samples = stress.read_samples(directory, year, "final")
    x, _ = stress.combined_arrays(directory, samples, year, bridge, weather)
    names = feature_names(directory)
    test = samples["test"]
    positions = choose_balanced(test, year, n_per_env)
    model = CatBoostRegressor()
    model.load_model(str(HERE / "results/2024/final/candidate_model.cbm"))
    selected = x["test"][positions]
    interactions = np.asarray(model.get_feature_importance(
        Pool(selected), type=EFstrType.ShapInteractionValues, thread_count=4), float)
    if interactions.shape != (len(positions), len(names)+1, len(names)+1):
        raise ValueError("Unexpected SHAP interaction dimensions")
    # CatBoost repeats the expected value in every cell of the final row/column.
    # Only the feature-by-feature submatrix is additive to the feature SHAP sums.
    shap = np.asarray(model.get_feature_importance(
        Pool(selected), type=EFstrType.ShapValues, thread_count=4), float)
    pair_main = interactions[:, :-1, :-1]
    if (np.max(np.abs(pair_main.sum(axis=2) - shap[:, :-1])) > 5e-4 or
            np.max(np.abs(pair_main.sum(axis=(1, 2)) + shap[:, -1] - model.predict(selected))) > 5e-4):
        raise ValueError("SHAP interaction additivity failed")
    pc_importance = importance.loc[(importance.Year == 2024) & importance.feature.str.startswith("PC")]
    top_pc = pc_importance.groupby("feature").mean_abs_shap_mgha.mean().idxmax()
    pairs = [("heat_excess35_cday", "vpd_proxy_mean_kpa"),
             ("heat_excess35_cday", top_pc),
             ("vpd_proxy_mean_kpa", "humidity_mean")]
    figure_rows = test.iloc[positions][["sample_id", "Env", "Hybrid", "Year"]].reset_index(drop=True)
    for a, b in pairs:
        i, j = names.index(a), names.index(b)
        figure_rows[a] = selected[:, i]
        figure_rows[b] = selected[:, j]
        figure_rows[f"interaction_{a}__{b}"] = interactions[:, i, j] + interactions[:, j, i]
    figure_rows.to_csv(OUT / "2024_pairwise_shap_interactions.csv", index=False)
    publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    for ax, (a, b) in zip(axes, pairs):
        values = figure_rows[f"interaction_{a}__{b}"]
        sc = ax.scatter(figure_rows[a], values, c=figure_rows[b], cmap="viridis", s=30,
                        alpha=0.83, edgecolors="white", linewidths=0.2)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xlabel(a.replace("_", " "))
        ax.set_ylabel("Pairwise SHAP interaction (Mg/ha)")
        ax.set_title(f"{a} x {b}".replace("_", " "), loc="left", fontsize=9.2)
        cb = fig.colorbar(sc, ax=ax, fraction=0.044, pad=0.02)
        cb.set_label(b.replace("_", " "), fontsize=8)
        ax.grid(alpha=0.13)
    fig.suptitle("Fig. 7-style interactions: diagnostic only", x=0.01, ha="left", fontsize=14)
    fig.text(0.01, 0.01, f"2024: {len(positions)} random scored hybrids from {figure_rows.Env.nunique()} environments."
             " No stable predictive gain; no causal claim.", fontsize=8.5, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(OUT / "fig7_style_interactions.png", dpi=190)
    plt.close(fig)
    return {"year": year, "n_hybrids": len(positions), "n_environments": figure_rows.Env.nunique(),
            "top_pc_chosen_after_shap_review": top_pc,
            "pairs": [{"a": a, "b": b, "mean_abs_pairwise_shap_mgha": float(
                np.abs(figure_rows[f"interaction_{a}__{b}"]).mean())} for a, b in pairs]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    weather, bridge = stress.load_weather_and_bridge()
    parts, importance_parts = [], []
    for year in stress.YEARS:
        rows, importance = explain_year(year, weather, bridge)
        parts.append(rows)
        importance_parts.append(importance)
    rows, importance = pd.concat(parts, ignore_index=True), pd.concat(importance_parts, ignore_index=True)
    importance.to_csv(OUT / "environment_feature_importance.csv", index=False)
    plot_rank(importance)
    plot_dependence(rows)
    pair_summary = interactions_2024(weather, bridge, importance)
    write_json({"sampled_rows": len(rows), "sampled_environment_years": rows[["Year", "Env"]].drop_duplicates().shape[0],
                "five_per_environment": True, "pairwise_2024": pair_summary,
                "interpretation": "post-hoc_model_diagnostic_not_validated_agronomic_or_causal_insight",
                "paper_analogy": "Figure 4 importance; Figure 6 dependence; Figure 7 pairwise model interaction"},
               OUT / "summary.json")
    print(json.dumps(pair_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
