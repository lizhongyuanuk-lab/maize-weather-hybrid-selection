"""Build/execute a reviewable results notebook without fitting any model.

Run after final experiment artifacts exist:
    PYTHONPATH=.runtime python modeling/build_research_notebook.py --execute
Without --execute, creates an explicitly unexecuted notebook scaffold only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path

import nbformat

PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports"


def markdown(text):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbformat.v4.new_code_cell(textwrap.dedent(text).strip())


def notebook():
    cells = [markdown("""
        # How Early Can Maize Yield Be Predicted?
        ### A weather-window study with multi-model, out-of-year evaluation

        **Research-results companion · G2F 2014–2024 · independent application project**

        This notebook audits saved predictions and produces the final figures. It does **not**
        retrain, tune, or select a model using 2024 outcomes. Re-execution recomputes the same
        locked evaluation and presentation; it is not another model-selection experiment.
        Missing final artifacts stop execution instead of producing illustrative scores.
        Units: **Mg ha⁻¹**, numerically equivalent to metric tonnes per hectare.
    """), markdown("""
        ### Setup and input integrity

        Run from this repository or its `reports/` directory. Dependencies are the local
        `.runtime` environment or the packages recorded in `modeling/results/final/run_manifest.json`.
        Plotting code and exact-value tables are intentionally retained for review.
    """), code(r'''
        import hashlib, json, sys
        from pathlib import Path
        ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                    if (p / "preparation/outputs/manifest.json").exists())
        if (ROOT / ".runtime").exists():
            sys.path.insert(0, str(ROOT / ".runtime"))
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        get_ipython().run_line_magic("matplotlib", "inline")
        from matplotlib.ticker import PercentFormatter
        from IPython.display import display, Markdown
        DEV, FINAL = ROOT / "modeling/results/development", ROOT / "modeling/results/final"
        REPORTS, FIGURES = ROOT / "reports", ROOT / "reports/figures"
        FIGURES.mkdir(parents=True, exist_ok=True)
        def sha256(path):
            result = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for block in iter(lambda: stream.read(1048576), b""):
                    result.update(block)
            return result.hexdigest()
        completed = json.loads((FINAL / "final_evaluation_manifest.json").read_text())
        for relative, expected in completed["artifacts"].items():
            assert sha256(ROOT / relative) == expected, f"Modified result: {relative}"
        frozen = json.loads((DEV / "frozen_selection.json").read_text())
        assert sha256(DEV / "frozen_selection.json") == completed["frozen_selection_sha256"]
        validation = pd.read_csv(DEV / "selected_metrics.csv")
        final = pd.read_csv(FINAL / "summary_metrics.csv")
        environments = pd.read_csv(FINAL / "environment_metrics.csv")
        deltas = pd.read_csv(FINAL / "weather_increment_comparisons.csv")
        subgroups = pd.read_csv(FINAL / "subgroup_metrics.csv")
        sufficiency = json.loads((FINAL / "earliest_sufficient_windows.json").read_text())
        cohorts = pd.read_csv(ROOT / "preparation/outputs/audit/split_summary.csv")
        selected = frozen["recommended_by_validation"]
        selected_family, selected_window = selected["model"], selected["window"]
        MODELS, WINDOWS = ["Ridge", "ExtraTrees", "CatBoost"], [0, 30, 60, 90, 120]
        expected = {(m, w) for m in MODELS for w in WINDOWS} | {("GlobalMean", 0), ("HistoricalSiteMean", 0)}
        assert set(zip(final.model, final.window)) == expected and len(final) == 17
        for choice in frozen["configs"]:
            observed = validation.loc[(validation.model == choice["model"]) & (validation.window == choice["window"])]
            assert len(observed) == 1
            assert np.isclose(observed.iloc[0].macro_rmse, choice["validation_macro_rmse"], atol=1e-10)
        COLORS = {"Ridge": "#2467A3", "ExtraTrees": "#C46C20", "CatBoost": "#667D33"}
        MARKERS = {"Ridge": "o", "ExtraTrees": "s", "CatBoost": "^"}
        plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 180, "font.size": 11,
                             "axes.titlesize": 13, "axes.labelsize": 11,
                             "axes.spines.top": False, "axes.spines.right": False,
                             "axes.grid": True, "grid.alpha": .18, "axes.axisbelow": True,
                             "font.family": "DejaVu Sans", "figure.facecolor": "white"})
        pd.options.display.float_format = "{:,.3f}".format
        def save_figure(fig, filename):
            fig.savefig(FIGURES / f"{filename}.png", bbox_inches="tight", facecolor="white")
            fig.savefig(FIGURES / f"{filename}.svg", bbox_inches="tight", facecolor="white")
            display(fig)
            plt.close(fig)
        def metric_row(table, model, window):
            rows = table.loc[(table.model == model) & (table.window == window)]
            assert len(rows) == 1
            return rows.iloc[0]
        print(f"Verified {len(completed['artifacts'])} final-artifact hashes; loaded {len(final)} preset evaluations.")
    '''), markdown("""
        ## tl;dr

        The statement below is generated from completed experimental outputs, not filled in
        before training. The highlighted configuration was chosen using **2023 only**.
    """), code(r'''
        chosen = metric_row(final, selected_family, selected_window)
        zero = metric_row(final, selected_family, 0)
        full = metric_row(final, selected_family, 120)
        delta_full = deltas.loc[(deltas.model == selected_family) & (deltas.window == 120)].iloc[0]
        pct = 100 * (zero.macro_rmse - full.macro_rmse) / zero.macro_rmse
        relation = "lower" if pct >= 0 else "higher"
        earliest = sufficiency["models"][selected_family]["earliest_sufficient_window"]
        earliest_text = (f"W{earliest} met the prespecified operational sufficiency rule in this test year."
                         if earliest is not None else
                         "No sufficiently early window was declared under the prespecified evidence rule.")
        summary = f"""**Validation-selected configuration: {selected_family}, W{selected_window}.**
        On the held-out 2024 observations, its environment-equal RMSE was **{chosen.macro_rmse:.3f} Mg ha⁻¹**
        (environment-bootstrap 95% interval {chosen.macro_rmse_ci95_low:.3f}–{chosen.macro_rmse_ci95_high:.3f}).
        In the same model family, W120 RMSE was **{abs(pct):.1f}% {relation} than W0**;
        the paired W120−W0 difference was {delta_full.vs_W0_delta_rmse:+.3f} Mg ha⁻¹
        (95% interval {delta_full.vs_W0_ci95_low:+.3f} to {delta_full.vs_W0_ci95_high:+.3f}).
        {earliest_text}

        All {int(chosen.n):,} observed environment–hybrid combinations across
        {int(chosen.n_environments)} environments were evaluated; {int(chosen.n_without_observed_target):,}
        test inputs without observed yield were not scored. These are observational predictive
        comparisons, not causal effects or evidence of operational deployment."""
        summary = "\n".join(line.lstrip() for line in summary.splitlines()).strip()
        display(Markdown(summary))
    '''), markdown("""
        ## Context & Methods

        **Question.** How much does weather available within 30, 60, 90, or 120 days after
        planting improve next-year maize yield prediction over genotype and static inputs?

        **Design.** Mean yield per environment × hybrid; train 2014–2022, select hyperparameters
        on 2023, then refit on 2014–2023 and evaluate 2024. Ridge, Extra Trees, and CatBoost
        each received two prespecified candidates per window. All five windows share one
        eligible cohort and the same genotype/static features. The final phase persisted
        all 15 model predictions plus two baselines before opening the observed target file.

        **Preprocessing.** Training-only genotype imputation and 100-component PCA, training-only
        static preprocessing, and weather clipped to `[planting, planting + window)` before
        imputation and aggregation. Multiple planting dates are represented by original-plot
        weights rather than an invented single date. Ridge adds training-fitted scaling.

        **Primary measure.** Compute RMSE within each environment, then average environments
        equally. Pooled RMSE, MAE, bias, environment-wise Pearson/Spearman, and top-20% recall
        are supplementary. Undefined constant-vector correlations remain missing. Ranking
        boundary ties receive fractional inclusion weights; a constant score is not credited
        with a deterministic hybrid-name ranking.

        **Uncertainty.** 2,000 environment-level paired bootstrap resamples, seed 20261018,
        percentile 95% intervals. Intervals are conditional on fitted predictions, unadjusted
        for multiple comparisons, and do not include model-selection uncertainty.

        ### Key Assumptions

        - The fixed-calendar common cohort defines scope; excluded trials are not represented.
        - Weather data are retrospective. Real-time publication delays have not been validated.
        - A material name not seen during fitting still has genotype information; this is not
          prediction without genetic inputs. New-site evidence may be very limited.
        - “Sufficient” requires W120 improvement ≥2%, a paired interval wholly below zero,
          then ≥90% point-estimate gain retention. This is not a formal noninferiority test.
        - One held-out year and a small number of environments cannot establish a universal
          forecast date or agricultural causality. No new model architecture is claimed.

        Implementation: [features](../modeling/build_features.py),
        [experiment runner](../modeling/run_experiments.py),
        [metrics](../modeling/metrics.py), [locked protocol](../modeling/PROTOCOL_LOCK.md).
    """), markdown("""
        ## Data

        Original release: [G2F competition data, DOI 10.25739/78mn-4394](https://dc.cyverse.org/dataset/genomes_to_fields_2024_maize_genotype_by_environment_prediction_competition).
        Some initial files came from the [public UFEED mirror](https://github.com/imbaterry11/UFEED_Maize_GxE).
        The provenance status below combines the original verification with subsequent checks;
        no unverified mirror is silently treated as an official byte-identical file.
    """), code(r'''
        provenance = json.loads((ROOT / "preparation/source_verification.json").read_text())
        latest_sources = {item["role"]: item for item in provenance["files"]}
        followup_path = ROOT / "preparation/provenance_followup/verification_report.json"
        if followup_path.exists():
            followup = json.loads(followup_path.read_text())
            for item in followup["files"]:
                if item.get("download_complete"):
                    assert sha256(ROOT / item["local_path"]) == item["local_sha256"]
                latest_sources[item["role"]] = item
        source_table = pd.DataFrame([{"input_role": role, "status": item.get("status"),
                                      "full_official_copy_obtained": item.get("download_complete", False),
                                      "semantic_match": item.get("semantic_match")}
                                     for role, item in latest_sources.items()])
        unresolved = source_table.loc[~source_table.full_official_copy_obtained, "input_role"].tolist()
        display(cohorts)
        display(source_table)
        display(Markdown("The known 2024 metadata mismatch was handled by using the official copy, "
                         "preserving its missing coordinates. Official 2024 observed targets were obtained directly. "
                         + (f"**Unresolved full-source checks: {', '.join(unresolved)}.**" if unresolved else
                            "All listed full-source comparisons are complete; the two weather files are byte-identical, "
                            "while traits and genotypes are cell-equivalent under documented representation changes.")))
        display(Markdown("The frozen runner carries a static provenance note written before the follow-up completed. "
                         "That 'pending' note was not refreshed when the selection file was emitted; it is not the current status. "
                         "Current evidence: [verification_report.json](../preparation/provenance_followup/verification_report.json). "
                         "These follow-up checks did not modify Stage 1 inputs or outputs."))
    '''), markdown("""
        ### Independent reconciliation from saved row-level predictions

        The following calculation does not call the experiment's metric functions. It independently
        recomputes primary errors, counts, and the paired bootstrap from saved, already-evaluated
        predictions. It does not re-open the raw holdout target file.
    """), code(r'''
        reconciled, reference_sample_ids = [], None
        for row in final.itertuples(index=False):
            path = FINAL / "predictions" / f"{row.model}_W{row.window}_evaluated.csv"
            frame = pd.read_csv(path)
            assert not frame.sample_id.duplicated().any()
            if reference_sample_ids is None:
                reference_sample_ids = frame.sample_id.tolist()
            assert frame.sample_id.tolist() == reference_sample_ids
            errors = frame.prediction.to_numpy() - frame.y_true.to_numpy()
            by_env = pd.DataFrame({"Env": frame.Env, "squared_error": errors**2, "absolute_error": np.abs(errors)})
            env_rmse = by_env.groupby("Env", sort=True).squared_error.mean().pow(.5)
            macro_rmse = float(env_rmse.mean())
            macro_mae = float(by_env.groupby("Env").absolute_error.mean().mean())
            assert np.isclose(macro_rmse, row.macro_rmse, rtol=1e-10, atol=1e-10)
            assert np.isclose(macro_mae, row.macro_mae, rtol=1e-10, atol=1e-10)
            assert np.isclose(np.sqrt(np.mean(errors**2)), row.pooled_rmse, rtol=1e-10, atol=1e-10)
            assert len(frame) == row.n and len(env_rmse) == row.n_environments
            reconciled.append({"model": row.model, "window": row.window,
                               "macro_rmse_recomputed": macro_rmse, "rows_checked": len(frame),
                               "environments_checked": len(env_rmse), "check": "PASS"})
        for model in MODELS:
            base = environments.loc[(environments.model == model) & (environments.window == 0)].set_index("Env").sort_index()
            for horizon in WINDOWS[1:]:
                candidate = environments.loc[(environments.model == model) & (environments.window == horizon)].set_index("Env").reindex(base.index)
                diff = (candidate.rmse - base.rmse).to_numpy()
                rng = np.random.default_rng(20261018)
                bootstrap_means = diff[rng.integers(0, len(diff), size=(2000, len(diff)))].mean(axis=1)
                limits = np.quantile(bootstrap_means, [.025, .975])
                recorded = deltas.loc[(deltas.model == model) & (deltas.window == horizon)].iloc[0]
                assert np.allclose(limits, [recorded.vs_W0_ci95_low, recorded.vs_W0_ci95_high], atol=1e-10)
                assert np.isclose(diff.mean(), recorded.vs_W0_delta_rmse, atol=1e-10)
        reconciliation = pd.DataFrame(reconciled)
        reconciliation.to_csv(REPORTS / "independent_metric_reconciliation.csv", index=False)
        display(reconciliation)
        print("PASS: 17 prediction sets, primary metrics and counts, and 12 paired bootstrap comparisons.")
    '''), markdown("""
        ## Results

        ### 1. Accuracy as the weather window grows

        Same scales across the panels. Shaded intervals are environment-bootstrap 95% intervals;
        the validation panel is used for selection and is not an independent test estimate.
        Sources: `selected_metrics.csv` and `summary_metrics.csv`.
    """), code(r'''
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
        for axis, table, title in zip(axes, [validation, final], ["2023 · validation / selection", "2024 · held-out test"]):
            for model in MODELS:
                part = table.loc[table.model == model].sort_values("window")
                assert part.window.tolist() == WINDOWS
                axis.plot(part.window, part.macro_rmse, label=model, color=COLORS[model], marker=MARKERS[model], lw=2)
                axis.fill_between(part.window, part.macro_rmse_ci95_low, part.macro_rmse_ci95_high, color=COLORS[model], alpha=.09)
            for baseline, style, color in [("GlobalMean", ":", "#999999"), ("HistoricalSiteMean", "--", "#333333")]:
                value = float(table.loc[table.model == baseline, "macro_rmse"].iloc[0])
                axis.axhline(value, color=color, linestyle=style, lw=1.6, label=baseline)
            axis.set(title=title, xlabel="Weather observed after planting (days)", xticks=WINDOWS)
        axes[0].set_ylabel("Environment-equal RMSE (Mg ha⁻¹) · lower is better")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=10, frameon=False)
        fig.suptitle("Prediction error across all prespecified models and windows", y=.99, fontsize=16)
        fig.tight_layout(rect=[0, .11, 1, .94])
        save_figure(fig, "01_accuracy_vs_weather_window")
        display(final[["model", "window", "macro_rmse", "macro_mae", "pooled_rmse", "macro_spearman", "macro_top20_recall"]])
    '''), markdown("""
        ### 2. Paired weather increment, with uncertainty

        Each estimate compares the **same model family and environments** with W0. Negative
        differences indicate lower error with weather; intervals crossing zero do not support
        a consistently signed effect under this bootstrap. These are predictive differences,
        not causal weather effects. Source: `weather_increment_comparisons.csv`.
    """), code(r'''
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharex=True, sharey=True)
        bound = deltas.loc[deltas.window > 0]
        left = min(float(bound.vs_W0_ci95_low.min()), 0)
        right = max(float(bound.vs_W0_ci95_high.max()), 0)
        margin = max(.05, (right-left)*.12)
        for axis, model in zip(axes, MODELS):
            part = bound.loc[bound.model == model].sort_values("window")
            y = np.arange(len(part))
            axis.hlines(y, part.vs_W0_ci95_low, part.vs_W0_ci95_high, color=COLORS[model], lw=1.8)
            axis.vlines(part.vs_W0_ci95_low, y-.08, y+.08, color=COLORS[model], lw=1.3)
            axis.vlines(part.vs_W0_ci95_high, y-.08, y+.08, color=COLORS[model], lw=1.3)
            axis.scatter(part.vs_W0_delta_rmse, y, marker=MARKERS[model], color=COLORS[model], s=48)
            axis.axvline(0, color="#333333", lw=1)
            axis.set(title=model, yticks=y, yticklabels=[f"W{w}" for w in part.window],
                     xlabel="Δ RMSE vs W0 (Mg ha⁻¹)", xlim=(left-margin, right+margin))
        axes[0].invert_yaxis()
        fig.suptitle("2024 weather increments · paired 95% environment-bootstrap intervals", fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, .92])
        save_figure(fig, "02_paired_weather_increment")
    '''), markdown("""
        ### 3. Environment heterogeneity

        W0 and W120 are paired within each test environment for the **validation-selected
        model family**, not the family that happens to look best on the final year. Each row
        is an environment; marker size does not encode sample count. Source: `environment_metrics.csv`.
    """), code(r'''
        subset = environments.loc[environments.model == selected_family]
        paired = subset.pivot(index="Env", columns="window", values="rmse")[[0, 120]]
        paired["difference"] = paired[120] - paired[0]
        paired = paired.sort_values("difference")
        y = np.arange(len(paired))
        fig, axis = plt.subplots(figsize=(9.5, max(6, len(paired)*.32)))
        for index, (_, row) in enumerate(paired.iterrows()):
            axis.plot([row[0], row[120]], [index, index], color="#AEB5BB", lw=1.5)
        axis.scatter(paired[0], y, facecolors="white", edgecolors="#555555", marker="o", s=48, linewidths=1.4, label="W0")
        axis.scatter(paired[120], y, color=COLORS[selected_family], marker=MARKERS[selected_family], s=48, label="W120")
        axis.set(yticks=y, yticklabels=paired.index, xlabel="Within-environment RMSE (Mg ha⁻¹)",
                 title=f"{selected_family}: W0 → W120 in each 2024 environment")
        axis.invert_yaxis()
        axis.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        save_figure(fig, "03_environment_heterogeneity")
        improved = int((paired.difference < 0).sum())
        display(Markdown(f"W120 reduced RMSE in **{improved}/{len(paired)}** environments for {selected_family}. "
                         "The variation is part of the result; the average is not a universal improvement guarantee."))
    '''), markdown("""
        ### 4. Absolute yield error versus material ranking

        Both panels show change relative to the **same model's W0**. Positive values are
        favorable. A reduction in absolute yield error does not by itself imply better
        within-environment selection of high-yielding hybrids. Source: `summary_metrics.csv`.
    """), code(r'''
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
        for model in MODELS:
            part = final.loc[final.model == model].sort_values("window")
            baseline = part.loc[part.window == 0].iloc[0]
            absolute_gain = (baseline.macro_rmse-part.macro_rmse)/baseline.macro_rmse
            ranking_gain = part.macro_spearman-baseline.macro_spearman
            for axis, values in zip(axes, [absolute_gain, ranking_gain]):
                axis.plot(part.window, values, color=COLORS[model], marker=MARKERS[model], lw=2, label=model)
        for axis in axes:
            axis.axhline(0, color="#444444", lw=1)
            axis.set(xticks=WINDOWS, xlabel="Weather observed after planting (days)")
        axes[0].set(title="Absolute-error improvement", ylabel="RMSE reduction relative to W0")
        axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1].set(title="Within-environment ranking improvement", ylabel="Change in macro Spearman vs W0")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .015),
                   ncol=3, frameon=False)
        fig.suptitle("2024: calibration and hybrid-ranking gains are different outcomes", fontsize=15)
        fig.tight_layout(rect=[0, .13, 1, .93])
        save_figure(fig, "04_error_vs_ranking")
    '''), markdown("""
        ### 5. Seen and previously unseen hybrids

        A hybrid is “seen” if its name appeared in the 2014–2023 fitting cohort. Both groups
        have genotype data. These are descriptive subgroup estimates, not a controlled
        experiment: their environment/sample composition may differ. No subgroup interval
        is invented from the aggregate interval. Source: `subgroup_metrics.csv`.
    """), code(r'''
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8), sharex=True, sharey=True)
        material = subgroups.loc[(subgroups.group_type == "hybrid") & subgroups.model.isin(MODELS)]
        group_styles = {"seen": ("#2467A3", "o", "-"), "unseen": ("#A67B12", "s", "--")}
        for axis, model in zip(axes, MODELS):
            for group, (color, marker, style) in group_styles.items():
                part = material.loc[(material.model == model) & (material.group == group)].sort_values("window")
                if len(part):
                    axis.plot(part.window, part.macro_rmse, color=color, marker=marker, linestyle=style, lw=2,
                              label=f"{group.capitalize()} · n={int(part.iloc[0].n):,}")
            axis.set(title=model, xlabel="Weather window (days)", xticks=WINDOWS)
            axis.legend(frameon=False, fontsize=10)
        axes[0].set_ylabel("Environment-equal subgroup RMSE (Mg ha⁻¹)")
        fig.suptitle("2024 generalization by prior hybrid-name exposure", fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, .93])
        save_figure(fig, "05_seen_unseen_hybrids")
        selected_groups = subgroups.loc[(subgroups.model == selected_family) & (subgroups.window == selected_window)]
        display(selected_groups[["group_type", "group", "n", "n_environments", "macro_rmse", "macro_spearman"]])
    '''), markdown("""
        ## Takeaways

        The following English summary is generated only after successful reconciliation and
        plotting. It distinguishes the validation-selected configuration from test-year
        descriptions and retains negative or inconclusive outcomes.
    """), code(r'''
        direction = "decreased" if pct >= 0 else "increased"
        unresolved_text = (f"Full official-versus-mirror verification remains incomplete for {', '.join(unresolved)}. "
                           if unresolved else "Complete official-source checks confirm the weather files are byte-identical and the trait/genotype values are equivalent under documented format changes; the official metadata override remains applicable. ")
        ranking_change = float(full.macro_spearman-zero.macro_spearman)
        brief = f"""# How Early Can Maize Yield Be Predicted?
        ## A multi-model weather-window study using G2F field trials

        **Objective.** Quantify the incremental predictive value of weather accumulated within 30, 60, 90, and 120 days after planting, relative to a genotype-plus-static-information baseline.

        **Design and implementation.** Public multi-environment maize trials were harmonized to environment–hybrid mean yield. The study uses 94,377 training combinations from 2014–2022 and 11,491 validation combinations from 2023. Training-only genotype imputation and 100-component PCA are combined with static inputs and strictly clipped weather-window features. Ridge, Extra Trees, and CatBoost each receive two prespecified configurations per window. Hyperparameters are chosen by 2023 environment-equal RMSE, frozen, and refitted on 2014–2023. All 15 model-window predictions and two simple baselines are saved before final scoring.

        **Observed results.** The validation-selected configuration, **{selected_family} W{selected_window}**, achieved a 2024 environment-equal RMSE of **{chosen.macro_rmse:.3f} Mg ha⁻¹** (95% environment-bootstrap interval {chosen.macro_rmse_ci95_low:.3f}–{chosen.macro_rmse_ci95_high:.3f}), evaluated on {int(chosen.n):,} labeled combinations across {int(chosen.n_environments)} environments. Within that model family, W120 {direction} RMSE by {abs(pct):.1f}% relative to W0. The paired difference was {delta_full.vs_W0_delta_rmse:+.3f} Mg ha⁻¹ (95% interval {delta_full.vs_W0_ci95_low:+.3f} to {delta_full.vs_W0_ci95_high:+.3f}); macro Spearman changed by {ranking_change:+.3f}. {earliest_text}

        **Contribution.** The project provides a reproducible temporal benchmark, leakage-aware feature assembly, paired weather ablations, environment-level uncertainty, and explicit separation of absolute-yield calibration from hybrid-ranking ability. It applies established models rather than claiming a new architecture.

        **Limitations and next steps.** Evidence covers one final year; bootstrap intervals do not include model-selection uncertainty and are not multiplicity-adjusted. Weather inputs are retrospective, real-time availability is unverified, and {int(chosen.n_without_observed_target):,} test inputs lack observed targets. {unresolved_text}Rolling-year evaluation and prospective validation are needed before deployment or general agricultural recommendations.
        """
        brief = "\n".join(line.lstrip() for line in brief.splitlines()).strip()+"\n"
        (REPORTS / "ENGLISH_PROJECT_SUMMARY.md").write_text(brief)
        display(Markdown(brief))
        display(Markdown("### Exact sources and reproduction\n"
            "- `preparation/outputs/manifest.json`: data assembly hashes and contract.\n"
            "- `preparation/provenance_followup/verification_report.json`: current complete official-source comparisons.\n"
            "- `modeling/features/{development,final}/feature_manifest.json`: training-fit boundaries and feature hashes.\n"
            "- `modeling/results/development/frozen_selection.json`: pre-test choices.\n"
            "- `modeling/results/final/prediction_lock_before_holdout.json`: prediction lock before scoring.\n"
            "- `modeling/results/final/final_evaluation_manifest.json`: final evidence hashes.\n"
            "- Run `PYTHONPATH=.runtime python modeling/build_research_notebook.py --execute` from the project root.\n"
            "- All notebook figures are also available as editable-code-derived PNG and SVG files in `reports/figures/`."))
    ''')]
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata.update({"kernelspec": {"display_name": "Maize research (local runtime)", "language": "python", "name": "maize-runtime"},
                        "language_info": {"name": "python", "version": sys.version.split()[0]},
                        "research_scope": "Saved-prediction analysis only; no model fitting or selection on test outcomes"})
    return nb


def build(execute=False):
    REPORTS.mkdir(exist_ok=True)
    nb = notebook()
    nbformat.validate(nb)
    path = REPORTS / "research_results.ipynb"
    if path.exists():
        existing = nbformat.read(path, as_version=4)
        if any(cell.get("execution_count") for cell in existing.cells) and not execute:
            raise ValueError("Refusing to overwrite an executed notebook with an unexecuted scaffold")
    if not execute:
        nbformat.write(nb, path)
        print(f"UNEXECUTED scaffold: {path}; run again with --execute after final results exist.")
        return
    if not (PROJECT/"modeling/results/final/final_evaluation_manifest.json").exists():
        raise FileNotFoundError("Final experiment completion manifest is required; no placeholder results will be generated")
    jupyter = REPORTS/".jupyter"
    kernel = jupyter/"kernels/maize-runtime"
    kernel.mkdir(parents=True, exist_ok=True)
    runtime = jupyter/"runtime"
    runtime.mkdir(exist_ok=True)
    matplotlib_cache = REPORTS/".matplotlib"
    matplotlib_cache.mkdir(exist_ok=True)
    kernel_spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                   "display_name": "Maize research (local runtime)", "language": "python",
                   "env": {"PYTHONPATH": str(PROJECT/".runtime"), "MPLCONFIGDIR": str(matplotlib_cache),
                           "JUPYTER_RUNTIME_DIR": str(runtime), "OPENBLAS_NUM_THREADS": "4"}}
    (kernel/"kernel.json").write_text(json.dumps(kernel_spec, indent=2)+"\n")
    # pip --target stores templates and their inherited assets outside sys.prefix.
    jupyter_paths = [str(jupyter)]
    local_jupyter_share = PROJECT / ".runtime/share/jupyter"
    if local_jupyter_share.is_dir():
        jupyter_paths.append(str(local_jupyter_share))
    jupyter_paths.extend(filter(None, os.environ.get("JUPYTER_PATH", "").split(os.pathsep)))
    os.environ["JUPYTER_PATH"] = os.pathsep.join(jupyter_paths)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    from nbclient import NotebookClient
    client = NotebookClient(nb, timeout=300, kernel_name="maize-runtime", allow_errors=False,
                            resources={"metadata": {"path": str(PROJECT)}})
    try:
        client.execute()
    except Exception:
        nbformat.write(nb, REPORTS/"research_results_FAILED.ipynb")
        raise
    nbformat.validate(nb)
    nbformat.write(nb, path)
    from nbconvert import HTMLExporter
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    html, _ = exporter.from_notebook_node(nb)
    html_path = REPORTS/"research_results.html"
    html_path.write_text(html)
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    figures = list((REPORTS/"figures").glob("*.png"))
    receipt = {"execution": "executed_top_to_bottom_with_nbclient", "code_cells_executed": len(code_cells),
               "all_cells_have_execution_counts": all(c.execution_count is not None for c in code_cells),
               "png_figures": len(figures), "html_preview": str(html_path.relative_to(PROJECT)),
               "visual_inspection": "pending human/agent rendered inspection; execution alone is not visual QA",
               "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "html_sha256": hashlib.sha256(html_path.read_bytes()).hexdigest()}
    (REPORTS/"notebook_execution_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))
    print(f"Executed notebook: {path}\nHTML preview: {html_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    build(args.execute)
