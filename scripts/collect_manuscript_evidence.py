"""Collect immutable manuscript evidence without fitting or changing experiments.

The existing brief's hash-chain guard is reused, but all aggregation checks below
are independently implemented from the frozen environment/annual CSV tables.
No prediction row, target file, feature matrix, or fitted model is loaded here.
Bootstrap endpoints are copied from audited tables, not resampled or re-estimated.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_research_brief import PROJECT, reviewed_evidence, sha


DEFAULT = PROJECT / "reports/manuscript/evidence.json"
MODELS = ("Ridge", "ExtraTrees", "CatBoost")
METRICS = ("rmse", "mae", "bias", "pearson", "spearman", "top20_recall")
YEARS = (2020, 2021, 2022, 2023, 2024)


def collect(output: Path = DEFAULT) -> dict:
    guarded, hashes = reviewed_evidence()
    checks = {}

    def check(condition, category, detail):
        if not bool(condition):
            raise AssertionError(f"{category}: {detail}")
        checks[category] = checks.get(category, 0) + 1

    def close(actual, expected, category, detail):
        check(np.allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True), category, detail)

    def read(relative):
        path = PROJECT / relative
        hashes[relative] = sha(path)
        return pd.read_csv(path) if path.suffix == ".csv" else json.loads(path.read_text())

    def table(experiment, name):
        return read(f"modeling/{experiment}/analysis/{name}.csv")

    def records(frame):
        # pandas handles NumPy values and maps all missing/nonfinite values to null.
        return json.loads(frame.to_json(orient="records", double_precision=15))

    environment = table("segmented_v1", "environment_metrics")
    annual = table("segmented_v1", "yearly_metrics")
    cumulative = table("rolling_v1", "cross_year_stability")
    segmented = table("segmented_v1", "cross_year_comparisons")
    cum_env = table("rolling_v1", "environment_weather_deltas")
    seg_env = table("segmented_v1", "environment_comparisons")
    cum_annual = table("rolling_v1", "yearly_stability")
    seg_annual = table("segmented_v1", "yearly_comparisons")
    subgroup = table("segmented_v1", "yearly_subgroup_metrics")
    cum_subgroup = table("rolling_v1", "cross_year_subgroup_stability")
    seg_subgroup = table("segmented_v1", "cross_year_subgroup_comparisons")
    coverage = table("segmented_v1", "coverage")
    sufficient = table("segmented_v1", "yearly_sufficient_windows")
    selections = table("segmented_v1", "validation_selections")
    primary_assessment = table("segmented_v1", "primary_assessment")
    upgrade_assessment = table("segmented_v1", "model_upgrade_assessment")

    check(len(environment) == 4386 and len(annual) == 170, "source_grain", "Complete frozen source tables")
    keys = ["Year", "representation", "model", "window"]
    check(not annual.duplicated(keys).any(), "source_grain", "Unique annual configuration keys")
    check(not environment.duplicated(keys + ["Env"]).any(), "source_grain", "Unique environment keys")
    for row in annual.itertuples(index=False):
        chosen = environment.loc[
            (environment.Year == row.Year) & (environment.representation == row.representation)
            & (environment.model == row.model) & (environment.window == row.window)
        ]
        context = f"{row.Year}/{row.representation}/{row.model}/W{row.window}"
        check(len(chosen) == row.n_environments and int(chosen.n.sum()) == row.n,
              "annual_support", context)
        for metric in METRICS:
            close(chosen[metric].mean(), getattr(row, "macro_" + metric), "environment_to_annual", context + "/" + metric)
        close(np.sqrt(np.average(chosen.rmse.to_numpy() ** 2, weights=chosen.n)), row.pooled_rmse,
              "pooled_metrics", context + "/rmse")
        for metric in ("mae", "bias"):
            close(np.average(chosen[metric], weights=chosen.n), getattr(row, "pooled_" + metric),
                  "pooled_metrics", context + "/" + metric)
        for metric in ("pearson", "spearman"):
            check(int(chosen[metric].notna().sum()) == getattr(row, metric + "_defined_environments"),
                  "undefined_metric_support", context + "/" + metric)

    for representation in ("cumulative", "segmented"):
        for year in YEARS:
            rows = annual.loc[(annual.Year == year) & (annual.representation == representation)]
            check(len(rows) == 17 and len(rows[["n_prediction_inputs", "n_without_observed_target", "n", "n_environments"]].drop_duplicates()) == 1,
                  "configuration_cohort_identity", f"{year}/{representation}")
    shared = annual.loc[annual.window.isin([0, 30])]
    shared_a = shared.loc[shared.representation == "cumulative"].set_index(["Year", "model", "window"])
    shared_b = shared.loc[shared.representation == "segmented"].set_index(["Year", "model", "window"])
    close(shared_a[["macro_" + m for m in METRICS]], shared_b[["macro_" + m for m in METRICS]],
          "shared_results", "W0/W30 annual metric equality")

    # Reconstruct all reported cross-year point estimates from paired environments.
    for frame, pairs, year_frame, kind in (
        (cumulative, cum_env, cum_annual, "cumulative"),
        (segmented, seg_env, seg_annual, "segmented"),
    ):
        for row in frame.itertuples(index=False):
            mask = (pairs.model == row.model) & (pairs.window == row.window)
            annual_mask = (year_frame.model == row.model) & (year_frame.window == row.window)
            if kind == "segmented":
                mask &= pairs.reference == row.reference
                annual_mask &= year_frame.reference == row.reference
                reference_column, candidate_column = "rmse_reference", "rmse_segmented"
                expected_ref, expected_cand = row.year_equal_rmse_reference, row.year_equal_rmse_segmented
            else:
                reference_column, candidate_column = "rmse_W0", "rmse_weather"
                expected_ref, expected_cand = row.year_equal_rmse_W0, row.year_equal_rmse_weather
            paired = pairs.loc[mask]
            by_year = paired.groupby("Year")[[reference_column, candidate_column, "delta_rmse"]].mean()
            reported_years = year_frame.loc[annual_mask].set_index("Year").sort_index()
            context = f"{kind}/{row.model}/W{row.window}/{getattr(row, 'reference', 'no_weather')}"
            check(len(by_year) == row.n_years == 5 and len(paired) == row.n_environment_years == 129,
                  "comparison_support", context)
            check(paired.Experiment_Code.nunique() == row.n_site_codes == 31,
                  "comparison_support", context + "/site codes")
            close(paired[candidate_column] - paired[reference_column], paired.delta_rmse,
                  "paired_environment_arithmetic", context)
            close(by_year.delta_rmse, reported_years.delta_rmse, "annual_comparison_arithmetic", context)
            close(by_year[reference_column].mean(), expected_ref, "cross_year_arithmetic", context + "/reference")
            close(by_year[candidate_column].mean(), expected_cand, "cross_year_arithmetic", context + "/candidate")
            close(by_year.delta_rmse.mean(), row.year_equal_delta_rmse, "cross_year_arithmetic", context + "/delta")
            close(paired.delta_rmse.mean(), row.environment_year_equal_delta_rmse,
                  "cross_year_arithmetic", context + "/alternative environment weighting")
            close(100 * (expected_ref - expected_cand) / expected_ref,
                  row.relative_gain_of_year_equal_rmse_percent, "cross_year_arithmetic", context + "/relative gain")
            check(int((by_year.delta_rmse < 0).sum()) == row.years_improved
                  and int((by_year.delta_rmse > 0).sum()) == row.years_worsened,
                  "year_direction_counts", context)
            close(by_year.delta_rmse.min(), row.year_min_delta, "cross_year_arithmetic", context + "/min delta")
            close(by_year.delta_rmse.max(), row.year_max_delta, "cross_year_arithmetic", context + "/max delta")

    # Secondary endpoints use exactly the same nested (environment -> year) weighting.
    secondary_columns = ["macro_" + metric for metric in METRICS]
    year_equal = annual.groupby(["representation", "model", "window"], as_index=False)[secondary_columns].mean()
    year_equal = year_equal.rename(columns={c: "year_equal_" + c.removeprefix("macro_") for c in secondary_columns})
    year_equal["n_years"] = 5
    year_equal["n_environment_years"] = 129
    # Defined correlation environments can differ from the 129 RMSE environments.
    defined = annual.groupby(["representation", "model", "window"], as_index=False)[
        ["pearson_defined_environments", "spearman_defined_environments"]].sum()
    year_equal = year_equal.merge(defined, on=["representation", "model", "window"], validate="one_to_one")
    primary = []
    for model in MODELS:
        def one(rows):
            check(len(rows) == 1, "primary_uniqueness", model)
            return records(rows)[0]
        cu = one(cumulative.loc[(cumulative.model == model) & (cumulative.window == 120)])
        sc = one(segmented.loc[(segmented.model == model) & (segmented.window == 120) & (segmented.reference == "cumulative")])
        sw = one(segmented.loc[(segmented.model == model) & (segmented.window == 120) & (segmented.reference == "no_weather")])
        primary.append({
            "model": model,
            "W0_year_equal_rmse": cu["year_equal_rmse_W0"],
            "cumulative_W120_year_equal_rmse": cu["year_equal_rmse_weather"],
            "segmented_W120_year_equal_rmse": sc["year_equal_rmse_segmented"],
            "cumulative_minus_W0": cu,
            "segmented_minus_cumulative": sc,
            "segmented_minus_W0": sw,
        })

    support = subgroup.loc[(subgroup.representation == "cumulative") & (subgroup.model == "Ridge")
                          & (subgroup.window == 0), ["Year", "group_type", "group", "n", "n_environments"]]
    for (year, group_type, group), frame in subgroup.groupby(["Year", "group_type", "group"]):
        check(len(frame[["n", "n_environments"]].drop_duplicates()) == 1, "subgroup_support_identity", f"{year}/{group_type}/{group}")
    for (year, group_type), frame in support.groupby(["Year", "group_type"]):
        expected = int(coverage.loc[coverage.Year == year, "n_scored"].iloc[0])
        check(int(frame.n.sum()) == expected, "subgroup_partition", f"{year}/{group_type}")
    check(sufficient.earliest_sufficient_window.isna().all(), "sufficiency", "No supported earliest window")
    check(not upgrade_assessment.both_reference_screens_pass.any(), "upgrade_screen", "No model passes both references")
    selections["params"] = selections.params.map(ast.literal_eval)

    original_manifest = read("modeling/results/final/run_manifest.json")
    candidates = original_manifest["candidates"]
    for year in YEARS:
        for stage in ("development", "final"):
            path = f"modeling/segmented_v1/results/{year}/{stage}/run_manifest.json"
            current = read(path)
            check(current["candidates"] == candidates, "equal_candidate_budget", path)
    prep = read("preparation/outputs/manifest.json")
    provenance = read("preparation/provenance_followup/verification_report.json")
    cohort_flow = read("preparation/outputs/audit/cohort_flow.csv")
    for relative in ("modeling/PROTOCOL_LOCK.md", "modeling/run_experiments.py", "modeling/metrics.py",
                     "preparation/README.md"):
        hashes[relative] = sha(PROJECT / relative)

    cohort_by_year = annual.loc[(annual.representation == "cumulative") & (annual.model == "Ridge")
                               & (annual.window == 0), ["Year", "n_prediction_inputs", "n_without_observed_target", "n", "n_environments"]]
    sites = sorted(cum_env.Experiment_Code.unique().tolist())
    totals = {
        "scored_environment_hybrid_observations": int(coverage.n_scored.sum()),
        "prediction_inputs": int(cohort_by_year.n_prediction_inputs.sum()),
        "missing_target_exclusions": int(cohort_by_year.n_without_observed_target.sum()),
        "environment_years": int(coverage.n_environments.sum()),
        "distinct_released_site_codes": len(sites),
        "test_years": list(YEARS), "released_site_codes": sites,
        "note": "Counts refer to one configuration, not totals multiplied by model/window/representation. Site codes are proxies, not independently verified locations.",
    }
    audits = {}
    for experiment, evidence in guarded.items():
        audit = evidence["audit"]
        audits[experiment] = {k: audit[k] for k in (
            "status", "checked_at_utc", "checks_passed", "audited_manifest_sha256", "arithmetic_source")}
    seg_audit = guarded["segmented_v1"]["audit"]
    replay = seg_audit["new_final_model_replays"]
    audits["segmented_v1"].update({
        "n_new_candidate_fits": seg_audit["n_new_candidate_fits"],
        "n_new_final_fits": seg_audit["n_new_final_fits"],
        "actual_fit_start_events": seg_audit["actual_fit_start_events"],
        "actual_fit_complete_events": seg_audit["actual_fit_complete_events"],
        "replayed_final_models": len(replay),
        "replayed_prediction_values": sum(r["n_predictions"] for r in replay),
        "maximum_replay_absolute_difference": max(r["max_absolute_difference"] for r in replay),
        "physical_feature_audit_scope": seg_audit["physical_feature_audit_scope"],
    })
    caveats = [
        "The rolling extension was specified after inspecting the original 2024 result; the segmented hypothesis was chosen after inspecting cumulative results. These are exploratory retrospective backtests, not independently blinded confirmation.",
        "W0 contains 100 training-fitted SNP principal components, static geographic/planting covariates and site encoding; a small weather increment does not mean weather is agriculturally unimportant.",
        "Weather is observed retrospective information within each post-planting horizon, not a forecast available at planting. Real-time release latency was not tested.",
        "Thirty-day blocks are calendar windows, not observed phenological stages. Replacement changes feature dimension and model regularization/subsampling, not only temporal resolution.",
        "Five years, overlapping training histories, site-code proxies and unadjusted multiple comparisons limit uncertainty interpretation. Bootstrap intervals are conditional sensitivity intervals excluding refitting and selection uncertainty.",
        "Cohort eligibility requires complete calendar coverage and observed valid targets. No claim applies to all released plots, all farms, national yield, or missing-target observations.",
        "Seen/unseen subgroup cuts are descriptive. Unseen site codes occur in only two years and three environment-years; no explicit geographic holdout or external agricultural validation was run.",
        "The engineering screen and sufficient-window rule are project-specific operational criteria, not published agronomic laws or noninferiority tests.",
        "Internal audit counts are integrity checks, not independent scientific experiments or a measure of publication-level novelty.",
        "GBLUP/RKHS, explicit genotype-by-weather models, measured-phenology alignment, tuning-budget sensitivity and prospective external validation are not part of these completed experiments.",
    ]
    result = {
        "schema_version": 1,
        "metadata": {
            "title": "Audited evidence for the maize weather-prediction manuscript",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Static manuscript figures/tables and independent aggregate verification; no new model training or bootstrap calculation",
            "yield_unit": "Mg ha^-1", "difference_sign": "candidate minus reference; negative RMSE difference is better",
            "primary_estimand": "Equal mean of five annual environment-equal RMSE values; not pooled row RMSE and not environment-year-equal RMSE",
            "interval_source": "Frozen independently audited CSV endpoints, 2,000 draws; copied unchanged, not recalculated",
        },
        "source_hashes": dict(sorted(hashes.items())),
        "audits": audits,
        "design": {
            "folds": [{"test_year": y, "development_train_first_year": 2014,
                       "development_train_last_year": y - 2, "validation_year": y - 1,
                       "final_train_first_year": 2014, "final_train_last_year": y - 1} for y in YEARS],
            "cumulative_weather_features": {str(h): 30 for h in (30, 60, 90, 120)},
            "segmented_weather_features": {str(h): 21 * (h // 30) + 9 for h in (30, 60, 90, 120)},
            "physical_features_per_block": 21, "whole_horizon_quality_features": 9,
            "base_snp_components": 100, "seed": original_manifest["seed"],
            "model_versions": original_manifest["versions"],
            "candidate_selection": "Two candidates per model/window selected by preceding-year environment-equal RMSE; independent choices per representation",
        },
        "totals": totals,
        "coverage": records(coverage), "cohort_by_year": records(cohort_by_year),
        "preparation": {k: prep[k] for k in ("split_summary", "source_trait_rows", "observed_plot_labels", "unusable_plot_labels", "marker_columns", "retained_genotypes", "test_template_rows", "test_labels", "test_missing_labels")},
        "cohort_flow": records(cohort_flow),
        "official_provenance_followup": {k: provenance[k] for k in ("verified_on", "source_catalog", "scope", "summary", "method", "raw_modified")},
        "primary_results": primary,
        "year_equal_metrics": records(year_equal),
        "yearly_metrics": records(annual),
        "environment_metrics": records(environment.loc[environment.window.isin([0, 120])]),
        "environment_metrics_export_scope": "W0 and W120 including both simple baselines, both representations; full 4,386-row source verified before filtering",
        "cumulative_cross_year_comparisons": records(cumulative),
        "segmented_cross_year_comparisons": records(segmented),
        "cumulative_yearly_comparisons": records(cum_annual),
        "segmented_yearly_comparisons": records(seg_annual),
        "cumulative_environment_comparisons_W120": records(cum_env.loc[cum_env.window == 120]),
        "segmented_environment_comparisons_W120": records(seg_env.loc[seg_env.window == 120]),
        "subgroup_support": records(support), "subgroup_yearly_metrics": records(subgroup),
        "cumulative_subgroup_comparisons": records(cum_subgroup),
        "segmented_subgroup_comparisons": records(seg_subgroup),
        "sufficient_windows": records(sufficient), "validation_selections": records(selections),
        "primary_assessment": records(primary_assessment), "upgrade_assessment": records(upgrade_assessment),
        "model_candidates": candidates,
        "verification": {"status": "PASS", "check_count": sum(checks.values()), "checks_by_category": checks,
                         "scope": "All 170 annual configuration records from 4,386 environment-configuration records; all 36 overall cross-year contrasts and year-direction counts; cohort/subgroup consistency. No raw-residual replay or bootstrap recomputation in this collector; those have separate existing audits."},
        "caveats": caveats,
    }
    output = Path(output).resolve()
    if not output.is_relative_to(PROJECT / "reports/manuscript"):
        raise ValueError("Evidence outputs must stay in reports/manuscript")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    receipt = {"status": "PASS", "evidence_path": str(output.relative_to(PROJECT)),
               "evidence_sha256": sha(output), "collector_sha256": sha(Path(__file__)),
               "source_hashes": result["source_hashes"], "verification": result["verification"],
               "output_bytes": output.stat().st_size,
               "frozen_experiments_modified": False, "model_fits": 0, "bootstrap_reruns": 0}
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    evidence = collect(args.output)
    print(json.dumps({"output": str(args.output.relative_to(PROJECT)),
                      "totals": evidence["totals"], "verification": evidence["verification"],
                      "primary_results": evidence["primary_results"],
                      "year_equal_metrics_W0_W120": [r for r in evidence["year_equal_metrics"] if r["window"] in (0, 120)]}, indent=2))
