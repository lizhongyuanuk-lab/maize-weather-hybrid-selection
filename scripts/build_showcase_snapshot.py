#!/usr/bin/env python3
"""Append reviewed five-year research queries without changing the frozen analyses.

Only showcase/src/data.json is written. This is a snapshot transformation, not a
model fit, metric-generation pipeline, or replacement for the independent audits.
Existing queries, narrative, identity, and presentation metadata are retained.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
PINNED = {
    "modeling/rolling_v1/analysis/analysis_manifest.json":
        "28b56456a512dff413b5d8a343daa2710df130f58e1fadd206f0126aaa0e575d",
    "modeling/segmented_v1/analysis/analysis_manifest.json":
        "b002c0050411dc1f3d957d8f32fc008020a01ef62420c0221d094d679c079aaa",
    "modeling/results/final/final_evaluation_manifest.json":
        "c9cce89304d08087b48fc0c2f0bfdb52794eaee4b89aeb97a1b7cee51d38581a",
}
MODELS = ["Ridge", "ExtraTrees", "CatBoost"]
CAVEATS = [
    "Retrospective, exploratory next-year backtests; the segmented hypothesis was chosen after inspecting cumulative results, not preregistered or tested on a new blind cohort.",
    "The target is aggregated environment × hybrid trial yield in Mg/ha, not national yield or individual-farm operational yield.",
    "Weather is observed up to each post-planting horizon, not a weather forecast; data availability latency has not been established.",
    "W0 retains genotype PCA, location/site-code and planting information; it is not an environment-free baseline.",
    "Only five years, overlapping training histories and 31 Experiment_Code proxies. A site code is not a verified independent physical location; no spatial external holdout was performed.",
    "Crossed year/site-code and year-block intervals are conditional on the fitted pipelines; they exclude fitting/selection uncertainty and have no multiplicity correction.",
    "Shared W0, W30 and baseline results appear under both representations for comparison, not as independent experiments. Do not sum scored observations across configurations.",
    "More detailed calendar blocks are not measured developmental stages. No GBLUP/G×E domain baseline or claim of a new model architecture is included.",
    "Negative or inconclusive incremental predictive performance does not establish that weather is agronomically unimportant or causally ineffective.",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT))


def local_path(value: str, base: Path = PROJECT) -> Path:
    path = Path(value)
    # Frozen manifests retain their original checkout paths. Only their exact
    # project suffix is portable; no arbitrary basename matching is permitted.
    if path.is_absolute():
        try:
            suffix = path.relative_to(PROJECT)
        except ValueError:
            marker = "/MaizeClimate_Research/"
            require(marker in value, f"Unrecognized source root: {value}")
            suffix = Path(value.split(marker, 1)[1])
        path = PROJECT / suffix
    else:
        path = base / path
    path = path.resolve()
    path.relative_to(PROJECT)
    require(path.is_file(), f"Missing source: {relative(path)}")
    return path


def file_record(path: Path, verified: dict[str, str]) -> dict:
    name = relative(path)
    return {"label": name, "path": name, "sha256": verified.get(name) or sha256(path)}


def verify_sources() -> tuple[dict, dict, dict]:
    verified: dict[str, str] = {}

    def check(path: Path, expected: str) -> None:
        name = relative(path)
        actual = verified.get(name) or sha256(path)
        require(actual == expected, f"Source SHA mismatch: {name}")
        verified[name] = actual

    for name, expected in PINNED.items():
        check(PROJECT / name, expected)
    bundles, receipts = {}, {}
    for kind, directory in [("cumulative", "rolling_v1"), ("segmented", "segmented_v1")]:
        root = PROJECT / "modeling" / directory
        manifest_path = root / "analysis/analysis_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        require(manifest["years"] == list(range(2020, 2025)), "Unexpected analysis years")
        require(manifest["n_scored_observations"] == 57673, "Unexpected scored support")
        require(manifest["n_environment_years"] == 129, "Unexpected environment support")
        for name, expected in manifest["source_hashes"].items():
            check(local_path(name), expected)
        for name, expected in manifest["artifacts"].items():
            check(local_path(name, root), expected)
        receipt_path = root / "validation_checks.json"
        receipt = json.loads(receipt_path.read_text())
        require(receipt["status"] == "PASS" and receipt.get("failure") is None,
                f"Independent audit is not PASS: {directory}")
        require(local_path(receipt["audited_manifest_path"]) == manifest_path,
                f"Audit path mismatch: {directory}")
        require(receipt["audited_manifest_sha256"] == sha256(manifest_path),
                f"Audit digest mismatch: {directory}")
        require(receipt["original_2024_manifest_sha256"] ==
                PINNED["modeling/results/final/final_evaluation_manifest.json"],
                f"Original 2024 binding mismatch: {directory}")
        validator = PROJECT / "modeling" / f"validate_{'rolling' if kind == 'cumulative' else 'segmented'}_results.py"
        check(validator, receipt["script_sha256"])
        if kind == "segmented":
            require(receipt["cumulative_analysis_sha256"] ==
                    PINNED["modeling/rolling_v1/analysis/analysis_manifest.json"],
                    "Cumulative audit dependency mismatch")
            replays = receipt["new_final_model_replays"]
            require(len(replays) == 45 and {(row["year"], row["model"], row["window"]) for row in replays}
                    == {(year, model, window) for year in range(2020, 2025)
                        for model in MODELS for window in [60, 90, 120]},
                    "Unexpected final-model replay support")
            check(PROJECT / "modeling/validate_rolling_results.py", receipt["independent_arithmetic_sha256"])
        verified[relative(receipt_path)] = sha256(receipt_path)
        bundles[kind] = {"root": root, "manifest": manifest, "manifest_path": manifest_path,
                         "receipt_path": receipt_path}
        receipts[kind] = {
            "status": receipt["status"], "checkedAt": receipt["checked_at_utc"],
            "checksPassed": receipt["checks_passed"],
            "path": relative(receipt_path), "sha256": verified[relative(receipt_path)],
            "auditedManifestPath": relative(manifest_path),
            "auditedManifestSha256": sha256(manifest_path),
            "scope": "Computational provenance and independent result arithmetic; not proof of scientific novelty, causality or external validity.",
        }
    return bundles, receipts, verified


def load_table(bundle: dict, name: str) -> pd.DataFrame:
    path = bundle["root"] / "analysis" / f"{name}.csv"
    frame = pd.read_csv(path)
    require(len(frame) == bundle["manifest"]["table_rows"][name], f"Row count mismatch: {name}")
    for column in ["evaluated_years", "missing_prespecified_years"]:
        if column in frame:
            frame[column] = frame[column].fillna("")
    return frame


def records(frame: pd.DataFrame) -> list[dict]:
    # Do not round the reviewed CSV values. JSON null, not zero, represents NA.
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def primary_rows(cumulative: pd.DataFrame, segmented: pd.DataFrame,
                 coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        c = cumulative.query("model == @model and window == 120").iloc[0]
        s = segmented.query("model == @model and window == 120 and reference == 'no_weather'").iloc[0]
        sc = segmented.query("model == @model and window == 120 and reference == 'cumulative'").iloc[0]
        row = {"model": model, "window": 120, "w0_rmse": c.year_equal_rmse_W0,
               "cumulative_rmse": c.year_equal_rmse_weather,
               "segmented_rmse": s.year_equal_rmse_segmented,
               "n_years": int(c.n_years), "n_environment_years": int(c.n_environment_years),
               "n_scored_observations": int(coverage.n_scored.sum()),
               "n_site_codes": int(c.n_site_codes), "evaluated_years": c.evaluated_years,
               "bootstrap_repeats": int(c.bootstrap_repeats),
               "ci_method": "crossed_year_site_code", "reference": "no_weather"}
        for prefix, source in [("cumulative", c), ("segmented", s), ("segmented_vs_cumulative", sc)]:
            row.update({
                f"{prefix}_delta_rmse": source.year_equal_delta_rmse,
                f"{prefix}_ci_low": source.crossed_year_site_ci95_low,
                f"{prefix}_ci_high": source.crossed_year_site_ci95_high,
                f"{prefix}_year_block_ci_low": source.year_block_ci95_low,
                f"{prefix}_year_block_ci_high": source.year_block_ci95_high,
                f"{prefix}_gain_percent": source.relative_gain_of_year_equal_rmse_percent,
                f"{prefix}_years_improved": int(source.years_improved),
                f"{prefix}_years_worsened": int(source.years_worsened),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def validate_tables(tables: dict[str, pd.DataFrame], primary: pd.DataFrame) -> dict:
    coverage, yearly, env = tables["coverage"], tables["yearly_metrics"], tables["environment_metrics"]
    require(coverage.Year.tolist() == list(range(2020, 2025)), "Coverage years mismatch")
    require(coverage.n_scored.tolist() == [11298, 13615, 11783, 11491, 9486], "Coverage samples mismatch")
    require(coverage.n_environments.tolist() == [25, 27, 28, 27, 22], "Coverage environments mismatch")
    require(len(yearly) == 170 and len(env) == 4386, "Configuration coverage mismatch")
    keys = ["Year", "representation", "model", "window"]
    require(not yearly.duplicated(keys).any(), "Duplicate yearly grain")
    require(not env.duplicated(keys + ["Env"]).any(), "Duplicate environment grain")
    for row in yearly.to_dict("records"):
        subset = env
        for key in keys:
            subset = subset[subset[key] == row[key]]
        cov = coverage[coverage.Year == row["Year"]].iloc[0]
        require(len(subset) == row["n_environments"] == cov.n_environments, "Environment support mismatch")
        require(subset.n.sum() == row["n"] == cov.n_scored, "Observation support mismatch")
        for metric in ["rmse", "mae", "bias"]:
            require(math.isclose(subset[metric].mean(), row[f"macro_{metric}"], rel_tol=1e-10, abs_tol=1e-10),
                    f"Environment-equal {metric} does not reconcile")
        pooled = math.sqrt(np.average(subset.rmse.to_numpy() ** 2, weights=subset.n))
        require(math.isclose(pooled, row["pooled_rmse"], rel_tol=1e-10, abs_tol=1e-10),
                "Pooled RMSE does not reconcile")
    for row in primary.to_dict("records"):
        model = row["model"]
        for representation, horizon, alias in [("cumulative", 0, "w0_rmse"),
                                                ("cumulative", 120, "cumulative_rmse"),
                                                ("segmented", 120, "segmented_rmse")]:
            vals = yearly.query("model == @model and representation == @representation and window == @horizon")
            require(len(vals) == 5, "Primary estimate missing a year")
            require(math.isclose(vals.macro_rmse.mean(), row[alias], abs_tol=1e-10), "Primary year-equal mean mismatch")
        for prefix, candidate, reference in [("cumulative", "cumulative_rmse", "w0_rmse"),
                                              ("segmented", "segmented_rmse", "w0_rmse"),
                                              ("segmented_vs_cumulative", "segmented_rmse", "cumulative_rmse")]:
            require(math.isclose(row[candidate] - row[reference], row[f"{prefix}_delta_rmse"], abs_tol=1e-10),
                    "Primary delta mismatch")
            require(math.isclose(100 * (1 - row[candidate] / row[reference]), row[f"{prefix}_gain_percent"], abs_tol=1e-10),
                    "Primary relative gain mismatch")
            require(row[f"{prefix}_ci_low"] <= row[f"{prefix}_ci_high"], "Reversed CI")
    group = tables["yearly_subgroup_metrics"]
    require(len(group) == 680 and not group.duplicated(keys + ["group_type", "group"]).any(), "Subgroup grain mismatch")
    support = group.groupby(keys + ["group_type"], dropna=False).n.sum().reset_index()
    support = support.merge(yearly[keys + ["n"]], on=keys, suffixes=("_subgroups", "_all"), validate="many_to_one")
    require((support.n_subgroups == support.n_all).all(), "Seen/unseen subgroup counts do not reconcile")
    sufficient = tables["yearly_sufficient_windows"]
    require(len(sufficient) == 30 and sufficient.earliest_sufficient_window.isna().all()
            and not sufficient.W120_gain_supported.any(), "Sufficiency differs from frozen evidence")
    return {"status": "PASS", "nYearConfigurationRows": len(yearly),
            "nEnvironmentMetricRows": len(env), "nSubgroupRows": len(group),
            "nObservationsPerConfiguration": int(coverage.n_scored.sum()),
            "nEnvironmentYearsPerConfiguration": int(coverage.n_environments.sum()),
            "checks": ["Pinned manifests, all manifest source hashes and all table artifacts verified",
                       "Audit receipts bind to current manifests and validator source hashes",
                       "170 annual rows reconcile with environment-level RMSE, MAE, bias, pooled RMSE and support",
                       "Three primary W120 rows reconcile with five annual environment-equal estimates",
                       "Seen/unseen sample counts reconcile separately within each group type",
                       "Thirty sufficiency rows preserve the unsupported/null conclusions"]}


def metric_definitions(files: list[dict]) -> list[dict]:
    lineage = [{"files": [item["path"] for item in files if item["path"].endswith(".csv")]}]
    return [
        {"label": "Prediction target and units", "definition": "Environment × hybrid aggregated trial yield, Mg/ha (tonnes per hectare). RMSE and MAE are lower-is-better; bias is prediction minus observed yield.", "sourceLineage": lineage},
        {"label": "Environment-equal / year-equal RMSE", "definition": "Compute RMSE within each environment, average environments equally within year, then average the five annual values equally. Not a pooled-row RMSE or a sample-count-weighted five-year mean.", "sourceLineage": lineage},
        {"label": "Delta and relative gain", "definition": "Delta = candidate RMSE minus reference RMSE (negative is better). Relative gain = 100 × (reference − candidate) / reference (positive is better). References must not be mixed; gain of the five-year means differs from mean annual percentage gain.", "sourceLineage": lineage},
        {"label": "Uncertainty", "definition": "The primary *_ci_low/high columns are 95% crossed year/site-code bootstrap intervals for paired delta RMSE; *_year_block_ci_low/high retain the year-only sensitivity intervals. Source tables retain both. 2,000 repetitions; five years and site-code proxies; conditional, descriptive and unadjusted for multiple comparisons.", "sourceLineage": lineage},
        {"label": "Coverage and subgroups", "definition": "n is scored environment × hybrid observations, not independent weather trials. Seen/unseen is training-set membership for hybrid or Experiment_Code separately; these two subgroup partitions overlap and must not be added together. 2024 excludes 571 inputs without observed yield.", "sourceLineage": lineage},
        {"label": "Earliest sufficient window", "definition": "Prespecified descriptive rule: W120 vs W0 improves RMSE by at least 2%, its paired 95% interval is wholly below zero, then choose the earliest positive-gain window retaining at least 90% of W120 point-estimate gain. All thirty model/representation/year rows are unsupported with null earliest window. Not a universal agronomic deadline or formal noninferiority test.", "sourceLineage": lineage},
    ]


def make_query(title: str, frame: pd.DataFrame, sources: list[tuple[dict, str]],
               verified: dict, generated: str, grain: str, method: str) -> dict:
    files = [file_record(bundle["root"] / "analysis" / f"{name}.csv", verified)
             for bundle, name in sources]
    bindings = {relative(bundle["manifest_path"]): sha256(bundle["manifest_path"])
                for bundle, _ in sources}
    return {
        "rows": records(frame),
        "source": {
            "label": title, "files": files, "classification": "reviewed_observations",
            "executedAt": generated, "period": "2020–2024", "timezone": "UTC",
            "grain": grain, "units": "Yield error: Mg/ha; relative_gain_*: percent unless explicitly named W120_relative_gain (fraction).",
            "analysisManifestSha256": bindings,
            "metricDefinitions": metric_definitions(files), "caveats": CAVEATS,
            "evidenceFlow": [
                {"title": "Verified frozen local evidence", "detail": "Read the exact CSV paths and full SHA-256 values listed under files. Verify pinned analysis manifests, all declared input/artifact hashes, and independent-audit manifest/source bindings before reading reviewed tables. No models or raw prediction rows are loaded."},
                {"title": "Snapshot transformation", "detail": method},
                {"title": "Reproduce and export", "detail": "Run scripts/build_showcase_snapshot.py with the project Python/pandas/NumPy runtime. CSV exports from the source inspector use these reviewed rows; original CSV paths are project-relative local evidence, not published download URLs."},
            ],
        },
        "methods": [{"language": "text", "code": method},
                    {"language": "bash", "code": "PYTHONPATH=.runtime python scripts/build_showcase_snapshot.py"}],
    }


def sanitize_legacy(value):
    """Remove local-machine/thread-link leaks without changing reviewed numbers."""
    if isinstance(value, dict):
        return {key: sanitize_legacy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_legacy(item) for item in value]
    if isinstance(value, str):
        value = value.replace(str(PROJECT) + "/", "")
        value = re.sub(r"\[[^\]]*\]\((?:codex://|https?://[^\s)]+/(?:threads|c)/)[^)]+\)", "[private conversation link omitted]", value)
        require("/Users/" not in value and "codex://" not in value, "Unmapped local path or private thread link in snapshot")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify and derive the snapshot without writing any file")
    args = parser.parse_args()
    target = PROJECT / "showcase/src/data.json"
    original_hash = sha256(target)
    original = json.loads(target.read_text())
    bundles, receipts, verified = verify_sources()
    cumulative, segmented = bundles["cumulative"], bundles["segmented"]
    tables = {name: load_table(segmented, name) for name in [
        "coverage", "yearly_metrics", "environment_metrics", "cross_year_comparisons",
        "yearly_subgroup_metrics", "yearly_sufficient_windows"]}
    incremental = load_table(cumulative, "cross_year_stability")
    primary = primary_rows(incremental, tables["cross_year_comparisons"], tables["coverage"])
    validation = validate_tables(tables, primary)
    generated = datetime.now(timezone.utc).isoformat()
    snapshot = sanitize_legacy(copy.deepcopy(original))
    specifications = [
        ("research_coverage", "Five-year paired research coverage", "coverage", "test year"),
        ("research_yearly", "All annual cumulative and segmented metrics", "yearly_metrics", "year × representation × model × weather window"),
        ("research_comparisons", "Segmented versus cumulative or W0: cross-year comparisons", "cross_year_comparisons", "model × weather window × explicit reference"),
        ("research_environment", "All scored environment-level metrics", "environment_metrics", "year × representation × model × weather window × environment"),
        ("research_subgroups", "Seen/unseen hybrid and site-code support and metrics", "yearly_subgroup_metrics", "year × representation × model × weather window × group type × seen/unseen"),
        ("research_sufficiency", "Annual earliest-sufficient-window assessments", "yearly_sufficient_windows", "year × representation × model"),
    ]
    for query_id, title, table, grain in specifications:
        snapshot["queries"][query_id] = make_query(
            title, tables[table], [(segmented, table)], verified, generated, grain,
            f"Read every row and original column of modeling/segmented_v1/analysis/{table}.csv; preserve CSV numeric precision and missing numeric values as JSON null. Preserve empty evaluated/missing-year lists as empty strings. No filtering, reselection or recalculation of intervals.")
    snapshot["queries"]["research_weather_increment"] = make_query(
        "Cumulative weather versus W0 at every horizon", incremental,
        [(cumulative, "cross_year_stability")], verified, generated, "model × weather window",
        "Read all 12 rows of rolling_v1/analysis/cross_year_stability.csv. Candidate is cumulative weather at the stated window; reference is the same model W0. Retain both year-block and crossed year/site-code intervals; do not replace them with segmented comparison intervals.")
    snapshot["queries"]["research_primary"] = make_query(
        "W120 three-way comparison, equal weight to each test year", primary,
        [(cumulative, "cross_year_stability"), (segmented, "cross_year_comparisons"),
         (segmented, "yearly_metrics"), (segmented, "coverage")], verified, generated, "model (fixed W120, all five years)",
        "For each Ridge/ExtraTrees/CatBoost model select W120 from cumulative cross_year_stability, plus segmented cross_year_comparisons reference=no_weather and reference=cumulative. Copy W0/cumulative/segmented year-equal RMSE and their paired delta/gain/interval/support fields into one wide row. *_ci_low/high means crossed_year_site_ci95_low/high; *_year_block_ci_low/high separately preserves year-only intervals. Validate means against the five annual macro_rmse rows and delta/gain algebra; do not average interval bounds or observed rows.")
    snapshot["buildStatus"] = "updating"
    snapshot["generatedAt"] = generated
    snapshot["researchSnapshotVersion"] = {
        "version": "five-year-cumulative-and-segmented-v1", "generatedAt": generated,
        "script": "scripts/build_showcase_snapshot.py", "scriptSha256": sha256(Path(__file__)),
        "analysisType": "exploratory_retrospective", "years": list(range(2020, 2025)),
        "analysisBindings": [{"path": key, "sha256": value} for key, value in PINNED.items()],
        "auditReceipts": receipts, "snapshotValidation": validation,
        "verifiedFiles": [{"path": key, "sha256": value} for key, value in sorted(verified.items())],
        "legacyScope": "The ten original queries and researchNarrative are retained as the original 2024-only evidence. New five-year consumers use research_* query IDs; the old prose is not a five-year summary.",
        "rowPopulation": "57,673 scored environment × hybrid observations and 129 environment-years per configuration; 170 representation/year/configuration rows include shared/reused configurations, not 170 independent experiments.",
        "intervalPolicy": "Primary CIs are crossed year/site-code paired-delta intervals; year-block sensitivity intervals also retained. No formal confirmatory or universal-future-year guarantee.",
        "dataClassification": "reviewed local research outputs, no synthetic values or raw prediction rows",
        "downloadPolicy": "Export reviewed query rows through the app's existing CSV action. Exact source CSV files are local project-relative evidence, not automatically published URLs.",
        "caveats": CAVEATS,
    }
    for key in ["id", "surface", "title"]:
        require(snapshot[key] == original[key], f"Identity changed: {key}")
    for key, value in original["queries"].items():
        if not key.startswith("research_"):
            require(snapshot["queries"][key] == sanitize_legacy(value), f"Legacy query changed: {key}")
    encoded = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    require("/Users/" not in encoded and "codex://" not in encoded, "Private machine or thread link leaked")
    require(sha256(target) == original_hash, "Concurrent snapshot edit detected; retry from current source")
    if not args.check_only:
        target.write_text(encoded)
    print(json.dumps({"status": "PASS", "check_only": args.check_only,
                      "output": relative(target), "buildStatus": snapshot["buildStatus"],
                      "bytes": len(encoded.encode()),
                      "queries": {key: len(value["rows"]) for key, value in snapshot["queries"].items()},
                      "primary": records(primary), "verifiedFiles": len(verified)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
