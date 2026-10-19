"""Paired cumulative/segmented comparisons; no fitting or narrative report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_rolling_stability import (
    YEARS, REPEATS, SEED, paired_environment_rows, within_year_stability,
    cross_year_stability, verify_artifacts, verify_run_inputs,
)
from metrics import evaluate_predictions
from run_experiments import CANDIDATES, HORIZONS, PROJECT, digest, now, write_json

ROOT = PROJECT / "modeling/segmented_v1"
CUMULATIVE_ROOT = PROJECT / "modeling/rolling_v1"
CONFIGS = {(m, h) for m in CANDIDATES for h in HORIZONS} | {
    ("GlobalMean", 0), ("HistoricalSiteMean", 0)}
SHARED = {(m, h) for m in CANDIDATES for h in (0, 30)} | {
    ("GlobalMean", 0), ("HistoricalSiteMean", 0)}
IDENTITY = ["sample_id", "Env", "Hybrid", "Year", "Experiment_Code", "y_true"]
REFERENCES = ("cumulative", "no_weather")


def source_directories(year, cumulative_root):
    if year == 2024:
        return PROJECT / "modeling/results/final", PROJECT / "modeling/features/final"
    return cumulative_root / f"results/{year}/final", cumulative_root / f"features/{year}/final"


def rename_reference_fields(record):
    """Old generic arithmetic names do not leak misleading baseline labels."""
    if isinstance(record, pd.DataFrame):
        return record.rename(columns={c: c.replace("_W0", "_reference").replace("_weather", "_segmented")
                                      for c in record.columns})
    return {k.replace("_W0", "_reference").replace("_weather", "_segmented"): v
            for k, v in record.items()}


def check_new_run(root, year):
    directory = root / f"results/{year}/final"
    feature_dir = root / f"features/{year}/final"
    completion_path = directory / "final_evaluation_manifest.json"
    completion = verify_artifacts(completion_path, root)
    if completion["test_year"] != year or completion["stage"] != "final":
        raise ValueError("Wrong segmented completion identity")
    run_path = directory / "run_manifest.json"
    if digest(run_path) != completion["run_manifest_sha256"]:
        raise ValueError("Segmented completed run changed")
    run = json.loads(run_path.read_text())
    for name, sha in run["inputs"].items():
        if not Path(name).is_absolute() or digest(Path(name)) != sha:
            raise ValueError(f"Segmented completed input changed: {name}")
    freeze = root / f"results/{year}/development/frozen_selection.json"
    if run["frozen_selection_sha256"] != digest(freeze):
        raise ValueError("Segmented frozen selection changed")
    if completion["frozen_selection_sha256"] != digest(freeze):
        raise ValueError("Segmented completion/selection mismatch")
    if run["inputs"].get(str(root / "PROTOCOL.md")) != digest(root / "PROTOCOL.md"):
        raise ValueError("Segmented analysis protocol differs from training protocol")
    feature_path = feature_dir / "feature_manifest.json"
    feature = json.loads(feature_path.read_text())
    if run["inputs"].get(str(feature_path)) != digest(feature_path):
        raise ValueError("Segmented run does not bind feature manifest")
    for name, sha in feature["input_hashes"].items():
        if not Path(name).is_absolute() or digest(Path(name)) != sha:
            raise ValueError(f"Segmented feature input changed: {name}")
    for name, sha in feature["outputs"].items():
        if digest(feature_dir / name) != sha:
            raise ValueError(f"Segmented feature output changed: {name}")
    required = {directory / f"predictions/{m}_W{h}_evaluated.csv" for m, h in CONFIGS}
    required |= {directory / name for name in (
        "run_manifest.json", "summary_metrics.csv", "environment_metrics.csv",
        "earliest_sufficient_windows.json", "prediction_lock_before_holdout.json")}
    bound = {(Path(completion["artifact_path_base"]) / n).resolve() for n in completion["artifacts"]}
    if not {p.resolve() for p in required}.issubset(bound):
        raise ValueError("Segmented completion does not bind all analysis inputs")
    return directory, feature_dir, completion_path, freeze


def engineering_screen(record):
    gain = float(record["relative_gain_of_year_equal_rmse_percent"])
    upper = float(record["crossed_year_site_ci95_high"])
    components = {
        "all_five_years": int(record["n_years"]) == 5,
        "gain_at_least_2_percent": bool(np.isfinite(gain) and gain >= 2.),
        "at_least_four_years_improve": int(record["years_improved"]) >= 4,
        "crossed_interval_upper_below_zero": bool(np.isfinite(upper) and upper < 0),
    }
    return {**components, "exploratory_engineering_signal": all(components.values()),
            "meaning": "prespecified_retrospective_screen_not_confirmatory_significance"}


def complete_subgroups(records, *, cross):
    names = ["model", "window", "reference", "group_type", "group"]
    axes = [CANDIDATES, HORIZONS[1:], REFERENCES, ("hybrid", "site_code"), ("seen", "unseen")]
    if not cross:
        names.insert(0, "Year")
        axes.insert(0, YEARS)
    idx = pd.MultiIndex.from_product(axes, names=names)
    result = pd.DataFrame(records).set_index(names).reindex(idx).reset_index()
    for column in (("n_years", "n_environment_years", "n_site_codes") if cross else
                   ("n_samples", "n_environments", "n_site_codes")):
        result[column] = result[column].fillna(0).astype(int)
    if cross:
        result["evaluated_years"] = result.evaluated_years.fillna("")
        result["missing_prespecified_years"] = result.missing_prespecified_years.fillna(";".join(map(str, YEARS)))
        result["bootstrap_status"] = result.bootstrap_status.fillna("empty_subgroup")
        result["interpretation"] = result.interpretation.fillna("empty_subgroup")
    else:
        result["interval_status"] = result.interval_status.fillna("empty_subgroup")
    return result


def analyze(root=ROOT, cumulative_root=CUMULATIVE_ROOT):
    root, cumulative_root = Path(root).resolve(), Path(cumulative_root).resolve()
    source_analysis = cumulative_root / "analysis/analysis_manifest.json"
    source_receipt = cumulative_root / "validation_checks.json"
    verified_source = verify_artifacts(source_analysis, cumulative_root)
    receipt = json.loads(source_receipt.read_text())
    if receipt["status"] != "PASS" or receipt["audited_manifest_sha256"] != digest(source_analysis):
        raise ValueError("Cumulative evidence is not bound to a passing independent audit")
    for path, sha in verified_source["source_hashes"].items():
        if digest(Path(path)) != sha:
            raise ValueError(f"Cumulative analysis source changed: {path}")
    source_paths = [source_analysis, source_receipt, root / "PROTOCOL.md", Path(__file__).resolve(),
                    PROJECT / "modeling/analyze_rolling_stability.py", PROJECT / "modeling/metrics.py",
                    PROJECT / "modeling/run_experiments.py"]
    sources = {str(p): digest(p) for p in source_paths}
    yearly, environment, coverage, subgroup_metrics, selection, sufficiency, reuse = ([] for _ in range(7))
    differences, group_differences = [], []
    for year in YEARS:
        cumulative_dir, cumulative_features = source_directories(year, cumulative_root)
        cumulative_completion = cumulative_dir / "final_evaluation_manifest.json"
        cumulative_record = verify_artifacts(cumulative_completion, PROJECT if year == 2024 else cumulative_root)
        cumulative_freeze = verify_run_inputs(cumulative_dir, cumulative_features,
                                              cumulative_record, cumulative_root, year)
        segmented_dir, segmented_features, segmented_completion, segmented_freeze = check_new_run(root, year)
        for path in (cumulative_completion, cumulative_freeze, segmented_completion, segmented_freeze):
            sources[str(path)] = digest(path)
        train = pd.read_csv(cumulative_features / "train_samples.csv")
        if train.Year.max() != year - 1 or (train.Year >= year).any():
            raise ValueError("Future year in reference training cohort")
        for split in ("train", "validation", "test"):
            old_path, new_path = cumulative_features / f"{split}_samples.csv", segmented_features / f"{split}_samples.csv"
            if digest(old_path) != digest(new_path):
                raise ValueError("Representation change altered temporal sample metadata")
            sources[str(old_path)] = digest(old_path)
            sources[str(new_path)] = digest(new_path)
        frames = {}
        for representation, directory, freeze_path in (
            ("cumulative", cumulative_dir, cumulative_freeze),
            ("segmented", segmented_dir, segmented_freeze)):
            table = pd.read_csv(directory / "summary_metrics.csv")
            if set(zip(table.model, table.window)) != CONFIGS or len(table) != 17:
                raise ValueError("Missing/extra representation configurations")
            yearly.append(table.assign(Year=year, representation=representation))
            environment.append(pd.read_csv(directory / "environment_metrics.csv").assign(Year=year, representation=representation))
            freeze = json.loads(freeze_path.read_text())
            selection.append({"Year": year, "representation": representation, **freeze["recommended_by_validation"]})
            rules = json.loads((directory / "earliest_sufficient_windows.json").read_text())
            sufficiency.extend({"Year": year, "representation": representation, "model": m, **r}
                               for m, r in rules["models"].items())
            for model, horizon in sorted(CONFIGS):
                p = directory / f"predictions/{model}_W{horizon}_evaluated.csv"
                frame = pd.read_csv(p)
                if set(frame.Year) != {year} or frame.sample_id.duplicated().any():
                    raise ValueError("Wrong evaluated year or repeated ID")
                frames[(representation, model, horizon)] = frame
                for kind, column in (("hybrid", "Hybrid"), ("site_code", "Experiment_Code")):
                    seen = frame[column].isin(set(train[column]))
                    for flag, label in ((True, "seen"), (False, "unseen")):
                        part = frame.loc[seen == flag]
                        metrics = evaluate_predictions(part)[0] if len(part) else {
                            "n": 0, "n_environments": 0,
                            **{f"macro_{m}": np.nan for m in ("rmse", "mae", "bias", "pearson", "spearman", "top20_recall")}}
                        subgroup_metrics.append({"Year": year, "representation": representation,
                            "model": model, "window": horizon, "group_type": kind, "group": label, **metrics})
        canonical = frames[("cumulative", "CatBoost", 0)]
        if any(not frame[IDENTITY].equals(canonical[IDENTITY]) for frame in frames.values()):
            raise ValueError("Representation/configuration comparisons changed scored cohort/order/outcomes")
        for model, horizon in sorted(SHARED):
            filename = f"{model}.csv" if model in ("GlobalMean", "HistoricalSiteMean") else f"{model}_W{horizon}.csv"
            if digest(cumulative_dir / "predictions" / filename) != digest(segmented_dir / "predictions" / filename):
                raise ValueError("Shared prediction unit was not byte-reused")
            a, b = frames[("cumulative", model, horizon)], frames[("segmented", model, horizon)]
            if not np.array_equal(a.prediction.to_numpy(), b.prediction.to_numpy()):
                raise ValueError("Scored shared unit predictions differ")
            reuse.append({"Year": year, "model": model, "window": horizon,
                          "raw_prediction_sha256": digest(cumulative_dir / "predictions" / filename),
                          "byte_identical": True, "new_fit_required": False})
        coverage.append({"Year": year, "n_scored": len(canonical), "n_environments": canonical.Env.nunique(),
                         "n_site_codes": canonical.Experiment_Code.nunique(), "n_hybrids": canonical.Hybrid.nunique(),
                         "n_training_samples": len(train), "training_last_year": int(train.Year.max()),
                         "representation_pairs": 17, "new_final_fits": 9})
        for model in CANDIDATES:
            for horizon in HORIZONS[1:]:
                candidate = frames[("segmented", model, horizon)]
                for reference in REFERENCES:
                    ref = frames[("cumulative", model, horizon if reference == "cumulative" else 0)]
                    paired = paired_environment_rows(ref, candidate).assign(model=model, window=horizon, reference=reference)
                    differences.append(paired)
                    for kind, column in (("hybrid", "Hybrid"), ("site_code", "Experiment_Code")):
                        seen = ref[column].isin(set(train[column]))
                        for flag, label in ((True, "seen"), (False, "unseen")):
                            a = ref.loc[seen == flag]
                            if len(a):
                                b = candidate.loc[candidate.sample_id.isin(set(a.sample_id))]
                                group_differences.append(paired_environment_rows(a, b).assign(
                                    model=model, window=horizon, reference=reference, group_type=kind, group=label))
    delta = pd.concat(differences, ignore_index=True)
    group_delta = pd.concat(group_differences, ignore_index=True)
    annual = [{"Year": y, "model": m, "window": h, "reference": r, **within_year_stability(g)}
              for (y, m, h, r), g in delta.groupby(["Year", "model", "window", "reference"])]
    multi = [{"model": m, "window": h, "reference": r, **cross_year_stability(g)}
             for (m, h, r), g in delta.groupby(["model", "window", "reference"])]
    group_annual = [{"Year": y, "model": m, "window": h, "reference": r, "group_type": k, "group": s,
                     **within_year_stability(g)} for (y, m, h, r, k, s), g in group_delta.groupby(
                         ["Year", "model", "window", "reference", "group_type", "group"])]
    group_multi = [{"model": m, "window": h, "reference": r, "group_type": k, "group": s,
                    **cross_year_stability(g)} for (m, h, r, k, s), g in group_delta.groupby(
                        ["model", "window", "reference", "group_type", "group"])]
    primary = [{**record, **engineering_screen(record)} for record in multi if record["window"] == 120]
    model_assessment = [{"model": model,
        "signal_vs_cumulative": next(r["exploratory_engineering_signal"] for r in primary if r["model"] == model and r["reference"] == "cumulative"),
        "signal_vs_no_weather": next(r["exploratory_engineering_signal"] for r in primary if r["model"] == model and r["reference"] == "no_weather"),
        "both_reference_screens_pass": all(r["exploratory_engineering_signal"] for r in primary if r["model"] == model),
        "interpretation": "exploratory_historical_signal_only_not_new_blind_confirmation"} for model in CANDIDATES]
    tables = {
        "coverage": pd.DataFrame(coverage), "yearly_metrics": pd.concat(yearly, ignore_index=True),
        "environment_metrics": pd.concat(environment, ignore_index=True),
        "yearly_subgroup_metrics": pd.DataFrame(subgroup_metrics),
        "environment_comparisons": rename_reference_fields(delta),
        "yearly_comparisons": rename_reference_fields(pd.DataFrame(annual)),
        "cross_year_comparisons": rename_reference_fields(pd.DataFrame(multi)),
        "primary_assessment": rename_reference_fields(pd.DataFrame(primary)),
        "model_upgrade_assessment": pd.DataFrame(model_assessment),
        "subgroup_environment_comparisons": rename_reference_fields(group_delta),
        "yearly_subgroup_comparisons": rename_reference_fields(complete_subgroups(group_annual, cross=False)),
        "cross_year_subgroup_comparisons": rename_reference_fields(complete_subgroups(group_multi, cross=True)),
        "validation_selections": pd.DataFrame(selection),
        "yearly_sufficient_windows": pd.DataFrame(sufficiency), "reuse_checks": pd.DataFrame(reuse),
    }
    if tables["coverage"].n_scored.sum() != 57673 or tables["coverage"].n_environments.sum() != 129:
        raise ValueError("The fixed five-year scored cohort changed")
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    for path, sha in sources.items():
        if digest(Path(path)) != sha:
            raise ValueError(f"Analysis source changed during analysis: {path}")
    manifest = {"completed_at_utc": now(), "analysis_type": "exploratory_retrospective_representation_comparison",
        "years": list(YEARS), "source_hashes": sources, "artifact_path_base": str(root),
        "artifacts": {str((out / f"{n}.csv").relative_to(root)): digest(out / f"{n}.csv") for n in tables},
        "table_rows": {n: len(f) for n, f in tables.items()}, "n_scored_observations": 57673,
        "n_environment_years": 129, "bootstrap_repeats": REPEATS, "seed": SEED,
        "caveats": ["Hypothesis chosen after inspecting cumulative outcomes; no new blind test.",
            "Equal search budget is not equal feature dimensionality or complexity.",
            "Only five years, overlapping training sets and site-code proxies; conditional intervals.",
            "No multiplicity adjustment, refitting uncertainty or true spatial holdout.",
            "Calendar blocks are not measured developmental stages."]}
    write_json(manifest, out / "analysis_manifest.json")
    print(json.dumps({"analysis": str(out), "table_rows": manifest["table_rows"]}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cumulative-root", type=Path, default=CUMULATIVE_ROOT)
    arguments = parser.parse_args()
    analyze(arguments.root, arguments.cumulative_root)
