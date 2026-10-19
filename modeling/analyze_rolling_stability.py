"""Supplementary, descriptive stability tables; does not train or write a report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import bootstrap_mean, evaluate_predictions
from run_experiments import CANDIDATES, HORIZONS, PROJECT, digest, now, write_json

DEFAULT_ROOT = PROJECT / "modeling/rolling_v1"
YEARS = (2020, 2021, 2022, 2023, 2024)
SEED = 20261018
REPEATS = 2000
METRICS = ("rmse", "mae", "bias", "pearson", "spearman", "top20_recall")


def verify_artifacts(manifest_path: Path, default_base: Path) -> dict:
    record = json.loads(manifest_path.read_text())
    base = Path(record.get("artifact_path_base", default_base))
    if not base.is_absolute():
        base = default_base / base
    for relative, expected in record["artifacts"].items():
        path = base / relative
        if digest(path) != expected:
            raise ValueError(f"Changed artifact: {path}")
    return record


def verify_run_inputs(directory, feature_directory, completion, root, year):
    """Bind subgroup membership/protocol to the already completed experiment."""
    run_path = directory/"run_manifest.json"
    artifact_base = Path(completion.get("artifact_path_base",PROJECT if year==2024 else root))
    recorded_artifacts = {(artifact_base/name).resolve() for name in completion["artifacts"]}
    required_artifacts = {directory/name for name in ("run_manifest.json","summary_metrics.csv",
        "environment_metrics.csv","weather_increment_comparisons.csv","earliest_sufficient_windows.json")}
    configurations = {(m,h) for m in CANDIDATES for h in HORIZONS} | {("GlobalMean",0),("HistoricalSiteMean",0)}
    required_artifacts |= {directory/"predictions"/f"{m}_W{h}_evaluated.csv" for m,h in configurations}
    if not {p.resolve() for p in required_artifacts}.issubset(recorded_artifacts):
        raise ValueError("Required analysis artifact is not bound to completion")
    if completion.get("run_manifest_sha256", digest(run_path)) != digest(run_path):
        raise ValueError("Completed run manifest changed")
    run = json.loads(run_path.read_text())
    input_paths = {(Path(name) if Path(name).is_absolute() else PROJECT/name).resolve() for name in run["inputs"]}
    if not {(feature_directory/"train_samples.csv").resolve(),(feature_directory/"feature_manifest.json").resolve()}.issubset(input_paths):
        raise ValueError("Training membership/features are not bound to the model run")
    for name, expected in run["inputs"].items():
        path = Path(name) if Path(name).is_absolute() else PROJECT/name
        if digest(path) != expected:
            raise ValueError(f"Completed-run input changed: {path}")
    feature_path = feature_directory/"feature_manifest.json"
    feature = json.loads(feature_path.read_text())
    for name,expected in feature["input_hashes"].items():
        path = Path(name) if Path(name).is_absolute() else PROJECT/name
        if digest(path) != expected:
            raise ValueError(f"Completed feature input changed: {path}")
    for name, expected in feature["outputs"].items():
        if digest(feature_directory/name) != expected:
            raise ValueError(f"Completed feature output changed: {name}")
    freeze_path = (PROJECT/"modeling/results/development/frozen_selection.json" if year==2024
                   else root/f"results/{year}/development/frozen_selection.json")
    if run["frozen_selection_sha256"] != digest(freeze_path):
        raise ValueError("Completed-run model selection changed")
    if completion.get("frozen_selection_sha256",digest(freeze_path)) != digest(freeze_path):
        raise ValueError("Completion selection binding changed")
    if year != 2024 and run["inputs"].get(str((root/"PROTOCOL.md").resolve())) != digest(root/"PROTOCOL.md"):
        raise ValueError("Analysis protocol was not the model-run protocol")
    return freeze_path


def paired_environment_rows(reference, candidate):
    """Require identical scored rows and observations, then compare by environment."""
    key = ["sample_id", "Env", "Hybrid", "Year", "Experiment_Code"]
    a = reference.sort_values("sample_id").reset_index(drop=True)
    b = candidate.sort_values("sample_id").reset_index(drop=True)
    if a.sample_id.duplicated().any() or not a[key].equals(b[key]):
        raise ValueError("Weather comparison changed sample membership or metadata")
    if not np.array_equal(a.y_true.to_numpy(), b.y_true.to_numpy()):
        raise ValueError("Weather comparison changed observed outcomes")
    _, am = evaluate_predictions(a)
    _, bm = evaluate_predictions(b)
    meta = a[["Env", "Year", "Experiment_Code"]].drop_duplicates()
    if meta.Env.duplicated().any():
        raise ValueError("Environment has conflicting year/site-code metadata")
    merged = am.merge(bm, on="Env", suffixes=("_W0", "_weather"), validate="one_to_one")
    merged = merged.merge(meta, on="Env", validate="one_to_one")
    if not (merged.n_W0 == merged.n_weather).all():
        raise ValueError("Paired environment supports differ")
    for metric in METRICS:
        merged[f"delta_{metric}"] = merged[f"{metric}_weather"]-merged[f"{metric}_W0"]
    return merged


def within_year_stability(frame, repeats=REPEATS):
    d = frame.delta_rmse.to_numpy(float)
    if not len(d) or not np.isfinite(d).all():
        raise ValueError("Nonempty finite paired environment differences required")
    lo, hi = (np.nan, np.nan)
    if len(d) > 1:
        interval = bootstrap_mean(d, repeats=repeats, seed=SEED)
        lo, hi = interval["ci95_low"], interval["ci95_high"]
    e0, ew = frame.rmse_W0.mean(), frame.rmse_weather.mean()
    improved, worsened = int((d < -1e-12).sum()), int((d > 1e-12).sum())
    return {"n_environments": len(d), "n_site_codes": frame.Experiment_Code.nunique(),
            "n_samples": int(frame.n_W0.sum()), "macro_rmse_W0": e0, "macro_rmse_weather": ew,
            "delta_rmse": d.mean(), "relative_gain_percent": 100*(e0-ew)/e0 if e0 > 0 else np.nan,
            "environment_ci95_low": lo, "environment_ci95_high": hi,
            "improved_environments": improved, "worsened_environments": worsened,
            "unchanged_environments": len(d)-improved-worsened,
            "improved_fraction": improved/len(d), "worsened_fraction": worsened/len(d),
            "delta_p10": np.quantile(d, .1), "delta_median": np.median(d),
            "delta_p90": np.quantile(d, .9), "delta_min": d.min(), "delta_max": d.max(),
            "interval_status": "conditional_environment_bootstrap" if len(d)>1 else "not_informative_single_environment"}


def cross_year_stability(frame, repeats=REPEATS, seed=SEED):
    """Year-equal estimator; crossed year/site multiplicities preserve Env weighting."""
    if frame.duplicated(["Year", "Env"]).any() or not len(frame):
        raise ValueError("One paired row per year/environment required")
    if not np.isfinite(frame[["delta_rmse", "rmse_W0", "rmse_weather"]].to_numpy()).all():
        raise ValueError("Finite RMSE inputs required")
    years = sorted(frame.Year.unique())
    sites = sorted(frame.Experiment_Code.unique())
    annual = frame.groupby("Year", sort=True)[["delta_rmse", "rmse_W0", "rmse_weather"]].mean()
    point = float(annual.delta_rmse.mean())
    gain = (annual.rmse_W0-annual.rmse_weather)/annual.rmse_W0.where(annual.rmse_W0>0)*100
    result = {"n_years": len(years), "n_environment_years": len(frame), "n_site_codes": len(sites),
              "evaluated_years": ";".join(str(y) for y in years),
              "missing_prespecified_years": ";".join(str(y) for y in YEARS if y not in years),
              "year_equal_delta_rmse": point,
              "environment_year_equal_delta_rmse": float(frame.delta_rmse.mean()),
              "year_equal_rmse_W0": annual.rmse_W0.mean(),
              "year_equal_rmse_weather": annual.rmse_weather.mean(),
              "relative_gain_of_year_equal_rmse_percent": 100*(annual.rmse_W0.mean()-annual.rmse_weather.mean())/annual.rmse_W0.mean() if annual.rmse_W0.mean()>0 else np.nan,
              "mean_annual_relative_gain_percent": gain.mean(),
              "years_improved": int((annual.delta_rmse < -1e-12).sum()),
              "years_worsened": int((annual.delta_rmse > 1e-12).sum()),
              "year_min_delta": annual.delta_rmse.min(), "year_max_delta": annual.delta_rmse.max(),
              "year_block_ci95_low": np.nan, "year_block_ci95_high": np.nan,
              "crossed_year_site_ci95_low": np.nan, "crossed_year_site_ci95_high": np.nan,
              "bootstrap_repeats": repeats, "seed": seed,
              "interpretation": "descriptive_conditional_sensitivity_not_future_year_guarantee"}
    if len(years) < 2:
        result["bootstrap_status"] = "insufficient_years"
        return result
    rng = np.random.default_rng(seed)
    a = annual.delta_rmse.to_numpy()
    year_means = a[rng.integers(0, len(years), (repeats, len(years)))].mean(axis=1)
    result["year_block_ci95_low"], result["year_block_ci95_high"] = np.quantile(year_means, [.025, .975])
    site_index = {site:i for i, site in enumerate(sites)}
    groups = [(g.delta_rmse.to_numpy(), np.array([site_index[s] for s in g.Experiment_Code]))
              for _, g in frame.groupby("Year", sort=True)]
    draws, attempts = [], 0
    while len(draws) < repeats and attempts < repeats*100:
        attempts += 1
        yw = rng.multinomial(len(years), np.full(len(years), 1/len(years)))
        sw = rng.multinomial(len(sites), np.full(len(sites), 1/len(sites)))
        weighted = []
        valid = True
        for j, (values, site_ids) in enumerate(groups):
            weights = sw[site_ids]
            if yw[j] and not weights.sum():
                valid = False
                break
            weighted.append(float(np.dot(weights, values)/weights.sum()) if weights.sum() else 0.)
        if valid:
            draws.append(float(np.dot(yw, weighted)/yw.sum()))
    if len(draws) != repeats:
        raise ValueError("Crossed bootstrap could not obtain sufficient nonempty draws")
    result["crossed_year_site_ci95_low"], result["crossed_year_site_ci95_high"] = np.quantile(draws, [.025, .975])
    result.update(bootstrap_status="complete", crossed_empty_draws_rejected=attempts-repeats)
    return result


def analyze(root=DEFAULT_ROOT, years=YEARS):
    root = Path(root)
    if tuple(years) != YEARS:
        raise ValueError("The frozen analysis includes all five years, without favorable-year selection")
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    protocol = root / "PROTOCOL.md"
    sources[str(protocol)] = digest(protocol)
    for name in ("analyze_rolling_stability.py", "metrics.py", "run_experiments.py"):
        path = PROJECT / "modeling" / name
        sources[str(path)] = digest(path)
    yearly, envs, comparisons, groups, deltas, subgroup_deltas, coverage, selections, rules = ([] for _ in range(9))
    expected_configs = {(m,h) for m in CANDIDATES for h in HORIZONS} | {("GlobalMean",0),("HistoricalSiteMean",0)}
    for year in years:
        directory = PROJECT/"modeling/results/final" if year == 2024 else root/f"results/{year}/final"
        feature_dir = PROJECT/"modeling/features/final" if year == 2024 else root/f"features/{year}/final"
        completion = directory/"final_evaluation_manifest.json"
        completed = verify_artifacts(completion, PROJECT if year == 2024 else root)
        freeze_path = verify_run_inputs(directory,feature_dir,completed,root,year)
        sources[str(completion)] = digest(completion)
        training = pd.read_csv(feature_dir/"train_samples.csv")
        if training.Year.max() != year-1 or (training.Year >= year).any():
            raise ValueError("Final-refit metadata includes test/future year")
        sources[str(feature_dir/"train_samples.csv")] = digest(feature_dir/"train_samples.csv")
        summary = pd.read_csv(directory/"summary_metrics.csv")
        if set(zip(summary.model, summary.window)) != expected_configs or len(summary) != 17:
            raise ValueError("Missing or duplicated preset configurations")
        summary.insert(0, "Year", year)
        yearly.append(summary)
        for file, collection in [("environment_metrics.csv",envs),("weather_increment_comparisons.csv",comparisons)]:
            f = pd.read_csv(directory/file)
            f.insert(0,"Year",year)
            collection.append(f)
        frozen = json.loads(freeze_path.read_text())
        selections.append({"Year":year, **frozen["recommended_by_validation"]})
        sources[str(freeze_path)] = digest(freeze_path)
        rule = json.loads((directory/"earliest_sufficient_windows.json").read_text())
        for model, value in rule["models"].items():
            rules.append({"Year":year,"model":model,**value})
        frames = {}
        for model, horizon in sorted(expected_configs):
            path = directory/"predictions"/f"{model}_W{horizon}_evaluated.csv"
            frame = pd.read_csv(path)
            if set(frame.Year) != {year} or frame.sample_id.duplicated().any():
                raise ValueError("Wrong year or duplicate predictions")
            frames[(model,horizon)] = frame
            for kind, column in [("hybrid","Hybrid"),("site_code","Experiment_Code")]:
                seen = frame[column].isin(set(training[column]))
                for flag, label in [(True,"seen"),(False,"unseen")]:
                    part = frame.loc[seen==flag]
                    metrics = evaluate_predictions(part)[0] if len(part) else {
                        "n":0,"n_environments":0, **{f"macro_{m}":np.nan for m in METRICS}}
                    groups.append({"Year":year,"model":model,"window":horizon,
                                   "group_type":kind,"group":label,**metrics})
        canonical = frames[("CatBoost",0)]
        for frame in frames.values():
            # Membership check also guards contrasts across model families.
            columns = ["sample_id","Env","Hybrid","Year","Experiment_Code","y_true"]
            if not frame[columns].equals(canonical[columns]):
                raise ValueError("Configurations scored different sample order, cohort, metadata or observed targets")
        coverage.append({"Year":year,"n_scored":len(canonical),"n_environments":canonical.Env.nunique(),
                         "n_site_codes":canonical.Experiment_Code.nunique(),"n_hybrids":canonical.Hybrid.nunique(),
                         "n_training_samples":len(training),"training_last_year":training.Year.max(),
                         "evaluation_role":"original_frozen_2024" if year==2024 else "supplementary_retrospective"})
        for model in CANDIDATES:
            reference = frames[(model,0)]
            for horizon in HORIZONS[1:]:
                candidate = frames[(model,horizon)]
                paired = paired_environment_rows(reference,candidate).assign(model=model,window=horizon)
                deltas.append(paired)
                for kind, column in [("hybrid","Hybrid"),("site_code","Experiment_Code")]:
                    seen = reference[column].isin(set(training[column]))
                    for flag,label in [(True,"seen"),(False,"unseen")]:
                        a = reference.loc[seen==flag]
                        b = candidate.loc[candidate.sample_id.isin(set(a.sample_id))]
                        if len(a):
                            subgroup_deltas.append(paired_environment_rows(a,b).assign(
                                model=model,window=horizon,group_type=kind,group=label))
    differences = pd.concat(deltas,ignore_index=True)
    group_differences = pd.concat(subgroup_deltas,ignore_index=True)
    annual_stats = [{"Year":y,"model":m,"window":w,**within_year_stability(g)}
                    for (y,m,w),g in differences.groupby(["Year","model","window"])]
    multi = [{"model":m,"window":w,**cross_year_stability(g)}
             for (m,w),g in differences.groupby(["model","window"])]
    group_annual = [{"Year":y,"model":m,"window":w,"group_type":k,"group":s,**within_year_stability(g)}
                    for (y,m,w,k,s),g in group_differences.groupby(["Year","model","window","group_type","group"])]
    group_multi = [{"model":m,"window":w,"group_type":k,"group":s,**cross_year_stability(g)}
                   for (m,w,k,s),g in group_differences.groupby(["model","window","group_type","group"])]
    # Keep unsupported subgroup-year cells explicitly empty rather than dropping them.
    complete_index = pd.MultiIndex.from_product([years,CANDIDATES,HORIZONS[1:],["hybrid","site_code"],["seen","unseen"]],
                                                names=["Year","model","window","group_type","group"])
    ga = pd.DataFrame(group_annual).set_index(list(complete_index.names)).reindex(complete_index).reset_index()
    for col in ("n_samples","n_environments","n_site_codes"):
        ga[col] = ga[col].fillna(0).astype(int)
    ga["interval_status"] = ga.interval_status.fillna("empty_subgroup")
    cross_index = pd.MultiIndex.from_product([CANDIDATES,HORIZONS[1:],["hybrid","site_code"],["seen","unseen"]],
                                            names=["model","window","group_type","group"])
    gm = pd.DataFrame(group_multi).set_index(list(cross_index.names)).reindex(cross_index).reset_index()
    for col in ("n_years","n_environment_years","n_site_codes"):
        gm[col] = gm[col].fillna(0).astype(int)
    gm["evaluated_years"] = gm.evaluated_years.fillna("")
    gm["missing_prespecified_years"] = gm.missing_prespecified_years.fillna(";".join(map(str,years)))
    gm["interpretation"] = gm.interpretation.fillna("empty_subgroup")
    gm["bootstrap_status"] = gm.bootstrap_status.fillna("empty_subgroup")
    tables = {"coverage":pd.DataFrame(coverage), "yearly_metrics":pd.concat(yearly,ignore_index=True),
              "environment_metrics":pd.concat(envs,ignore_index=True),
              "yearly_weather_comparisons":pd.concat(comparisons,ignore_index=True),
              "yearly_subgroup_metrics":pd.DataFrame(groups), "environment_weather_deltas":differences,
              "yearly_stability":pd.DataFrame(annual_stats), "cross_year_stability":pd.DataFrame(multi),
              "subgroup_environment_weather_deltas":group_differences,
              "yearly_subgroup_stability":ga,"cross_year_subgroup_stability":gm,
              "validation_selections":pd.DataFrame(selections), "yearly_sufficient_windows":pd.DataFrame(rules)}
    for name,frame in tables.items():
        frame.to_csv(out/f"{name}.csv",index=False)
    manifest = {"completed_at_utc":now(),"analysis_type":"supplementary_retrospective",
                "years":list(years),"source_hashes":sources,"n_scored_observations":int(tables["coverage"].n_scored.sum()),
                "n_environment_years":int(tables["coverage"].n_environments.sum()),
                "table_rows":{name:len(f) for name,f in tables.items()},
                "artifacts":{str(p.relative_to(root)):digest(p) for p in sorted(out.glob("*.csv"))},
                "artifact_path_base":str(root),"bootstrap_repeats":REPEATS,
                "caveats":["Only five years; training sets overlap; intervals conditional on fitted pipelines.",
                           "Site code is a proxy, not verified physical location; no spatial holdout.",
                           "Historical cohort is selected for target availability and common calendar coverage.",
                           "All models/windows shown; secondary comparisons unadjusted for multiplicity.",
                           "Original 2024 results already inspected before this extension was designed."]}
    write_json(manifest,out/"analysis_manifest.json")
    print(json.dumps({"analysis":str(out),"rows":manifest["table_rows"],
                      "n_scored":manifest["n_scored_observations"],"environments":manifest["n_environment_years"]}),flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    analyze(parser.parse_args().root)
