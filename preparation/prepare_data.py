"""Stage 1: auditable sample assembly, without fitted transforms or model training.

Run from any directory: python preparation/prepare_data.py
Raw files are read-only inputs. Only --output-dir receives derived files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
WINDOWS = (30, 60, 90, 120)
KEY = ["Env", "Hybrid"]
CORE_WEATHER = ["PS", "RH2M", "T2MWET", "T2M", "WS2M", "QV2M",
                "T2M_MIN", "T2MDEW", "T2M_MAX", "PRECTOTCORR"]
LAT = "Weather_Station_Latitude (in decimal numbers NOT DMS)"
LON = "Weather_Station_Longitude (in decimal numbers NOT DMS)"
FILES = {
    "trait": "1_Training_Trait_Data_2014_2023.csv",
    "train_meta": "2_Training_Meta_Data_2014_2023.csv",
    "test_meta": "2_Testing_Meta_Data_2024.csv",
    "train_weather": "4_Training_Weather_Data_2014_2023_full_year.csv",
    "test_weather": "4_Testing_Weather_Data_2024_full_year.csv",
    "genotype": "5_Genotype_Data_All_2014_2025_Hybrids_numerical.csv",
    "template": "1_Submission_Template_2024.csv",
    "observed": "7_Testing_Observed_Values.csv",
}
SAMPLE_COLUMNS = [
    "sample_id", "Env", "Hybrid", "Year", "Experiment_Code", "Latitude", "Longitude",
    "Planting_DOY", "Planting_Date_Min", "Planting_Date_Max",
]
FORBIDDEN = {
    "Yield_Mg_ha", "Date_Harvested", "Stand_Count_plants", "Pollen_DAP_days",
    "Silk_DAP_days", "Plant_Height_cm", "Ear_Height_cm", "Root_Lodging_plants",
    "Stalk_Lodging_plants", "Grain_Moisture", "Twt_kg_m3", "yield_sd",
    "yield_min", "yield_max", "labeled_plot_count", "has_observed_label",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    require(len(header) == len(set(header)), f"Duplicate column names: {path.name}")
    return pd.read_csv(path, low_memory=False,
                       dtype={"Env": "string", "Hybrid": "string", "Geno": "string"})


def normalize_ids(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = result[column].astype("string").str.strip().replace("", pd.NA)
    return result


def unique(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    require(not frame[columns].isna().any().any(), f"Missing {name} keys")
    require(not frame.duplicated(columns).any(), f"Duplicate {name} keys: {columns}")


def sample_ids(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        ["s_" + hashlib.sha256((str(e) + "\0" + str(h)).encode()).hexdigest()[:24]
         for e, h in frame[KEY].itertuples(index=False, name=None)], index=frame.index,
    )


def assign_split(year: pd.Series) -> pd.Series:
    require(year.notna().all() and year.between(2014, 2024).all(), "Unexpected sample year")
    return pd.Series(np.select([year.le(2022), year.eq(2023)],
                              ["train", "validation"], default="test"), index=year.index)


def resolve_planting_dates(trait: pd.DataFrame) -> pd.DataFrame:
    """Use pre-outcome records only; never choose dates based on yield availability."""
    result = trait.copy()
    original = result["Date_Planted"]
    dates = pd.to_datetime(original, format="mixed", errors="coerce").dt.normalize()
    result["Planting_Date"] = dates
    result["planting_source"] = np.where(dates.notna(), "record", "unresolved")
    result["planting_parse_failed"] = original.notna() & dates.isna()
    for columns, label in [(KEY, "unique_env_hybrid"), (["Env"], "unique_environment")]:
        groups = result.groupby(columns, dropna=False)["Planting_Date"]
        counts, first = groups.transform("nunique"), groups.transform("first")
        use = result["Planting_Date"].isna() & counts.eq(1)
        result.loc[use, "Planting_Date"] = first[use]
        result.loc[use, "planting_source"] = label
    return result


def aggregate_traits(trait: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Targets use observed yields; planting weights use ALL input plot records."""
    rows = resolve_planting_dates(normalize_ids(trait, KEY))
    rows["source_record"] = np.arange(1, len(rows) + 1)
    rows["Year"] = pd.to_numeric(rows["Year"], errors="coerce")
    require(rows["Year"].notna().all() and rows["Year"].between(2014, 2023).all(),
            "Training years must be 2014–2023")
    require(rows[KEY].notna().all().all(), "Missing trait key: requires explicit source repair")
    env_year = pd.to_numeric(rows["Env"].str.extract(r"_(\d{4})$", expand=False))
    require(env_year.eq(rows["Year"]).all(), "Trait Env/year mismatch")
    known = rows["Planting_Date"].notna()
    require(rows.loc[known, "Planting_Date"].dt.year.eq(rows.loc[known, "Year"]).all(),
            "Planting date outside sample year")
    raw_y = pd.to_numeric(rows["Yield_Mg_ha"], errors="coerce")
    valid_y = raw_y.notna() & np.isfinite(raw_y) & raw_y.gt(0)
    rows["target_status"] = np.where(valid_y, "observed",
        np.where(rows["Yield_Mg_ha"].isna(), "missing", "invalid"))
    rows["_y"] = raw_y.where(valid_y)
    rows["_doy"] = rows["Planting_Date"].dt.dayofyear
    agg = rows.groupby(KEY, as_index=False, sort=True).agg(
        Year=("Year", "first"), raw_plot_count=("source_record", "size"),
        labeled_plot_count=("_y", "count"), Yield_Mg_ha=("_y", "mean"),
        yield_sd=("_y", "std"), yield_min=("_y", "min"), yield_max=("_y", "max"),
        Planting_Date_Count=("Planting_Date", "nunique"),
        Planting_Date_Min=("Planting_Date", "min"), Planting_Date_Max=("Planting_Date", "max"),
        Planting_DOY=("_doy", "mean"),
    )
    missing_dates = rows["Planting_Date"].isna().groupby([rows[k] for k in KEY]).sum()
    agg = agg.merge(missing_dates.rename("unresolved_planting_rows").reset_index(),
                    on=KEY, validate="one_to_one")
    agg["Year"] = agg["Year"].astype(int)
    agg["sample_id"] = sample_ids(agg)
    rows = rows.merge(agg[KEY + ["sample_id", "raw_plot_count"]], on=KEY, validate="many_to_one")
    bridge = rows.loc[rows["Planting_Date"].notna()].groupby(
        ["sample_id", "Env", "Planting_Date"], as_index=False, sort=True,
    ).agg(plot_count=("source_record", "size"), total_plots=("raw_plot_count", "first"),
          labeled_at_date=("_y", "count"))
    bridge["planting_weight"] = bridge["plot_count"] / bridge["total_plots"]
    total_labels = bridge.groupby("sample_id")["labeled_at_date"].transform("sum")
    bridge["observed_label_date_weight"] = bridge["labeled_at_date"] / total_labels.replace(0, np.nan)
    bridge["weight_mismatch"] = total_labels.gt(0) & ~np.isclose(
        bridge["planting_weight"], bridge["observed_label_date_weight"], equal_nan=False)
    mismatched = bridge.groupby("sample_id")["weight_mismatch"].any()
    agg["planting_weight_target_mismatch"] = agg["sample_id"].map(mismatched).fillna(False).astype(bool)
    membership = rows[["source_record", "sample_id", "Env", "Hybrid", "Planting_Date",
                       "planting_source", "planting_parse_failed", "target_status"]]
    return agg, bridge, membership


def prepare_metadata(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = pd.concat([train, test], ignore_index=True)
    result = normalize_ids(result, ["Env", "Experiment_Code"])
    unique(result, ["Env"], "metadata")
    result["Year"] = pd.to_numeric(result["Year"], errors="raise").astype(int)
    require(result["Env"].str.extract(r"_(\d{4})$", expand=False).astype(int)
            .eq(result["Year"]).all(), "Metadata Env/year mismatch")
    issues = []
    for original, renamed, bound in [(LAT, "Latitude", 90), (LON, "Longitude", 180)]:
        numeric = pd.to_numeric(result[original], errors="coerce")
        valid = numeric.notna() & np.isfinite(numeric) & numeric.between(-bound, bound)
        bad = result[original].notna() & ~valid
        for idx in result.index[bad]:
            issues.append({"Env": result.at[idx, "Env"], "field": original,
                           "raw_value": str(result.at[idx, original]), "action": "set_missing"})
        result[renamed] = numeric.where(valid)
    return (result[["Env", "Year", "Experiment_Code", "Latitude", "Longitude"]],
            pd.DataFrame(issues, columns=["Env", "field", "raw_value", "action"]))


def prepare_weather(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require(set(train.columns) == set(test.columns), "Weather schema differs between periods")
    columns = sorted(set(train.columns) - {"Env", "Date"})
    require(set(CORE_WEATHER).issubset(columns), "Required core weather variables absent")
    result = normalize_ids(pd.concat([train, test], ignore_index=True), ["Env"])
    result["Date"] = pd.to_datetime(result["Date"].astype(str), format="%Y%m%d", errors="raise")
    unique(result, ["Env", "Date"], "weather")
    require(result["Date"].dt.year.eq(result["Env"].str.extract(r"_(\d{4})$", expand=False)
            .astype(int)).all(), "Weather date outside environment year")
    profile = []
    for column in columns:
        old = result[column]
        numeric = pd.to_numeric(old, errors="coerce")
        require(not (old.notna() & numeric.isna()).any(), f"Non-numeric weather: {column}")
        sentinel = numeric.isin([-999.0, -9999.0])
        nonfinite = numeric.notna() & ~np.isfinite(numeric)
        numeric = numeric.mask(sentinel | nonfinite)
        result[column] = numeric
        profile.append({"variable": column, "native_missing": int(old.isna().sum()),
                        "sentinel_to_missing": int(sentinel.sum()),
                        "nonfinite_to_missing": int(nonfinite.sum()),
                        "missing_after_normalization": int(numeric.isna().sum())})
    return result[["Env", "Date"] + columns].sort_values(["Env", "Date"]), pd.DataFrame(profile)


def prepare_genotypes(frame: pd.DataFrame) -> pd.DataFrame:
    result = normalize_ids(frame, ["Geno"]).rename(columns={"Geno": "Hybrid"})
    unique(result, ["Hybrid"], "genotype")
    columns = list(result.columns.drop("Hybrid"))
    numeric = result[columns].apply(pd.to_numeric, errors="coerce")
    require(not (result[columns].notna() & numeric.isna()).any().any(), "Non-numeric SNP value")
    require(not np.isinf(numeric.to_numpy()).any(), "Infinite SNP value")
    result[columns] = numeric
    return result.sort_values("Hybrid")


def window_coverage(bridge: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Calendar coverage only: never interpolate from outside a declared window."""
    variables = list(weather.columns.drop(["Env", "Date"]))
    groups = {env: part.set_index("Date")[variables] for env, part in weather.groupby("Env")}
    results = []
    for env, date in bridge[["Env", "Planting_Date"]].drop_duplicates().sort_values(
            ["Env", "Planting_Date"]).itertuples(index=False, name=None):
        daily = groups.get(env, pd.DataFrame(columns=variables, index=pd.DatetimeIndex([])))
        for horizon in WINDOWS:
            expected = pd.date_range(date, periods=horizon, freq="D")
            window = daily.reindex(expected)
            present = int(expected.isin(daily.index).sum())
            row = {"Env": env, "Planting_Date": date, "horizon_days": horizon,
                   "start_inclusive": date, "end_exclusive": date + pd.Timedelta(days=horizon),
                   "expected_days": horizon, "present_days": present,
                   "missing_calendar_days": horizon - present,
                   "calendar_complete": present == horizon,
                   "core_missing_days": int(window[CORE_WEATHER].isna().any(axis=1).sum())}
            row.update({f"missing_{col}": int(window[col].isna().sum()) for col in variables})
            results.append(row)
    return pd.DataFrame(results)


def assemble_registry(samples: pd.DataFrame, metadata: pd.DataFrame,
                      genotypes: pd.DataFrame, bridge: pd.DataFrame,
                      coverage: pd.DataFrame) -> pd.DataFrame:
    unique(samples, KEY, "sample")
    unique(samples, ["sample_id"], "sample ID")
    result = samples.merge(metadata.rename(columns={"Year": "metadata_year"}),
                           on="Env", how="left", validate="many_to_one", indicator=True)
    result["metadata_available"] = result["_merge"].eq("both")
    require(result.loc[result["metadata_available"], "Year"].eq(
        result.loc[result["metadata_available"], "metadata_year"]).all(), "Joined year mismatch")
    result = result.drop(columns="_merge")
    result["split"] = assign_split(result["Year"])
    result["genotype_available"] = result["Hybrid"].isin(genotypes["Hybrid"])
    pair_complete = coverage.groupby(["Env", "Planting_Date"], as_index=False).agg(
        all_windows_complete=("calendar_complete", "all"), window_count=("horizon_days", "size"))
    require(pair_complete["window_count"].eq(len(WINDOWS)).all(), "Missing coverage windows")
    linked = bridge.merge(pair_complete, on=["Env", "Planting_Date"], how="left", validate="many_to_one")
    sample_complete = linked.groupby("sample_id")["all_windows_complete"].all()
    result["weather_calendar_complete"] = result["sample_id"].map(sample_complete).fillna(False).astype(bool)
    result["planting_complete"] = result["unresolved_planting_rows"].eq(0) & result["Planting_Date_Count"].gt(0)
    # Selection does NOT depend on 2024 outcomes or their availability.
    gates = {
        "no_observed_yield": result["split"].eq("test") | result["labeled_plot_count"].gt(0),
        "missing_genotype": result["genotype_available"],
        "missing_metadata": result["metadata_available"] & result["Experiment_Code"].notna(),
        "unresolved_planting_date": result["planting_complete"],
        "planting_target_weight_mismatch": ~result["planting_weight_target_mismatch"],
        "incomplete_weather_calendar": result["weather_calendar_complete"],
    }
    for reason, passed in gates.items():
        result[f"exclude_{reason}"] = ~passed
    reason_cols = [f"exclude_{reason}" for reason in gates]
    result["eligible_common"] = ~result[reason_cols].any(axis=1)
    result["exclusion_reasons"] = result[reason_cols].apply(
        lambda row: "|".join(c.removeprefix("exclude_") for c, value in row.items() if value), axis=1)
    result["first_exclusion_reason"] = result["exclusion_reasons"].str.split("|").str[0]
    return result.sort_values(["Year", "Env", "Hybrid"])


def validate_labels(observed: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    labels = normalize_ids(observed, KEY)
    unique(labels, KEY, "2024 label")
    labels["Yield_Mg_ha"] = pd.to_numeric(labels["Yield_Mg_ha"], errors="raise")
    require(labels["Yield_Mg_ha"].notna().all() and np.isfinite(labels["Yield_Mg_ha"]).all()
            and labels["Yield_Mg_ha"].gt(0).all(), "Invalid official 2024 label")
    joined = labels.merge(registry.loc[registry["split"].eq("test"),
        KEY + ["sample_id"]], on=KEY, how="left", validate="one_to_one")
    require(joined["sample_id"].notna().all(), "Official label outside test template")
    return joined[["sample_id"] + KEY + ["Yield_Mg_ha"]].sort_values(KEY)


def save_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0} if destination.suffix == ".gz" else None
    frame.to_csv(destination, index=False, date_format="%Y-%m-%d", compression=compression)


def write_json(data: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build(raw_dir: Path, output: Path, official_dir: Path | None = None) -> dict:
    raw_dir, output = raw_dir.resolve(), output.resolve()
    require(output != raw_dir and raw_dir not in output.parents,
            "Derived output must not be inside the raw data directory")
    source_paths = {key: (official_dir / filename if official_dir and (official_dir / filename).exists()
                         else raw_dir / filename) for key, filename in FILES.items()}
    mirror_hashes = {key: sha256(raw_dir / filename) for key, filename in FILES.items()}
    source_hashes = {key: {"file": filename, "bytes": source_paths[key].stat().st_size,
                          "sha256": sha256(source_paths[key]),
                          "source_kind": "official_override" if source_paths[key] != raw_dir / filename
                          else ("official_cyverse" if key == "observed" else "public_mirror"),
                          "original_local_sha256": mirror_hashes[key]}
                     for key, filename in FILES.items()}
    raw = {key: read_csv(path) for key, path in source_paths.items()}
    require(not raw["trait"].duplicated().any(), "Exact duplicate trait records require adjudication")
    train, train_bridge, membership = aggregate_traits(raw["trait"])
    metadata, coordinate_issues = prepare_metadata(raw["train_meta"], raw["test_meta"])
    weather, weather_profile = prepare_weather(raw["train_weather"], raw["test_weather"])
    genotypes = prepare_genotypes(raw["genotype"])
    template = normalize_ids(raw["template"], KEY)
    unique(template, KEY, "test template")
    require(template["Yield_Mg_ha"].isna().all(), "Test template unexpectedly contains yield values")
    test = template[KEY].copy()
    test["sample_id"] = sample_ids(test)
    test["Year"] = 2024
    test_dates = normalize_ids(raw["test_meta"], ["Env"])[["Env", "Date_Planted"]].copy()
    test_dates["Planting_Date_Min"] = pd.to_datetime(test_dates["Date_Planted"], format="mixed", errors="coerce")
    test = test.merge(test_dates[["Env", "Planting_Date_Min"]], on="Env", how="left", validate="many_to_one")
    require(test["Env"].str.endswith("_2024").all(), "Non-2024 test environment")
    require(test.loc[test["Planting_Date_Min"].notna(), "Planting_Date_Min"].dt.year.eq(2024).all(),
            "Test planting date outside 2024")
    test["Planting_Date_Max"] = test["Planting_Date_Min"]
    test["Planting_DOY"] = test["Planting_Date_Min"].dt.dayofyear
    test["Planting_Date_Count"] = test["Planting_Date_Min"].notna().astype(int)
    test["unresolved_planting_rows"] = test["Planting_Date_Min"].isna().astype(int)
    test["planting_weight_target_mismatch"] = False  # Test labels never determine input eligibility.
    for column in ["raw_plot_count", "labeled_plot_count", "Yield_Mg_ha", "yield_sd", "yield_min", "yield_max"]:
        test[column] = np.nan
    test_bridge = test.loc[test["Planting_Date_Min"].notna(), ["sample_id", "Env", "Planting_Date_Min"]]
    test_bridge = test_bridge.rename(columns={"Planting_Date_Min": "Planting_Date"}).copy()
    test_bridge["plot_count"] = pd.NA  # Test release does not specify replicate counts.
    test_bridge["total_plots"] = pd.NA
    test_bridge["planting_weight"] = 1.0
    bridge_columns = ["sample_id", "Env", "Planting_Date", "plot_count", "total_plots", "planting_weight"]
    bridge = pd.concat([train_bridge[bridge_columns], test_bridge], ignore_index=True)
    coverage = window_coverage(bridge, weather)
    registry = assemble_registry(pd.concat([train, test], ignore_index=True), metadata,
                                 genotypes, bridge, coverage)
    eligible = registry.loc[registry["eligible_common"]].copy()
    selected_bridge = bridge.loc[bridge["sample_id"].isin(eligible["sample_id"])].sort_values(
        ["Env", "sample_id", "Planting_Date"])
    weights = selected_bridge.groupby("sample_id")["planting_weight"].sum()
    require(np.isclose(weights.to_numpy(), 1.0).all(), "Planting weights do not sum to one")
    require(len(weights) == len(eligible), "Eligible sample without planting link")
    unique(selected_bridge, ["sample_id", "Planting_Date"], "planting bridge")

    # This is the ONLY point at which held-out outcomes are read into derived data.
    # They do not enter any eligibility gate, model input, or fitted preprocessing.
    labels = validate_labels(raw["observed"], registry)
    test_registry = registry.loc[registry["split"].eq("test")]
    test_coverage = test_registry[["sample_id", "Env", "Hybrid", "eligible_common"]].copy()
    test_coverage["has_observed_label"] = test_coverage["sample_id"].isin(labels["sample_id"])
    test_coverage["evaluable"] = test_coverage["eligible_common"] & test_coverage["has_observed_label"]

    for split in ["train", "validation", "test"]:
        part = eligible.loc[eligible["split"].eq(split)]
        features = part[SAMPLE_COLUMNS]
        require(not (FORBIDDEN & set(features.columns)), "Forbidden field in model-facing sample table")
        unique(features, KEY, f"{split} output")
        save_csv(features, output / f"samples/{split}.csv")
        if split != "test":
            save_csv(part[["sample_id", "Yield_Mg_ha"]], output / f"targets/{split}.csv")
    save_csv(labels, output / "holdout/test_targets.csv")
    save_csv(selected_bridge, output / "links/sample_plantings.csv.gz")
    used_pairs = selected_bridge[["Env", "Planting_Date"]].drop_duplicates()
    selected_windows = coverage.merge(used_pairs, on=["Env", "Planting_Date"], validate="many_to_one")
    save_csv(selected_windows, output / "links/weather_windows.csv")
    used_envs = set(eligible["Env"])
    save_csv(metadata.loc[metadata["Env"].isin(used_envs)].sort_values(["Year", "Env"]),
             output / "dimensions/environments.csv")
    save_csv(genotypes.loc[genotypes["Hybrid"].isin(eligible["Hybrid"])],
             output / "dimensions/genotypes.csv.gz")
    save_csv(weather.loc[weather["Env"].isin(used_envs)], output / "dimensions/weather_daily.csv.gz")
    save_csv(registry, output / "audit/sample_registry.csv.gz")
    save_csv(registry.loc[~registry["eligible_common"]], output / "audit/excluded_samples.csv")
    save_csv(membership, output / "audit/plot_membership.csv.gz")
    save_csv(membership.loc[membership["target_status"].ne("observed")],
             output / "audit/unusable_plot_labels.csv")
    save_csv(train[["sample_id"] + KEY + ["raw_plot_count", "labeled_plot_count", "yield_sd", "yield_min", "yield_max"]],
             output / "audit/target_diagnostics.csv.gz")
    mismatch_ids = set(train.loc[train["planting_weight_target_mismatch"], "sample_id"])
    save_csv(train_bridge.loc[train_bridge["sample_id"].isin(mismatch_ids)],
             output / "audit/planting_target_weight_mismatch.csv")
    save_csv(test_coverage, output / "audit/test_label_coverage.csv")
    save_csv(coordinate_issues, output / "audit/coordinate_issues.csv")
    save_csv(weather_profile, output / "audit/weather_missingness.csv")
    save_csv(coverage, output / "audit/all_weather_windows.csv")

    counts = []
    reason_columns = [c for c in registry if c.startswith("exclude_")]
    flow = []
    for split in ["train", "validation", "test"]:
        all_part = registry.loc[registry["split"].eq(split)]
        part = all_part.loc[all_part["eligible_common"]]
        counts.append({"split": split, "first_year": int(all_part["Year"].min()),
                       "last_year": int(all_part["Year"].max()), "candidate_combinations": len(all_part),
                       "retained_combinations": len(part), "excluded_combinations": len(all_part)-len(part),
                       "environments": int(part["Env"].nunique()), "hybrids": int(part["Hybrid"].nunique()),
                       "multi_planting_date_combinations": int(part["Planting_Date_Count"].gt(1).sum()),
                       "rows_with_missing_coordinates": int(part[["Latitude", "Longitude"]].isna().any(axis=1).sum())})
        remaining = pd.Series(True, index=all_part.index)
        for column in reason_columns:
            removed = remaining & all_part[column]
            remaining &= ~all_part[column]
            flow.append({"split": split, "gate": column.removeprefix("exclude_"),
                         "removed_at_gate": int(removed.sum()), "remaining": int(remaining.sum()),
                         "fails_gate_in_total": int(all_part[column].sum())})
    save_csv(pd.DataFrame(counts), output / "audit/split_summary.csv")
    save_csv(pd.DataFrame(flow), output / "audit/cohort_flow.csv")
    for key, filename in FILES.items():
        require(sha256(raw_dir / filename) == mirror_hashes[key], "Raw input changed during preparation")
        require(sha256(source_paths[key]) == source_hashes[key]["sha256"], "Selected source changed during preparation")

    contract = {
        "stage": "1: sample assembly only", "grain": ["Env", "Hybrid"],
        "splits": {"train": [2014, 2022], "validation": [2023, 2023], "test": [2024, 2024]},
        "windows_days": list(WINDOWS), "weather_bounds": "[planting_date, planting_date + horizon)",
        "target": "Arithmetic mean of finite positive observed plot yields; missing targets are never imputed.",
        "input_aggregation": "Planting weights use all source plots, independently of yield availability.",
        "label_exposure_alignment": "Exclude development combinations when the planting-date distribution of labeled plots differs from that of all plots; preserve them in diagnostics for sensitivity analysis. Test selection never uses outcome availability.",
        "planting_resolution": ["original record", "unique known date in Env-Hybrid", "unique known date in Env", "exclude if unresolved"],
        "paired_cohort": "All W0/W30/W60/W90/W120 use the exact same samples per split.",
        "static_predictor_allowlist": ["Experiment_Code", "Latitude", "Longitude", "Planting_DOY"],
        "identifier_columns_not_predictors": ["sample_id", "Env", "Hybrid", "Year", "Planting_Date_Min", "Planting_Date_Max"],
        "forbidden_predictors": sorted(FORBIDDEN),
        "weather_predictors": "Not yet constructed. weather_daily contains full raw dates; ALWAYS join through weather_windows and enforce its bounds before imputation or aggregation.",
        "genotype_preprocessing": "Not fitted. Genotype missing values and numeric marker values are retained; fit imputation and PCA on the development training split only.",
        "coordinate_preprocessing": "Missing/invalid coordinates remain missing; no future-year coordinate backfill.",
        "holdout_policy": "holdout/test_targets.csv is for final scoring after freezing model selection. Separation is a workflow convention, not an OS access restriction.",
        "diagnostics_policy": "audit files contain outcome-derived diagnostics and MUST NOT be merged into model predictors.",
        "mixed_plantings": "For mixed dates retain all per-date windows and pre-outcome plot weights. The aggregate horizon is available at the latest constituent cutoff; do not relabel it as a single original planting date.",
        "weather_missing_sentinels": [-999, -9999],
    }
    write_json(contract, output / "data_contract.json")
    summary = {
        "stage": "stage1_completed_no_models_trained", "split_summary": counts,
        "source_trait_rows": len(raw["trait"]),
        "observed_plot_labels": int(membership["target_status"].eq("observed").sum()),
        "unusable_plot_labels": int(membership["target_status"].ne("observed").sum()),
        "planting_sources": {str(k): int(v) for k, v in membership["planting_source"].value_counts().items()},
        "test_template_rows": len(template), "test_labels": len(labels),
        "test_evaluable_rows": int(test_coverage["evaluable"].sum()),
        "test_missing_labels": int((~test_coverage["has_observed_label"]).sum()),
        "shared_environment_rows": len(metadata.loc[metadata["Env"].isin(used_envs)]),
        "marker_columns": len(genotypes.columns)-1,
        "retained_genotypes": int(genotypes["Hybrid"].isin(eligible["Hybrid"]).sum()),
        "all_raw_hashes_unchanged": True,
        "checks": {"unique_sample_keys": True, "join_cardinality_validated": True,
                   "calendar_windows_complete_for_retained_samples": bool(selected_windows["calendar_complete"].all()),
                   "planting_weights_sum_to_one": True, "test_selection_independent_of_labels": True,
                   "test_targets_separate": True, "forbidden_fields_absent_from_samples": True},
        "raw_inputs": source_hashes,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "script_sha256": sha256(Path(__file__)),
        "official_source_verification": "../source_verification.json (separate byte-comparison receipt)",
    }
    summary["output_files"] = {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                               for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    write_json(summary, output / "manifest.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT / "audit/data/raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "preparation/outputs")
    parser.add_argument("--official-dir", type=Path, default=PROJECT / "preparation/official_inputs")
    args = parser.parse_args()
    result = build(args.raw_dir, args.output_dir, args.official_dir)
    print(pd.DataFrame(result["split_summary"]).to_string(index=False))
    print(f"Final test labels available: {result['test_evaluable_rows']} (not used for modeling)")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
