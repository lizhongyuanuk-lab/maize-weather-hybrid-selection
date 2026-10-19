from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw"
OUTPUT = ROOT / "outputs" / "project_risk_profile.json"

CORE_WEATHER = [
    "PS",
    "RH2M",
    "T2MWET",
    "T2M",
    "WS2M",
    "QV2M",
    "T2M_MIN",
    "T2MDEW",
    "T2M_MAX",
    "PRECTOTCORR",
]


def pct(n: int | float, d: int | float) -> float:
    return round(100 * n / d, 3) if d else float("nan")


def as_int_dict(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.items()}


def main() -> None:
    trait = pd.read_csv(DATA / "1_Training_Trait_Data_2014_2023.csv", low_memory=False)
    train_meta = pd.read_csv(DATA / "2_Training_Meta_Data_2014_2023.csv", low_memory=False)
    train_weather = pd.read_csv(
        DATA / "4_Training_Weather_Data_2014_2023_full_year.csv", low_memory=False
    )
    test = pd.read_csv(DATA / "1_Submission_Template_2024.csv", low_memory=False)
    test_meta = pd.read_csv(DATA / "2_Testing_Meta_Data_2024.csv", low_memory=False)
    test_weather = pd.read_csv(
        DATA / "4_Testing_Weather_Data_2024_full_year.csv", low_memory=False
    )
    genotype = pd.read_csv(
        DATA / "5_Genotype_Data_All_2014_2025_Hybrids_numerical.csv",
        low_memory=False,
        na_values=["NA"],
    )
    train_soil = pd.read_csv(DATA / "3_Training_Soil_Data_2015_2023.csv", low_memory=False)
    test_soil = pd.read_csv(DATA / "3_Testing_Soil_Data_2024.csv", low_memory=False)

    trait["Yield_Mg_ha"] = pd.to_numeric(trait["Yield_Mg_ha"], errors="coerce")
    labeled = trait.loc[trait["Yield_Mg_ha"].notna()].copy()

    # Grain: the public training trait file is plot-level, while the submission is
    # one prediction per Env-Hybrid. Quantify how much aggregation is required.
    train_key_counts = labeled.groupby(["Env", "Hybrid"], dropna=False).size()
    test_key_counts = test.groupby(["Env", "Hybrid"], dropna=False).size()
    within_key_std = labeled.groupby(["Env", "Hybrid"])["Yield_Mg_ha"].std().dropna()

    # Date validity and whether any nominated early window would extend beyond harvest.
    planted = pd.to_datetime(labeled["Date_Planted"], format="mixed", errors="coerce")
    harvested = pd.to_datetime(labeled["Date_Harvested"], format="mixed", errors="coerce")
    days_to_harvest = (harvested - planted).dt.days

    # Metadata integrity.
    train_meta_envs = set(train_meta["Env"].dropna())
    trait_envs = set(labeled["Env"].dropna())
    test_meta_envs = set(test_meta["Env"].dropna())
    test_envs = set(test["Env"].dropna())

    # Genotype coverage and dimensionality.
    geno_ids = set(genotype["Geno"].dropna())
    train_hybrids = set(labeled["Hybrid"].dropna())
    test_hybrids = set(test["Hybrid"].dropna())
    marker = genotype.drop(columns=["Geno"])
    marker_missing_by_row = marker.isna().mean(axis=1)
    marker_unique = marker.nunique(dropna=True)

    # Soil coverage. It starts in 2015 and remains sparse by environment and field.
    train_soil_envs = set(train_soil["Env"].dropna())
    test_soil_envs = set(test_soil["Env"].dropna())
    train_soil_numeric = train_soil.drop(
        columns=[c for c in ["Year", "Env", "LabID", "Date Received", "Date Reported", "Texture", "Comments"] if c in train_soil.columns]
    ).apply(pd.to_numeric, errors="coerce")
    test_soil_numeric = test_soil.drop(
        columns=[c for c in ["Year", "Env", "LabID", "Date Received", "Date Reported", "Texture", "Comments"] if c in test_soil.columns]
    ).apply(pd.to_numeric, errors="coerce")

    # Weather window coverage at the actual Env-planting-date grain. This catches
    # environments with multiple planting dates that an Env-only join would miss.
    train_weather["_date"] = pd.to_datetime(
        train_weather["Date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    test_weather["_date"] = pd.to_datetime(
        test_weather["Date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    planting_pairs = (
        labeled.assign(PlantingDate=planted)[["Env", "PlantingDate"]]
        .drop_duplicates()
        .dropna()
    )
    train_weather_by_env = {
        env: frame.set_index("_date") for env, frame in train_weather.groupby("Env")
    }
    window_checks: dict[str, dict] = {}
    for days in [30, 60, 90, 120]:
        complete_pairs = 0
        pairs_with_any_missing = 0
        unavailable_pairs = 0
        for row in planting_pairs.itertuples(index=False):
            frame = train_weather_by_env.get(row.Env)
            if frame is None:
                unavailable_pairs += 1
                continue
            idx = pd.date_range(row.PlantingDate, periods=days, freq="D")
            window = frame.reindex(idx)
            if len(window) == days and window["Env"].notna().all():
                complete_pairs += 1
            else:
                unavailable_pairs += 1
            if window[CORE_WEATHER].isna().any(axis=None):
                pairs_with_any_missing += 1
        window_checks[str(days)] = {
            "env_planting_pairs": int(len(planting_pairs)),
            "complete_date_pairs": complete_pairs,
            "complete_date_pair_pct": pct(complete_pairs, len(planting_pairs)),
            "unavailable_pairs": unavailable_pairs,
            "pairs_with_any_core_weather_missing": pairs_with_any_missing,
            "pairs_with_any_core_weather_missing_pct": pct(
                pairs_with_any_missing, len(planting_pairs)
            ),
        }

    # Test weather coverage uses the one planting date supplied per environment.
    test_meta_dates = test_meta[["Env", "Date_Planted"]].copy()
    test_meta_dates["PlantingDate"] = pd.to_datetime(
        test_meta_dates["Date_Planted"], format="mixed", errors="coerce"
    )
    test_weather_by_env = {
        env: frame.set_index("_date") for env, frame in test_weather.groupby("Env")
    }
    test_window_checks: dict[str, dict] = {}
    for days in [30, 60, 90, 120]:
        complete = 0
        any_missing = 0
        for row in test_meta_dates.itertuples(index=False):
            frame = test_weather_by_env.get(row.Env)
            if frame is None or pd.isna(row.PlantingDate):
                continue
            idx = pd.date_range(row.PlantingDate, periods=days, freq="D")
            window = frame.reindex(idx)
            complete += int(len(window) == days and window["Env"].notna().all())
            any_missing += int(window[CORE_WEATHER].isna().any(axis=None))
        test_window_checks[str(days)] = {
            "environments": int(len(test_meta_dates)),
            "complete_date_environments": complete,
            "environments_with_any_core_weather_missing": any_missing,
        }

    profile = {
        "modeling_grain": {
            "labeled_plot_rows": int(len(labeled)),
            "unique_train_env_hybrid_keys": int(len(train_key_counts)),
            "extra_plot_rows_beyond_env_hybrid_grain": int(len(labeled) - len(train_key_counts)),
            "extra_plot_row_pct": pct(len(labeled) - len(train_key_counts), len(labeled)),
            "train_env_hybrid_replicate_count_quantiles": {
                str(q): float(train_key_counts.quantile(q)) for q in [0, 0.5, 0.9, 0.99, 1]
            },
            "train_env_hybrid_keys_with_replicates": int((train_key_counts > 1).sum()),
            "train_env_hybrid_keys_with_replicates_pct": pct(
                (train_key_counts > 1).sum(), len(train_key_counts)
            ),
            "median_within_key_yield_sd": round(float(within_key_std.median()), 4),
            "test_rows": int(len(test)),
            "unique_test_env_hybrid_keys": int(len(test_key_counts)),
            "duplicate_test_keys": int((test_key_counts > 1).sum()),
        },
        "dates": {
            "labeled_rows_with_planting_date": int(planted.notna().sum()),
            "labeled_rows_with_harvest_date": int(harvested.notna().sum()),
            "days_to_harvest_quantiles": {
                str(q): float(days_to_harvest.dropna().quantile(q))
                for q in [0, 0.01, 0.5, 0.99, 1]
            },
            "rows_harvested_before_day_120": int((days_to_harvest < 120).sum()),
            "rows_harvested_before_day_90": int((days_to_harvest < 90).sum()),
        },
        "metadata_integrity": {
            "train_meta_rows": int(len(train_meta)),
            "train_meta_unique_envs": int(train_meta["Env"].nunique()),
            "train_meta_duplicate_env_rows": int(train_meta.duplicated("Env").sum()),
            "labeled_trait_envs": int(len(trait_envs)),
            "labeled_trait_envs_missing_meta": sorted(trait_envs - train_meta_envs),
            "test_meta_rows": int(len(test_meta)),
            "test_meta_unique_envs": int(test_meta["Env"].nunique()),
            "test_envs_missing_meta": sorted(test_envs - test_meta_envs),
            "train_location_codes": int(train_meta["Experiment_Code"].nunique()),
            "test_location_codes": int(test_meta["Experiment_Code"].nunique()),
            "test_location_codes_seen_in_training": int(
                len(
                    set(test_meta["Experiment_Code"].dropna())
                    & set(train_meta["Experiment_Code"].dropna())
                )
            ),
            "unseen_2024_location_codes": sorted(
                set(test_meta["Experiment_Code"].dropna())
                - set(train_meta["Experiment_Code"].dropna())
            ),
            "train_weather_station_coordinate_completeness_pct": pct(
                train_meta[
                    "Weather_Station_Latitude (in decimal numbers NOT DMS)"
                ].notna().sum(),
                len(train_meta),
            ),
            "test_weather_station_coordinate_completeness_pct": pct(
                test_meta[
                    "Weather_Station_Latitude (in decimal numbers NOT DMS)"
                ].notna().sum(),
                len(test_meta),
            ),
            "train_irrigated_completeness_pct": pct(
                train_meta["Irrigated"].notna().sum(), len(train_meta)
            ),
            "test_irrigated_completeness_pct": pct(
                test_meta["Irrigated"].notna().sum(), len(test_meta)
            ),
        },
        "genotype": {
            "genotype_rows": int(len(genotype)),
            "unique_genotype_ids": int(genotype["Geno"].nunique()),
            "duplicate_genotype_ids": int(genotype.duplicated("Geno").sum()),
            "marker_columns": int(marker.shape[1]),
            "constant_or_all_missing_markers": int((marker_unique <= 1).sum()),
            "overall_marker_missing_pct": round(float(marker.isna().mean().mean() * 100), 3),
            "median_hybrid_marker_missing_pct": round(float(marker_missing_by_row.median() * 100), 3),
            "max_hybrid_marker_missing_pct": round(float(marker_missing_by_row.max() * 100), 3),
            "train_hybrids": int(len(train_hybrids)),
            "train_hybrids_with_genotype": int(len(train_hybrids & geno_ids)),
            "train_hybrid_genotype_coverage_pct": pct(len(train_hybrids & geno_ids), len(train_hybrids)),
            "train_labeled_rows_with_genotype_pct": pct(labeled["Hybrid"].isin(geno_ids).sum(), len(labeled)),
            "test_hybrids": int(len(test_hybrids)),
            "test_hybrids_with_genotype": int(len(test_hybrids & geno_ids)),
            "test_hybrid_genotype_coverage_pct": pct(len(test_hybrids & geno_ids), len(test_hybrids)),
            "test_rows_with_genotype_pct": pct(test["Hybrid"].isin(geno_ids).sum(), len(test)),
            "test_hybrids_without_genotype_sample": sorted(test_hybrids - geno_ids)[:20],
        },
        "soil": {
            "train_soil_rows": int(len(train_soil)),
            "train_soil_unique_envs": int(train_soil["Env"].nunique()),
            "train_soil_duplicate_env_rows": int(train_soil.duplicated("Env").sum()),
            "labeled_env_soil_coverage_pct": pct(len(trait_envs & train_soil_envs), len(trait_envs)),
            "train_soil_numeric_median_missing_pct": round(float(train_soil_numeric.isna().mean().median() * 100), 3),
            "train_soil_numeric_max_missing_pct": round(float(train_soil_numeric.isna().mean().max() * 100), 3),
            "test_soil_rows": int(len(test_soil)),
            "test_soil_unique_envs": int(test_soil["Env"].nunique()),
            "test_soil_duplicate_env_rows": int(test_soil.duplicated("Env").sum()),
            "test_soil_envs_with_multiple_rows": int(
                (test_soil.groupby("Env").size() > 1).sum()
            ),
            "test_env_soil_coverage_pct": pct(len(test_envs & test_soil_envs), len(test_envs)),
            "test_envs_without_soil": sorted(test_envs - test_soil_envs),
            "test_soil_numeric_median_missing_pct": round(float(test_soil_numeric.isna().mean().median() * 100), 3),
            "test_soil_numeric_max_missing_pct": round(float(test_soil_numeric.isna().mean().max() * 100), 3),
        },
        "train_weather_windows": window_checks,
        "test_weather_windows": test_window_checks,
        "known_2024_outcome_gap": {
            "requested_rows": 10057,
            "externally_reported_observed_rows": 9486,
            "externally_reported_missing_rows": 571,
            "environment_without_outcomes": "SCH1_2024",
            "official_observed_values_present_locally": (
                DATA / "7_Testing_Observed_Values.csv"
            ).exists(),
        },
        "post_prediction_columns_to_exclude": [
            "Date_Harvested",
            "Stand_Count_plants",
            "Pollen_DAP_days",
            "Silk_DAP_days",
            "Plant_Height_cm",
            "Ear_Height_cm",
            "Root_Lodging_plants",
            "Stalk_Lodging_plants",
            "Grain_Moisture",
            "Twt_kg_m3",
        ],
    }

    OUTPUT.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
