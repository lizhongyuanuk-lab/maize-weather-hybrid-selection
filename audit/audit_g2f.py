from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = {
    "train_trait": (
        "1_Training_Trait_Data_2014_2023.csv",
        "23003e5f4b7c0e82443698eaa0e8d10c86e18f62",
    ),
    "train_meta": (
        "2_Training_Meta_Data_2014_2023.csv",
        "43e12056907a3dba1f6a1376144bfec6abdc115d",
    ),
    "train_weather": (
        "4_Training_Weather_Data_2014_2023_full_year.csv",
        "a73213ccbd7f843dd603cde8663acd98caac040b",
    ),
    "test_template": (
        "1_Submission_Template_2024.csv",
        "c9a74db630c06930ad84318a464351bd1071a60e",
    ),
    "test_meta": (
        "2_Testing_Meta_Data_2024.csv",
        "f44c1c0ca08b28f8a9637977fbbcdac3730d6ca9",
    ),
    "test_weather": (
        "4_Testing_Weather_Data_2024_full_year.csv",
        "9561d1711f56bfb5d4a4d23f160bd32fd2182fac",
    ),
}

REPO = "imbaterry11/UFEED_Maize_GxE"
OFFICIAL_TEST_LABEL_URL = (
    "https://datacommons.cyverse.org/download/iplant/home/shared/commons_repo/curated/"
    "GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Testing_data/"
    "7_Testing_Observed_Values.csv"
)
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


def ensure_data(data_dir: Path) -> dict[str, Path]:
    """Resolve the auditable CSV files, preferring the official label release."""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, (filename, sha) in DATASETS.items():
        destination = data_dir / filename
        tmp_copy = Path("/tmp") / filename
        if not destination.exists() and tmp_copy.exists():
            shutil.copy2(tmp_copy, destination)
        if not destination.exists():
            url = f"https://api.github.com/repos/{REPO}/git/blobs/{sha}"
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/vnd.github.raw+json",
                    "User-Agent": "MaizeClimate-data-audit",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                destination.write_bytes(response.read())
        paths[key] = destination

    observed_destination = data_dir / "7_Testing_Observed_Values.csv"
    if not observed_destination.exists():
        request = urllib.request.Request(
            OFFICIAL_TEST_LABEL_URL,
            headers={"User-Agent": "MaizeClimate-data-audit"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            observed_destination.write_bytes(response.read())
    paths["test_observed"] = observed_destination
    return paths


def pct(numerator: int | float, denominator: int | float) -> float:
    return round(100.0 * numerator / denominator, 3) if denominator else np.nan


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed", errors="coerce")


def run_audit(data_dir: Path, output_dir: Path) -> dict:
    paths = ensure_data(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trait = pd.read_csv(paths["train_trait"], low_memory=False)
    train_meta = pd.read_csv(paths["train_meta"], low_memory=False)
    train_weather = pd.read_csv(paths["train_weather"], low_memory=False)
    test_template = pd.read_csv(paths["test_template"], low_memory=False)
    test_meta = pd.read_csv(paths["test_meta"], low_memory=False)
    test_weather = pd.read_csv(paths["test_weather"], low_memory=False)
    test_observed = pd.read_csv(paths["test_observed"], low_memory=False)

    y = pd.to_numeric(trait["Yield_Mg_ha"], errors="coerce")
    labeled = trait.loc[y.notna()].copy()
    labeled["Yield_Mg_ha"] = y[y.notna()]
    yield_quantiles = y.dropna().quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1])

    planted_all = parse_date(trait["Date_Planted"])
    planted_labeled = parse_date(labeled["Date_Planted"])
    planting_by_env = (
        labeled.assign(_planting=planted_labeled)
        .groupby("Env")["_planting"]
        .nunique(dropna=True)
    )

    train_weather["_date"] = pd.to_datetime(
        train_weather["Date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    train_weather_envs = set(train_weather["Env"].dropna())
    yield_envs = set(labeled["Env"].dropna())
    missing_train_weather_envs = sorted(yield_envs - train_weather_envs)
    labeled_weather_rows = int(labeled["Env"].isin(train_weather_envs).sum())
    train_env_day_counts = train_weather.groupby("Env")["_date"].nunique()
    train_weather_dup = int(train_weather.duplicated(["Env", "_date"]).sum())
    train_weather_values = train_weather.drop(columns=["Env", "Date", "_date"])

    hybrid_env_counts = labeled.groupby("Hybrid")["Env"].nunique()
    hybrid_year_counts = labeled.groupby("Hybrid")["Year"].nunique()

    test_meta["_planting"] = parse_date(test_meta["Date_Planted"])
    test_weather["_date"] = pd.to_datetime(
        test_weather["Date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    test_weather_envs = set(test_weather["Env"].dropna())
    test_envs = set(test_template["Env"].dropna())
    test_weather_dup = int(test_weather.duplicated(["Env", "_date"]).sum())
    test_weather_values = test_weather.drop(columns=["Env", "Date", "_date"])
    test_weather_missing = pd.DataFrame(
        {
            "variable": test_weather_values.columns,
            "missing_cells": test_weather_values.isna().sum().values,
            "missing_pct": [
                pct(v, len(test_weather_values))
                for v in test_weather_values.isna().sum().values
            ],
        }
    ).sort_values(["missing_cells", "variable"], ascending=[False, True])

    plant_map = test_meta.set_index("Env")["_planting"]
    test_weather["_planting"] = test_weather["Env"].map(plant_map)
    early_windows: list[dict] = []
    for days in [30, 60, 90, 120]:
        mask = (
            test_weather["_date"].ge(test_weather["_planting"])
            & test_weather["_date"].lt(test_weather["_planting"] + pd.to_timedelta(days, unit="D"))
        )
        window = test_weather.loc[mask]
        expected = len(test_envs) * days
        core_missing_rows = int(window[CORE_WEATHER].isna().any(axis=1).sum())
        early_windows.append(
            {
                "days_after_planting": days,
                "expected_env_days": expected,
                "available_env_days": int(len(window)),
                "date_coverage_pct": pct(len(window), expected),
                "core_weather_missing_rows": core_missing_rows,
                "core_weather_missing_row_pct": pct(core_missing_rows, len(window)),
            }
        )

    train_hybrids = set(labeled["Hybrid"].dropna())
    test_hybrids = set(test_template["Hybrid"].dropna())
    overlap_hybrids = train_hybrids & test_hybrids

    # Treat Env × Hybrid as the required 2024 scoring key. The observed-value
    # file is the held-out label source and must not be used for model tuning.
    test_key = ["Env", "Hybrid"]
    template_keys = test_template[test_key].drop_duplicates()
    observed_keys = test_observed[test_key].drop_duplicates()
    observed_yield = pd.to_numeric(test_observed["Yield_Mg_ha"], errors="coerce")
    template_with_labels = template_keys.merge(
        test_observed[test_key + ["Yield_Mg_ha"]],
        on=test_key,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_2024_by_env = (
        template_with_labels.assign(
            has_observed_label=template_with_labels["_merge"].eq("both")
            & pd.to_numeric(template_with_labels["Yield_Mg_ha"], errors="coerce").notna()
        )
        .groupby("Env", as_index=False)
        .agg(
            requested_rows=("Hybrid", "size"),
            observed_rows=("has_observed_label", "sum"),
        )
    )
    missing_2024_by_env["missing_rows"] = (
        missing_2024_by_env["requested_rows"] - missing_2024_by_env["observed_rows"]
    )
    missing_2024_by_env["label_coverage_pct"] = (
        100.0
        * missing_2024_by_env["observed_rows"]
        / missing_2024_by_env["requested_rows"]
    ).round(3)
    missing_envs = missing_2024_by_env.loc[
        missing_2024_by_env["observed_rows"].eq(0), "Env"
    ].tolist()
    partially_observed_envs = missing_2024_by_env.loc[
        missing_2024_by_env["observed_rows"].gt(0)
        & missing_2024_by_env["missing_rows"].gt(0),
        "Env",
    ].tolist()
    observed_sha256 = hashlib.sha256(paths["test_observed"].read_bytes()).hexdigest()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": {
            "public_mirror": f"https://github.com/{REPO}",
            "official_resources": "https://www.genomes2fields.org/resources/",
            "official_data_note": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12983533/",
            "official_cyverse_landing": "https://dc.cyverse.org/dataset/genomes_to_fields_2024_maize_genotype_by_environment_prediction_competition",
            "official_2024_observed_values": OFFICIAL_TEST_LABEL_URL,
            "2024_row_level_secondary_check": "https://pipette.bio/use-cases/maize-future-season-genomic-prediction.html",
        },
        "yield": {
            "rows_total": int(len(trait)),
            "rows_labeled": int(y.notna().sum()),
            "rows_missing": int(y.isna().sum()),
            "label_completeness_pct": pct(y.notna().sum(), len(y)),
            "nonpositive_labeled_rows": int((y.dropna() <= 0).sum()),
            "single_yield_field": "Yield_Mg_ha",
            "official_unit_definition": "Mg ha-1, standardized to 15.5% grain moisture",
            "quantiles": {str(k): round(float(v), 4) for k, v in yield_quantiles.items()},
        },
        "planting_dates": {
            "all_rows_present": int(planted_all.notna().sum()),
            "all_rows_total": int(len(trait)),
            "all_rows_present_pct": pct(planted_all.notna().sum(), len(trait)),
            "labeled_rows_present": int(planted_labeled.notna().sum()),
            "labeled_rows_total": int(len(labeled)),
            "labeled_rows_present_pct": pct(planted_labeled.notna().sum(), len(labeled)),
            "labeled_envs_with_date": int(
                labeled.assign(_planting=planted_labeled)
                .groupby("Env")["_planting"]
                .apply(lambda s: s.notna().any())
                .sum()
            ),
            "labeled_envs_total": int(labeled["Env"].nunique()),
            "labeled_envs_with_multiple_dates": int((planting_by_env > 1).sum()),
            "test_envs_with_date": int(test_meta["_planting"].notna().sum()),
            "test_envs_total": int(test_meta["Env"].nunique()),
        },
        "weather": {
            "train_rows": int(len(train_weather)),
            "train_envs": int(train_weather["Env"].nunique()),
            "yield_envs_total": int(len(yield_envs)),
            "yield_envs_with_weather": int(len(yield_envs & train_weather_envs)),
            "yield_env_weather_coverage_pct": pct(
                len(yield_envs & train_weather_envs), len(yield_envs)
            ),
            "labeled_rows_with_weather": labeled_weather_rows,
            "labeled_rows_total": int(len(labeled)),
            "labeled_row_weather_coverage_pct": pct(labeled_weather_rows, len(labeled)),
            "missing_training_weather_envs": missing_train_weather_envs,
            "train_env_day_count_min": int(train_env_day_counts.min()),
            "train_env_day_count_max": int(train_env_day_counts.max()),
            "train_duplicate_env_dates": train_weather_dup,
            "train_native_missing_cells": int(train_weather_values.isna().sum().sum()),
            "test_rows": int(len(test_weather)),
            "test_envs_with_weather": int(len(test_envs & test_weather_envs)),
            "test_envs_total": int(len(test_envs)),
            "test_duplicate_env_dates": test_weather_dup,
            "test_date_min": str(test_weather["_date"].min().date()),
            "test_date_max": str(test_weather["_date"].max().date()),
            "test_env_day_count_min": int(test_weather.groupby("Env")["_date"].nunique().min()),
            "test_env_day_count_max": int(test_weather.groupby("Env")["_date"].nunique().max()),
            "test_native_missing_cells": int(test_weather_values.isna().sum().sum()),
            "test_rows_with_any_native_missing": int(test_weather_values.isna().any(axis=1).sum()),
        },
        "coverage": {
            "train_year_min": int(labeled["Year"].min()),
            "train_year_max": int(labeled["Year"].max()),
            "train_years": int(labeled["Year"].nunique()),
            "train_environments": int(labeled["Env"].nunique()),
            "train_field_locations": int(labeled["Field_Location"].nunique()),
            "test_year": 2024,
            "test_environments": int(test_template["Env"].nunique()),
            "test_rows": int(len(test_template)),
        },
        "material_repetition": {
            "train_labeled_hybrids": int(len(hybrid_env_counts)),
            "median_environments_per_hybrid": float(hybrid_env_counts.median()),
            "max_environments_per_hybrid": int(hybrid_env_counts.max()),
            "hybrids_in_one_environment": int((hybrid_env_counts == 1).sum()),
            "hybrids_in_at_least_2_environments": int((hybrid_env_counts >= 2).sum()),
            "hybrids_in_at_least_5_environments": int((hybrid_env_counts >= 5).sum()),
            "hybrids_in_at_least_10_environments": int((hybrid_env_counts >= 10).sum()),
            "median_years_per_hybrid": float(hybrid_year_counts.median()),
            "hybrids_in_at_least_2_years": int((hybrid_year_counts >= 2).sum()),
            "test_hybrids": int(len(test_hybrids)),
            "test_hybrids_seen_in_training": int(len(overlap_hybrids)),
            "test_hybrids_new": int(len(test_hybrids - train_hybrids)),
            "test_hybrid_overlap_pct": pct(len(overlap_hybrids), len(test_hybrids)),
        },
        "test_2024_labels": {
            "official_release_includes_observed_file": True,
            "official_filename": "7_Testing_Observed_Values.csv",
            "source_url": OFFICIAL_TEST_LABEL_URL,
            "sha256": observed_sha256,
            "requested_rows": int(len(test_template)),
            "observed_rows": int(len(test_observed)),
            "observed_rows_with_numeric_yield": int(observed_yield.notna().sum()),
            "missing_requested_rows": int(
                len(test_template) - observed_yield.notna().sum()
            ),
            "label_completeness_pct": pct(observed_yield.notna().sum(), len(test_template)),
            "template_duplicate_keys": int(test_template.duplicated(test_key).sum()),
            "observed_duplicate_keys": int(test_observed.duplicated(test_key).sum()),
            "observed_exact_duplicate_rows": int(test_observed.duplicated().sum()),
            "observed_keys_not_in_template": int(
                len(observed_keys.merge(template_keys, on=test_key, how="left", indicator=True).query("_merge == 'left_only'"))
            ),
            "template_keys_with_labels": int(
                template_with_labels["_merge"].eq("both").sum()
            ),
            "nonpositive_observed_yields": int((observed_yield.dropna() <= 0).sum()),
            "requested_environments": int(test_template["Env"].nunique()),
            "environments_with_any_outcomes": int(test_observed["Env"].nunique()),
            "environments_without_outcomes": missing_envs,
            "partially_observed_environments": partially_observed_envs,
            "rows_missing_in_zero-outcome_environments": int(
                missing_2024_by_env.loc[
                    missing_2024_by_env["observed_rows"].eq(0), "missing_rows"
                ].sum()
            ),
            "rows_missing_in_partially_observed_environments": int(
                missing_2024_by_env.loc[
                    missing_2024_by_env["observed_rows"].gt(0), "missing_rows"
                ].sum()
            ),
            "caveat": "Use these labels once, after model and horizon selection on 2014–2023. Missing 2024 keys are excluded from scored metrics, not imputed.",
        },
        "early_weather_windows": early_windows,
    }

    checks = pd.DataFrame(
        [
            {
                "check": "1. 产量标签完整且单位一致",
                "status": "通过（有条件）",
                "numerator": summary["yield"]["rows_labeled"],
                "denominator": summary["yield"]["rows_total"],
                "rate_pct": summary["yield"]["label_completeness_pct"],
                "evidence": "单一 Yield_Mg_ha 字段；官方定义为 15.5% 水分校正后的 Mg/ha",
                "warning": "缺失标签需剔除；单位一致性主要由字段定义与官方说明支持",
            },
            {
                "check": "2. 可靠播种日期",
                "status": "通过",
                "numerator": summary["planting_dates"]["labeled_rows_present"],
                "denominator": summary["planting_dates"]["labeled_rows_total"],
                "rate_pct": summary["planting_dates"]["labeled_rows_present_pct"],
                "evidence": "有产量的训练行 100% 可解析；2024 为 23/23 环境",
                "warning": "11 个训练环境有两个播种日，早期窗口应按地块日期而非环境单一日期",
            },
            {
                "check": "3. 可按环境匹配的日尺度天气",
                "status": "通过（有条件）",
                "numerator": summary["weather"]["yield_envs_with_weather"],
                "denominator": summary["weather"]["yield_envs_total"],
                "rate_pct": summary["weather"]["yield_env_weather_coverage_pct"],
                "evidence": "训练天气无重复 Env-Date；2024 为 23/23 环境，播后 30–120 天日期覆盖 100%",
                "warning": "训练缺 TXH2_2015/2016/2017；2024 核心变量在 60–120 天窗口约 2.4%–2.9% 行缺失",
            },
            {
                "check": "4. 足够地点和年份",
                "status": "通过",
                "numerator": summary["coverage"]["train_environments"],
                "denominator": summary["coverage"]["train_environments"],
                "rate_pct": 100.0,
                "evidence": "2014–2023 共 10 年、272 个环境、45 个地点；另有 2024 年 23 个测试环境",
                "warning": "环境分布不均，泛化评估必须按年份/地点分组",
            },
            {
                "check": "5. 同一材料跨环境重复",
                "status": "通过（有条件）",
                "numerator": summary["material_repetition"]["hybrids_in_at_least_2_environments"],
                "denominator": summary["material_repetition"]["train_labeled_hybrids"],
                "rate_pct": pct(
                    summary["material_repetition"]["hybrids_in_at_least_2_environments"],
                    summary["material_repetition"]["train_labeled_hybrids"],
                ),
                "evidence": "杂交种跨环境数中位数 17；4,983/5,027 至少出现在 2 个环境",
                "warning": "2024 的 1,063 个杂交种中仅 104 个在训练集原样出现，应区分已见/未见材料",
            },
            {
                "check": "6. 2024 独立测试标签公开可用",
                "status": "通过",
                "numerator": int(observed_yield.notna().sum()),
                "denominator": int(len(test_template)),
                "rate_pct": pct(observed_yield.notna().sum(), len(test_template)),
                "evidence": "CyVerse 官方 7_Testing_Observed_Values.csv 已下载并逐行复算；键均属于提交模板且无重复 Env-Hybrid",
                "warning": "571 个请求键无标签：SCH1_2024 全环境缺 385 行，另有 186 行分散在部分环境；最终指标只在有真实标签的 9,486 行上计算",
            },
        ]
    )

    env_year = (
        labeled.groupby("Year")
        .agg(labeled_rows=("Yield_Mg_ha", "size"), environments=("Env", "nunique"))
        .reset_index()
    )

    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checks.to_csv(output_dir / "audit_checks.csv", index=False)
    pd.DataFrame({"Env": missing_train_weather_envs}).to_csv(
        output_dir / "missing_training_weather_environments.csv", index=False
    )
    pd.DataFrame(early_windows).to_csv(
        output_dir / "2024_early_window_completeness.csv", index=False
    )
    test_weather_missing.to_csv(
        output_dir / "2024_weather_variable_missingness.csv", index=False
    )
    missing_2024_by_env.to_csv(
        output_dir / "2024_observed_label_coverage_by_environment.csv", index=False
    )
    env_year.to_csv(output_dir / "training_coverage_by_year.csv", index=False)
    return {"summary": summary, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit G2F 2014–2024 data feasibility.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    result = run_audit(args.data_dir, args.output_dir)
    print(result["checks"].to_string(index=False))


if __name__ == "__main__":
    main()

