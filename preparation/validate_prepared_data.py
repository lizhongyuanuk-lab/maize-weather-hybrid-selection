"""Independently reconcile exported Stage 1 tables against raw records.

This validator never trains a model or computes 2024 yield performance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import PROJECT, FILES, FORBIDDEN, SAMPLE_COLUMNS, sha256


def validate() -> dict:
    output = PROJECT / "preparation/outputs"
    manifest = json.loads((output / "manifest.json").read_text())
    checks = []

    def check(name: str, condition: bool, evidence: str) -> None:
        if not condition:
            raise AssertionError(f"{name}: {evidence}")
        checks.append({"check": name, "status": "PASS", "evidence": evidence})

    for key, source in manifest["raw_inputs"].items():
        check(f"raw_hash_{key}", sha256(PROJECT / "audit/data/raw" / FILES[key]) == source["original_local_sha256"],
              "Original input hash matches pre-build fingerprint")
        if source["source_kind"] == "official_override":
            check(f"official_hash_{key}", sha256(PROJECT / "preparation/official_inputs" / FILES[key]) == source["sha256"],
                  "Official override matches recorded fingerprint")
    for relative, entry in manifest["output_files"].items():
        check(f"output_hash_{relative}", sha256(output / relative) == entry["sha256"], "Export hash matches manifest")

    split_tables = {name: pd.read_csv(output / f"samples/{name}.csv") for name in ["train", "validation", "test"]}
    all_samples = pd.concat(split_tables.values(), ignore_index=True)
    check("sample_id_unique", all_samples["sample_id"].is_unique, f"{len(all_samples)} distinct sample IDs")
    check("environment_hybrid_unique", not all_samples.duplicated(["Env", "Hybrid"]).any(), "Env × Hybrid is unique across all splits")
    for name, frame in split_tables.items():
        check(f"schema_{name}", list(frame.columns) == SAMPLE_COLUMNS and not (set(frame.columns) & FORBIDDEN),
              "Model-facing sample table uses the exact permitted schema")
        valid_years = frame["Year"].between(2014, 2022) if name == "train" else frame["Year"].eq(2023 if name == "validation" else 2024)
        check(f"years_{name}", valid_years.all(), f"{len(frame)} rows in declared temporal split")
    env_sets = [set(frame["Env"]) for frame in split_tables.values()]
    check("environment_split_disjoint", all(not env_sets[i] & env_sets[j] for i in range(3) for j in range(i+1, 3)),
          "No environment-year shared between splits; shared hybrids are allowed")

    trait = pd.read_csv(PROJECT / "audit/data/raw" / FILES["trait"], low_memory=False)
    for col in ["Env", "Hybrid"]:
        trait[col] = trait[col].astype("string").str.strip()
    values = pd.to_numeric(trait["Yield_Mg_ha"], errors="coerce")
    trait["_target"] = values.where(values.notna() & np.isfinite(values) & values.gt(0))
    independent_means = trait.groupby(["Env", "Hybrid"], as_index=False)["_target"].mean()
    for name in ["train", "validation"]:
        target = pd.read_csv(output / f"targets/{name}.csv")
        frame = split_tables[name].merge(target, on="sample_id", how="left", validate="one_to_one")
        frame = frame.merge(independent_means, on=["Env", "Hybrid"], how="left", validate="one_to_one")
        check(f"target_alignment_{name}", set(target["sample_id"]) == set(split_tables[name]["sample_id"]), "Targets match sample IDs exactly")
        check(f"raw_target_mean_{name}", np.allclose(frame["Yield_Mg_ha"], frame["_target"], rtol=0, atol=1e-10),
              f"All {len(frame)} exported targets equal independently recomputed raw plot means")

    registry = pd.read_csv(output / "audit/sample_registry.csv.gz", low_memory=False)
    selected = registry.loc[registry["eligible_common"]]
    check("registry_reconciles", set(selected["sample_id"]) == set(all_samples["sample_id"]), "Registry eligibility equals exported sample population")
    check("sample_accounting", len(selected) + len(registry.loc[~registry["eligible_common"]]) == len(registry),
          f"{len(selected)} retained + {len(registry)-len(selected)} excluded = {len(registry)} candidate combinations")
    check("label_exposure_consistency", not selected["planting_weight_target_mismatch"].any(),
          "No retained development combination has different planting-date weights for inputs and observed targets")

    genotype = pd.read_csv(output / "dimensions/genotypes.csv.gz")
    check("genotype_key_unique", genotype["Hybrid"].is_unique, "One genotype row per material")
    check("genotype_join_coverage", set(all_samples["Hybrid"]).issubset(set(genotype["Hybrid"])), "Every retained sample has a matching raw genotype vector")
    check("genotype_dimensions", len(genotype.columns)-1 == manifest["marker_columns"], f"{manifest['marker_columns']} raw markers; no PCA fitted")
    weather = pd.read_csv(output / "dimensions/weather_daily.csv.gz", parse_dates=["Date"])
    check("weather_key_unique", not weather.duplicated(["Env", "Date"]).any(), "Daily weather has unique Env-Date keys")
    bridge = pd.read_csv(output / "links/sample_plantings.csv.gz", parse_dates=["Planting_Date"])
    check("bridge_coverage", set(bridge["sample_id"]) == set(all_samples["sample_id"]), "Every retained sample is linked to its planting dates")
    check("bridge_weights", np.allclose(bridge.groupby("sample_id")["planting_weight"].sum(), 1), "Planting weights sum to one for each sample")
    windows = pd.read_csv(output / "links/weather_windows.csv", parse_dates=["Planting_Date", "start_inclusive", "end_exclusive"])
    check("window_lengths", (windows["end_exclusive"]-windows["start_inclusive"]).dt.days.eq(windows["horizon_days"]).all(),
          "All intervals have the specified left-closed/right-open length")
    check("window_start", windows["start_inclusive"].eq(windows["Planting_Date"]).all(), "Every weather window starts on its actual planting date")
    pair_windows = windows.groupby(["Env", "Planting_Date"])["horizon_days"].agg(set)
    check("four_windows_per_planting", pair_windows.map(lambda x: x == {30, 60, 90, 120}).all(), "Every planting pair has exactly the four planned horizons")
    check("window_key_unique", not windows.duplicated(["Env", "Planting_Date", "horizon_days"]).any(), "Window keys are unique")
    dates_by_env = {env: set(g["Date"]) for env, g in weather.groupby("Env")}
    actual_coverage = all(set(pd.date_range(row.start_inclusive, periods=row.horizon_days)).issubset(dates_by_env[row.Env])
                          for row in windows.itertuples())
    check("window_calendar_recomputed", actual_coverage, "Every expected date exists in the exported daily weather table")

    template = pd.read_csv(PROJECT / "audit/data/raw" / FILES["template"])
    test = split_tables["test"]
    check("test_template_retained", set(zip(test["Env"], test["Hybrid"])) == set(zip(template["Env"], template["Hybrid"])),
          "All 10,057 template inputs remain, including combinations without labels")
    labels = pd.read_csv(output / "holdout/test_targets.csv")
    official = pd.read_csv(PROJECT / "audit/data/raw" / FILES["observed"])
    check("test_label_keys_preserved", set(zip(labels["Env"], labels["Hybrid"])) == set(zip(official["Env"], official["Hybrid"])),
          "Official observed keys preserved; no outcome distribution or model performance was examined")
    check("test_labels_separate", not (output / "targets/test.csv").exists() and labels["sample_id"].is_unique,
          "Test labels exist only in the explicitly separate holdout file, not in model-facing inputs")
    official_meta = pd.read_csv(PROJECT / "preparation/official_inputs" / FILES["test_meta"])
    station_cols = {"Latitude": "Weather_Station_Latitude (in decimal numbers NOT DMS)",
                    "Longitude": "Weather_Station_Longitude (in decimal numbers NOT DMS)"}
    for clean, original in station_cols.items():
        reference = pd.to_numeric(official_meta.set_index("Env")[original], errors="coerce")
        check(f"official_test_{clean}", np.allclose(test[clean], test["Env"].map(reference), equal_nan=True),
              "Test coordinates match official values, including missing values; mirror supplements are not used")

    result = {"status": "PASS", "checks_passed": len(checks), "checks": checks,
              "scope": "Stage 1 schema, targets, identity, joins, time bounds, provenance fingerprints; no models trained.",
              "source_limit": "Full official-vs-mirror comparison remains incomplete for four large input files; see source_verification.json."}
    path = PROJECT / "preparation/validation_report.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(checks)} export reconciliation checks")
    print(path)
    return result


if __name__ == "__main__":
    validate()
