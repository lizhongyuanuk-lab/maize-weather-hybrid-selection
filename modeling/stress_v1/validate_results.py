"""Independent checks of frozen comparison artifacts; never refits models."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
YEARS = range(2020, 2025)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def top20(y: np.ndarray, p: np.ndarray) -> float:
    k = max(1, math.ceil(len(y) * .2))

    def weights(a: np.ndarray) -> np.ndarray:
        cutoff = np.partition(a, len(a) - k)[len(a) - k]
        higher, tied = a > cutoff, a == cutoff
        w = higher.astype(float)
        w[tied] = (k - higher.sum()) / tied.sum()
        return w

    return float(np.dot(weights(y), weights(p)) / k)


def main() -> None:
    primary = json.loads((ROOT / "results/summary.json").read_text())
    gxe = json.loads((ROOT / "gxe_results/summary.json").read_text())
    yearly, env = [], []
    for year in YEARS:
        base = ROOT / "results" / str(year) / "final"
        lock = json.loads((base / "prediction_lock.json").read_text())
        candidate_path = base / "predictions.csv"
        reference_path = (PROJECT / "modeling/results/final/predictions/CatBoost_W120.csv"
                          if year == 2024 else PROJECT / f"modeling/rolling_v1/results/{year}/final/predictions/CatBoost_W120.csv")
        assert sha256(candidate_path) == lock["prediction_sha256"]
        assert sha256(reference_path) == lock["reference_prediction_sha256"]
        candidate = pd.read_csv(candidate_path)
        reference = pd.read_csv(reference_path)
        assert candidate.sample_id.is_unique and reference.sample_id.is_unique
        assert set(candidate.sample_id) == set(reference.sample_id)
        assert (candidate.Year == year).all()
        annual = json.loads((base / "annual_metrics.json").read_text())
        paired = pd.read_csv(base / "paired_environments.csv")
        assert paired.Env.is_unique and int(paired.n_reference.sum()) == annual["n_scored_rows"]
        assert (paired.n_reference == paired.n_candidate).all()
        for role in ("reference", "candidate"):
            for metric in ("rmse", "top20_recall"):
                observed = paired[f"{metric}_{role}"].mean()
                expected = annual[f"{role}_macro_{metric}"]
                assert abs(observed - expected) < 1e-12, (year, role, metric)
        yearly.append(annual)
        env.append(paired)

    annuals = pd.DataFrame(yearly)
    environments = pd.concat(env, ignore_index=True)
    assert not environments.duplicated(["Year", "Env"]).any()
    assert len(environments) == primary["n_environment_years"] == 129
    assert int(annuals.n_scored_rows.sum()) == primary["n_scored_rows"] == 57673
    for metric in ("rmse", "top20_recall"):
        for role in ("reference", "candidate"):
            observed = annuals[f"{role}_macro_{metric}"].mean()
            expected = primary[metric][f"year_equal_{role}"]
            assert abs(observed - expected) < 1e-12, (metric, role)
        delta = (annuals[f"candidate_macro_{metric}"] - annuals[f"reference_macro_{metric}"]).mean()
        assert abs(delta - primary[metric]["year_equal_delta"]) < 1e-12

    # Independent raw-row check for the previously held-out 2024 target file.
    c = pd.read_csv(ROOT / "results/2024/final/predictions.csv")
    b = pd.read_csv(PROJECT / "modeling/results/final/predictions/CatBoost_W120.csv")
    t = pd.read_csv(PROJECT / "preparation/outputs/holdout/test_targets.csv")
    raw = c[["sample_id", "Env", "prediction"]].merge(
        b[["sample_id", "prediction"]], on="sample_id", suffixes=("_candidate", "_reference"), validate="one_to_one")
    raw = raw.merge(t[["sample_id", "Yield_Mg_ha"]], on="sample_id", validate="one_to_one")
    reported = pd.read_csv(ROOT / "results/2024/final/paired_environments.csv").set_index("Env")
    max_rmse_error = max_top20_error = 0.0
    for env_id, group in raw.groupby("Env"):
        y = group.Yield_Mg_ha.to_numpy(float)
        for role in ("candidate", "reference"):
            p = group[f"prediction_{role}"].to_numpy(float)
            rmse = float(np.sqrt(np.mean((p-y)**2)))
            recall = top20(y, p)
            max_rmse_error = max(max_rmse_error, abs(rmse - reported.loc[env_id, f"rmse_{role}"]))
            max_top20_error = max(max_top20_error, abs(recall - reported.loc[env_id, f"top20_recall_{role}"]))
    assert max_rmse_error < 1e-12 and max_top20_error < 1e-12

    # The exploratory GxE model is separate; verify its reported direction.
    gxe_annuals = pd.read_csv(ROOT / "gxe_results/yearly_metrics.csv")
    assert len(gxe_annuals) == 5
    for year in YEARS:
        folder = ROOT / "gxe_results" / str(year) / "final"
        lock = json.loads((folder / "prediction_lock.json").read_text())
        for role in ("additive", "gxe"):
            assert sha256(folder / f"{role}_predictions.csv") == lock["prediction_hashes"][role]
    for metric in ("rmse", "top20_recall"):
        assert abs(gxe_annuals[f"delta_{metric}"].mean() - gxe[metric]["delta_year_equal"]) < 1e-12

    result = {"pass": True, "n_scored_rows": int(annuals.n_scored_rows.sum()),
              "n_environment_years": len(environments), "n_years": len(annuals),
              "verified_locked_candidate_and_reference_hashes": True,
              "verified_2024_raw_rmse_and_top20": True,
              "max_2024_rmse_recalculation_error": max_rmse_error,
              "max_2024_top20_recalculation_error": max_top20_error,
              "verified_gxe_prediction_hashes_and_annual_summary": True}
    (ROOT / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
