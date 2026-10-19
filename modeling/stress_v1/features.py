"""Three prespecified stress features on the frozen W120 cohort.

The weather transformation deliberately mirrors build_features.aggregate_window:
crop first, interpolate only bounded interior gaps, then fill with medians
computed from distinct training environment-date rows in the same horizon.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

MODELING = Path(__file__).resolve().parents[1]
PROJECT = MODELING.parent
sys.path.insert(0, str(MODELING))
import build_features as original

NAMES = ["vpd_proxy_mean_kpa", "vpd_proxy_p90_kpa", "heat_excess35_cday"]
HORIZON = 120


def vpd_proxy(temp_c: np.ndarray, relative_humidity: np.ndarray) -> np.ndarray:
    """Daily-mean atmospheric-demand proxy in kPa, not midday canopy VPD."""
    t, rh = np.asarray(temp_c, float), np.asarray(relative_humidity, float)
    if not np.isfinite(t).all() or not np.isfinite(rh).all() or ((rh < 0) | (rh > 100)).any():
        raise ValueError("Invalid temperature or relative humidity for VPD")
    return 0.6108 * np.exp(17.27 * t / (t + 237.3)) * (1.0 - rh / 100.0)


def pair_features(daily: pd.DataFrame, planting: pd.Timestamp,
                  medians: pd.Series) -> np.ndarray:
    dates = pd.date_range(planting, periods=HORIZON, freq="D")
    raw = daily.reindex(dates)[original.WEATHER].copy()
    clean = raw.interpolate(method="linear", limit_area="inside").fillna(medians)
    if not np.isfinite(clean.to_numpy(float)).all():
        raise ValueError("Nonfinite reconstructed weather")
    dry_air = vpd_proxy(clean.T2M.to_numpy(), clean.RH2M.to_numpy())
    heat = np.maximum(clean.T2M_MAX.to_numpy(float) - 35.0, 0.0)
    values = np.array([dry_air.mean(), np.quantile(dry_air, 0.9), heat.sum()], dtype=np.float32)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid derived stress features")
    return values


def build_extra(samples: dict[str, pd.DataFrame], test_year: int,
                bridge: pd.DataFrame, weather: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict]:
    cohort = pd.concat(samples.values(), ignore_index=True)
    ids = set(cohort.sample_id)
    if cohort.sample_id.isna().any() or cohort.sample_id.duplicated().any():
        raise ValueError("Sample identifiers must be unique and nonmissing")
    links = bridge.loc[bridge.sample_id.isin(ids)].copy()
    if set(links.sample_id) != ids or links.duplicated(["sample_id", "Env", "Planting_Date"]).any():
        raise ValueError("Planting bridge coverage or uniqueness failed")
    links["Planting_Date"] = pd.to_datetime(links.Planting_Date, errors="raise")
    if not np.allclose(links.groupby("sample_id").planting_weight.sum(), 1.0, atol=1e-10, rtol=0):
        raise ValueError("Planting-date weights do not sum to one")
    env_by_id = cohort.set_index("sample_id").Env
    if not (links.Env.to_numpy() == links.sample_id.map(env_by_id).to_numpy()).all():
        raise ValueError("Planting bridge environment mismatch")

    envs = set(cohort.Env)
    available = weather.loc[weather.Env.isin(envs)].copy()
    available["Date"] = pd.to_datetime(available.Date, errors="raise")
    if available.Date.dt.year.gt(test_year).any() or available.duplicated(["Env", "Date"]).any():
        raise ValueError("Future weather or duplicate environment-date rows")
    if set(available.Env) != envs:
        raise ValueError("Weather environment coverage failed")
    daily = {env: frame.set_index("Date")[original.WEATHER].sort_index()
             for env, frame in available.groupby("Env", sort=False)}
    medians, n_training_weather_days = original.training_weather_medians(
        links, daily, set(samples["train"].sample_id), HORIZON)
    if not np.isfinite(medians.to_numpy(float)).all():
        raise ValueError("Training weather median unavailable")

    pairs = links[["Env", "Planting_Date"]].drop_duplicates().sort_values(["Env", "Planting_Date"])
    rows = []
    for env, date in pairs.itertuples(index=False, name=None):
        rows.append((env, date, *pair_features(daily[env], date, medians)))
    by_pair = pd.DataFrame(rows, columns=["Env", "Planting_Date", *NAMES])
    joined = links.merge(by_pair, on=["Env", "Planting_Date"], validate="many_to_one")
    if len(joined) != len(links):
        raise ValueError("Stress feature join changed planting rows")
    for name in NAMES:
        joined[name] *= joined.planting_weight
    mixed = joined.groupby("sample_id", sort=False)[NAMES].sum()
    extras = {}
    for split, sample in samples.items():
        values = mixed.reindex(sample.sample_id).to_numpy(np.float32)
        if values.shape != (len(sample), len(NAMES)) or not np.isfinite(values).all():
            raise ValueError(f"Invalid stress feature array for {split}")
        extras[split] = values
    metadata = {"test_year": test_year, "horizon_days": HORIZON,
                "feature_names": NAMES, "n_environment_planting_pairs": len(by_pair),
                "n_training_weather_days": n_training_weather_days,
                "training_medians": {k: float(v) for k, v in medians.items()},
                "value_ranges": {n: [float(by_pair[n].min()), float(by_pair[n].max())] for n in NAMES},
                "daily_vpd_is_proxy": True, "targets_read": False}
    return extras, metadata
