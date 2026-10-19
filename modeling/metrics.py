"""Environment-equal maize regression metrics and paired uncertainty.

Correlations of constant vectors are undefined, not zero. Top-k boundary ties
receive fractional inclusion weights on *both* observed and predicted ranks.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

SEED = 20261018


def correlation(a, b, ranked=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    if ranked:
        a = pd.Series(a).rank(method="average").to_numpy()
        b = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def fractional_top_k(values, k):
    """Expected inclusion under uniform random resolution of boundary ties."""
    values = np.asarray(values, float)
    if not 1 <= k <= len(values) or not np.isfinite(values).all():
        raise ValueError("Finite values and 1 <= k <= n required")
    cutoff = np.partition(values, len(values)-k)[len(values)-k]
    above, tied = values > cutoff, values == cutoff
    weights = above.astype(float)
    weights[tied] = (k-int(above.sum()))/int(tied.sum())
    return weights


def top20_recall(observed, predicted):
    n = len(observed)
    if n == 0:
        return float("nan")
    k = max(1, math.ceil(n*.2))
    return float(np.dot(fractional_top_k(observed, k), fractional_top_k(predicted, k))/k)


def evaluate_predictions(frame):
    """Return summary dict + one row per environment (not pooled pseudo-reps)."""
    required = {"Env", "y_true", "prediction"}
    if not required.issubset(frame):
        raise ValueError(f"Required columns missing: {required-set(frame)}")
    if len(frame) == 0 or frame.Env.isna().any():
        raise ValueError("Evaluation requires nonempty rows with environments")
    if not np.isfinite(frame[["y_true", "prediction"]].to_numpy(float)).all():
        raise ValueError("Non-finite observed or predicted value")
    rows = []
    for env, group in frame.groupby("Env", sort=True):
        observed, predicted = group.y_true.to_numpy(), group.prediction.to_numpy()
        error = predicted-observed
        rows.append({"Env": env, "n": len(group), "rmse": float(np.sqrt(np.mean(error**2))),
                     "mae": float(np.abs(error).mean()), "bias": float(error.mean()),
                     "pearson": correlation(observed, predicted),
                     "spearman": correlation(observed, predicted, ranked=True),
                     "top20_recall": top20_recall(observed, predicted)})
    environments = pd.DataFrame(rows)
    error = frame.prediction.to_numpy()-frame.y_true.to_numpy()
    summary = {"n": len(frame), "n_environments": len(environments),
               "pooled_rmse": float(np.sqrt(np.mean(error**2))),
               "pooled_mae": float(np.abs(error).mean()), "pooled_bias": float(error.mean())}
    for name in ["rmse", "mae", "bias", "pearson", "spearman", "top20_recall"]:
        summary[f"macro_{name}"] = float(environments[name].mean())
    summary["pearson_defined_environments"] = int(environments.pearson.notna().sum())
    summary["spearman_defined_environments"] = int(environments.spearman.notna().sum())
    return summary, environments


def bootstrap_mean(values, repeats=2000, seed=SEED):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty environment values required")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
    lower, upper = np.quantile(means, [.025, .975])
    return {"estimate": float(values.mean()), "ci95_low": float(lower), "ci95_high": float(upper),
            "n_environments": len(values), "bootstrap_repeats": repeats, "seed": seed}


def paired_difference(reference, candidate, metric="rmse", repeats=2000, seed=SEED):
    """Candidate minus reference. Equal environment identities are mandatory."""
    if reference.Env.duplicated().any() or candidate.Env.duplicated().any():
        raise ValueError("Duplicate environment rows")
    if set(reference.Env) != set(candidate.Env):
        raise ValueError("Paired comparisons require identical environments")
    a = reference.set_index("Env")[metric].sort_index()
    b = candidate.set_index("Env")[metric].reindex(a.index)
    result = bootstrap_mean((b-a).to_numpy(), repeats, seed)
    result["definition"] = "candidate minus reference; negative RMSE difference is better"
    return result


def window_analysis(environment_metrics, repeats=2000, seed=SEED):
    """Operational sufficiency requires >=2% W120 gain and paired CI <0."""
    if set(environment_metrics) != {0, 30, 60, 90, 120}:
        raise ValueError("All five windows required")
    reference = environment_metrics[0]
    e0, e120 = reference.rmse.mean(), environment_metrics[120].rmse.mean()
    full_gain = e0-e120
    full_comparison = paired_difference(reference, environment_metrics[120], repeats=repeats, seed=seed)
    relative_gain = float(full_gain/e0) if e0 > 0 else float("nan")
    supported = bool(full_gain > 0 and relative_gain >= .02 and full_comparison["ci95_high"] < 0)
    rows = []
    earliest = None
    for horizon in [0, 30, 60, 90, 120]:
        metrics = environment_metrics[horizon]
        current = metrics.rmse.mean()
        vs0 = paired_difference(reference, metrics, repeats=repeats, seed=seed)
        vs120 = paired_difference(environment_metrics[120], metrics, repeats=repeats, seed=seed)
        retention = float((e0-current)/full_gain) if full_gain > 0 else float("nan")
        sufficient = bool(supported and horizon > 0 and retention >= .9)
        if sufficient and earliest is None:
            earliest = horizon
        rows.append({"window": horizon, "macro_rmse": float(current),
                     "gain_retention": retention, "full_window_relative_gain": relative_gain,
                     "vs_W0_delta_rmse": vs0["estimate"], "vs_W0_ci95_low": vs0["ci95_low"],
                     "vs_W0_ci95_high": vs0["ci95_high"],
                     "vs_W120_delta_rmse": vs120["estimate"], "vs_W120_ci95_low": vs120["ci95_low"],
                     "vs_W120_ci95_high": vs120["ci95_high"], "sufficient_by_operational_rule": sufficient})
    return pd.DataFrame(rows), {"earliest_sufficient_window": earliest,
                               "W120_gain_supported": supported,
                               "W120_relative_gain": relative_gain,
                               "criterion": ">=2% W120 vs W0 RMSE improvement, paired 95% CI strictly below zero, then >=90% point-estimate gain retention; not proof of noninferiority or a universal agronomic rule"}
