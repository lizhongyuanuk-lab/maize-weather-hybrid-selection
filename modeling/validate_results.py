"""Independent audit of completed experiments, without importing scoring helpers.

Development audits only development targets. Final refuses to read holdout
targets or evaluated prediction CSVs until final_evaluation_manifest.json exists.
No fitting, prediction generation, or Stage 1 rebuild is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
PREP = PROJECT / "preparation/outputs"
HORIZONS = (0, 30, 60, 90, 120)
MODELS = ("Ridge", "ExtraTrees", "CatBoost")
BASELINES = ("GlobalMean", "HistoricalSiteMean")
SEED = 20261018
REPEATS = 2000
KEY_COLUMNS = ["sample_id", "Env", "Hybrid", "Year", "Experiment_Code"]
EXPECTED_PAIRS = {(model, horizon) for model in MODELS for horizon in HORIZONS}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def instant(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


class AuditFailure(ValueError):
    pass


class Audit:
    def __init__(self, stage):
        self.stage = stage
        self.checks = []
        self.notes = []
        self.file_hashes = {}
        self.holdout_read = False

    def require(self, condition, name, detail=None):
        record = {"name": name, "passed": bool(condition)}
        if detail is not None:
            record["detail"] = detail
        self.checks.append(record)
        if not condition:
            raise AuditFailure(name + (f": {detail}" if detail else ""))

    def close(self, actual, expected, name):
        actual = np.asarray(actual, dtype=float)
        expected = np.asarray(expected, dtype=float)
        valid = actual.shape == expected.shape and np.allclose(
            actual, expected, rtol=1e-9, atol=1e-9, equal_nan=True)
        self.require(valid, name)

    def hashes(self, expected, name):
        for relative, expected_hash in expected.items():
            path = PROJECT / relative
            self.require(path.is_file(), f"{name}: file exists: {relative}")
            # All audits run after completion; avoid hashing identical artifacts twice.
            if str(path) not in self.file_hashes:
                self.file_hashes[str(path)] = digest(path)
            self.require(self.file_hashes[str(path)] == expected_hash,
                         f"{name}: SHA-256: {relative}")


def independent_correlation(observed, predicted, ranks=False):
    a = np.asarray(observed, float)
    b = np.asarray(predicted, float)
    if len(a) < 2 or np.max(a) == np.min(a) or np.max(b) == np.min(b):
        return np.nan
    if ranks:
        a = pd.Series(a).rank(method="average").to_numpy()
        b = pd.Series(b).rank(method="average").to_numpy()
    a = a - a.mean()
    b = b - b.mean()
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


def independent_top20(observed, predicted):
    """Construct tied inclusion by descending distinct-value groups, not partition."""
    count = len(observed)
    k = math.ceil(count / 5)

    def weights(values):
        values = np.asarray(values, float)
        result = np.zeros(len(values))
        remaining = k
        for value in np.unique(values)[::-1]:
            group = values == value
            slots = int(np.sum(group))
            result[group] = min(1.0, remaining / slots)
            remaining -= min(remaining, slots)
            if remaining == 0:
                break
        return result

    return float(np.sum(weights(observed) * weights(predicted)) / k)


def independent_metrics(frame):
    """Direct residual arithmetic and independently implemented rank metrics."""
    env_rows = []
    for env in sorted(frame.Env.unique()):
        part = frame.loc[frame.Env == env]
        observed = part.y_true.to_numpy(float)
        predicted = part.prediction.to_numpy(float)
        error = predicted - observed
        n = len(part)
        env_rows.append({
            "Env": env, "n": n, "rmse": float(math.sqrt(np.dot(error, error) / n)),
            "mae": float(np.sum(np.abs(error)) / n), "bias": float(np.sum(error) / n),
            "pearson": independent_correlation(observed, predicted),
            "spearman": independent_correlation(observed, predicted, ranks=True),
            "top20_recall": independent_top20(observed, predicted),
        })
    envs = pd.DataFrame(env_rows)
    residuals = frame.prediction.to_numpy(float) - frame.y_true.to_numpy(float)
    summary = {
        "n": len(frame), "n_environments": len(envs),
        "pooled_rmse": float(math.sqrt(np.dot(residuals, residuals) / len(frame))),
        "pooled_mae": float(np.sum(np.abs(residuals)) / len(frame)),
        "pooled_bias": float(np.sum(residuals) / len(frame)),
    }
    for metric in ("rmse", "mae", "bias", "pearson", "spearman", "top20_recall"):
        values = envs[metric].to_numpy(float)
        defined = values[np.isfinite(values)]
        summary[f"macro_{metric}"] = float(np.sum(defined) / len(defined)) if len(defined) else np.nan
    summary["pearson_defined_environments"] = int(envs.pearson.notna().sum())
    summary["spearman_defined_environments"] = int(envs.spearman.notna().sum())
    return summary, envs


def independent_bootstrap(values):
    """Environment resampling, one replicate at a time with the locked RNG seed."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    replicates = np.empty(REPEATS)
    for repeat in range(REPEATS):
        indices = rng.integers(0, len(values), size=len(values))
        replicates[repeat] = sum(values[indices]) / len(values)
    low, high = np.quantile(replicates, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def add_interval(summary, environments):
    _, low, high = independent_bootstrap(environments.rmse.to_numpy())
    return {**summary, "macro_rmse_ci95_low": low, "macro_rmse_ci95_high": high}


def compare_summary(audit, actual, expected, context):
    for metric, expected_value in expected.items():
        audit.require(metric in actual, f"{context}: contains {metric}")
        actual_value = actual[metric]
        audit.close(np.nan if actual_value is None else actual_value, expected_value,
                    f"{context}: independently recomputed {metric}")


def compare_environments(audit, actual, expected, context):
    audit.require(actual.Env.is_unique, f"{context}: unique environment rows")
    audit.require(set(actual.Env) == set(expected.Env), f"{context}: identical environment coverage")
    actual = actual.set_index("Env").reindex(expected.Env)
    for metric in ("n", "rmse", "mae", "bias", "pearson", "spearman", "top20_recall"):
        audit.close(actual[metric].to_numpy(), expected[metric].to_numpy(), f"{context}: {metric}")


def load_prediction(path, samples, audit, context):
    frame = pd.read_csv(path)
    audit.require(frame.sample_id.is_unique and frame.sample_id.notna().all(), f"{context}: unique sample IDs")
    for column in KEY_COLUMNS:
        audit.require(frame[column].tolist() == samples[column].tolist(),
                      f"{context}: original sample order and {column}")
    audit.require("y_true" not in frame and "Yield_Mg_ha" not in frame,
                  f"{context}: prediction file contains no observed targets")
    audit.require(np.isfinite(frame.prediction.to_numpy(float)).all(), f"{context}: finite predictions")
    return frame


def score_against_labels(predictions, targets, audit, context, allow_unlabeled=False):
    audit.require(targets.sample_id.is_unique, f"{context}: target IDs unique")
    audit.require(set(targets.sample_id).issubset(set(predictions.sample_id)), f"{context}: target IDs belong to prediction cohort")
    merged = predictions.merge(targets[["sample_id", "Yield_Mg_ha"]], on="sample_id",
                               how="left", validate="one_to_one")
    if not allow_unlabeled:
        audit.require(merged.Yield_Mg_ha.notna().all(), f"{context}: all development labels matched")
    scored = merged.loc[merged.Yield_Mg_ha.notna()].rename(columns={"Yield_Mg_ha": "y_true"})
    audit.require(np.isfinite(scored.y_true.to_numpy(float)).all() and len(scored) > 0,
                  f"{context}: valid nonempty scored cohort")
    summary, environments = independent_metrics(scored)
    return scored, add_interval(summary, environments), environments


def baseline_reference(name, train_samples, train_targets, target_samples, audit):
    aligned = train_samples[["sample_id", "Experiment_Code"]].merge(
        train_targets[["sample_id", "Yield_Mg_ha"]], on="sample_id", how="left", validate="one_to_one")
    audit.require(aligned.Yield_Mg_ha.notna().all(), f"{name}: all baseline training labels align")
    global_value = float(aligned.Yield_Mg_ha.sum() / len(aligned))
    if name == "GlobalMean":
        return np.repeat(global_value, len(target_samples))
    grouped = aligned.groupby("Experiment_Code").Yield_Mg_ha.agg(["sum", "count"])
    mapping = (grouped["sum"] / grouped["count"]).to_dict()
    return np.array([mapping.get(site, global_value) for site in target_samples.Experiment_Code])


def verify_feature_inputs(audit, feature_directory, result_directory, stage):
    run = read_json(result_directory / "run_manifest.json")
    audit.require(run["stage"] == stage, f"{stage}: run manifest stage")
    audit.require(run["seed"] == SEED, f"{stage}: fixed seed")
    audit.hashes(run["inputs"], f"{stage}: run inputs")
    manifest = read_json(feature_directory / "feature_manifest.json")
    audit.require(manifest["mode"] == stage, f"{stage}: feature manifest mode")
    audit.hashes({str((feature_directory / name).relative_to(PROJECT)): sha
                  for name, sha in manifest["outputs"].items()}, f"{stage}: feature artifacts")
    samples = {s: pd.read_csv(feature_directory / f"{s}_samples.csv") for s in ("train", "validation", "test")}
    for split, frame in samples.items():
        audit.require(frame.sample_id.is_unique and frame.sample_id.notna().all(), f"{stage}: {split} unique IDs")
        audit.require("y_true" not in frame and "Yield_Mg_ha" not in frame, f"{stage}: {split} has no target")
    audit.require(set(samples["test"].Year) == {2024}, f"{stage}: test is 2024")
    audit.require(set(samples["train"].Year).issubset(set(range(2014, 2023 if stage == "development" else 2024))),
                  f"{stage}: training year boundary")
    audit.require((set(samples["validation"].Year) == {2023}) if stage == "development" else len(samples["validation"]) == 0,
                  f"{stage}: validation year/empty boundary")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        audit.require(not (set(samples[left].sample_id) & set(samples[right].sample_id)),
                      f"{stage}: no {left}/{right} sample overlap")
    for split, frame in samples.items():
        source = pd.read_csv(PREP / f"samples/{split}.csv")
        if stage == "final" and split == "train":
            source = pd.concat([source, pd.read_csv(PREP / "samples/validation.csv")], ignore_index=True)
        if stage == "final" and split == "validation":
            source = source.iloc[:0]
        for column in KEY_COLUMNS:
            audit.require(source[column].tolist() == frame[column].tolist(), f"{stage}: {split} Stage 1 order/{column}")
    return run, samples


def quality_column_profile(feature_directory):
    records = []
    for horizon in HORIZONS[1:]:
        names = read_json(feature_directory / f"W{horizon}_columns.json")["columns"][-9:]
        with np.load(feature_directory / f"W{horizon}.npz", allow_pickle=False) as saved:
            quality = saved["X_train"][:, -9:]
        for index, name in enumerate(names):
            values = quality[:, index]
            records.append({"window": horizon, "feature": name,
                            "train_min": float(values.min()), "train_max": float(values.max()),
                            "constant_in_training": bool(np.ptp(values) == 0)})
    return records


def validate_development(audit, feature_directory, out):
    # Require completed development before doing substantive result reads.
    freeze_path = out / "frozen_selection.json"
    audit.require(freeze_path.is_file(), "development: completed selection freeze exists")
    run, samples = verify_feature_inputs(audit, feature_directory, out, "development")
    frozen = read_json(freeze_path)
    audit.require(not frozen["holdout_labels_read"], "freeze: no final labels used for selection")
    audit.require(frozen["windows"] == list(HORIZONS) and frozen["bootstrap_repeats"] == REPEATS,
                  "freeze: all five windows and 2000 bootstrap replicates")
    audit.require(frozen["development_run_manifest_sha256"] == digest(out / "run_manifest.json"), "freeze: development manifest hash")
    audit.require(frozen["development_input_hashes"] == run["inputs"], "freeze: exact development inputs")
    config_keys = [(c["model"], c["window"]) for c in frozen["configs"]]
    audit.require(len(config_keys) == 15 and set(config_keys) == EXPECTED_PAIRS, "freeze: all 15 unique model/window configurations")
    audit.require(set(frozen["baselines"]) == set(BASELINES), "freeze: both predetermined baselines")
    labels = pd.read_csv(PREP / "targets/validation.csv")
    train_labels = pd.read_csv(PREP / "targets/train.csv")
    selected = pd.read_csv(out / "selected_metrics.csv")
    candidates = pd.read_csv(out / "candidate_metrics.csv")
    all_pairs = EXPECTED_PAIRS | {(name, 0) for name in BASELINES}
    audit.require(len(selected) == 17 and set(zip(selected.model, selected.window)) == all_pairs,
                  "development: all 17 selected summaries")
    audit.require(len(candidates) == 30 and not candidates.duplicated(["model", "window", "candidate"]).any(),
                  "development: exactly 30 unique candidate rows")
    candidate_keys = set(zip(candidates.model, candidates.window, candidates.candidate))
    audit.require(candidate_keys == {(m, h, c) for m, h in EXPECTED_PAIRS for c in (0, 1)},
                  "development: equal two-candidate budget")
    selected_lookup = selected.set_index(["model", "window"])
    independently_chosen = []
    for name in BASELINES:
        frame = load_prediction(out / "predictions" / f"{name}.csv", samples["validation"], audit, name)
        expected = baseline_reference(name, samples["train"], train_labels, samples["validation"], audit)
        audit.close(frame.prediction.to_numpy(), expected, f"{name}: training-only baseline mapping")
        _, summary, envs = score_against_labels(frame, labels, audit, name)
        compare_summary(audit, selected_lookup.loc[(name, 0)], summary, name)
        compare_environments(audit, pd.read_csv(out / f"{name}_environment_metrics.csv"), envs, name)
    for horizon in HORIZONS:
        for model in MODELS:
            context = f"{model}_W{horizon}"
            unit = read_json(out / "units" / f"{context}.json")
            audit.hashes(unit["artifacts"], context)
            scores, predictions = [], []
            for candidate in (0, 1):
                cid = f"{context}_c{candidate}"
                frame = load_prediction(out / "candidate_predictions" / f"{cid}.csv", samples["validation"], audit, cid)
                _, summary, envs = score_against_labels(frame, labels, audit, cid)
                row = candidates.loc[(candidates.model == model) & (candidates.window == horizon) & (candidates.candidate == candidate)].iloc[0]
                compare_summary(audit, row, summary, cid)
                compare_environments(audit, pd.read_csv(out / "candidate_predictions" / f"{cid}_environments.csv"), envs, cid)
                audit.require(json.loads(row.params) == run["candidates"][model][candidate], f"{cid}: preset candidate parameters")
                record = [c for c in unit["candidates"] if c["candidate"] == candidate]
                audit.require(len(record) == 1, f"{cid}: exactly one unit candidate")
                compare_summary(audit, record[0], summary, f"{cid}: unit record")
                scores.append(summary)
                predictions.append(frame)
            best = min(range(2), key=lambda index: scores[index]["macro_rmse"])
            config = [c for c in frozen["configs"] if (c["model"], c["window"]) == (model, horizon)][0]
            audit.require(int(config["candidate"]) == best == int(unit["selected"]["candidate"]), f"{context}: independently selected minimum environment-equal RMSE")
            audit.require(config["params"] == run["candidates"][model][best], f"{context}: frozen parameters match minimum")
            audit.close(config["validation_macro_rmse"], scores[best]["macro_rmse"], f"{context}: frozen score")
            compare_summary(audit, selected_lookup.loc[(model, horizon)], scores[best], f"{context}: selected summary")
            chosen = load_prediction(out / "predictions" / f"{context}.csv", samples["validation"], audit, context)
            audit.close(chosen.prediction.to_numpy(), predictions[best].prediction.to_numpy(), f"{context}: selected predictions equal chosen candidate")
            _, expected_envs = independent_metrics(chosen.assign(y_true=labels.set_index("sample_id").Yield_Mg_ha.reindex(chosen.sample_id).to_numpy()))
            compare_environments(audit, pd.read_csv(out / f"{context}_environment_metrics.csv"), expected_envs, f"{context}: selected environment metrics")
            independently_chosen.append(config)
    best = min(independently_chosen, key=lambda item: item["validation_macro_rmse"])
    audit.require(best == frozen["recommended_by_validation"], "freeze: recommendation minimizes independently confirmed validation RMSE")
    quality = quality_column_profile(feature_directory)
    audit.notes.append({"development_quality_columns": quality,
                        "interpretation": "Constant training columns cannot supply learned predictive discrimination in these fitted model families; no causal weather attribution is claimed."})
    return frozen


def final_ordering_checks(audit, out, feature_directory, freeze_path, completion):
    lock_path = out / "prediction_lock_before_holdout.json"
    lock = read_json(lock_path)
    audit.require(len(lock["prediction_hashes"]) == 17, "final: 17 prediction files locked before holdout")
    audit.hashes(lock["prediction_hashes"], "final prediction lock")
    expected_names = {f"{m}_W{h}.csv" for m, h in EXPECTED_PAIRS} | {f"{m}.csv" for m in BASELINES}
    audit.require({Path(path).name for path in lock["prediction_hashes"]} == expected_names,
                  "final: prediction lock covers exactly prescribed models and baselines")
    freeze_hash = digest(freeze_path)
    audit.require(lock["frozen_selection_sha256"] == completion["frozen_selection_sha256"] == freeze_hash,
                  "final: freeze hash shared by lock and completion")
    audit.require(lock["test_sample_ids_sha256"] == digest(feature_directory / "test_samples.csv"), "final: locked input cohort hash")
    events = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines() if line.strip()]
    opens = [(i, e) for i, e in enumerate(events) if e["event"] == "holdout_labels_opened_for_preset_evaluation"]
    audit.require(len(opens) == 1, "final: exactly one experiment-runner holdout-open event")
    open_index, opening = opens[0]
    locks = [(i, e) for i, e in enumerate(events) if e["event"] == "all_predictions_persisted_before_holdout"]
    audit.require(len(locks) == 1 and locks[0][0] < open_index, "final: all-predictions event occurs before holdout-open event")
    lock_index, lock_event = locks[0]
    audit.require(lock_event["count"] == 17 and lock_event["prediction_lock_sha256"] == digest(lock_path), "final: pre-holdout event attests current prediction lock")
    audit.require(instant(lock_event["utc"]) <= instant(opening["utc"]), "final: prediction lock timestamp precedes labels")
    audit.require(instant(read_json(freeze_path)["frozen_at_utc"]) <= instant(lock_event["utc"]), "final: model selection frozen before final prediction lock")
    audit.require(instant(opening["utc"]) <= instant(completion["completed_at_utc"]), "final: holdout opening precedes evaluation completion")
    fits = {(e["model"], e["window"]) for e in events[:lock_index]
            if e["event"] in {"final_fit_complete", "resume_completed_final_prediction"}}
    audit.require(fits == EXPECTED_PAIRS, "final: all 15 model predictions complete before holdout")
    for path in lock["prediction_hashes"]:
        audit.require((PROJECT / path).stat().st_mtime <= instant(opening["utc"]) + 0.001,
                      f"final: prediction file mtime before label-open: {Path(path).name}")
    audit.require(lock_path.stat().st_mtime <= instant(opening["utc"]) + 0.001, "final: lock file mtime before label-open")
    audit.notes.append({"ordering_scope": "Hashes, log sequence and local file timestamps agree. This is workflow provenance, not a security guarantee that no other process accessed labels."})
    return opening


def verify_window_comparisons(audit, out, environment_tables, frozen):
    declared = pd.read_csv(out / "weather_increment_comparisons.csv")
    audit.require(len(declared) == 15 and set(zip(declared.model, declared.window)) == EXPECTED_PAIRS,
                  "final: all 15 weather comparisons present")
    decisions = read_json(out / "earliest_sufficient_windows.json")
    audit.require(decisions["recommended_by_2023_validation"] == frozen["recommended_by_validation"], "final: recommendation unchanged from validation")
    audit.require(decisions["test_result_selection_permitted"] is False, "final: no test-result model selection")
    for model in MODELS:
        tables = environment_tables[model]
        reference = tables[0].set_index("Env").sort_index().rmse
        full = tables[120].set_index("Env").reindex(reference.index).rmse
        gain = float(reference.mean() - full.mean())
        relative = gain / float(reference.mean())
        _, _, full_upper = independent_bootstrap((full - reference).to_numpy())
        supported = bool(gain > 0 and relative >= 0.02 and full_upper < 0)
        earliest = None
        for horizon in HORIZONS:
            candidate = tables[horizon].set_index("Env").reindex(reference.index).rmse
            zero_delta, zero_low, zero_high = independent_bootstrap((candidate - reference).to_numpy())
            full_delta, full_low, full_high = independent_bootstrap((candidate - full).to_numpy())
            retention = float((reference.mean() - candidate.mean()) / gain) if gain > 0 else np.nan
            enough = bool(supported and horizon > 0 and retention >= 0.9)
            if enough and earliest is None:
                earliest = horizon
            expected = {
                "macro_rmse": float(candidate.mean()), "gain_retention": retention,
                "full_window_relative_gain": relative,
                "vs_W0_delta_rmse": zero_delta, "vs_W0_ci95_low": zero_low, "vs_W0_ci95_high": zero_high,
                "vs_W120_delta_rmse": full_delta, "vs_W120_ci95_low": full_low, "vs_W120_ci95_high": full_high,
            }
            row = declared.loc[(declared.model == model) & (declared.window == horizon)].iloc[0]
            compare_summary(audit, row, expected, f"{model} W{horizon}: paired environment bootstrap")
            audit.require(bool(row.sufficient_by_operational_rule) == enough, f"{model} W{horizon}: operational sufficiency rule")
        decision = decisions["models"][model]
        audit.require(decision["earliest_sufficient_window"] == earliest and decision["W120_gain_supported"] == supported,
                      f"{model}: earliest sufficient window recomputed")
        audit.close(decision["W120_relative_gain"], relative, f"{model}: complete window gain")


def validate_final(audit, feature_directory, out, development_out, development_features):
    completion_path = out / "final_evaluation_manifest.json"
    audit.require(completion_path.is_file(), "final: completion manifest must exist before audit reads evaluated predictions or holdout")
    completion = read_json(completion_path)
    frozen = validate_development(audit, development_features, development_out)
    freeze_path = development_out / "frozen_selection.json"
    run, samples = verify_feature_inputs(audit, feature_directory, out, "final")
    audit.require(run["frozen_selection_sha256"] == digest(freeze_path), "final: run freeze hash")
    audit.hashes(completion["artifacts"], "final completion artifacts")
    opening = final_ordering_checks(audit, out, feature_directory, freeze_path, completion)
    # This is the only holdout read in this audit and is unreachable in development.
    holdout = PREP / "holdout/test_targets.csv"
    audit.require(digest(holdout) == completion["holdout_sha256"] == opening["sha256"], "final: exact holdout file fingerprint")
    labels = pd.read_csv(holdout)
    audit.holdout_read = True
    audit.require(labels.sample_id.is_unique, "final: unique held-out target IDs")
    audit.require(len(labels) == completion["n_scored"] and len(samples["test"]) == completion["n_prediction_inputs"], "final: completion sample accounting")
    audit.require(completion["preset_models_scored"] == 17, "final: 17 models scored")
    audit.require(completion["recommended_by_validation"] == frozen["recommended_by_validation"], "final: manifest preserves validation recommendation")
    train_labels = pd.concat([pd.read_csv(PREP / "targets/train.csv"), pd.read_csv(PREP / "targets/validation.csv")], ignore_index=True)
    summaries = pd.read_csv(out / "summary_metrics.csv")
    environments = pd.read_csv(out / "environment_metrics.csv")
    groups = pd.read_csv(out / "subgroup_metrics.csv")
    all_pairs = EXPECTED_PAIRS | {(name, 0) for name in BASELINES}
    audit.require(len(summaries) == 17 and set(zip(summaries.model, summaries.window)) == all_pairs, "final: complete unique summary table")
    audit.require(not environments.duplicated(["model", "window", "Env"]).any(), "final: environment metric keys unique")
    audit.require(not groups.duplicated(["model", "window", "group_type", "group"]).any(), "final: subgroup keys unique")
    tables = {m: {} for m in MODELS}
    expected_group_keys = set()
    expected_environment_keys = set()
    for model, horizon in sorted(all_pairs):
        context = f"{model}_W{horizon}"
        name = f"{model}.csv" if model in BASELINES else f"{context}.csv"
        frame = load_prediction(out / "predictions" / name, samples["test"], audit, f"final {context}")
        if model in BASELINES:
            audit.close(frame.prediction.to_numpy(), baseline_reference(model, samples["train"], train_labels, samples["test"], audit), f"final {model}: refitted baseline mapping")
        else:
            unit = read_json(out / "units" / f"{context}.json")
            audit.require(unit["frozen_selection_sha256"] == digest(freeze_path), f"final {context}: unit frozen selection hash")
            audit.hashes(unit["artifacts"], f"final {context}")
        scored, summary, envs = score_against_labels(frame, labels, audit, context, allow_unlabeled=True)
        # Existing scored CSVs are read only now, after the completion gate.
        saved = pd.read_csv(out / "predictions" / f"{context}_evaluated.csv")
        for column in KEY_COLUMNS:
            audit.require(saved[column].tolist() == scored[column].tolist(), f"{context}: evaluated sample order/{column}")
        audit.close(saved[["prediction", "y_true"]].to_numpy(), scored[["prediction", "y_true"]].to_numpy(), f"{context}: evaluated predictions and original labels")
        row = summaries.loc[(summaries.model == model) & (summaries.window == horizon)].iloc[0]
        compare_summary(audit, row, summary, f"final {context}")
        audit.require(row.n_prediction_inputs == len(frame) and row.n_without_observed_target == len(frame) - len(labels), f"{context}: labeled/unlabeled accounting")
        actual_env = environments.loc[(environments.model == model) & (environments.window == horizon)]
        compare_environments(audit, actual_env, envs, f"final {context}")
        expected_environment_keys.update((model, horizon, env) for env in envs.Env)
        if model in MODELS:
            tables[model][horizon] = envs
        for kind, column in (("hybrid", "Hybrid"), ("site", "Experiment_Code")):
            known = set(samples["train"][column])
            mask = scored[column].isin(known)
            for seen, label in ((True, "seen"), (False, "unseen")):
                part = scored.loc[mask == seen]
                if not len(part):
                    continue
                expected_group_keys.add((model, horizon, kind, label))
                expected, _ = independent_metrics(part)
                actual = groups.loc[(groups.model == model) & (groups.window == horizon) & (groups.group_type == kind) & (groups.group == label)]
                audit.require(len(actual) == 1, f"{context} {kind} {label}: exactly one subgroup row")
                compare_summary(audit, actual.iloc[0], expected, f"{context} {kind} {label}: independently recomputed subgroup")
    audit.require(set(zip(environments.model, environments.window, environments.Env)) == expected_environment_keys, "final: no missing or extra environment metric groups")
    audit.require(set(zip(groups.model, groups.window, groups.group_type, groups.group)) == expected_group_keys, "final: no missing or extra seen/unseen groups")
    verify_window_comparisons(audit, out, tables, frozen)
    audit.notes.append({"final_training_quality_columns": quality_column_profile(feature_directory)})
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "final"), required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--development-results", type=Path, default=PROJECT / "modeling/results/development")
    parser.add_argument("--development-features", type=Path, default=PROJECT / "modeling/features/development")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    audit = Audit(args.stage)
    out = args.results or PROJECT / "modeling/results" / args.stage
    feature_directory = args.features or PROJECT / "modeling/features" / args.stage
    report = args.report or PROJECT / "modeling" / f"{args.stage}_validation_report.json"
    failed = None
    try:
        if args.stage == "development":
            validate_development(audit, feature_directory, out)
        else:
            validate_final(audit, feature_directory, out, args.development_results, args.development_features)
    except Exception as error:
        failed = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
    result = {
        "stage": args.stage, "status": "FAIL" if failed else "PASS",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": digest(Path(__file__)), "holdout_read_by_this_audit": audit.holdout_read,
        "independent_calculations": "NumPy residual sums, pandas average ranks, tie-group fractional Top-20%, and explicit 2000-loop environment bootstrap; no imports from metrics or runner.",
        "checks_passed": sum(c["passed"] for c in audit.checks), "checks": audit.checks,
        "notes": audit.notes, "failure": failed,
        "limitations": ["Does not refit models or recompute their predictions from model objects.",
                        "Hash/log/timestamp agreement supports local workflow provenance, not secure access isolation.",
                        "Passing checks is not a measure of model accuracy or generalizability."],
    }
    audited_manifest = out / ("run_manifest.json" if args.stage == "development" else "final_evaluation_manifest.json")
    if failed is None and audited_manifest.is_file():
        result["audited_manifest_sha256"] = digest(audited_manifest)
        result["audited_manifest_path"] = str(audited_manifest.relative_to(PROJECT))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"{result['status']}: {result['checks_passed']} independent checks; holdout_read={audit.holdout_read}; report={report}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
