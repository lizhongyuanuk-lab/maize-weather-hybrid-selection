"""Read-only, hash-bound access to completed cumulative-weather experiments.

Source verification may hash evaluated CSVs, but never parses their label values.
The original 2024 run and rolling_v1 remain immutable. No training entry point is
called here. Relative original paths use PROJECT; rolling paths use their root.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from run_experiments import BASELINES, CANDIDATES, HORIZONS, PROJECT, SEED, digest

ORIGINAL_2024_SHA256 = "c9cce89304d08087b48fc0c2f0bfdb52794eaee4b89aeb97a1b7cee51d38581a"
ROLLING_ANALYSIS_SHA256 = "28b56456a512dff413b5d8a343daa2710df130f58e1fadd206f0126aaa0e575d"
YEARS = (2020, 2021, 2022, 2023, 2024)
IDENTITY = ["sample_id", "Env", "Hybrid", "Year", "Experiment_Code"]


def read_json(path):
    return json.loads(Path(path).read_text())


class SourceVerifier:
    """Record every verified source hash for the new experiment's own manifest."""
    def __init__(self):
        self.hashes = {}
        self.stats = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        actual = self.hashes.get(str(path)) if self.stats.get(str(path)) == stamp else digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Cumulative source hash mismatch: {path}")
        self.hashes[str(path)] = actual
        self.stats[str(path)] = stamp
        return actual

    def mapping(self, mapping, base):
        for name, expected in mapping.items():
            path = Path(name)
            self.check(path if path.is_absolute() else base/path, expected)

    def artifacts(self, path, base):
        self.check(path)
        record = read_json(path)
        if Path(record.get("artifact_path_base", base)).resolve() != base.resolve():
            raise ValueError(f"Unexpected cumulative artifact base: {path}")
        for name, expected in record["artifacts"].items():
            artifact = (base/name).resolve()
            if Path(name).is_absolute() or not artifact.is_relative_to(base.resolve()):
                raise ValueError("Escaping cumulative artifact path")
            self.check(artifact, expected)
        return record


@dataclass
class SourceFold:
    year: int
    stage: str
    feature_directory: Path
    directory: Path
    development_directory: Path
    final_directory: Path
    freeze: dict
    completion: dict
    units: dict
    development_units: dict
    baseline_selected: dict
    hashes: dict
    artifact_base: Path

    def prediction_path(self, model, window):
        filename = f"{model}.csv" if model in BASELINES else f"{model}_W{window}.csv"
        return self.directory/"predictions"/filename

    def configuration(self, model, window):
        return next(c for c in self.freeze["configs"] if c["model"] == model and c["window"] == window)

    @property
    def evaluated_reference(self):
        return self.final_directory/"predictions/GlobalMean_W0_evaluated.csv"


def source_fold(test_year, stage, cumulative_root, *, project=PROJECT):
    """Verify completed sources without opening outcome-bearing CSV values."""
    if test_year not in YEARS or stage not in ("development", "final"):
        raise ValueError("Invalid segmented source fold")
    project, cumulative_root = Path(project).resolve(), Path(cumulative_root).resolve()
    verify = SourceVerifier()
    audit_path = cumulative_root/"validation_checks.json"
    analysis_path = cumulative_root/"analysis/analysis_manifest.json"
    audit = read_json(audit_path)
    verify.check(audit_path)
    if (audit.get("status") != "PASS" or audit.get("failure") is not None
            or audit.get("years") != list(YEARS)
            or audit.get("original_2024_manifest_sha256") != ORIGINAL_2024_SHA256
            or Path(audit["audited_manifest_path"]).resolve() != analysis_path):
        raise ValueError("Completed rolling_v1 PASS audit and fixed original 2024 binding are required")
    if audit["audited_manifest_sha256"] != ROLLING_ANALYSIS_SHA256:
        raise ValueError("Cumulative analysis differs from the prespecified version")
    verify.check(analysis_path, ROLLING_ANALYSIS_SHA256)
    analysis = verify.artifacts(analysis_path, cumulative_root)
    if analysis.get("years") != list(YEARS):
        raise ValueError("Cumulative analysis must retain all five prescribed years")
    verify.mapping(analysis["source_hashes"], project)
    original_final = project/"modeling/results/final/final_evaluation_manifest.json"
    verify.check(original_final, ORIGINAL_2024_SHA256)
    if analysis["source_hashes"].get(str(original_final)) != ORIGINAL_2024_SHA256:
        raise ValueError("Original 2024 completion is not bound to the audited source analysis")

    original = test_year == 2024
    base = project if original else cumulative_root
    result_root = project/"modeling/results" if original else cumulative_root/"results"/str(test_year)
    feature_root = project/"modeling/features" if original else cumulative_root/"features"/str(test_year)
    directory, development, final = result_root/stage, result_root/"development", result_root/"final"
    feature_directory = feature_root/stage
    completion_path = final/"final_evaluation_manifest.json"
    completion = verify.artifacts(completion_path, base)
    if analysis["source_hashes"].get(str(completion_path)) != verify.hashes[str(completion_path)]:
        raise ValueError("Fold completion is not bound to audited cumulative analysis")
    freeze_path = development/"frozen_selection.json"
    verify.check(freeze_path, completion["frozen_selection_sha256"])
    frozen = read_json(freeze_path)
    expected = {(m, h) for m in CANDIDATES for h in HORIZONS}
    if (len(frozen["configs"]) != 15
            or {(c["model"], c["window"]) for c in frozen["configs"]} != expected
            or frozen["recommended_by_validation"] != min(frozen["configs"], key=lambda c: c["validation_macro_rmse"])):
        raise ValueError("Invalid cumulative validation freeze")
    for config in frozen["configs"]:
        if config["candidate"] not in (0, 1) or config["params"] != CANDIDATES[config["model"]][config["candidate"]]:
            raise ValueError("Cumulative candidate budget differs")
    dev_run_path = development/"run_manifest.json"
    verify.check(dev_run_path, frozen["development_run_manifest_sha256"])
    verify.mapping(frozen["development_input_hashes"], project)
    if original:
        for phase, manifest_path in [("development", dev_run_path), ("final", completion_path)]:
            receipt_path = project/"modeling"/f"{phase}_validation_report.json"
            receipt = read_json(receipt_path)
            verify.check(receipt_path)
            receipt_target = Path(receipt["audited_manifest_path"])
            receipt_target = receipt_target if receipt_target.is_absolute() else project/receipt_target
            if receipt.get("status") != "PASS" or receipt_target.resolve() != manifest_path:
                raise ValueError("Original 2024 independent audit is missing or misbound")
            verify.check(manifest_path, receipt["audited_manifest_sha256"])
        replay_path = project/"modeling/prediction_replay_report.json"
        replay = read_json(replay_path)
        verify.check(replay_path)
        if replay.get("status") != "PASS" or replay.get("audited_manifest_sha256") != ORIGINAL_2024_SHA256:
            raise ValueError("Original 2024 saved-model replay is not PASS/bound")
        verify.mapping(replay["verified_hashes"], project)
    else:
        dev_complete = verify.artifacts(development/"development_completion_manifest.json", base)
        if dev_complete["artifacts"].get(str(freeze_path.relative_to(base))) != digest(freeze_path):
            raise ValueError("Rolling source freeze not bound to development completion")

    for phase in {"development", stage}:
        phase_run_path = result_root/phase/"run_manifest.json"
        phase_run = read_json(phase_run_path)
        verify.check(phase_run_path)
        if phase_run["stage"] != phase or phase_run["candidates"] != CANDIDATES or phase_run["seed"] != SEED:
            raise ValueError("Cumulative source stage/candidate budget/seed changed")
        verify.mapping(phase_run["inputs"], project)
        if phase == "final" and phase_run["frozen_selection_sha256"] != digest(freeze_path):
            raise ValueError("Source final run does not bind its validation selection")
    run = read_json(directory/"run_manifest.json")
    input_paths = {(Path(p) if Path(p).is_absolute() else project/p).resolve() for p in run["inputs"]}
    required_features = {feature_directory/"feature_manifest.json"}
    required_features |= {feature_directory/f"{s}_samples.csv" for s in ("train", "validation", "test")}
    required_features |= {feature_directory/f"W{h}{suffix}" for h in HORIZONS for suffix in (".npz", "_columns.json")}
    if not required_features.issubset(input_paths):
        raise ValueError("Source feature inputs are not bound to the model run")
    feature = read_json(feature_directory/"feature_manifest.json")
    if feature["mode"] != stage or feature["fit_years"] != [2014, test_year-(2 if stage == "development" else 1)]:
        raise ValueError("Source feature fitting period differs")
    verify.mapping(feature["input_hashes"], project)
    verify.mapping(feature["outputs"], feature_directory)
    helper = project/"modeling"/("build_features.py" if original else "build_rolling_features.py")
    verify.check(helper, feature["script_sha256"])
    units, development_units = {}, {}
    for model, horizon in sorted(expected):
        dev_path = development/"units"/f"{model}_W{horizon}.json"
        dev_unit = verify.artifacts(dev_path, base)
        candidates = dev_unit["candidates"]
        selected = dev_unit["selected"]
        if (len(candidates) != 2 or {c["candidate"] for c in candidates} != {0, 1}
                or selected != min(candidates, key=lambda c: c["macro_rmse"])):
            raise ValueError("Source candidate selection does not follow validation-only minimum")
        config = next(c for c in frozen["configs"] if c["model"] == model and c["window"] == horizon)
        if selected["params"] != config["params"] or selected["macro_rmse"] != config["validation_macro_rmse"]:
            raise ValueError("Source selected unit and freeze disagree")
        for c in candidates:
            if c["model"] != model or c["window"] != horizon or c["params"] != CANDIDATES[model][c["candidate"]]:
                raise ValueError("Source candidate identity differs")
            for suffix in (".csv", "_environments.csv"):
                verify.check(development/"candidate_predictions"/f"{model}_W{horizon}_c{c['candidate']}{suffix}")
        development_units[(model, horizon)] = dev_unit
        unit = dev_unit if stage == "development" else verify.artifacts(directory/"units"/f"{model}_W{horizon}.json", base)
        if stage == "final" and unit["frozen_selection_sha256"] != digest(freeze_path):
            raise ValueError("Source final model unit bound to a different freeze")
        suffix = ".cbm" if model == "CatBoost" else ".joblib"
        required = [directory/"models"/f"{model}_W{horizon}{suffix}", directory/"predictions"/f"{model}_W{horizon}.csv"]
        if not {str(p.relative_to(base)) for p in required}.issubset(unit["artifacts"]):
            raise ValueError("Source unit does not bind model and prediction")
        units[(model, horizon)] = unit
    # This table contains validation metrics only, never test labels/results.
    selected_path = development/"selected_metrics.csv"
    verify.check(selected_path)
    selected = pd.read_csv(selected_path)
    baseline_selected = {}
    for name in BASELINES:
        rows = selected.loc[selected.model == name]
        if len(rows) != 1:
            raise ValueError("Source baseline validation metric absent/duplicated")
        baseline_selected[name] = rows.iloc[0].to_dict()
        prediction_path = directory/"predictions"/f"{name}.csv"
        verify.check(prediction_path)
        if stage == "final" and str(prediction_path.relative_to(base)) not in completion["artifacts"]:
            raise ValueError("Source final baseline not bound to completion")
    return SourceFold(test_year, stage, feature_directory, directory, development, final,
                      frozen, completion, units, development_units, baseline_selected,
                      verify.hashes, base)
