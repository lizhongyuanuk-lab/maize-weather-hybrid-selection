"""Bounded, resumable five-year segmented-weather experiment; no final report."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_experiments import PROJECT, digest, now, write_json

ROOT = PROJECT / "modeling/segmented_v1"
CUMULATIVE_ROOT = PROJECT / "modeling/rolling_v1"


def pipeline(root=ROOT, cumulative_root=CUMULATIVE_ROOT, workers=2):
    root, cumulative_root = Path(root).resolve(), Path(cumulative_root).resolve()
    if root == cumulative_root or workers not in (1, 2):
        raise ValueError("Use a separate root and at most two four-thread workers")
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scripts = [PROJECT / "modeling" / name for name in (
        "build_segmented_features.py", "run_segmented_experiments.py", "segmented_sources.py",
        "run_segmented_pipeline.py", "build_features.py", "run_experiments.py", "metrics.py",
        "build_rolling_features.py", "run_rolling_experiments.py", "analyze_rolling_stability.py")]
    preserved = [cumulative_root / "analysis/analysis_manifest.json", cumulative_root / "validation_checks.json",
                 cumulative_root / "PROTOCOL.md", PROJECT / "modeling/results/final/final_evaluation_manifest.json",
                 PROJECT / "modeling/results/development/frozen_selection.json"]
    inputs = {str(p): digest(p) for p in [root / "PROTOCOL.md", *scripts, *preserved]}
    manifest = {"test_years": [2020, 2021, 2022, 2023, 2024], "new_windows": [60, 90, 120],
                "shared_windows": [0, 30], "planned_new_fits": 135, "workers": workers,
                "threads_per_model": 4, "input_hashes": inputs, "cumulative_root": str(cumulative_root)}
    manifest_path = root / "pipeline_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Pipeline identity changed; preserve the existing experiment")
    if not manifest_path.exists():
        write_json(manifest, manifest_path)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(PROJECT / ".runtime"), str(PROJECT / "modeling")])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "4"

    def step(year, stage, script, *arguments):
        record = {"utc": now(), "event": "stage_start", "year": year, "stage": stage}
        print(json.dumps(record), flush=True)
        started = time.monotonic()
        with (logs / f"{year}_{stage}.log").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            subprocess.run([sys.executable, str(PROJECT / "modeling" / script), *arguments],
                           env=environment, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({"utc": now(), "event": "stage_complete", "year": year,
                          "stage": stage, "elapsed_seconds": time.monotonic() - started}), flush=True)

    def fold(year):
        shared_args = ("--root", str(root), "--cumulative-root", str(cumulative_root))
        for stage in ("development", "final"):
            step(year, f"features_{stage}", "build_segmented_features.py", "--test-year", str(year), "--mode", stage, *shared_args)
            step(year, f"models_{stage}", "run_segmented_experiments.py", "--year", str(year), "--stage", stage, *shared_args)
        return year

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fold, year) for year in (2020, 2021, 2022, 2023, 2024)]):
            print(json.dumps({"event": "fold_complete", "year": future.result(), "utc": now()}), flush=True)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError(f"Frozen input changed during training: {path}")
    step("all", "representation_analysis", "analyze_segmented_results.py", "--root", str(root),
         "--cumulative-root", str(cumulative_root))
    write_json({"completed_at_utc": now(), "pipeline_manifest_sha256": digest(manifest_path),
                "analysis_manifest_sha256": digest(root / "analysis/analysis_manifest.json"),
                "final_report_created": False, "scope": "segmented_weather_core_comparison"},
               root / "pipeline_completion.json")
    print(json.dumps({"event": "pipeline_complete", "utc": now(), "root": str(root)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cumulative-root", type=Path, default=CUMULATIVE_ROOT)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    pipeline(args.root, args.cumulative_root, args.workers)
