"""Run four new backtests with bounded parallelism, then machine-readable analysis.

No original experiment, final report, notebook or showcase is rebuilt.
"""
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

ROOT = PROJECT / "modeling/rolling_v1"


def pipeline(root=ROOT, workers=2):
    root = Path(root).resolve()
    if workers not in (1,2):
        raise ValueError("At most two folds, four threads per fold")
    root.mkdir(parents=True,exist_ok=True)
    lock = (root/"pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    scripts = [PROJECT/"modeling"/name for name in (
        "build_rolling_features.py","run_rolling_experiments.py","run_rolling_pipeline.py",
        "build_features.py","run_experiments.py","metrics.py")]
    preserved = [PROJECT/"modeling/results/final/final_evaluation_manifest.json",
                 PROJECT/"modeling/results/development/frozen_selection.json",
                 PROJECT/"modeling/features/final/feature_manifest.json"]
    inputs = {str(p):digest(p) for p in [root/"PROTOCOL.md",*scripts,*preserved]}
    manifest = {"new_test_years":[2020,2021,2022,2023],"reuse_original_year":2024,
                "input_hashes":inputs,"workers":workers,"threads_per_model":4}
    manifest_path = root/"pipeline_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Pipeline identity changed; preserve outputs and adjudicate, do not overwrite")
    if not manifest_path.exists():
        write_json(manifest,manifest_path)
    logs = root/"logs"
    logs.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(PROJECT/".runtime"),str(PROJECT/"modeling")])
    for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
        environment[variable] = "4"
    def step(year,name,script,*arguments):
        record = {"utc":now(),"event":"stage_start","year":year,"stage":name}
        print(json.dumps(record),flush=True)
        start = time.monotonic()
        with (logs/f"{year}_{name}.log").open("a") as stream:
            stream.write(json.dumps(record)+"\n")
            stream.flush()
            subprocess.run([sys.executable,str(PROJECT/"modeling"/script),*arguments],
                           cwd=PROJECT,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(json.dumps({"utc":now(),"event":"stage_complete","year":year,"stage":name,
                          "elapsed_seconds":time.monotonic()-start}),flush=True)
    def fold(year):
        for mode in ("development","final"):
            step(year,f"features_{mode}","build_rolling_features.py","--test-year",str(year),"--mode",mode,"--root",str(root))
            step(year,f"models_{mode}","run_rolling_experiments.py","--year",str(year),"--stage",mode,"--root",str(root))
        return year
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fold,y):y for y in (2020,2021,2022,2023)}
        for future in as_completed(futures):
            print(json.dumps({"event":"fold_complete","year":future.result(),"utc":now()}),flush=True)
    for path,expected in inputs.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Code/protocol changed while running: {path}")
    step("all","stability_analysis","analyze_rolling_stability.py","--root",str(root))
    write_json({"completed_at_utc":now(),"pipeline_manifest_sha256":digest(manifest_path),
                "analysis_manifest_sha256":digest(root/"analysis/analysis_manifest.json"),
                "final_report_created":False},root/"pipeline_completion.json")
    print(json.dumps({"event":"pipeline_complete","utc":now(),"root":str(root)}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--workers",type=int,choices=(1,2),default=2)
    args = parser.parse_args()
    pipeline(args.root,args.workers)
