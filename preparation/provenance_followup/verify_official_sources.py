"""Download public official inputs without altering raw data; compare full values.

Uses ordinary HTTPS and documented HTTP byte ranges. Incomplete chunks never
count as official files. Run with Python 3.12 and pandas 2.2.3 from project root.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
BASE = "https://de.cyverse.org/anon-files/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/"
SOURCES = [
    ("test_weather", "Testing_data/4_Testing_Weather_Data_2024_full_year.csv", 702794),
    ("training_weather", "Training_data/4_Training_Weather_Data_2014_2023_full_year.csv", 10460386),
    ("training_traits", "Training_data/1_Training_Trait_Data_2014_2023.csv", 31590253),
    ("genotype", "Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt", 40765753),
]


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def get_part(source, start, end, timeout):
    role, relative, size = source
    path = HERE / "parts" / f"{role}.{start}-{end}.part"
    headers = path.with_suffix(".headers")
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected and headers.exists():
        header = headers.read_text()
        if re.search(rf"content-range:\s*bytes {start}-{end}/{size}", header, re.I):
            return role, start, True, "reused verified complete range"
    result = subprocess.run([
        "curl", "-sS", "--fail", "--connect-timeout", "10", "--max-time", str(timeout),
        "--range", f"{start}-{end}", "--dump-header", str(headers), "--output", str(path),
        BASE + relative,
    ], capture_output=True, text=True)
    header = headers.read_text() if headers.exists() else ""
    length = path.stat().st_size if path.exists() else 0
    valid = (
        result.returncode == 0 and length == expected
        and re.search(rf"content-range:\s*bytes {start}-{end}/{size}", header, re.I) is not None
    )
    return role, start, valid, result.stderr.strip() or f"received {length}/{expected} bytes"


def compare(source, official):
    role, relative, size = source
    local = ROOT / "audit" / "data" / "raw" / pathlib.Path(relative).name.replace(".txt", ".csv")
    info = dict(role=role, official_url=BASE + relative, official_path=str(official.relative_to(ROOT)),
                local_path=str(local.relative_to(ROOT)), official_bytes=official.stat().st_size,
                local_bytes=local.stat().st_size, official_sha256=sha256(official), local_sha256=sha256(local),
                download_complete=True)
    info["byte_match"] = info["official_sha256"] == info["local_sha256"]
    with official.open(newline="", encoding="utf-8-sig") as a, local.open(newline="", encoding="utf-8-sig") as b:
        if role == "genotype":
            info["official_format_declaration"] = a.readline().strip()
            if info["official_format_declaration"] != "<Numeric>":
                raise ValueError("Unexpected official genotype format declaration")
            aa, bb = csv.reader(a, delimiter="\t"), csv.reader(b)
        else:
            aa, bb = csv.reader(a), csv.reader(b)
        ah, bh = next(aa), next(bb)
        if role == "genotype":
            info["official_identifier_header"] = ah[0]
            info["local_identifier_header"] = bh[0]
            if ah[0] != "<Marker>" or bh[0] != "Geno":
                raise ValueError("Unexpected genotype identifier headers")
            ah[0] = bh[0]  # <Marker> -> Geno: only allowed representation change.
        info["headers_equal_after_documented_identifier_rename"] = ah == bh
        from itertools import zip_longest
        rows = 0
        differs = 0
        samples = []
        official_rows = 0
        local_rows = 0
        different_row_lengths = 0
        key_columns = [i for i, name in enumerate(bh) if name in ("Env", "Hybrid", "Geno", "Date", "Plot")]
        ordered_keys_equal = True
        for ar, br in zip_longest(aa, bb):
            official_rows += ar is not None
            local_rows += br is not None
            rows += 1
            if ar is None or br is None:
                differs += 1
                ordered_keys_equal = False
                if len(samples) < 10:
                    samples.append(dict(row=rows, type="row_absent_one_side"))
                continue
            if len(ar) != len(br):
                different_row_lengths += 1
            if [ar[i] if i < len(ar) else None for i in key_columns] != [br[i] if i < len(br) else None for i in key_columns]:
                ordered_keys_equal = False
            for col, (av, bv) in enumerate(zip_longest(ar, br)):
                if av != bv:
                    differs += 1
                    if len(samples) < 10:
                        samples.append(dict(row=rows, column=bh[col] if col < len(bh) else col, official=av, local=bv))
        info.update(official_rows=official_rows, local_rows=local_rows, official_columns=len(ah), local_columns=len(bh),
                    ordered_keys_equal=ordered_keys_equal, key_columns=[bh[i] for i in key_columns],
                    different_row_lengths=different_row_lengths, different_cells_or_absent_rows=differs,
                    difference_examples=samples, semantic_match=ah == bh and differs == 0)
    info["status"] = "byte_match" if info["byte_match"] else ("semantic_match_documented_representation" if info["semantic_match"] else "content_difference")
    if role == "training_traits":
        transformed = official.read_bytes().replace(b"\r\n", b"\n")
        info["documented_byte_transform"] = "Replace CRLF line endings with LF; no other byte transformation."
    elif role == "genotype":
        original = official.read_bytes()
        prefix = b"<Numeric>\n<Marker>"
        if not original.startswith(prefix):
            raise ValueError("Unexpected genotype prefix for exact byte conversion")
        transformed = b"Geno" + original[len(prefix):]
        transformed = transformed.replace(b"\t", b",")
        info["documented_byte_transform"] = "Remove <Numeric> declaration line, rename first <Marker> header to Geno, and replace tab delimiters by commas; no other byte transformation."
    else:
        transformed = None
    if transformed is not None:
        info["transformed_official_bytes"] = len(transformed)
        info["transformed_official_sha256"] = hashlib.sha256(transformed).hexdigest()
        info["documented_byte_transform_matches_mirror"] = info["transformed_official_sha256"] == info["local_sha256"]
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--timeout", type=int, default=110)
    parser.add_argument("--roles", nargs="*")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    (HERE / "official").mkdir(exist_ok=True)
    (HERE / "parts").mkdir(exist_ok=True)
    sources = [s for s in SOURCES if not args.roles or s[0] in args.roles]
    started = time.time()
    jobs = []
    results = []
    if not args.compare_only:
        for source in sources:
            official = HERE / "official" / pathlib.Path(source[1]).name
            if official.exists() and official.stat().st_size == source[2]:
                continue
            for start in range(0, source[2], args.chunk_bytes):
                jobs.append((source, start, min(start + args.chunk_bytes - 1, source[2] - 1), args.timeout))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(get_part, *job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                role, start, ok, detail = future.result()
                results.append(dict(role=role, start=start, complete=ok, detail=detail))
                print(role, start, "OK" if ok else "INCOMPLETE", detail, flush=True)
        for source in sources:
            role, relative, size = source
            official = HERE / "official" / pathlib.Path(relative).name
            if official.exists() and official.stat().st_size == size:
                continue
            source_jobs = [j for j in jobs if j[0][0] == role]
            good = {r["start"] for r in results if r["role"] == role and r["complete"]}
            if len(good) == len(source_jobs) and source_jobs:
                with official.open("wb") as target:
                    for _, start, end, _ in sorted(source_jobs, key=lambda x: x[1]):
                        target.write((HERE / "parts" / f"{role}.{start}-{end}.part").read_bytes())
                assert official.stat().st_size == size
    comparisons = []
    for source in sources:
        official = HERE / "official" / pathlib.Path(source[1]).name
        if official.exists() and official.stat().st_size == source[2]:
            comparisons.append(compare(source, official))
        else:
            comparisons.append(dict(role=source[0], official_url=BASE + source[1], download_complete=False,
                                    status="unverified_download_incomplete"))
    complete = [f for f in comparisons if f.get("download_complete")]
    equal = [f for f in comparisons if f.get("semantic_match")]
    historical = json.loads((ROOT / "preparation/source_verification.json").read_text())
    historical_hashes = {f["role"]: f["local_sha256"] for f in historical["files"]}
    for item in complete:
        item["local_sha256_matches_stage1_receipt"] = item["local_sha256"] == historical_hashes[item["role"]]
    report = dict(schema_version=1, verified_on="2026-10-18", source_catalog="https://dc.cyverse.org/dataset/genomes_to_fields_2024_maize_genotype_by_environment_prediction_competition",
                  scope="Follow-up for the four previously incomplete official-source comparisons; not a replacement for the historical three-file metadata/template receipt.",
                  summary=dict(requested_files=len(comparisons), complete_official_downloads=len(complete),
                               byte_matches=sum(bool(f.get("byte_match")) for f in comparisons),
                               semantic_matches_including_byte_matches=len(equal),
                               content_differences=sum(f.get("status") == "content_difference" for f in comparisons),
                               unverified_files=[f["role"] for f in comparisons if not f.get("download_complete")],
                               all_followup_files_semantically_verified=len(equal) == len(comparisons)),
                  method="Public official HTTPS byte ranges, exact Content-Range validation, full size and SHA-256, csv-reader ordered header/key/cell equality; genotype <Numeric> declaration stripped and <Marker> identifier renamed to mirror Geno.",
                  raw_modified=False, elapsed_seconds=round(time.time() - started, 2), files=comparisons,
                  script_sha256=sha256(pathlib.Path(__file__)),
                  raw_inputs_unchanged_from_stage1_receipt=all(f["local_sha256_matches_stage1_receipt"] for f in complete) and len(complete) == len(sources),
                  download_results=results)
    output = HERE / ("verification_" + ("_".join(args.roles) if args.roles else "report") + ".json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(output), "files": [{"role": f["role"], "status": f["status"]} for f in comparisons]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
