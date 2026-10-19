"""Read-only publication preflight for this project, never its parent workspace.

Standard library only. No staging, Git initialization, deletion, redaction,
training, network access or publication. Findings contain locations, not matched
secret values. This is not a complete security, rights or Git-history audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SIZE_LIMIT = 10 * 1024 * 1024  # Project policy, not a claim about a hosting limit.
TEXT_LIMIT = SIZE_LIMIT
PATTERNS = {
    "LOCAL_HOME_PATH": re.compile(r"(?<![A-Za-z0-9])/(?:Users|home)/[^/\s\"'<>]+/"),
    "WINDOWS_HOME_PATH": re.compile(r"\b[A-Za-z]:\\Users\\[^\\\s]+\\"),
    "PRIVATE_KEY": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "GITHUB_TOKEN": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "AWS_ACCESS_KEY": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "TASK_IDENTIFIER": re.compile(r"codex://(?:threads/|[^\"'\s]*threadId[=:])[0-9a-f-]{20,}", re.I),
}
PRIVATE_SUFFIXES = {".joblib", ".cbm", ".npz", ".npy", ".pkl", ".pickle", ".pem", ".key"}
PRIVATE_PARTS = {".runtime", ".venv", "node_modules", "__pycache__", ".git"}
PRIVATE_NAME = re.compile(r"passport|employment[_ -]?certificate|recommendation[_ -]?letter|transcript", re.I)
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def text_findings(text, relative):
    findings = []
    for kind, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({"code": kind, "file": relative, "line": text.count("\n", 0, match.start())+1})
    return findings


def git_inventory(root):
    def git(*arguments):
        return subprocess.run(["git", "-C", str(root), *arguments], check=True,
                              capture_output=True).stdout
    top = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
    raw = git("ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", ".")
    # Git paths are relative to -C (unless --full-name is explicitly requested).
    return top, sorted({Path(p.decode()) for p in raw.split(b"\0") if p})


def check_file(root, relative):
    root = Path(root).resolve()
    path = root / relative
    name = relative.as_posix()
    findings = []
    if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(root):
        return [{"code": "OUTSIDE_PROJECT", "file": name}], 0
    if path.is_symlink():
        return [{"code": "SYMLINK_REQUIRES_REVIEW", "file": name}], 0
    if not path.is_file():
        return [{"code": "MISSING_CANDIDATE", "file": name}], 0
    if (path.suffix.lower() in PRIVATE_SUFFIXES or set(relative.parts) & PRIVATE_PARTS
            or path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example")
            or PRIVATE_NAME.search(path.name)):
        findings.append({"code": "PRIVATE_OR_GENERATED_ARTIFACT", "file": name})
    size = path.stat().st_size
    if size > SIZE_LIMIT:
        findings.append({"code": "FILE_EXCEEDS_PROJECT_10_MIB_POLICY", "file": name, "bytes": size})
    if size > TEXT_LIMIT:
        findings.append({"code": "CONTENT_TOO_LARGE_FOR_LIGHT_SCAN", "file": name, "bytes": size})
        return findings, size
    data = path.read_bytes()
    if b"\0" in data:
        return findings, size  # Binary assets need a separate visual/rights check.
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return findings, size
    findings += text_findings(text, name)
    return findings, size


def check_links(root, candidates):
    findings = []
    documents = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    for path in documents:
        if not path.is_file():
            continue
        text = path.read_text()
        for match in LINK.finditer(text):
            target = match.group(1).strip().split(" ", 1)[0].strip("<>")
            if target.startswith(("#", "http:", "https:", "mailto:")):
                continue
            linked = (path.parent / unquote(target.split("#", 1)[0])).resolve()
            location = {"file": str(path.relative_to(root)), "line": text.count("\n", 0, match.start())+1}
            if not linked.is_relative_to(root):
                findings.append({"code": "DOCUMENT_LINK_OUTSIDE_PROJECT", **location})
            elif not linked.exists():
                findings.append({"code": "BROKEN_LOCAL_DOCUMENT_LINK", **location})
            else:
                rel = linked.relative_to(root)
                included = rel in candidates if linked.is_file() else any(p.is_relative_to(rel) for p in candidates)
                if not included:
                    findings.append({"code": "DOCUMENT_LINK_NOT_IN_RELEASE_CANDIDATES", **location})
    return findings


def check_readme_results(root):
    """Compare headline rounding to the already independently audited CSVs."""
    findings = []
    try:
        def rows(name):
            with (root / name).open(newline="") as stream:
                return list(csv.DictReader(stream))
        cumulative = rows("modeling/rolling_v1/analysis/cross_year_stability.csv")
        segmented = rows("modeling/segmented_v1/analysis/primary_assessment.csv")
        readme = (root / "README.md").read_text()
        if len(cumulative) != 12 or len(segmented) != 6:
            raise ValueError("Unexpected aggregate comparison inventory")
        for model, label in (("Ridge", "Ridge"), ("ExtraTrees", "Extra Trees"), ("CatBoost", "CatBoost")):
            old = next(r for r in cumulative if r["model"] == model and r["window"] == "120")
            new = next(r for r in segmented if r["model"] == model and r["reference"] == "cumulative")
            base, cum, seg = map(float, (old["year_equal_rmse_W0"], old["year_equal_rmse_weather"], new["year_equal_rmse_segmented"]))
            expected = f"| {label} | {base:.3f} | {cum:.3f} | {seg:.3f} | +{100*(seg/cum-1):.2f}% error |"
            if expected not in readme:
                findings.append({"code": "README_RESULT_TABLE_DIFFERS_FROM_CSV", "file": "README.md", "model": model})
            if int(new["n_years"]) != 5 or int(new["n_environment_years"]) != 129:
                raise ValueError("Unexpected support")
        if any(r["exploratory_engineering_signal"].lower() != "false" for r in segmented):
            findings.append({"code": "README_NEGATIVE_SCREEN_CLAIM_REQUIRES_UPDATE", "file": "README.md"})
    except (OSError, KeyError, ValueError, StopIteration) as error:
        findings.append({"code": "AGGREGATE_EVIDENCE_CHECK_UNAVAILABLE", "type": type(error).__name__})
    return findings


def check(root=ROOT):
    root = Path(root).resolve()
    findings, notes, total = [], [], 0
    try:
        top, inventory = git_inventory(root)
    except (OSError, subprocess.CalledProcessError):
        return {"status": "NEEDS_REVIEW", "findings": [{"code": "NO_READABLE_GIT_INVENTORY"}],
                "notes": ["No files staged, initialized, modified or published."]}
    if top != root:
        findings.append({"code": "ENCLOSING_REPOSITORY_IS_NOT_PROJECT_ROOT"})
    has_license = any((root / name).is_file() for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"))
    rights_note = root / "docs/CODE_RIGHTS.md"
    documented_no_license = (rights_note.is_file() and
                             "No license is granted for this repository's original code."
                             in rights_note.read_text())
    if not has_license and not documented_no_license:
        findings.append({"code": "ORIGINAL_CODE_LICENSE_DECISION_PENDING"})
    for relative in inventory:
        found, size = check_file(root, relative)
        findings += found
        total += size
    findings += check_links(root, set(inventory))
    findings += check_readme_results(root)
    notes.extend([
        "Read-only candidate scan; not a Git-history, complete secret, rights or binary-asset audit.",
        "Git ignore rules do not exclude already tracked files; those were also scanned.",
        "10 MiB is this project's lightweight-file policy, not a hosting-provider limit.",
        "Frozen local manifests must not be edited to redact paths; create separately identified release copies if needed.",
        "No Git initialization, staging, commit, deletion, upload, model fitting or publication performed.",
    ])
    return {"status": "NEEDS_REVIEW" if findings else "READY_WITHIN_STATIC_SCAN_SCOPE",
            "candidate_files": len(inventory), "candidate_bytes": total, "findings": findings, "notes": notes}


class SyntheticTests(unittest.TestCase):
    def test_home_path_without_leaking_matched_value(self):
        sample = "/" + "Users" + "/" + "synthetic_person" + "/project/file"
        found = text_findings(sample, "synthetic.txt")
        self.assertEqual(found, [{"code": "LOCAL_HOME_PATH", "file": "synthetic.txt", "line": 1}])
        self.assertNotIn("synthetic_person", json.dumps(found))

    def test_official_url_not_local_home(self):
        self.assertEqual(text_findings("https://data.cyverse.org/iplant/home/shared/example.csv", "source.md"), [])

    def test_high_signal_secret_pattern(self):
        self.assertEqual(text_findings("ghp_" + "a"*36, "synthetic.txt")[0]["code"], "GITHUB_TOKEN")

    def test_private_artifact_and_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.joblib").write_bytes(b"\0test")
            found, _ = check_file(root, Path("model.joblib"))
            self.assertEqual(found[0]["code"], "PRIVATE_OR_GENERATED_ARTIFACT")
            with (root / "large.bin").open("wb") as stream:
                stream.truncate(SIZE_LIMIT + 1)
            found, _ = check_file(root, Path("large.bin"))
            self.assertIn("FILE_EXCEEDS_PROJECT_10_MIB_POLICY", {f["code"] for f in found})

    def test_outside_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            found, _ = check_file(Path(temporary), Path("../personal.txt"))
            self.assertEqual(found[0]["code"], "OUTSIDE_PROJECT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests)
        raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    result = check(args.root)
    # Never print matched text or file contents. Long finding lists remain bounded.
    print(json.dumps({**result, "findings": result["findings"][:50],
                      "total_findings": len(result["findings"]),
                      "finding_counts": {code: sum(f["code"] == code for f in result["findings"])
                                         for code in sorted({f["code"] for f in result["findings"]})}},
                     ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "READY_WITHIN_STATIC_SCAN_SCOPE" else 1)
