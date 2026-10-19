# Data, environment and reproducibility

## Three different reproducibility claims

1. **Read the evidence:** small aggregate CSVs under modeling/rolling_v1/analysis/
   and modeling/segmented_v1/analysis/ support the README comparisons without
   loading a model.
2. **Audit the completed local archive:** original inputs, features, saved
   models, predictions and hash-linked manifests are required. Independent
   validators recalculate metrics and inspect provenance; the segmented
   validator also replays its 45 new final models.
3. **Recompute on a new machine:** retrieve data, install the recorded
   dependencies and run preparation and the original experiment in a new
   workspace. Numerical results and byte-identical historical manifests are
   different goals. Timestamps, absolute paths and platform-dependent floating
   point behavior mean fresh manifests need not have the archived hashes.

The rolling and segmented guards pin the completed historical experiment.
**A fresh clone is not a one-command reconstruction of that archive.**
A provenance migration for an independently regenerated base run requires a
new, explicitly versioned protocol and source bindings. Do not edit historical
hash anchors or silently label a new run as the old one.

## Official data and attribution

Genomes to Fields (2025). *Genomes to Fields 2024 Maize Genotype by Environment
Prediction Competition*. CyVerse Data Commons. DOI **10.25739/78mn-4394**.

- [Official dataset and resource directory](https://dc.cyverse.org/dataset/genomes_to_fields_2024_maize_genotype_by_environment_prediction_competition)
- [Genomes to Fields resources](https://www.genomes2fields.org/resources/)
- [Initial public mirror](https://github.com/imbaterry11/UFEED_Maize_GxE)

The official catalog listed **Open Data Commons Public Domain Dedication and
License (PDDL)** when checked on 2026-10-18. This is the catalog's statement about
the dataset, not a license for this project's code, every mirror repository,
article, website asset or dependency. Read resource notes and retain attribution.
Raw data are intentionally not bundled here.

The archived provenance comparison found:

- Training and 2024 weather files are byte-identical to the official files.
- Training trait values match after the documented line-ending conversion.
- Genotype cells and ordered keys match after the documented declaration/header/
  delimiter conversion.
- The 2024 metadata mirror contained eight differing cells. Modeling used the
  official metadata copy, including its missing coordinates.

This establishes source equivalence under those transformations, not that all
field observations are error-free. See the [provenance workflow](../preparation/provenance_followup/README.md)
and [data preparation](../preparation/README.md).

## Obtaining inputs

The official catalog contains a competition README and training/testing
directories. The original workflow reads audit/data/raw/ and prefers matching
official replacements under preparation/official_inputs/. Exact local input
names and SHA-256 values are recorded in preparation/outputs/manifest.json in
the complete archive.

The initial mirror-blob and official-target retrieval logic is in
audit/audit_g2f.py, an audit script **not a complete clean-install downloader**.
The provenance follow-up supports resumable retrieval/comparison of four large
official files, not every required input. Follow its README. With those copies
present, its offline comparison command is:

    python preparation/provenance_followup/verify_official_sources.py --compare-only

No all-input download wrapper or redistributed dataset is provided. Retrieve
and check required sources before preparation; do not substitute fabricated
data. Keep raw inputs unchanged.

## Recorded execution environment

Scientific runs recorded Python **3.12.14**, NumPy **2.3.5**, pandas **2.2.3**,
SciPy **1.18.1**, scikit-learn **1.7.2**, CatBoost **1.2.8** and joblib **1.6.0**.
These are archived manifest values, not a newly verified availability claim for
every platform.

[Modeling requirements](../modeling/requirements.txt) pin observed direct
dependencies, including notebook/PDF tooling; [preparation requirements](../preparation/requirements.txt)
are the smaller preparation set. Neither is a complete transitive lockfile.
Install separately, without replacing a working experiment's dependencies:

    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r modeling/requirements.txt

Use that environment's interpreter as python below. The original machine's
.runtime/ dependency directory is not portable source and is excluded from Git.
The local presentation under showcase/ has separate runtime/build requirements
and is excluded from this public research-code repository.

## Lightweight checks: no real training or data download

From the project root:

    python scripts/check_publication.py --self-test
    python scripts/check_publication.py
    python modeling/validate_segmented_results.py --self-test
    python -m unittest discover -s preparation -p 'test_prepare_data.py' -v
    python -m unittest discover -s modeling -p 'test_*.py' -v

The publication checker only reads candidate files and Git state. It never
stages, deletes, initializes, commits, uploads or alters experiment artifacts.
Its static scan is a guardrail, not a complete security/privacy audit.

## Full local-archive verification: no fitting

These commands require the excluded data/features/predictions/models; aggregate
CSVs alone are insufficient:

    python preparation/validate_prepared_data.py
    python preparation/provenance_followup/test_verify_official_sources.py
    python modeling/validate_results.py --stage development
    python modeling/validate_results.py --stage final
    python modeling/replay_saved_predictions.py
    python modeling/validate_rolling_results.py
    python modeling/validate_segmented_results.py

Validators read historical outcomes to verify completed evaluations; they do
not train or select a new model. The segmented validator replays 45 saved new
final models, including all 10,057 prediction inputs in 2024, not only 9,486
scored rows. Physical-weather calculations are an independent deterministic
**sample**; base/quality matrix equality and shared-file reuse are checked in full.

## Original experiment: deliberate recomputation

**These commands build features and train models.** Use a fresh, explicitly
identified workspace with all inputs; never overwrite the frozen archive to
test installation.

    python preparation/prepare_data.py
    python modeling/build_features.py --mode development
    python modeling/run_experiments.py --stage development
    python modeling/build_features.py --mode final
    python modeling/run_experiments.py --stage final

Development selects on 2023 only. Frozen choices are required before final
feature construction and the 2014–2023 refit. The completed model×window unit is
the resumption boundary: matching completed units are reused, but an incomplete
unit can retry a candidate. Do not delete artifacts or relax hashes to bypass a
provenance mismatch.

In the matching completed archive, the versioned extensions use:

    python modeling/run_rolling_pipeline.py --workers 2
    python modeling/run_segmented_pipeline.py --workers 2

These verify source chains and preserve completed units. They are not default
viewing commands or promises of portability to a newly generated source archive.
Each model uses four threads; two workers can use up to eight threads. The
segmented run completed 90 candidate fits and 45 final fits; W0/W30 and both
baselines were reused.

## Legacy presentation regeneration

Original notebook/PDF builders describe the original 2024 experiment.
Running them does **not** automatically add the later five-year results and
may replace presentation files. Preserve current materials and inspect any
outputs before using these historical commands:

    python modeling/build_research_notebook.py --execute
    python modeling/build_project_brief.py
    python modeling/build_showcase.py

Notebook execution needs a working Jupyter kernel; PDF visual verification
additionally needs an external renderer such as pdftoppm. Generator success
is not visual or scientific validation.
