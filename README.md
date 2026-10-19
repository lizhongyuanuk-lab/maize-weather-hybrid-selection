# Maize weather-aware yield prediction and hybrid selection

An independent, retrospective research project using public Genomes to Fields
multi-environment trials. It tests **whether lower yield-prediction error also
helps identify high-yielding hybrids within an environment**. Earlier benchmark
experiments ask whether weather adds predictive value and whether fixed 30-day
summaries improve on cumulative summaries. This is not a new large language
model, a published paper, or a deployed agricultural decision service.

## Find your way around

The repository follows the research workflow, not a web-app layout:

| Stage | Start with | What it does |
|---|---|---|
| Data audit | [`audit/audit_g2f.py`](audit/audit_g2f.py), [`preparation/README.md`](preparation/README.md) | Inspect source files, eligibility and source provenance. |
| Cohort preparation | [`preparation/prepare_data.py`](preparation/prepare_data.py) | Build environment–hybrid targets and aligned trial inputs. |
| Original benchmark | [`modeling/build_features.py`](modeling/build_features.py), [`modeling/run_experiments.py`](modeling/run_experiments.py) | Fit the initial 2024 Ridge/Extra Trees/CatBoost weather-window comparisons. |
| Forward-year validation | [`modeling/run_rolling_pipeline.py`](modeling/run_rolling_pipeline.py), [`modeling/rolling_v1/PROTOCOL.md`](modeling/rolling_v1/PROTOCOL.md) | Repeat the locked comparison over test years 2020–2024. |
| Weather representation | [`modeling/run_segmented_pipeline.py`](modeling/run_segmented_pipeline.py), [`modeling/segmented_v1/PROTOCOL.md`](modeling/segmented_v1/PROTOCOL.md) | Compare cumulative and 30-day-block weather summaries on matched cohorts. |
| Stress/selection follow-up | [`modeling/stress_v1/RESULTS.md`](modeling/stress_v1/RESULTS.md), [`modeling/selection_v1/PROTOCOL.md`](modeling/selection_v1/PROTOCOL.md) | Test the VPD/heat extension, within-environment selection and error decomposition. |
| Manuscript | [`reports/selection_paper/manuscript.md`](reports/selection_paper/manuscript.md), [`scripts/build_selection_paper.py`](scripts/build_selection_paper.py) | Present the current decision-focused synthesis and figures. |

If you are reviewing the work rather than rerunning it, start with the result
below, then read the [manuscript](reports/selection_paper/manuscript.md), the
[rolling aggregate tables](modeling/rolling_v1/analysis/) and the
[segmented aggregate tables](modeling/segmented_v1/analysis/). The source data,
saved models and row-level predictions are intentionally **not** in GitHub;
this public snapshot supports inspection and lightweight tests, not a
one-command reconstruction of the historical model archive.

## Start here: selection-focused manuscript

The current [English research draft](reports/selection_paper/manuscript.md) and
[review PDF](output/pdf/MAIZE_SELECTION_MANUSCRIPT_DRAFT.pdf) analyze matched
held-out predictions from five forward-year tests. They ask a decision-oriented
question: does adding weather improve the selection of the highest-yielding
fifth of hybrids in each trial environment, not just absolute-yield RMSE?

- The genetic-and-static CatBoost baseline (W0) had **28.72%** top-20% recall;
  adding 120 days of *observed* post-planting weather (W120) raised it to
  **29.77%**. The point-estimate gain was **1.05 percentage points**, but the
  crossed year/site-code sensitivity interval was **[-0.38, +2.89] points**.
- Most of the point-estimated squared-error reduction reflected better
  calibration of environment means, not a comparably large improvement in
  discrimination among hybrids within an environment.
- An exploratory VPD/heat extension lowered RMSE slightly but reduced observed
  top-20% recall to **29.13%**. These increments are uncertain; SHAP attribution
  is a model diagnostic, not a causal stress finding.

The selection-centered analysis was formulated after the original RMSE
benchmark. It is **retrospective and exploratory**, with no independent new-site
confirmation or validated pre-planting use. Read the [methods and caveats](reports/selection_paper/manuscript.md)
before treating a point estimate as an agricultural recommendation.

## Summary

- Five chronological test years: **2020–2024**, **57,673 scored environment×hybrid
  observations** and **129 environment-years**, spanning 31 site-code proxies.
  These are not 57,673 independent weather experiments or 129 independent sites.
- Ridge, Extra Trees and CatBoost use W0/W30/W60/W90/W120 and two simple
  baselines. Hyperparameters are selected on the preceding year; each test year
  is scored only after predictions are saved and locked.
- Cumulative W120 weather did **not demonstrate stable improvement** over W0.
  CatBoost's year-equal RMSE reduction was **0.72%**, with a crossed year/site-code
  95% sensitivity interval for ΔRMSE of **[−0.213, +0.165] Mg/ha**.
- The 30-day segmented representation **did not improve the primary W120
  comparison**. Its five-year mean RMSE was higher than cumulative weather for
  all three models; none passed the prespecified two-reference improvement
  screen. This negative result is retained.
- The completed segmented run passed **38,785 independent checks**. All **45
  new final models** were reloaded, reproducing **524,196 predictions** with a
  maximum absolute difference of **7.11×10⁻¹⁵**. Verification demonstrates
  computational consistency, not general agricultural validity.
- A separately isolated [VPD/heat stress extension](modeling/stress_v1/RESULTS.md)
  tested three physically motivated features against the saved CatBoost W120
  reference across the same five years. Its **0.51% point-estimate RMSE gain was
  not stable**, and Top-20% hybrid selection did not improve. Held-out SHAP
  figures are retained as model diagnostics, not causal findings.

## Main results

Environment-equal RMSE within each year, then equal weight to all five years;
units are Mg/ha (numerically tonnes/hectare). Lower is better.

| Model | No weather, W0 | Cumulative W120 | Segmented W120 | Segmented change vs cumulative |
|---|---:|---:|---:|---:|
| Ridge | 2.540 | 2.602 | 2.692 | +3.48% error |
| Extra Trees | 2.579 | 2.578 | 2.613 | +1.37% error |
| CatBoost | 2.533 | 2.515 | 2.628 | +4.50% error |

Sources: [cumulative cross-year comparisons](modeling/rolling_v1/analysis/cross_year_stability.csv),
[segmented primary comparisons](modeling/segmented_v1/analysis/primary_assessment.csv)
and [all yearly metrics](modeling/segmented_v1/analysis/yearly_metrics.csv).
These are descriptive comparisons, not permission to replace a validation-frozen
recommendation using test results.

For segmented W120 minus cumulative W120, crossed 95% sensitivity intervals are
Ridge **[−0.026, +0.216]**, Extra Trees **[−0.099, +0.183]** and CatBoost
**[−0.052, +0.247] Mg/ha**. All cross zero: the point estimates do not establish a
universal penalty for segmentation. Ridge segmented W120 versus W0 was worse in
all five years; its conditional crossed interval was **[+0.011, +0.337] Mg/ha**.

The model-level screen requires at least 2% relative RMSE improvement, improvement
in at least four of five years, and a crossed interval entirely below zero,
**against both cumulative W120 and W0**. [No model passed](modeling/segmented_v1/analysis/model_upgrade_assessment.csv).
The unchanged within-year rule found [no supported earliest sufficient window](modeling/segmented_v1/analysis/yearly_sufficient_windows.csv)
in either representation.

## Study design and evidence boundaries

- **Target:** mean valid positive plot yield for each environment×hybrid
  combination. Missing 2024 yields are not zero-filled: 9,486 of 10,057 inputs
  are scored; 571 remain unscored.
- **Temporal splits:** for test year t, development uses 2014–(t−2), validation
  uses t−1, and final refitting uses 2014–(t−1).
- **Common base:** training-fitted genotype PCA (100 components), site-code
  encoding, coordinates and planting day-of-year. Preprocessing uses training
  data only.
- **Weather:** each available horizon is cropped before interpolation/imputation.
  The segmented extension replaces 21 cumulative physical features with 21
  features per 30-day block, preserving the same nine whole-horizon quality
  features. Actual planting-date features are combined using fixed plot weights.
- **Equal search budget:** two candidates per model/window. Segmentation changes
  feature dimensionality and model regularization/subsampling behavior; it is
  not a constant-complexity causal isolation of temporal resolution.
- **Evaluation:** paired environments, year-equal effects, 2,000-draw environment,
  year-block and crossed year/site-code sensitivity intervals. Rows are not
  bootstrapped as independent weather evidence.

The original 2024 experiment is preserved. Historical rolling backtests were
added subsequently; the segmented hypothesis was chosen **after reviewing the
cumulative results**. The extensions are exploratory retrospective validation,
not fresh blind confirmation.

Five years, overlapping training sets, limited candidate searches, selected
complete-calendar/observed-target cohorts and site codes that are not verified
physical locations limit generalization. Intervals condition on fitted models,
omit selection/refit uncertainty and are not multiplicity-adjusted. Fixed
calendar blocks are not measured growth stages. Real-time weather availability,
true spatial holdout and prospective deployment safety were not evaluated.

## Repository map

| Path | Purpose |
|---|---|
| [Research design](RESEARCH_DESIGN.md) | Original question and design history |
| [Original protocol](modeling/PROTOCOL_LOCK.md) | Initial 2024 evaluation |
| [Rolling protocol](modeling/rolling_v1/PROTOCOL.md) | Five-year temporal extension |
| [Segmented protocol](modeling/segmented_v1/PROTOCOL.md) | Fixed representation comparison and decision rules |
| [Stress extension results](modeling/stress_v1/RESULTS.md) | Isolated VPD/heat comparison, G×E follow-up, SHAP diagnostics and validation |
| [Preparation](preparation/README.md) | Eligibility, aggregation and joins |
| [Rolling aggregate tables](modeling/rolling_v1/analysis/) | Year/environment/subgroup stability |
| [Segmented aggregate tables](modeling/segmented_v1/analysis/) | Both references, all windows, positive and negative results |
| [Reproduction guide](docs/DATA_AND_REPRODUCIBILITY.md) | Data, recorded dependencies, audit and training commands |
| [Release checklist](docs/PUBLICATION_CHECKLIST.md) | Repository boundary, privacy, rights and publication decisions |
| [Selection manuscript](reports/selection_paper/manuscript.md) | Current decision-focused paper draft and journal-style figures |
| [Application summary](reports/ENGLISH_PROJECT_SUMMARY.md) | Short, evidence-bounded English project description |
| [Code rights](docs/CODE_RIGHTS.md) | Public inspection without an original-code license |
| [Research materials](reports/) | Briefs and historical supporting presentations; check each one's scope |

The lightweight repository excludes raw data, row-level prediction archives,
models, feature matrices and machine-specific manifests. They remain in the
complete local archive; aggregate CSVs alone cannot reproduce a full saved-model
audit. Original 2024-only notebook/PDF generators remain historical and do not
automatically include the later five-year study.

## Quick checks

Clone the code and run the standard-library publication checks (no source data
download or model fitting):

```bash
git clone https://github.com/lizhongyuanuk-lab/maize-weather-hybrid-selection.git
cd maize-weather-hybrid-selection
python3 scripts/check_publication.py --self-test
python3 scripts/check_publication.py
```

For synthetic preparation/modeling tests, use Python 3.12 and install the
[recorded direct dependencies](modeling/requirements.txt) into a separate
environment first. These tests do not download the G2F dataset or fit the
completed study's real models:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r modeling/requirements.txt
.venv/bin/python -m unittest discover -s preparation -p 'test_*.py'
.venv/bin/python -m unittest discover -s modeling -p 'test_*.py'
.venv/bin/python modeling/validate_segmented_results.py --self-test
```

Full archive checks and the retained original training commands are in
[Data and reproducibility](docs/DATA_AND_REPRODUCIBILITY.md). A fresh clone is
**not** a one-command reproduction of the pinned historical manifests; their
absolute paths and source hashes require a deliberate versioned migration.

## Data, rights and project status

Data: Genomes to Fields (2025), *Genomes to Fields 2024 Maize Genotype by
Environment Prediction Competition*, CyVerse Data Commons, DOI
**10.25739/78mn-4394**. The [official catalog](https://dc.cyverse.org/dataset/genomes_to_fields_2024_maize_genotype_by_environment_prediction_competition)
lists PDDL for the dataset (checked 2026-10-18). This does not license every
third-party asset or this project's code. The owner chose public access without
granting an original-code license; this is not advertised as an open-source
release. Third-party rights must be checked before adding further assets.

This directory is the publication boundary: the enclosing workspace contains
unrelated personal materials and must not be uploaded. The independent local
presentation and admission-site repositories are excluded from this code
release. Public visibility does not grant reuse rights; see
[Code rights](docs/CODE_RIGHTS.md). Review the [release checks](docs/PUBLICATION_CHECKLIST.md)
before publishing additional artifacts.
