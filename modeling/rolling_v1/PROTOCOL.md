# Supplementary rolling-origin evaluation v1

Fixed on 2026-10-18 (Asia/Shanghai), before running the new folds.

## Scope and status

The user authorized (1) multi-year temporal backtesting and (2) year/environment
and seen/unseen subgroup stability analysis. No new model families, new weather
features, hyperparameter-budget expansion, final report, website refresh or
publication belongs to this run. The completed 2024 experiment is immutable.

This extension was designed after inspecting the original 2024 results. The
historical labels were previously used in that experiment's training/validation.
Consequently these are supplementary retrospective, computationally time-separated
backtests, NOT newly blinded independent confirmation. No favorable-year selection,
post-test tuning, or replacement of the original 2024 recommendation is allowed.

## Prespecified folds

| Test year | Development training | Parameter selection | Final refit |
|---|---|---|---|
| 2020 | 2014–2018 | 2019 | 2014–2019 |
| 2021 | 2014–2019 | 2020 | 2014–2020 |
| 2022 | 2014–2020 | 2021 | 2014–2021 |
| 2023 | 2014–2021 | 2022 | 2014–2022 |
| 2024 | Existing frozen experiment; reuse unchanged | 2023 | 2014–2023 |

Use the existing prepared eligible Env × Hybrid cohort and positive observed-yield
targets. All five windows within a fold use identical sample IDs. Do not rerun raw
data cleaning or change eligibility. This is a fixed complete-calendar/available-
target cohort, not a prospective all-farm population. Count actual evaluated rows
and environments from outputs; original 2024 has 10,057 inputs but only 9,486
observations in 22 scored environments. Retain the 571 unlabeled exclusions.

## Features and models

Reuse the original feature definitions and pure functions. Fit genotype mean
imputation and 100-component PCA on unique training hybrids only; fit static
medians, site one-hot encoding and Ridge standardization on the corresponding
training split only. Never reuse a preprocessing estimator fitted on later years.
Weather windows remain [planting, planting + horizon), with clipping before
interior interpolation, then training-window medians. Retain original planting-
date mixture weights, 21 physical weather summaries and 9 quality features.
Only samples up to the fold test year enter feature generation. Target columns
must not enter feature generation.

Keep W0/W30/W60/W90/W120 and Ridge, ExtraTrees, CatBoost. Import the original
CANDIDATES unchanged: 2 candidates per model/window, 30 development fits and 15
final refits per new fold, plus GlobalMean and HistoricalSiteMean baselines.
Seed 20261018 and 4 model/BLAS threads. No early stopping or test-dependent search.
Each model/window is selected on the preceding year's environment-equal RMSE;
the overall validation recommendation is retained, never selected on test scores.
W0 already contains genotype, location and planting information. Window contrasts
measure equally budgeted prediction pipelines, not fixed-parameter causal effects.

Persist and hash the selection, all 17 test prediction files, then score test
outcomes. Historical target source files contain several years: selection/training
must explicitly restrict target joins to permitted IDs. This ordering is a
computational audit safeguard, not a claim that historical labels were unseen.

## Prespecified analysis

1. Retain all 17 configurations, all five test years and all evaluated environments.
   Primary metric: environment-equal RMSE (Mg/ha). Retain MAE, pooled RMSE, bias,
   within-environment Pearson/Spearman and fractional-tie Top-20% Recall.
2. Primary weather contrast: W120 minus W0 within EACH of the three model families.
   Negative RMSE differences mean improvement. W30/60/90 are secondary contrasts.
   Compute paired differences on identical environments and sample membership.
3. Per-year results: paired environment bootstrap (2,000 draws, fixed seed),
   relative gain, improving/worsening/unchanged environment counts and proportions,
   and distribution of environment-level differences. Preserve all negative results.
4. Cross-year primary point estimate: equal-weight mean of the five per-year
   environment-equal estimates. Also provide environment-year-weighted means,
   explicitly labeled as a different estimand, not row-weighted pseudo-replication.
   Report both relative difference of year-equal mean RMSE and the mean of annual
   relative gains in separately named columns; never treat them as interchangeable.
5. Cross-year uncertainty: report year-block bootstrap and a crossed year/site-code
   cluster-bootstrap sensitivity analysis (2,000 draws). Resample years as whole
   blocks; for the crossed version additionally draw one common set of site-code
   multiplicities, shared across all years. Within each sampled year normalize
   site weights, then average using year multiplicities. Reject/redraw an empty
   sampled-year/site intersection. Do not bootstrap individual hybrid rows.
   Apply each site multiplicity to each environment row, preserving environment-
   equal weighting rather than collapsing multiple same-code environments first.
   The original within-year environment bootstrap remains for comparability, but
   does not itself account for same-code environments within a year.
   Only five years, overlapping training sets and site-code proxies limit the
   intervals. These conditional, approximate sensitivity intervals do not include
   preprocessing/model refitting or selection uncertainty and are not a guarantee
   about future years. No formal significance or universal threshold claims.
6. Apply the original within-year earliest-sufficient-window rule unchanged; report
   its frequency across years. Do NOT define a new pooled universal sufficient day.
7. Seen/unseen hybrids and site codes are relative to each fold's final-refit
   training data. Compute subgroup metrics and paired weather differences within
   identical subgroup members. Show sample and environment support; one-environment
   groups have no informative resampling interval. Experiment_Code is a released
   site-code proxy, not verified physical-location identity. These descriptive cuts
   are not leave-location-out experiments or evidence of geographically independent
   transfer. No new spatial holdout experiment is in scope.

## Integrity, resumption and deliverables

All new features, results and machine-readable checks live under modeling/rolling_v1.
Hash the protocol, source inputs, code dependencies and completed unit artifacts;
resume only after verifying hashes. Do not overwrite old experiments or silently
reuse changed caches. A completed model/window unit must not be retrained merely
because a later unit failed. Refit only the genuinely new earlier-year folds.

Deliver executable extension scripts, resumable logs/manifests, complete CSV/JSON
result tables and independent checks. No final narrative report or PDF is requested.
Do not change the existing showcase or original report in this run.
