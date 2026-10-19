# Segmented-weather representation experiment v1

Specified on 2026-10-18, before building or fitting the new representation.

## Question, scope and status

Does replacing cumulative physical weather summaries with fixed 30-day summaries
improve maize yield prediction under the same temporal split and search budget?
This is the core of the authorized third upgrade direction. This run does not add
GBLUP/RKHS, genotype-by-weather interactions, spatial holdouts, a final report,
website changes, publication, or further hyperparameter searches.

All five cumulative-weather test-year results have already been inspected. This
is an exploratory, retrospective representation experiment, not a fresh blind
test, a new algorithm, or independent confirmation. Its thresholds are fixed
before new results, but the hypothesis itself was selected after the earlier
results. Keep all original and rolling_v1 artifacts unchanged.

## Fixed population and time splits

Use the exact prepared samples, ordering, scored-target intersection and splits
from the completed cumulative experiment for each year and stage:

| Test year | Development training | Validation | Final training |
|---|---|---|---|
| 2020 | 2014–2018 | 2019 | 2014–2019 |
| 2021 | 2014–2019 | 2020 | 2014–2020 |
| 2022 | 2014–2020 | 2021 | 2014–2021 |
| 2023 | 2014–2021 | 2022 | 2014–2022 |
| 2024 | 2014–2022 | 2023 | 2014–2023 |

Years 2020–2023 map to modeling/rolling_v1; 2024 maps to the original
modeling/features and modeling/results directories. The original 2024 completion
SHA-256 must remain c9cce89304d08087b48fc0c2f0bfdb52794eaee4b89aeb97a1b7cee51d38581a.
The controlling rolling analysis SHA-256 is
28b56456a512dff413b5d8a343daa2710df130f58e1fadd206f0126aaa0e575d and its independent
validation is PASS. Verify source chains, not just the text of that status.

Expected scored support remains 57,673 environment-hybrid observations and 129
environment-years. These are not 57,673 independent weather experiments or 129
independent locations. The original 2024 cohort has 10,057 prediction inputs;
retain the 571 missing-target exclusions and score the same 9,486 observations.
Do not rerun cleaning or change eligibility, yield aggregation or planting weights.

## Weather representation and preprocessing

1. Reuse each fold/stage's exact W0 matrix, sample metadata, fitted genetic PCA,
   imputation and static/site encoding. Do not refit these identical components.
2. For horizon H in 30/60/90/120, crop daily data to [planting, planting + H).
   Apply the same bounded interior interpolation as the cumulative pipeline to
   this entire available horizon, then use that existing fold/stage/H's training-
   only weather medians. Never use H120-cleaned data for shorter horizons.
3. Split the cleaned horizon into nonoverlapping [0,30), [30,60), [60,90),
   [90,120) day blocks as available. Compute the existing 21 physical summaries
   per block, in the original order and with unchanged definitions/thresholds.
4. Keep the original nine whole-horizon quality features unchanged and append
   them once. Do not add segment-specific missingness or availability features.
   Weather feature counts are 30 at H30, 51 at H60, 72 at H90 and 93 at H120.
5. Compute nonlinear summaries separately for every actual planting date, then
   use the original planting_weight mixture. Do not summarize an averaged date.
6. Recompute H30 across all source samples and require exact float32 equality
   with the original H30 matrix before byte-copy reuse. W0/H30 source files and
   all sample IDs/order must remain byte-identical. Assert whole-horizon quality
   columns and all base columns match the original arrays at every H.

These are fixed post-planting calendar blocks, NOT measured vegetative/flowering/
grain-filling stages. No flowering date, future weather, harvest measurement or
test outcome enters feature generation. The representation replaces cumulative
physical features rather than appending to them: e.g. a dry spell spanning a
block boundary is no longer represented as one full-horizon longest dry spell.

Feature dimensionality and the behavior of regularization/feature subsampling
change. Consequently this is an equally budgeted prediction-pipeline comparison,
not a constant-complexity test or a causal isolation of temporal resolution.

## Fitting, selection and reuse

Keep Ridge, ExtraTrees and CatBoost, their two original candidates per window,
seed 20261018 and four threads. Keep W0/W30/W60/W90/W120 and both simple baselines.
Reuse verified source W0/H30 predictions, candidates, fitted-model references and
choices, plus both simple baselines. Never invoke a fit for these shared units.
Only H60/H90/H120 segmented models are new: 18 development fits and 9 final fits
per year, 135 total. The segmented representation selects its own candidate on
the preceding validation year's environment-equal RMSE; it must not inherit the
cumulative H60/H90/H120 choice merely for convenience.

Freeze all 15 model/window choices and the validation recommendation before final
prediction. Persist/hash-lock all 17 final prediction files before connecting
test outcomes for scoring. Earlier access to historical outcomes is acknowledged;
this ordering is only a computational safeguard. Restrict training/validation
target joins to permitted sample IDs. Access 2024 held-out outcome values only
after that fold's new prediction lock. Missing targets remain missing.

Reuse a completed unit only after verifying source, protocol, code, input and
artifact hashes. Keep stage locks, append-only progress, atomic feature staging,
and explicit reused-versus-new unit metadata. Checkpoint at complete model/window
units: never refit a completed unit. An interrupted incomplete unit may retry its
candidate fits; this is not candidate-level checkpointing. The 135-fit count is
the planned unique configuration budget; progress events retain any extra attempts
and must be reported if retries occur. Record feature counts and fit times for
inspection, not as a controlled runtime benchmark.

## Prespecified comparisons and interpretation

- Primary comparisons: separately for each model, H120 segmented minus H120
  cumulative; also necessarily H120 segmented minus W0. Negative RMSE differences
  mean improvement. The two references answer different questions.
- H60/H90 comparisons are secondary. H30 and W0 are identity/reuse checks.
  Show all models, horizons and years, including negative results.
- Primary metric is environment-equal RMSE (Mg/ha), then equal weight to each
  of the five years. Also retain MAE, bias, within-environment correlations and
  fractional-tie Top-20% Recall. Pair identical rows, outcomes and environments.
- Retain annual paired environment bootstrap and cross-year year-block plus
  crossed year/site-code sensitivity intervals, 2,000 draws, seed 20261018, using
  the existing rolling_v1 definitions. Do not resample hybrid rows as independent
  weather evidence. Preserve improvement/worsening proportions and magnitudes.
- For each primary model/reference comparison, flag an exploratory engineering
  improvement signal only if year-equal relative RMSE reduction is at least 2%,
  at least 4 of 5 annual point estimates improve, and the crossed sensitivity
  interval's upper bound is below zero. This is a project-specific operational
  screen, not a published statistical law or family-wise significance test.
  Report the components even when the conjunction fails. To call a model's new
  weather representation promising, both cumulative and W0 screens must pass.
- Apply the existing within-year sufficient-window rule unchanged to the new
  complete W0–W120 pipeline. Do not create a pooled universal sufficient day.
- Provide descriptive seen/unseen hybrid and site-code cuts relative to each
  fold's final training data. Include empty groups, counts and evaluated years.
  These are not spatial holdouts; site codes are not verified physical locations.

Only five years, overlapping training histories, selected complete-calendar/
observed-target cohorts, post-result hypothesis choice and repeated comparisons
limit inference. Intervals are conditional on fitted models, exclude selection/
refit uncertainty and are unadjusted for multiplicity. Even passing the screen
supports only a retrospective exploratory signal within this cohort. Neither a
failure nor a success establishes a causal weather effect or agricultural safety.

## Outputs and verification

Save protocol, new feature matrices, model artifacts/references, frozen choices,
paired predictions, complete CSV/JSON comparisons and independent checks under
modeling/segmented_v1. The final report and presentation remain deferred.
Independent verification must cover target/time isolation, shared-unit exact
reuse, new-only fit counts, source/artifact chains, selection from validation,
prediction-before-scoring order, direct metric arithmetic, bootstrap arithmetic,
full five-year coverage, subgroup support and model replay for the new saved models.

## Methodological context

Fixed temporal weather windows are established, not a novel method. Rogers and
Holland compared several window widths, including 30 days, and studied transfer
to untested environments: [G3, jkab440](https://academic.oup.com/g3journal/article/12/2/jkab440/6486423).
Westhues et al. used observed phenology to define developmental-stage covariates;
our fixed calendar blocks deliberately do not claim that alignment:
[Frontiers in Plant Science, 699589](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2021.699589/full).
Neither paper establishes a guaranteed gain for our data, features or validation.
