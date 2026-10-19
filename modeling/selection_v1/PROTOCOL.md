# Selection-oriented G×E pilot (locked before any new fit)

Date: 2026-10-19 Asia/Shanghai. This is an **exploratory retrospective**
follow-up, designed after the original and stress-extension test results were
seen. It is neither a new blinded evaluation nor external site/year
validation. All new code and artifacts live in `modeling/selection_v1/`;
original feature matrices and saved baselines remain read-only.

## Question and decision target

Can a model trained to predict a hybrid's *deviation from its training
environment mean* select high-yielding hybrids in future test years better
than the existing absolute-yield CatBoost W120 model? Does the frozen W120
weather matrix add selection value to this centered-target model relative to
the same model with W0 features? The final comparison is made within each
held-out environment, not over pooled rows.

## Fixed arms and fitting

- Frozen reference: saved CatBoost W120 absolute-yield test predictions.
- Centered W0: CatBoost regression on the existing W0 matrix, trained on
  `yield - mean(training yield within Env)`.
- Centered W120: the same learner and target on the existing W120 matrix.

The centered target is calculated from **training environments only**. Test
environment means are never used to generate predictions; an environment-
constant offset would not change within-environment rankings anyway. The
W120 model can learn nonlinear genotype × environment patterns from the
existing genomic PCs and weather features, but no biological interaction is
presumed. Both new arms use the original CatBoost two-depth budget (depth 6
or 8, 600 iterations, seed 20261018). In each chronological fold, the
immediately preceding year selects depth using environment-equal Top-20%
recall, with environment-equal Spearman as the first tie-breaker and lower
depth as the second. After selection, refit on all prior years and predict
the test year. All five test years 2020–2024 must be kept; do not choose a
favorable subset. Reuse the exact frozen sample IDs and W0/W120 matrices.

## Endpoints and interpretation

Primary: equal-year mean of environment-equal Top-20% recall. Report paired
candidate-minus-reference differences, yearly signs, and year-block plus
crossed year/site-code sensitivity intervals. Secondary: environment-equal
Spearman and centered within-environment MSE. Centered predictions are not
absolute yield forecasts and must not be compared to absolute-yield RMSE.

The two comparisons are (1) centered W120 versus frozen absolute-yield W120
and (2) centered W120 versus centered W0. A practically compelling pilot
would improve Top-20% recall by at least 0.02 absolute, favor at least four
of five years, and have a crossed year/site-code interval excluding zero.
These are *descriptive screening criteria*, not a claim of statistical
significance or fresh hypothesis confirmation, because outcomes informed
this follow-up's design. Report all results even if neither comparison passes.

## Leakage and reproducibility

Select hyperparameters from the preceding validation year only. For every
final year, save all test predictions and their SHA-256 locks **before**
joining that year's observed yields. Verify reference/candidate cohort
identity, finite values, no duplicate sample IDs, and year partitions.
Site codes are proxies, not guaranteed unique physical locations. Full-season
W120 weather is retrospective, not a pre-planting input. This pilot cannot
support a claim of generalization to genuinely untouched years or sites.
