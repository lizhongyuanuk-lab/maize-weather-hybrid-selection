# Stress-representation extension v1 (locked before new model fits)

Date: 2026-10-19 Asia/Shanghai. This is a retrospective, post-hoc extension
designed after reading the frozen W0/W120 and segmented results. It is not a new
blind confirmation and must not overwrite those experiments.

## Question and target

Does a compact representation of atmospheric dryness and heat intensity improve
out-of-year maize yield prediction, relative to the existing cumulative W120
weather representation? The target and eligible environment x hybrid cohort are
unchanged: valid positive plot-mean yield in Mg/ha, not plot-level independent
weather samples. This experiment tests predictive utility, not a causal effect.

## Prespecified features

Use the fixed 120-day calendar window beginning at each actual planting date.
Apply the original within-window interpolation and training-window median
imputation separately in each chronological fold. Add exactly three physical
features to the frozen W120 matrix:

1. `vpd_proxy_mean_kpa`: mean of daily 0.6108*exp(17.27*T2M/(T2M+237.3))
   *(1-RH2M/100), where T2M is degrees C and RH2M is percent.
2. `vpd_proxy_p90_kpa`: 90th percentile of the same daily proxy.
3. `heat_excess35_cday`: sum(max(T2M_MAX-35, 0)) in C-days.

The daily-mean temperature/humidity VPD is an atmospheric-demand proxy, not
measured midday canopy VPD. These features are nonlinear re-expressions of
already available weather variables, not new independent meteorological data.
The original W120 matrix already contains days >=30/35 C, humidity summaries,
and weather-quality fields. No explicit VPD x heat product enters the primary
model, so pairwise model interactions remain interpretable. For mixed planting
dates, calculate nonlinear features per date first, then use the existing
plot-count planting weights. Do not use yield, post-harvest variables, future
years, or an imputer fitted on validation/test weather.

## Fixed comparisons

- **Reference**: saved CatBoost W120 predictions and selected 2-candidate model
  from the original/rolling experiments; verify sample IDs and file hashes.
- **Candidate**: CatBoost on W120 plus the three features above, with the same
  two candidates, seed and four-thread budget as the reference. Select candidate
  using the immediately preceding validation year; refit on all earlier years.
- Test years 2020-2024 retain the exact existing chronological folds. Historical
  backtests and the new 2024 feature hypothesis are post-hoc relative to
  previously available outcomes. All five folds must be reported.

Primary numeric endpoint: year-equal average of environment-equal RMSE; paired
candidate-minus-reference difference (negative favors candidate). Secondary:
environment-equal Top-20% recall and within-environment Spearman; report their
year-equal paired differences, with denominators and undefined correlations.
Do not claim a selection benefit unless Top-20% recall itself improves; do not
claim yield-accuracy benefit based on Top-20% recall alone. Report per-year signs,
environment counts, 2,000-draw paired environment intervals and year/site-code
cluster sensitivity intervals. A useful gain should have a nontrivial effect,
consistent annual direction and an interval excluding zero; isolated favorable
years or overlapping intervals are exploratory, not demonstrated improvement.

Pre-existing baseline files are read-only. Persist candidate predictions and
their SHA-256 hashes before joining test targets. Never select features, model
depth, plot panels, subgroups or thresholds using test outcomes.

## Explanation gate

Only after the primary predictive comparison is complete, compute held-out
TreeSHAP for the candidate. Show (i) grouped/individual held-out importance,
(ii) SHAP dependence for VPD proxy and heat intensity (analogous to reference
article Fig. 6), and (iii) pairwise VPD x heat and weather x genetic-PC model
interaction contributions (analogous to its Fig. 7). The article's Fig. 4,
not Fig. 6, is the global feature-ranking analogue. Explanations must retain
environment/year support, use environment-aware sampling/aggregation, and be
checked against actual held-out observations before making a condition-specific
claim. Genetic PCs cannot be named as causal loci. SHAP does not prove causality.

If the candidate lacks stable predictive gain, any SHAP plots are diagnostic
only and must not be presented as validated agronomic mechanisms. Full-season
observed W120 weather is retrospective and not an available pre-planting input.
