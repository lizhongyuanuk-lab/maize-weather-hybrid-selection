# Stress-aware maize yield prediction: completed exploratory extension

Completed 2026-10-19. This extends the existing Genomes to Fields (G2F)
project; it does **not** replace the frozen original, five-year rolling, or
segmented experiments. New code and outputs live only in `modeling/stress_v1/`.

## One-sentence answer

Adding a compact VPD/heat representation to the established 120-day CatBoost
baseline **did not demonstrate a reliable predictive improvement** across five
held-out years, and it **did not improve within-environment top-20% hybrid
selection**. SHAP confirms the model uses the new VPD feature, but provides no
validated stress-response threshold or causal mechanism.

## Research question and comparison

Can atmospheric dryness and heat intensity, represented by mean and 90th
percentile daily VPD *proxies* plus heat excess above 35 °C, add predictive
information beyond the existing cumulative 120-day weather features, genotype
PCs and site/planting metadata? The comparison holds the eligible
environment×hybrid cohort, five chronological folds and CatBoost's two-option
search budget fixed. The reference is the **saved W120 CatBoost predictions**;
the candidate appends only the three features specified in [PROTOCOL.md](PROTOCOL.md).
No measured canopy VPD or new weather source was added: these are nonlinear
re-expressions of existing daily temperature and humidity data.

The outcome is plot-mean yield, Mg/ha. The primary metric is mean RMSE across
environments within a year and then across years with equal year weights. The
secondary metric is the analogous Top-20% recall of high-yielding hybrids
*within each environment*. Smaller RMSE and larger recall are better. This is
retrospective model evaluation, not an intervention or causal test.

## Primary result

| Held-out year | Scored rows | Environments | W120 RMSE | +Stress RMSE | ΔRMSE | W120 Top-20% | +Stress Top-20% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 11,298 | 25 | 2.2178 | 2.1850 | −0.0328 | 0.2389 | 0.2350 |
| 2021 | 13,615 | 27 | 2.3510 | 2.3436 | −0.0074 | 0.3570 | 0.3612 |
| 2022 | 11,783 | 28 | 2.8554 | 2.8808 | +0.0254 | 0.2351 | 0.2238 |
| 2023 | 11,491 | 27 | 2.4570 | 2.4429 | −0.0141 | 0.3388 | 0.3361 |
| 2024 | 9,486 | 22 | 2.6934 | 2.6578 | −0.0356 | 0.3187 | 0.3004 |
| **Equal-year mean** | **57,673** | **129 environment-years** | **2.5149** | **2.5020** | **−0.0129** | **0.2977** | **0.2913** |

The point-estimate RMSE gain is **0.51%**, with four of five years favorable.
The year-block 95% resampling interval for candidate minus reference is
**[−0.0302, +0.0066] Mg/ha**; the crossed year/site-code sensitivity interval
is **[−0.0628, +0.0338] Mg/ha**. Both include zero. Top-20% recall changes by
**−0.0064** (−0.64 percentage points) and improves in only one year. Thus the
accuracy signal is too small/uncertain to support an upgrade, and it cannot be
described as better hybrid selection. The 31 site codes are proxies, not 31
verified independent geographic sites. These resampling intervals condition on
the fitted models and do not include model-selection or hypothesis-search
uncertainty. See the [machine-readable summary](results/summary.json),
[yearly results](results/yearly_metrics.csv), and [paired environments](results/paired_environments.csv).

## Explicit genetic × stress follow-up

After seeing the primary outcome, we ran a separately labelled **post-hoc**
test: matched Ridge pipelines, additive versus 100 genomic PCs multiplied by
train-standardized mean VPD and heat excess. This tests one regularized
interaction implementation, not the existence of biological G×E.
The interaction model's year-equal RMSE was 2.5980 versus 2.6114 Mg/ha
(Δ −0.0135; year-block interval [−0.1504, +0.1079], inconclusive). Its Top-20%
recall was **0.2296 versus 0.2445** (Δ −0.0150), worse in **all five years**.
Therefore this implementation does not rescue the breeding-selection objective.
See [exploratory design](GXE_EXPLORATORY.md) and [results](gxe_results/summary.json).

## SHAP: what the fitted model used, not what weather caused

Held-out TreeSHAP was computed on **five random scored hybrids per environment**:
645 rows across all 129 environment-years. Environment means receive equal
weight within each year; years receive equal weight in the ranking. Among 182
candidate-model features, VPD p90 ranked **11th** (mean |SHAP| 0.099 Mg/ha),
mean VPD ranked **26th** (0.042), and heat excess ranked **48th** (0.019).
The older `hot30_days` feature ranked 3rd (0.251). VPD p90 is therefore used by
the model, but its SHAP rank does **not** establish an out-of-year accuracy gain.

The same sampled data contain strong overlap among environmental features:
the 129 environment-year means have Spearman correlation **0.977** for VPD
mean versus VPD p90, **0.899** for VPD p90 versus heat excess, and **0.988**
for heat excess versus existing `hot35_days`. Redundancy is a plausible
*explanation* of limited incremental predictive value, not a demonstrated
cause. Exact pairwise CatBoost SHAP interactions were computed separately for
**44 scored hybrids in 22 environments in 2024**. In that sample, mean absolute
heat-excess × mean-VPD interaction was **0**, heat-excess × selected PC002 was
**0.00019 Mg/ha**, and mean-VPD × existing humidity mean was **0.00581 Mg/ha**.
PC002 was selected after inspecting the SHAP ranking, so its panel is purely
illustrative. Twenty-six of the 44 sampled rows had zero heat excess. These
plots do not justify a VPD/heat threshold or a causal yield statement.

The visual sequence follows the supplied paper's roles: [global ranking
(Fig. 4 style)](explain/fig4_style_feature_rank.png), [single-feature dependence
(Fig. 6 style)](explain/fig6_style_dependence.png), and [pairwise model
interactions (Fig. 7 style)](explain/fig7_style_interactions.png). The labels
explicitly say *model diagnostic* because the predictive gate failed. Numeric
[SHAP summary](explain/summary.json) and [sample-level values](explain/) are
retained; the latter must not be mistaken for a census of all 57,673 rows.

## Architecture and audit trail

```text
Public G2F trial/weather/genotype data
  -> existing preparation and eligible environment×hybrid target
  -> frozen W120 feature matrices + saved CatBoost reference predictions
  -> stress_v1/features.py: append 3 predeclared VPD/heat summaries
  -> stress_v1/run.py: preceding-year selection, five final-year predictions,
                       SHA locks before target joins, paired evaluation
  -> stress_v1/gxe.py: separately labelled post-hoc interaction test
  -> stress_v1/explain.py: held-out SHAP diagnostics after accuracy assessment
  -> stress_v1/validate_results.py: independent raw-2024 metric/hash checks
```

The [validation record](validation.json) passes: all five candidate and
reference prediction hashes match their locks; yearly/environment summaries
reconcile; and independent recomputation of raw 2024 RMSE and Top-20% recall
matches all 22 reported environments to floating-point precision. It confirms
computational consistency, **not** external validity. The original W120 and
other extension files were not overwritten.

## Interpretation for an MSc application

The strongest honest contribution is a **well-controlled negative result**:
physically motivated stress features can receive meaningful model attribution
without supplying a stable incremental prediction or selection benefit.
The project demonstrates dataset construction, leakage-aware temporal
evaluation, paired baselines, uncertainty, G×E hypothesis testing, and careful
interpretation of explainability. It is **not** a demonstrated new state-of-the-
art predictor, an agricultural decision product, or a Q1-journal-ready causal
discovery. The next scientific step would be a locked external-year/site
validation with phenology-aware stress windows and genuinely independent
weather/soil or management information—not another post-hoc threshold mined
from these five years. Full-season observed weather also cannot be claimed as
available for pre-planting prediction.

For presentation, lead with the research question, the matched W120 baseline,
the five-year table, and the negative conclusion; use the SHAP panels to show
*why a visually interesting explanation is insufficient on its own*. Keep the
original [project README](../../README.md) as the broader project history.

## Data and method references

- [G2F 2024 public dataset catalog and DOI](https://dc.cyverse.org/dataset/genomes_to_fields_2024_maize_genotype_by_environment_prediction_competition).
- [FAO-56](https://www.wcc.nrcs.usda.gov/ftpref/wntsc/waterMgt/irrigation/fao56.pdf)
  for the saturation-vapor-pressure relation; our mean-temperature/RH VPD is
  explicitly a proxy, not a measured canopy or midday value.
- [CatBoost SHAP definition](https://catboost.ai/docs/en/concepts/shap-values)
  and [feature-importance types](https://catboost.ai/docs/en/features/feature-importances-calculation).
- [SHAP's warning on causal interpretation of predictive-model explanations](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html).
- The figure roles are adapted from the user-supplied *Energy* (2025) article;
  the models, data and conclusions here are independent of that article.
