# Summary

Maize yield prediction: the incremental value and temporal representation of weather

- **Core question.** How much does weather observed within 30, 60, 90 or 120 days after planting improve maize yield prediction beyond genetic and static information? Does retaining fixed 30-day weather blocks help more than cumulative summaries?
- **Main finding.** At W120, cumulative weather reduced CatBoost's five-year, year-equal mean RMSE by **0.72%**, from **2.5332 to 2.5149 Mg/ha**. The paired uncertainty interval includes zero; the evidence does not establish a dependable weather gain.
- **Representation extension.** Replacing cumulative physical summaries with fixed 30-day blocks did not produce stable improvement. At W120, segmented RMSE was higher on average for all three model families; CatBoost was worse than its cumulative counterpart in all five test years.
- **Evidence base.** Public Genomes to Fields (G2F) trials support five test years, 2020-2024: **57,673 environment-hybrid observations, 129 environment-years and 31 released site codes**. These are not 129 independent physical sites. The rolling and segmented extensions are retrospective and exploratory.

## Research design

- **Prediction unit and inputs.** The target is mean grain yield for an environment-hybrid combination, measured in Mg/ha. W0 is a rich baseline: 100 genetic principal components plus location/site and planting information, with no weather features. It is not an uninformed mean predictor; two simple mean baselines are also retained.
- **Temporal evaluation.** For test year t, fit on 2014 through t-2, select parameters on t-1, freeze the choice, then refit through t-1. Compare Ridge, Extra Trees and CatBoost at W0/W30/W60/W90/W120, with two preset candidates per model-window. All 17 predictions are locked before each final scoring step; test scores never select parameters.
- **Fair representation comparison.** The segmented extension reuses identical samples, fitted static/genetic preprocessing, W0/W30 predictions and simple baselines. Only W60/W90/W120 are newly fitted, with independent validation-based choices: 90 candidate fits and 45 final refits. Each 30-day block has 21 physical summaries; nine whole-window quality features remain unchanged. Calendar blocks are not measured crop-development stages.

## Three research difficulties and the corresponding design

- **Avoiding temporal leakage.** Fit genetic imputation/PCA and other preprocessing on each fold's training data only. Clip weather at the available horizon before interpolation; never use later-window weather to construct an earlier prediction.
- **Separating information value from changed samples or tuning.** Keep the cohort, sample order, target aggregation and two-candidate search budget paired. Preserve planting-date mixture weights and verify shared arrays and predictions exactly. This controls the comparison, but not model complexity: segmentation changes dimensionality and feature-subsampling behavior.
- **Respecting correlated observations.** Score each environment by RMSE, average environments equally within year, then average the five years equally. Use paired environment, year-block and crossed year/site-code bootstrap sensitivity analyses with 2,000 draws, rather than treating hybrid rows as independent weather replicates. Track within-environment ranking separately from absolute yield error.

## Contribution

- A reproducible, leakage-aware comparison of established models, paired weather ablations and a falsifiable representation extension, including negative findings. No new algorithm, state-of-the-art performance, publication or institutional collaboration is claimed. Implementation was AI-assisted; individual contribution statements require separate verification.

<!-- PAGEBREAK -->

## Main results

Five-year, year-equal environment RMSE (Mg/ha; lower is better). W120 uses weather within 120 days after planting in the year being predicted.

| Model | W0 | Cumulative W120 | Segmented W120 |
| --- | ---: | ---: | ---: |
| Ridge | 2.5398 | 2.6020 | 2.6925 |
| Extra Trees | 2.5794 | 2.5781 | 2.6135 |
| CatBoost | 2.5332 | 2.5149 | 2.6281 |

Paired RMSE difference and 95% crossed year/site-code sensitivity interval (Mg/ha). Negative differences favor the added-weather or segmented pipeline.

| Model | Cumulative W120 minus W0 | Segmented minus cumulative W120 |
| --- | ---: | ---: |
| Ridge | +0.062 [-0.049, +0.224] | +0.090 [-0.026, +0.216] |
| Extra Trees | -0.001 [-0.132, +0.150] | +0.035 [-0.099, +0.183] |
| CatBoost | -0.018 [-0.213, +0.165] | +0.113 [-0.052, +0.247] |

- **Weather increment.** Cumulative W120 changed mean RMSE by +2.45% for Ridge, -0.05% for Extra Trees and -0.72% for CatBoost relative to W0. CatBoost improved in three of five years. Small average gains and intervals spanning zero do not support a universal weather benefit in this pipeline.
- **Segmentation result.** Relative to cumulative W120, segmented RMSE rose by 3.48%, 1.37% and 4.50%, respectively. No model passed the predefined exploratory improvement screen against both cumulative W120 and W0. No sufficiently early window was declared in either representation under the existing within-year rule. These are descriptive comparisons, not a new test-selected recommendation.

## Limitations and next step

- **Retrospective evidence.** Earlier-year outcomes had already been available to the project, and the segmented hypothesis followed inspection of cumulative results. Five years, overlapping training histories, repeated site codes and a selected observed-target cohort limit generalization. Keep the 571 unlabeled 2024 inputs out of scoring; do not impute their yield.
- **Conditional uncertainty.** Bootstrap intervals are approximate sensitivity summaries, conditional on fitted pipelines. They exclude refitting and model-selection uncertainty and are not multiplicity-adjusted. A site code is only a location proxy; this is not an independent spatial-transfer experiment.
- **Meaning of the result.** An unsuccessful representation does not show that weather is agriculturally unimportant. Fixed calendar blocks, simple genetic features and a small parameter budget may miss useful relationships. Historical weather availability has not been verified in real time, so this is not a validated operational prediction service or a causal agronomic result. A useful next step is a prospectively frozen external-year evaluation with verified input availability.

## Reproducibility

- **Evidence trail.** Numerical audits passed for both extensions; the segmented audit also replayed all 45 new final models. Tables and protocols are under `modeling/rolling_v1/` and `modeling/segmented_v1/`; independent receipts are each directory's `validation_checks.json`. The exact summary values are in `analysis/cross_year_stability.csv` and `analysis/cross_year_comparisons.csv`, respectively.

Data: Genomes to Fields, DOI [10.25739/78mn-4394](https://doi.org/10.25739/78mn-4394). Rebuild this brief from verified evidence with `python scripts/build_research_brief.py` in the documented project environment; no models are retrained.
