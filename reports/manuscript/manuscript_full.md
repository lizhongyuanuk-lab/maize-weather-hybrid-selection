# Incremental value of within-season weather information for maize yield prediction: a five-year retrospective benchmark

## Abstract

Weather covariates are increasingly available for crop prediction, but their incremental predictive value depends on the information already represented in a model and on the evaluation setting. We assessed whether weather observed during the first 30, 60, 90 or 120 days after planting improved maize yield prediction beyond genetic and static site information, and whether fixed 30-day weather blocks were more useful than cumulative summaries. Public Genomes to Fields data were evaluated in five temporally ordered test years, 2020-2024, comprising 57,673 environment-hybrid observations in 129 environment-years. Ridge regression, extremely randomized trees and CatBoost were compared using identical eligible samples, training-only preprocessing and equal, limited hyperparameter-search budgets. Parameters were selected using the preceding year; predictions were generated from refits containing only earlier years. At 120 days, cumulative weather changed year-equal, environment-averaged root mean squared error from 2.5398 to 2.6020 Mg/ha for Ridge, from 2.5794 to 2.5781 Mg/ha for extremely randomized trees, and from 2.5332 to 2.5149 Mg/ha for CatBoost. The CatBoost reduction was 0.72%, with a paired difference of -0.018 Mg/ha and a 95% crossed year/site-code bootstrap sensitivity interval of [-0.213, 0.165]. Replacing cumulative features with calendar-block features increased average error at 120 days for all three families. No representation satisfied the study's sufficient-window rule. These retrospective results do not establish a dependable weather increment under the tested pipelines. They motivate evaluating weather representations against informative non-weather baselines and reporting temporal instability, rather than interpreting additional environmental detail as a guaranteed predictive improvement.

**Keywords:** maize; yield prediction; environmental covariates; temporal validation; weather representation; genomic information

## 1. Introduction

Predicting the performance of maize hybrids in future field trials requires combining information about genetic material with information about the environments in which that material will be grown. These information sources are not interchangeable. A model may distinguish high- and low-yielding environments without accurately ranking hybrids within an environment, while a model with useful within-environment rankings may remain poorly calibrated for absolute yield. Consequently, the value of an environmental predictor should be evaluated against a clearly specified prediction target, baseline and deployment setting, rather than inferred from its agronomic relevance alone.

The Genomes to Fields (G2F) initiative provides a public setting for such evaluation. Its multi-environment trials connect phenotypic observations, genetic information and environmental records. The public 2024 yield-prediction competition release additionally supplies historical records and subsequent test-year observations, allowing a reproducible retrospective assessment (Genomes to Fields, 2025; Chen et al., 2026). This study uses environment-hybrid mean yield as its target. It does not address county-level crop production, field-wide commercial yield or the causal response to an agronomic intervention.

Prior work has already integrated genetic and environmental predictors in maize. Rogers and Holland (2022) compared environmental aggregation windows of 5, 10, 15 and 30 days and examined prediction under several genetic and environmental separation schemes. Their results showed that improvements from environmental information and marker-by-environment terms depended on the relationship between training and test environments. Westhues et al. (2021) investigated gradient boosting with genomic and environmental predictors and used observed phenology to construct developmental-stage weather covariates. These studies establish that environmental representation and validation design both matter. They also mean that neither adding weather nor dividing a season into temporal windows constitutes a methodological novelty by itself.

Covariance-based reaction norms offer another established approach for jointly representing genetic, environmental and interaction effects, initially demonstrated in wheat by Jarquín et al. (2014). Within G2F maize, Jarquin et al. (2021) evaluated climatic information in combining-ability models and did not obtain a consistent additional benefit from their naive weather-covariate representation. These results motivate a baseline-relative question: the usefulness of a particular environmental encoding must be established empirically, and a negative increment should not automatically be dismissed as an implementation failure.

Three practical questions remain important for a specific predictive pipeline. First, does observed weather add information beyond a baseline that already contains genetic structure, location and planting date? Such a contrast is more demanding than comparison with a global mean, because static features may already encode substantial differences among environments. Second, is an apparent benefit maintained across successive test years, or is it driven by a favorable year? Third, does preserving coarse temporal ordering through calendar blocks improve prediction at an equivalent observation horizon? The last question involves a trade-off: retaining timing information also changes dimensionality, regularization and the summaries available to a learner.

We examined these questions using three established model families, four post-planting weather horizons and five temporally ordered test years. The original 2024 evaluation was extended with retrospective rolling-origin backtests for 2020-2023. After examining the cumulative-weather results, we specified an exploratory comparison using nonoverlapping 30-day calendar blocks. All comparisons retained the same eligible observations within each fold, separated preprocessing and parameter selection from the corresponding test year, and assessed paired changes within model families. This design emphasizes the incremental performance of equally budgeted pipelines rather than a competition to select a winner from the test results.

The contribution is a bounded empirical benchmark, not a new learning architecture. It combines an informative non-weather baseline, explicit information horizons, temporal evaluation, paired representation comparisons and a transparent account of negative findings. We ask whether cumulative weather improves absolute yield error, whether the direction of that improvement is stable across years, and whether a more detailed representation remains useful after accounting for its uncertainty and computational design. We separately examine within-environment ranking and the support available for previously unseen hybrids and site codes. All claims are conditional on the observed cohort, model families and restricted search space.

## 2. Materials and methods

### 2.1. Data sources, prediction target and common cohort

We used public Genomes to Fields (G2F) multi-environment maize trial data, including numerical genotypes, trial metadata, yield observations and released daily weather derived from NASA POWER (Genomes to Fields, 2025; Chen et al., 2026). The target was yield in Mg ha−1 for a unique environment-hybrid combination, not individual-plot yield or regional production. For 2014-2023, repeated plots were aggregated by the arithmetic mean of finite, positive observed yields; missing, non-finite and non-positive values were excluded from that mean without target imputation. The 2024 target was the official released environment-hybrid observation. Of 10,057 prediction inputs across 23 environments, 9,486 observations across 22 environments were scored; 571 unlabeled inputs remained in prediction files but were excluded from every scored comparison.

Eligibility required genotype and environment records, a site code, resolved planting dates and all calendar dates needed through 120 days after planting. Numerical weather values could nevertheless be missing. Models and horizons used identical eligible samples and scored targets within each fold. Consequently, shorter-window estimates were conditional on complete 120-day calendar coverage, rather than prospective coverage of all trials. Historical data retained 94,377 combinations from 2014-2022 and 11,491 from 2023. The five test years, 2020-2024, contributed 57,673 scored combinations in 129 environment-years and 31 site codes, not 57,673 independent weather experiments or 129 distinct physical sites.

Missing planting dates were resolved using a unique known date within the environment-hybrid combination, then within the environment; unresolved combinations were excluded. Multiple dates retained their original plot-count weights, independent of outcomes. Historical combinations whose labeled-plot date distribution differed from the all-plot distribution were excluded to align exposure and target mixtures; this rule did not select 2024 inputs by label availability. Yield-derived diagnostics were excluded from predictors. Official metadata replaced differing mirror entries, preserving missing official coordinates. Weather sentinels −999 and −9999 and non-finite values were treated as missing. Source identity or documented format equivalence, join cardinalities and file fingerprints were verified. These checks established source traceability, not the measurement accuracy of every field observation. Each environment identifier belonged to one year, while hybrids and site codes could recur; the latter were not assumed to identify unchanged physical locations.

### 2.2. Genetic and static predictors

The no-weather reference, W0, contained 100 genetic principal components plus latitude, longitude, planting day of year, coordinate-missingness indicators and one-hot encoded site code. Marker-wise mean imputation and principal component analysis of 2,425 numerical markers were fitted on unique training hybrids, avoiding repeated weighting of hybrids present in multiple environments. Principal component analysis used centering without marker variance standardization, a randomized solver, four power iterations and seed 20261018.

Static imputation medians and site categories were training-fitted; unseen codes received an all-zero encoding. Historical planting day of year was averaged over source plots. Hybrid names, environment identifiers and year were not predictors. W0 therefore already represented genetic, location and planting information. Preprocessing was independently fitted within each development or final training period, then applied to validation and test inputs. All models shared the resulting numerical inputs; the site-encoding dimension varied with the training vocabulary, while genetic and other static feature definitions remained fixed.

### 2.3. Time-bounded weather representations

Horizons H ∈ {30, 60, 90, 120} used [planting date, planting date + H), excluding day H. Seven daily variables were retained: mean, maximum and minimum 2-m air temperature, precipitation, downward surface shortwave radiation, 2-m relative humidity and 2-m wind speed. Data were cropped before imputation. Bounded interior gaps were linearly interpolated; remaining values were filled using variable-specific medians from distinct training environment-date observations within the relevant horizon. Repeated hybrids did not multiply observations used to estimate these medians. Inputs were observed retrospective weather, not weather forecasts; real-time release latency was not evaluated.

Each cumulative window contained 21 physical summaries: temperature mean and population standard deviation; mean and maximum daily maximum temperature; mean and minimum daily minimum temperature; counts of maximum temperature ≥30 °C and ≥35 °C; accumulated growing degree days; precipitation total; wet-, heavy-rain- and dry-day counts; longest dry spell; wet-day precipitation intensity; radiation sum and mean; humidity mean and population standard deviation; wind mean; and mean daily temperature range. Wet, dry and heavy-rain thresholds were respectively ≥1, <1 and ≥10 mm. Wet-day intensity was zero without wet days. Daily growing degree days were max{[min(Tmax, 30) + max(Tmin, 10)]/2 − 10, 0}, with temperatures in °C. Radiation retained released units without an unverified conversion.

Nine quality predictors comprised the seven original missing fractions, any-interpolation indicator and count of original days complete for all seven variables. Cumulative weather therefore added 30 features to W0. For segmentation, each H-day horizon was cleaned identically before splitting into available non-overlapping 30-day blocks. The 21 physical summaries were computed per block, while the nine whole-horizon quality features appeared once, yielding 30, 51, 72 and 93 weather predictors at H = 30, 60, 90 and 120. Interpolation could cross block boundaries within H, but not exceed H. Blocks represented calendar time, not measured phenological stages.

Nonlinear summaries were calculated separately for each actual planting date before plot-count-weighted averaging. These mixture features became fully available at the latest constituent cutoff; they did not describe one averaged planting date or trajectory. Quality indicators could become fractional after averaging. Segmented physical summaries replaced cumulative summaries, potentially losing cross-boundary longest-spell information. W0 and W30 were exactly equivalent across representations and reused after verification. Thus, segmentation changed dimensionality and information encoding; equal search budgets did not ensure equal model complexity.

### 2.4. Rolling-origin training and model selection

For test year t ∈ {2020, 2021, 2022, 2023, 2024}, development training used 2014 through t − 2, parameter selection used t − 1, and final refitting used 2014 through t − 1. Preprocessing was refitted at each stage; random row splitting across years was avoided. This design addresses structured-data validation and leakage concerns (Roberts et al., 2017; Kapoor and Narayanan, 2023).

![Figure 1. Temporally ordered development, validation and test splits. Each row is one test-year evaluation. Model parameters and preprocessing are fitted using development training data, candidates are selected using the immediately preceding year, and final refits include both earlier portions before test prediction. The original 2024 evaluation was reused unchanged for cumulative weather. Historical backtests and the representation extension are retrospective, not prospectively independent confirmation.](figures/figure1_temporal_design.png)

Ridge regression (Hoerl and Kennard, 1970) compared α = 100 and 10,000 after training-fitted standardization. Extremely randomized trees (Extra Trees; Geurts et al., 2006) used 160 trees, depth 18 and a maximum-feature fraction of 0.8, comparing minimum leaf sizes of 5 and 20. CatBoost (Prokhorenkova et al., 2018) used 600 iterations, learning rate 0.05, L2 leaf regularization 5 and RMSE loss, comparing depths 6 and 8 without early stopping. Applicable stochastic operations used seed 20261018 and four threads. Training observations were not explicitly environment-weighted; environment-equal weighting governed selection and evaluation.

Each model-window pair compared two candidates using preceding-year environment-equal RMSE. Segmented H60-H120 selected candidates independently of cumulative versions under the same budget. Both representations retained all model-window results. Simple baselines predicted the training-wide mean or historical site-code mean, falling back to the former for unseen codes. Validation-selected recommendations and all final prediction files were frozen before scoring. Each fold retained 15 model-window predictions and two baseline predictions. Only segmented H60-H120 required new fitting: three models, three horizons and two development candidates plus one final refit over five years produced 135 new fits. Shared W0/W30 and baseline units were not refitted.

The original 2024 experiment remained unchanged. However, earlier-year outcomes had already been available in its development data, and the rolling extension followed inspection of its results. The segmentation hypothesis was selected after inspecting cumulative five-year outcomes. Prediction-before-scoring safeguards therefore do not make these extensions newly blinded confirmation: they are retrospective exploratory evaluations.

### 2.5. Metrics and paired estimands

The primary annual metric averaged environment-specific RMSE equally; the primary cross-year metric then averaged the five annual values equally:

RMSE<sub>e,t</sub> = [(1/n<sub>e,t</sub>) Σ<sub>i</sub> (ŷ<sub>i,e,t</sub> - y<sub>i,e,t</sub>)<sup>2</sup>]<sup>1/2</sup>

R<sub>t</sub> = (1/E<sub>t</sub>) Σ<sub>e</sub> RMSE<sub>e,t</sub>;     R = (1/5) Σ<sub>t</sub> R<sub>t</sub>

ΔR = R<sub>candidate</sub> - R<sub>reference</sub>;     Gain (%) = 100 (R<sub>reference</sub> - R<sub>candidate</sub>)/R<sub>reference</sub>

Here, n_et denotes scored hybrids in environment e and year t, E_t denotes that year's environment count, y and ŷ are observed and predicted yields, and R_t and R are annual and year-equal RMSE. Environments, not hybrid rows, received equal annual weight; years received equal cross-year weight. Pooled-row RMSE and the environment-year-weighted mean were distinct secondary estimands.

Contrasts were paired candidate-minus-reference RMSE differences, with negative values favoring the candidate. Primary cumulative contrasts compared W120 against W0 within each model family. Primary segmented contrasts compared W120 against both cumulative W120 and W0; H30-H90 comparisons were secondary. All comparisons preserved identical scored members and outcomes. Relative reduction was 100 × (R_reference − R_candidate)/R_reference, distinguished from the average of annual percentage reductions.

Secondary metrics were MAE, signed bias (prediction minus observation), pooled errors, within-environment Pearson and Spearman correlations, and Top-20% Recall. Correlations with fewer than two observations or constant vectors remained undefined; averages retained only defined environments with support reported. For Top-20% Recall, k = max{1, ceiling(0.2n)}. Both observed and predicted rankings assigned unit inclusion above the kth-value boundary, zero below it and equal fractional inclusion at boundary ties, totaling k. Recall was the dot product of inclusion vectors divided by k, preventing identifier-based tie-breaking.

### 2.6. Uncertainty and exploratory decision rules

Within-year uncertainty used 2,000 paired environment bootstrap replicates and percentile 95% intervals; hybrid rows were not resampled independently. Cross-year uncertainty included a year-block bootstrap and a crossed year/site-code sensitivity analysis motivated by crossed-array resampling (Owen, 2007), without claiming equivalence to that estimator or guaranteed coverage. The latter independently drew year and site-code multiplicities, sharing site multiplicities across years. Each environment received its site's multiplicity; weights were normalized within each sampled year before averaging with year multiplicities. Same-code environments were not first collapsed. Empty sampled year-site intersections were rejected and redrawn. Both procedures used 2,000 retained replicates and seed 20261018.

Intervals were conditional on fitted models, excluding preprocessing/refitting, selection and data-assembly uncertainty. Five years, overlapping training histories, site-code proxies and unadjusted multiple comparisons limited inference. Accordingly, intervals were approximate sensitivity summaries rather than confirmatory significance tests or future-performance guarantees.

The earliest-sufficient-window rule was applied separately to each model-year pair. W120 first had to reduce RMSE by ≥2% against W0, with its paired within-year interval's upper bound below zero. Only then was the earliest positive horizon retaining ≥90% of that W120 point-estimate gain selected. No separate non-inferiority test was applied to earlier windows, and no universal prediction day was inferred.

For segmented W120, an exploratory improvement screen required ≥2% lower year-equal RMSE, improvement in ≥4 of 5 years and a crossed interval upper bound below zero. A promising representation required passing separately against cumulative W120 and W0. These project-specific thresholds preceded segmented fitting but followed the earlier results that motivated the hypothesis; they were not a confirmatory error-control procedure.

Seen/unseen hybrids and site codes were defined relative to each fold's final training membership. Paired subgroup contrasts reported sample, environment and year support, including empty groups. Single-environment groups had no informative resampling interval. These descriptive temporal-test cuts were not spatial holdouts or verified geographic transfer tests.

### 2.7. Reproducibility and interpretation

Frozen protocols, input hashes, configurations, sample identities, predictions and saved models were retained. Completed units were reused only after source and artifact checks. Independent verification recomputed metrics, checked temporal and target boundaries and shared-unit equivalence, and replayed saved segmented models against their prediction inputs. These checks establish computational consistency, not biological replication.

The design compares equally budgeted predictive pipelines, not causal weather effects. Weather-quality features accompany physical covariates, windows select their own candidates, and segmentation changes representation complexity. No genomic relationship-matrix baseline, explicit parameterized genotype-by-weather interaction model or independent spatial validation was performed. Therefore, limited incremental gains cannot establish that weather is biologically unimportant or that all temporally resolved representations are ineffective.

## 3. Results

### 3.1. Evaluation coverage and baseline performance

The five test years comprised 57,673 scored environment-hybrid observations and 129 environment-years, with 22-28 scored environments per year (Table 1). The records represented 31 distinct released site codes. These counts describe different units: repeated hybrids within an environment share weather, and a site code is not a verified independent physical location. The 2024 prediction-input cohort contained 10,057 observations, of which 9,486 had an observed outcome suitable for scoring. The remaining 571 observations were retained as unlabeled inputs and were not assigned synthetic outcomes.

Table 1. Training and scored-test coverage by year.

| Test year | Final-fit observations | Scored observations | Environment-years | Site codes |
| --- | --- | --- | --- | --- |
| 2020 | 57,681 | 11,298 | 25 | 24 |
| 2021 | 68,979 | 13,615 | 27 | 27 |
| 2022 | 82,594 | 11,783 | 28 | 28 |
| 2023 | 94,377 | 11,491 | 27 | 27 |
| 2024 | 105,868 | 9,486 | 22 | 22 |

Final-fit observations include only years preceding the test year. Site codes are not necessarily independent physical locations; their unique count across all test years is 31, not the sum of this column. Scored observations total 57,673; environment-years total 129.

The non-weather models already achieved year-equal mean environment RMSE values between 2.5332 and 2.5794 Mg/ha (Table 2). Their similar average errors are descriptive and do not establish that the algorithms are equivalent. Each W0 model used the same genetic and static feature representation, with its own validation-selected parameter candidate. The global-mean and historical-site-mean baselines were retained as additional reference methods; their complete aggregate results are provided in Supplementary Table S1. W0 should therefore not be interpreted as a naive baseline or as a model without environmental context.

### 3.2. Incremental value of cumulative weather

At W120, cumulative weather reduced CatBoost's mean RMSE from 2.5332 to 2.5149 Mg/ha, a relative reduction of 0.72%. Extremely randomized trees changed from 2.5794 to 2.5781 Mg/ha, a reduction of approximately 0.05%. Ridge changed from 2.5398 to 2.6020 Mg/ha, an increase of 2.45%. The paired crossed year/site-code intervals for all three primary comparisons included zero (Table 3; Figure 2a). Thus, the small average improvements for the two tree ensembles were not accompanied by a clearly separated conditional sensitivity interval.

Table 2. Primary five-year mean environment RMSE (Mg/ha).

| Model | W0 | Cumulative W120 | Segmented W120 |
| --- | --- | --- | --- |
| Ridge | 2.5398 | 2.6020 | 2.6925 |
| Extra Trees | 2.5794 | 2.5781 | 2.6135 |
| CatBoost | 2.5332 | 2.5149 | 2.6281 |

Each environment receives equal weight within its test year; each of the five test years receives equal weight. Lower is better. W0 contains genetic and static information, not post-planting weather.

![Figure 2. Primary W120 paired comparisons. Points show year-equal means of environment-level RMSE differences; whiskers show 95% crossed year/site-code bootstrap sensitivity intervals from 2,000 draws. (a) Cumulative weather versus the same-family W0 baseline. (b) Segmented versus cumulative weather. Negative values favor the first pipeline. Each comparison contains five years and 129 environment-years. The intervals are conditional on fitted models and do not include selection/refitting uncertainty or multiplicity adjustment.](figures/figure2_primary_contrasts.png)

Annual differences were not uniform (Figure 3). Cumulative W120 improved the annual environment-averaged RMSE in three of five years for CatBoost, two of five for extremely randomized trees and three of five for Ridge. For Ridge, improvements in more than half of the years coexisted with a worse five-year average, demonstrating that improvement frequency does not summarize improvement magnitude. Across the 12 cumulative model-by-horizon contrasts against W0, all crossed year/site-code intervals included zero. These findings do not prove absence of useful weather information; they show that a dependable increment was not established by the tested representation, models and evaluation period.

Table 3. W120 paired differences and conditional sensitivity intervals.

| Model | Contrast | RMSE difference [95% interval] | Years improved |
| --- | --- | --- | --- |
| Ridge | Cumulative - W0 | +0.062 [-0.049, +0.224] | 3/5 |
| Ridge | Segmented - cumulative | +0.090 [-0.026, +0.216] | 1/5 |
| Ridge | Segmented - W0 | +0.153 [+0.011, +0.337] | 0/5 |
| Extra Trees | Cumulative - W0 | -0.001 [-0.132, +0.150] | 2/5 |
| Extra Trees | Segmented - cumulative | +0.035 [-0.099, +0.183] | 2/5 |
| Extra Trees | Segmented - W0 | +0.034 [-0.167, +0.243] | 2/5 |
| CatBoost | Cumulative - W0 | -0.018 [-0.213, +0.165] | 3/5 |
| CatBoost | Segmented - cumulative | +0.113 [-0.052, +0.247] | 0/5 |
| CatBoost | Segmented - W0 | +0.095 [-0.106, +0.304] | 1/5 |

Units are Mg/ha. Negative differences favor the first pipeline. Intervals use the crossed year/site-code bootstrap (2,000 draws); they are conditional on fitted models, approximate and not multiplicity-adjusted. An interval is not a prospective performance guarantee.

![Figure 3. Annual W120 weather increments relative to the same-family W0 model. Each mark is one test year's environment-equal RMSE difference; negative values indicate improvement. Circles denote cumulative weather and squares denote segmented weather. Vertical stems connect descriptive point estimates to zero and are not uncertainty intervals. All five test years are displayed; aggregate uncertainty is provided in Figure 2 and Table 3.](figures/figure3_annual_differences.png)

### 3.3. Calendar-block representation did not improve the primary comparison

At W120, the segmented representation produced mean RMSE values of 2.6925, 2.6135 and 2.6281 Mg/ha for Ridge, extremely randomized trees and CatBoost, respectively. Relative to the corresponding cumulative representation, these values represented error increases of 3.48%, 1.37% and 4.50%. CatBoost's segmented W120 error was higher than its cumulative counterpart in all five test years. The segmented-cumulative difference was favorable in one year for Ridge and two years for extremely randomized trees. Crossed intervals for the three primary segmented-cumulative differences included zero; the numerical increases should not be described as uniformly established population-level deterioration.

The pattern was not confined to the primary horizon. All nine newly fitted model-by-horizon combinations at W60, W90 and W120 had a worse five-year mean RMSE than their cumulative counterparts and their same-family W0 reference. Nevertheless, some annual comparisons favored segmentation, illustrating why neither a favorable individual year nor a single mean is sufficient to characterize transfer stability. Figure 4 reports all observation horizons, with W30 unchanged because its single block is identical to the cumulative window. None of the three model families passed the exploratory improvement screen against both required references.

![Figure 4. Weather increments at all evaluated observation horizons. Points show year-equal means of paired environment RMSE differences from W0; whiskers show 95% crossed year/site-code bootstrap sensitivity intervals. Circles denote cumulative weather and squares denote segmented weather. Horizontal offsets separate the two representations visually without changing their true horizons of 30, 60, 90 and 120 days. W30 is identical across representations. The three panels use common scales; no intermediate observation days were evaluated. All intervals are conditional and unadjusted for multiple comparisons.](figures/figure4_weather_horizons.png)

### 3.4. No sufficient observation horizon was identified

The sufficient-window rule required a supported W120 improvement of at least 2% before searching for an earlier window that retained at least 90% of its point-estimated gain. No model-year combination satisfied this rule in either representation: zero of 15 cumulative and zero of 15 segmented assessments yielded a sufficient window. This is a result about the prespecified rule, not a claim that useful prediction is impossible before day 120. In particular, W0 predictions already existed at the no-post-planting-weather baseline, and some earlier weather windows performed better in individual years. The experiment did not evaluate every possible day, nor did it test a formal noninferiority margin for an earlier horizon.

### 3.5. Ranking and subgroup support qualify the application claim

Absolute yield error and within-environment material selection describe different tasks. Table 4 therefore reports MAE, Spearman correlation and fractional-tie top-20% recall alongside the W120 comparisons. These are year-equal averages of environment-level summaries, not pooled correlations over all hybrids. For Extra Trees, changing cumulative W120 to segmented W120 increased mean Spearman correlation from 0.192 to 0.209 and top-20% recall from 0.279 to 0.287, despite the higher RMSE. CatBoost's top-20% recall changed from 0.287 at W0 to 0.298 with cumulative W120 and 0.296 with segmented W120. It would therefore be inaccurate to describe the segmented representation as worse on every metric. Undefined correlations were retained as undefined rather than replaced with zero. These secondary comparisons are descriptive; the primary uncertainty analysis concerned paired RMSE.

Table 4. Secondary error and within-environment ranking measures.

| Model | Pipeline | MAE (Mg/ha) | Spearman | Top-20% recall |
| --- | --- | --- | --- | --- |
| Ridge | W0 | 2.153 | 0.161 | 0.247 |
| Ridge | Cumulative W120 | 2.221 | 0.147 | 0.244 |
| Ridge | Segmented W120 | 2.312 | 0.163 | 0.248 |
| Extra Trees | W0 | 2.201 | 0.174 | 0.256 |
| Extra Trees | Cumulative W120 | 2.206 | 0.192 | 0.279 |
| Extra Trees | Segmented W120 | 2.246 | 0.209 | 0.287 |
| CatBoost | W0 | 2.162 | 0.221 | 0.287 |
| CatBoost | Cumulative W120 | 2.151 | 0.235 | 0.298 |
| CatBoost | Segmented W120 | 2.259 | 0.235 | 0.296 |

MAE is lower-is-better; Spearman and recall are higher-is-better. Recall is a fraction, not a percentage. For these nine model summaries, Spearman is defined in all 129 environment-years. No secondary-metric uncertainty or confirmatory superiority claim is implied.

Support for unseen material and unseen locations was highly uneven (Table 5). Only 30 scored observations belonged to previously unseen hybrids in 2021, while no unseen-hybrid observations were present in 2023. Unseen site codes occurred only in 2020 and 2024, totaling 861 observations in three environment-years. Their performance cannot establish reliable geographical transfer. These subsets were defined relative to each fold's training history rather than by an independently held-out spatial design, and they should not be interpreted as a uniform five-year validation of new-hybrid or new-location prediction.

Table 5. Scored support for previously unseen hybrids and site codes.

| Test year | Unseen-hybrid observations | Environments | Unseen-site observations | Environments |
| --- | --- | --- | --- | --- |
| 2020 | 10,551 | 25 | 439 | 2 |
| 2021 | 30 | 12 | 0 | 0 |
| 2022 | 10,999 | 27 | 0 | 0 |
| 2023 | 0 | 0 | 0 | 0 |
| 2024 | 8,314 | 22 | 422 | 1 |

Unseen status is relative to the corresponding final-fit training data. Zero indicates an empty evaluated subgroup, not zero prediction error. These cuts are descriptive and are not independent spatial holdouts.

## 4. Discussion

### 4.1. Agronomic relevance is not the same as incremental predictive value

Weather affects the conditions in which a crop develops, but this does not imply that an arbitrary weather representation must improve every predictive model. Here, the relevant estimand was the change in prediction error after adding weather to an existing genetic-and-static pipeline. Site identifiers, coordinates and planting date may already capture part of the environmental structure available in these trials. A small residual weather increment is therefore compatible with substantial environmental influence on yield. It cannot be interpreted as a causal estimate of the contribution of weather to maize production.

The observed lack of a stable gain is also compatible with limitations in the tested feature representation, training support or model search. Genetic principal components summarize broad marker variation but do not reproduce a complete genomic relationship model or explicit genotype-by-environment interaction structure. The two-candidate search was intentionally limited and equally applied; it was not intended to characterize the best achievable performance of any model family. These explanations are plausible hypotheses rather than mechanisms identified by the experiment. Distinguishing them would require additional, separately specified comparisons.

The findings align with the broader lesson that environmental information must be evaluated under the intended transfer setting. Rogers and Holland (2022) showed that the usefulness of environmental covariates and interaction terms depended on training-test environmental similarity. Our results cannot be compared directly with their reported predictive correlations: the data periods, target preparation, feature sets, algorithms and validation schemes differ. The present five-year RMSE estimates are consequently not a ranking against that study and do not support a state-of-the-art claim.

### 4.2. More temporal detail is a hypothesis, not a guaranteed upgrade

The segmented extension was motivated by the possibility that the timing of weather exposure matters beyond a whole-window average. Its negative result is informative because the representation was compared at the same observation horizons and on the same observations, with shared genetic/static preprocessing. However, it was not a pure intervention on temporal resolution. At W120 the weather block increased from 30 to 93 features, including the unchanged quality descriptors. The same Ridge penalties and tree feature-subsampling fractions therefore operated in a different-dimensional input space. Independent validation selection maintained the search budget, but not effective model complexity.

Segmentation also replaced rather than augmented cumulative physical summaries. A dry period spanning two calendar blocks was no longer represented by one whole-window longest-dry-spell feature. Accordingly, segmentation simultaneously retained some timing information, removed some cross-block information and altered the learner's input geometry. The experiment does not identify which of these changes caused the observed differences. Testing additive cumulative-plus-block representations, dimension-matched summaries or a broader training-only tuning budget would address distinct hypotheses and should not be retrofitted into the present results.

Calendar alignment is another limitation. Westhues et al. (2021) constructed stage-based covariates using observed phenology, whereas the present blocks were fixed intervals after planting. The latter avoid relying on future stage observations in feature construction, but they need not correspond to equivalent developmental states across hybrids or environments. A future phenology-aware analysis would need to specify how stage information is available at prediction time; using retrospectively observed flowering dates without such a rule could change the intended forecasting problem.

### 4.3. Temporal evaluation improves relevance but does not remove all dependence

The rolling-origin design prevented the corresponding test year's outcomes from entering its parameter selection or final model fit. It also exposed discrepancies between average improvement and year-to-year consistency. Nevertheless, the extensions remain retrospective. Historical outcomes had already been available during the original project, and the segmented hypothesis was chosen after viewing the cumulative results. Freezing computations before each extension prevents within-run adaptation but does not convert the extension into independent prospective confirmation.

Five test years provide limited information about a broader population of future seasons. Training histories overlap, site codes recur, and some codes are only proxies for physical trial locations. The year-block and crossed year/site-code bootstrap intervals describe sensitivity to reweighting the observed evaluation units, conditional on the fitted models. They do not account for the complete learning procedure, hyperparameter selection or unobserved environmental regimes. When an interval includes zero, this should be read as limited support for an increment, not as proof of equivalence. Conversely, an isolated interval excluding zero among many exploratory comparisons would not, without further analysis, constitute a family-wise confirmatory finding.

### 4.4. Scope, external validity and remaining controls

The analysis concerns selected public research trials, with weather-calendar availability assessed through the longest horizon and scoring restricted to observed outcomes. Selection into this paired cohort can limit transfer to trials with incomplete calendars or different reporting practices. Environment-hybrid arithmetic means also do not adjust for all field-design, block or spatial effects. They provide a transparent target but are not interchangeable with adjusted genotype performance estimates from a trial-level mixed model. The 2024 missing-target observations cannot be used to characterize prediction error without additional outcome information.

The experiments did not include GBLUP, reaction-norm or RKHS baselines, an explicit genotype-by-weather interaction model, soil predictors, an external data provider or a strict leave-location-out evaluation. Their absence is a substantive limitation for comparisons with the genomic-prediction literature, not an omission that can be repaired by stronger prose. The three implemented families provide linear and nonlinear reference learners, but their comparison does not exhaust either agricultural prediction methods or plausible feature encodings. Similarly, subgroup summaries cannot substitute for a deliberately designed new-location test.

Historical weather was restricted to a specified crop-relative horizon, but its publication and operational availability delays were not audited. Multi-planting-date observations represent weighted mixtures, whose complete information is not available until the latest constituent horizon ends. Thus, these results describe retrospective prediction with observed weather, not a validated real-time forecasting service. No causal agronomic intervention, planting recommendation or deployment safety claim follows from the reported accuracy.

### 4.5. A bounded use for a negative benchmark

The practical contribution is a reproducible test of whether a proposed information or representation change survives an informative baseline and temporal stress tests. The same evidence supports retaining an unmodified baseline when an upgrade lacks stable benefit, while exposing exactly what would need to change to justify a new decision. This is different from claiming that W0 is universally optimal. The outcome is specific to the tested population, information availability assumptions, preprocessing and limited tuning budget.

A stronger follow-up would combine a prospectively frozen external-year test with verified information availability, agricultural reference models and a separately specified spatial-transfer evaluation. Representation experiments should distinguish calendar segmentation from phenological alignment and control, or explicitly study, changes in dimensionality. These are future analyses, not results of the present study. The current record supplies the baselines, frozen comparison rules and complete negative outcomes against which such extensions can be assessed without rewriting the original conclusions.

## 5. Conclusions

In five retrospective test years of public G2F maize trials, cumulative weather observed through 120 days after planting produced small average improvements for CatBoost and extremely randomized trees and a higher average error for Ridge, relative to their genetic-and-static baselines. The crossed year/site-code sensitivity intervals did not establish a dependable positive weather increment. Fixed 30-day calendar-block representations did not improve the five-year mean error at the newly evaluated horizons, and neither representation yielded a sufficient observation window under the study's rule. These findings support testing environmental information as an incremental predictive hypothesis, with explicit temporal boundaries and complete reporting of unfavorable results. They do not imply that weather is unimportant to crop development, nor that the tested methods define the limit of maize yield prediction.

## Data and code availability

The underlying public competition data are identified by DOI 10.25739/78mn-4394 (Genomes to Fields, 2025). Local preparation, protocols, code and frozen result tables form the companion research project. The manuscript figures and tables are generated from the audited analytical outputs; their relative source paths and SHA-256 digests are recorded with the PDF build receipt. No public repository URL or archive accession is claimed in this draft because public code release has not yet been completed. Before submission, the companion code, permissible derived results and an immutable release identifier should be deposited, with the data provider's license and attribution retained.

## Declarations

This is an unpublished research manuscript prepared for review, not a peer-reviewed publication. Author identity, affiliation, contributions, funding and competing-interest declarations require confirmation before submission and have not been invented. Generative AI tools assisted implementation, numerical review and manuscript drafting. Final scientific interpretation, authorship and responsibility require review by the submitting author. No institutional collaboration is implied.

## References

Chen, Q., Washburn, J. D., Lima, D. C., Romay, M. C., Gage, J. L., Holland, J. B., et al. (2026). Genomes to fields 2024 maize genotype by environment prediction competition. BMC Research Notes, 19, 113. [https://doi.org/10.1186/s13104-026-07629-5](https://doi.org/10.1186/s13104-026-07629-5)

Genomes to Fields. (2025). Genomes to Fields 2024 Maize Genotype by Environment Prediction Competition [Data set]. CyVerse Data Commons. [https://doi.org/10.25739/78mn-4394](https://doi.org/10.25739/78mn-4394)

Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. Machine Learning, 63, 3–42. [https://doi.org/10.1007/s10994-006-6226-1](https://doi.org/10.1007/s10994-006-6226-1)

Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1), 55–67. [https://doi.org/10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)

Jarquín, D., Crossa, J., Lacaze, X., Du Cheyron, P., Daucourt, J., Lorgeou, J., et al. (2014). A reaction norm model for genomic selection using high-dimensional genomic and environmental data. Theoretical and Applied Genetics, 127, 595–607. [https://doi.org/10.1007/s00122-013-2243-1](https://doi.org/10.1007/s00122-013-2243-1)

Jarquin, D., de Leon, N., Romay, C., Bohn, M., Buckler, E. S., Ciampitti, I., et al. (2021). Utility of climatic information via combining ability models to improve genomic prediction for yield within the Genomes to Fields maize project. Frontiers in Genetics, 11, 592769. [https://doi.org/10.3389/fgene.2020.592769](https://doi.org/10.3389/fgene.2020.592769)

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9), 100804. [https://doi.org/10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804)

Owen, A. B. (2007). The pigeonhole bootstrap. The Annals of Applied Statistics, 1(2), 386–411. [https://doi.org/10.1214/07-AOAS122](https://doi.org/10.1214/07-AOAS122)

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. Advances in Neural Information Processing Systems, 31, 6638–6648. [Publisher record](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40(8), 913–929. [https://doi.org/10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881)

Rogers, A. R., & Holland, J. B. (2022). Environment-specific genomic prediction ability in maize using environmental covariates depends on environmental similarity to training data. G3 Genes|Genomes|Genetics, 12(2), jkab440. [https://doi.org/10.1093/g3journal/jkab440](https://doi.org/10.1093/g3journal/jkab440)

Westhues, C. C., Mahone, G. S., da Silva, S., Thorwarth, P., Schmidt, M., Richter, J.-C., et al. (2021). Prediction of maize phenotypic traits with genomic and environmental predictors using gradient boosting frameworks. Frontiers in Plant Science, 12, 699589. [https://doi.org/10.3389/fpls.2021.699589](https://doi.org/10.3389/fpls.2021.699589)

## Supplementary material

### Supplementary Table S1. Complete year-equal mean RMSE

All implemented model-window combinations are retained. Cumulative and segmented W0/W30 share identical inputs and predictions by construction. Duplicate baseline values across representations are shown only once for the two simple baselines. These are descriptive point estimates; paired sensitivity intervals are shown in Figures 2 and 4 and retained in the companion CSV tables.



| Model | Horizon | Cumulative RMSE | Segmented RMSE |
| --- | --- | --- | --- |
| Global mean | W0 | 2.9377 | Shared |
| Historical site mean | W0 | 2.6243 | Shared |
| Ridge | W0 | 2.5398 | 2.5398 |
| Ridge | W30 | 2.6161 | 2.6161 |
| Ridge | W60 | 2.6009 | 2.6744 |
| Ridge | W90 | 2.6090 | 2.6485 |
| Ridge | W120 | 2.6020 | 2.6925 |
| Extra Trees | W0 | 2.5794 | 2.5794 |
| Extra Trees | W30 | 2.6432 | 2.6432 |
| Extra Trees | W60 | 2.6444 | 2.6679 |
| Extra Trees | W90 | 2.6020 | 2.6136 |
| Extra Trees | W120 | 2.5781 | 2.6135 |
| CatBoost | W0 | 2.5332 | 2.5332 |
| CatBoost | W30 | 2.6893 | 2.6893 |
| CatBoost | W60 | 2.6697 | 2.7132 |
| CatBoost | W90 | 2.5644 | 2.6483 |
| CatBoost | W120 | 2.5149 | 2.6281 |

RMSE units: Mg/ha. The full yearly and subgroup tables are available in the companion analytical outputs; no configurations are omitted from the means above.

### Supplementary Table S2. Fixed hyperparameter candidates

Only the listed choice varied within each model family. Both candidates were evaluated independently for each newly fitted model-window-representation unit using the preceding validation year. There was no early stopping or test-driven search. Seed 20261018 and four threads were used where supported.



| Family | Candidate values | Fixed configuration |
| --- | --- | --- |
| Ridge | alpha = 100 or 10,000 | Training-only standardization |
| Extra Trees | min_samples_leaf = 5 or 20 | 160 trees; maximum depth 18; max_features = 0.8 |
| CatBoost | depth = 6 or 8 | 600 iterations; learning rate 0.05; L2 = 5; RMSE loss |

Model versions recorded in the frozen runs: scikit-learn 1.7.2 and CatBoost 1.2.8. The candidates have equal search counts, not equal statistical complexity across representations.

### Supplementary Table S3. Reproducibility map



| Evidence | Companion project path |
| --- | --- |
| Cohort and source verification | preparation/outputs/; preparation/provenance_followup/verification_report.json |
| Original model protocol | modeling/PROTOCOL_LOCK.md |
| Rolling design and results | modeling/rolling_v1/PROTOCOL.md; modeling/rolling_v1/analysis/ |
| Segmented design and results | modeling/segmented_v1/PROTOCOL.md; modeling/segmented_v1/analysis/ |
| Independent numerical audits | modeling/rolling_v1/validation_checks.json; modeling/segmented_v1/validation_checks.json |
| Manuscript evidence export | reports/manuscript/evidence.json; evidence.receipt.json |
| Reproduction of this PDF | scripts/collect_manuscript_evidence.py; scripts/build_manuscript.py |

The manuscript build verifies the source hashes before rendering. It does not rerun model fitting or replace the frozen experimental records. A public code-release URL is not yet available.
