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

{{include:methods_draft.md}}

## 3. Results

### 3.1. Evaluation coverage and baseline performance

The five test years comprised 57,673 scored environment-hybrid observations and 129 environment-years, with 22-28 scored environments per year (Table 1). The records represented 31 distinct released site codes. These counts describe different units: repeated hybrids within an environment share weather, and a site code is not a verified independent physical location. The 2024 prediction-input cohort contained 10,057 observations, of which 9,486 had an observed outcome suitable for scoring. The remaining 571 observations were retained as unlabeled inputs and were not assigned synthetic outcomes.

{{table:coverage}}

The non-weather models already achieved year-equal mean environment RMSE values between 2.5332 and 2.5794 Mg/ha (Table 2). Their similar average errors are descriptive and do not establish that the algorithms are equivalent. Each W0 model used the same genetic and static feature representation, with its own validation-selected parameter candidate. The global-mean and historical-site-mean baselines were retained as additional reference methods; their complete aggregate results are provided in Supplementary Table S1. W0 should therefore not be interpreted as a naive baseline or as a model without environmental context.

### 3.2. Incremental value of cumulative weather

At W120, cumulative weather reduced CatBoost's mean RMSE from 2.5332 to 2.5149 Mg/ha, a relative reduction of 0.72%. Extremely randomized trees changed from 2.5794 to 2.5781 Mg/ha, a reduction of approximately 0.05%. Ridge changed from 2.5398 to 2.6020 Mg/ha, an increase of 2.45%. The paired crossed year/site-code intervals for all three primary comparisons included zero (Table 3; Figure 2a). Thus, the small average improvements for the two tree ensembles were not accompanied by a clearly separated conditional sensitivity interval.

{{table:primary}}

{{figure:primary}}

Annual differences were not uniform (Figure 3). Cumulative W120 improved the annual environment-averaged RMSE in three of five years for CatBoost, two of five for extremely randomized trees and three of five for Ridge. For Ridge, improvements in more than half of the years coexisted with a worse five-year average, demonstrating that improvement frequency does not summarize improvement magnitude. Across the 12 cumulative model-by-horizon contrasts against W0, all crossed year/site-code intervals included zero. These findings do not prove absence of useful weather information; they show that a dependable increment was not established by the tested representation, models and evaluation period.

{{table:contrasts}}

{{figure:annual}}

### 3.3. Calendar-block representation did not improve the primary comparison

At W120, the segmented representation produced mean RMSE values of 2.6925, 2.6135 and 2.6281 Mg/ha for Ridge, extremely randomized trees and CatBoost, respectively. Relative to the corresponding cumulative representation, these values represented error increases of 3.48%, 1.37% and 4.50%. CatBoost's segmented W120 error was higher than its cumulative counterpart in all five test years. The segmented-cumulative difference was favorable in one year for Ridge and two years for extremely randomized trees. Crossed intervals for the three primary segmented-cumulative differences included zero; the numerical increases should not be described as uniformly established population-level deterioration.

The pattern was not confined to the primary horizon. All nine newly fitted model-by-horizon combinations at W60, W90 and W120 had a worse five-year mean RMSE than their cumulative counterparts and their same-family W0 reference. Nevertheless, some annual comparisons favored segmentation, illustrating why neither a favorable individual year nor a single mean is sufficient to characterize transfer stability. Figure 4 reports all observation horizons, with W30 unchanged because its single block is identical to the cumulative window. None of the three model families passed the exploratory improvement screen against both required references.

{{figure:windows}}

### 3.4. No sufficient observation horizon was identified

The sufficient-window rule required a supported W120 improvement of at least 2% before searching for an earlier window that retained at least 90% of its point-estimated gain. No model-year combination satisfied this rule in either representation: zero of 15 cumulative and zero of 15 segmented assessments yielded a sufficient window. This is a result about the prespecified rule, not a claim that useful prediction is impossible before day 120. In particular, W0 predictions already existed at the no-post-planting-weather baseline, and some earlier weather windows performed better in individual years. The experiment did not evaluate every possible day, nor did it test a formal noninferiority margin for an earlier horizon.

### 3.5. Ranking and subgroup support qualify the application claim

Absolute yield error and within-environment material selection describe different tasks. Table 4 therefore reports MAE, Spearman correlation and fractional-tie top-20% recall alongside the W120 comparisons. These are year-equal averages of environment-level summaries, not pooled correlations over all hybrids. For Extra Trees, changing cumulative W120 to segmented W120 increased mean Spearman correlation from 0.192 to 0.209 and top-20% recall from 0.279 to 0.287, despite the higher RMSE. CatBoost's top-20% recall changed from 0.287 at W0 to 0.298 with cumulative W120 and 0.296 with segmented W120. It would therefore be inaccurate to describe the segmented representation as worse on every metric. Undefined correlations were retained as undefined rather than replaced with zero. These secondary comparisons are descriptive; the primary uncertainty analysis concerned paired RMSE.

{{table:secondary}}

Support for unseen material and unseen locations was highly uneven (Table 5). Only 30 scored observations belonged to previously unseen hybrids in 2021, while no unseen-hybrid observations were present in 2023. Unseen site codes occurred only in 2020 and 2024, totaling 861 observations in three environment-years. Their performance cannot establish reliable geographical transfer. These subsets were defined relative to each fold's training history rather than by an independently held-out spatial design, and they should not be interpreted as a uniform five-year validation of new-hybrid or new-location prediction.

{{table:subgroups}}

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

{{references}}

## Supplementary material

### Supplementary Table S1. Complete year-equal mean RMSE

All implemented model-window combinations are retained. Cumulative and segmented W0/W30 share identical inputs and predictions by construction. Duplicate baseline values across representations are shown only once for the two simple baselines. These are descriptive point estimates; paired sensitivity intervals are shown in Figures 2 and 4 and retained in the companion CSV tables.

{{table:all_means}}

### Supplementary Table S2. Fixed hyperparameter candidates

Only the listed choice varied within each model family. Both candidates were evaluated independently for each newly fitted model-window-representation unit using the preceding validation year. There was no early stopping or test-driven search. Seed 20261018 and four threads were used where supported.

{{table:hyperparameters}}

### Supplementary Table S3. Reproducibility map

{{table:provenance}}
