# Verified literature and citation guardrails

Verified against primary publisher, author, or official dataset sources on 18 October 2026. This is a targeted background check, not a systematic review. Bibliographic metadata for manuscript generation is in `references.json`.

## Positioning that the literature supports

- Environmental covariates, temporal weather windows, phenology-aligned weather features, and genotype-by-environment prediction are established approaches. This study does not introduce them.
- The defensible contribution is a controlled, retrospective evaluation of the incremental value of observed weather under five forward-year test folds, with paired comparisons across model families and an explicitly exploratory representation extension.
- Prior papers' predictive correlations, accuracy improvements, and datasets are not directly comparable to this study's year-equal mean of environment-level RMSE. Do not make a superiority claim across papers.
- Failure to obtain a stable increment over a genomic-plus-static baseline does not imply that weather is biologically unimportant. The inference is conditional on the data, information horizon, feature representation, learners, and tuning budget.

## Closely related primary studies

### `rogers2022` — environmental similarity and weather resolution

Rogers, A. R., and Holland, J. B. (2022). Environment-specific genomic prediction ability in maize using environmental covariates depends on environmental similarity to training data. *G3 Genes|Genomes|Genetics*, 12(2), jkab440. DOI: [10.1093/g3journal/jkab440](https://doi.org/10.1093/g3journal/jkab440).

- Verified: [publisher full text](https://academic.oup.com/g3journal/article/12/2/jkab440/6486423), abstract, environmental-main-effect methods, and stratified validation sections.
- Supports: 5-, 10-, 15-, and 30-day weather windows have already been compared in G2F maize; the utility of environmental information depends on training–test environmental similarity.
- Difference here: limited-horizon cumulative versus calendar-block features, newer data, and forward-year rolling evaluation; no equivalent explicit marker-by-environment model is fitted here.
- Guardrail: do not present 30-day segmentation or holding out years as a new methodological invention.

### `westhues2021` — boosting and phenology-aware environmental features

Westhues, C. C., Mahone, G. S., da Silva, S., Thorwarth, P., Schmidt, M., Richter, J.-C., Simianer, H., and Beissinger, T. M. (2021). Prediction of Maize Phenotypic Traits With Genomic and Environmental Predictors Using Gradient Boosting Frameworks. *Frontiers in Plant Science*, 12, 699589. DOI: [10.3389/fpls.2021.699589](https://doi.org/10.3389/fpls.2021.699589).

- Verified: [publisher full text](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2021.699589/full), abstract and weather-feature methods.
- Supports: G2F work already compared linear models with XGBoost and LightGBM; environmental summaries were aligned to growth periods using observed planting and 50% silking dates.
- Difference here: fixed elapsed-day blocks need no observed silking dates, but do not identify actual developmental stages. This is a distinct, less biologically aligned representation, not proof of improved phenology modeling.

### `jarquin2020` — weather inclusion can fail to improve G2F prediction

Jarquin, D., de Leon, N., Romay, C., Bohn, M., Buckler, E. S., Ciampitti, I., et al. (2021). Utility of Climatic Information via Combining Ability Models to Improve Genomic Prediction for Yield Within the Genomes to Fields Maize Project. *Frontiers in Genetics*, 11, 592769. DOI: [10.3389/fgene.2020.592769](https://doi.org/10.3389/fgene.2020.592769).

- Verified: [publisher full text](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2020.592769/full), abstract, discussion, and suggested citation.
- Supports: a G2F analysis found benefits from genotype-by-environment terms without consistent added benefit from its naive weather-covariate representation.
- Difference here: this benchmark tests other representations and learners over later forward-year folds. Concordant negative increments are not a replication of their exact model or causal explanation.
- Date note: retain the stable key `jarquin2020`, but cite **2021**, matching the publisher's suggested citation and 8 March 2021 publication date. Volume 11 and the DOI contain 2020.

### `jarquin2014` — established reaction-norm baseline family

Jarquín, D., Crossa, J., Lacaze, X., Du Cheyron, P., Daucourt, J., Lorgeou, J., et al. (2014). A reaction norm model for genomic selection using high-dimensional genomic and environmental data. *Theoretical and Applied Genetics*, 127, 595–607. DOI: [10.1007/s00122-013-2243-1](https://doi.org/10.1007/s00122-013-2243-1).

- Verified: [publisher article](https://link.springer.com/article/10.1007/s00122-013-2243-1), abstract and methodological framing.
- Supports: covariance-based reaction norms provide an established way to include genomic, environmental, and interaction effects without explicit enumeration of all interactions.
- Guardrail: the demonstration was in wheat, not G2F maize. It motivates a missing comparator/future extension; it is not a method implemented by this study.

## Data provenance

### `g2f2025` — exact competition data release

Genomes to Fields (2025). *Genomes to Fields 2024 Maize Genotype by Environment Prediction Competition* [Data set]. CyVerse Data Commons. DOI: [10.25739/78mn-4394](https://doi.org/10.25739/78mn-4394).

- Verified: [official CyVerse dataset landing page](https://dc.cyverse.org/dataset/genomes_to_fields_2024_maize_genotype_by_environment_prediction_competition) and [G2F resources](https://www.genomes2fields.org/resources/).
- The landing page recommends citation year **2025**, even though the competition refers to 2024. Current catalog publication metadata is 8 April 2026; original `de_created_date` is 21 February 2025. Do not silently treat the catalog migration date as the dataset citation year.
- The current landing page identifies **Open Data Commons Public Domain Dedication and License (PDDL)**. Do not substitute CC BY or CC0. Record checksums/access date for the exact local files, because the page is not a versioned snapshot of those local bytes.
- The dataset release includes training/test data; this manuscript's filtered analytical sample must be described separately from release-level counts.

### `chen2026` — formal description of the same data release

Chen, Q., Washburn, J. D., Lima, D. C., Romay, M. C., Gage, J. L., Holland, J. B., et al. (2026). Genomes to fields 2024 maize genotype by environment prediction competition. *BMC Research Notes*, 19, 113. DOI: [10.1186/s13104-026-07629-5](https://doi.org/10.1186/s13104-026-07629-5).

- Verified: [publisher article](https://link.springer.com/article/10.1186/s13104-026-07629-5), published 9 February 2026, version of record 13 March 2026.
- Supports: competition training years 2014–2023 and test year 2024; yield expressed in Mg ha⁻¹ at 15.5% moisture; now-public 2024 labels; competition-package weather obtained from NASA POWER; missingness and approximate coordinates require care.
- Guardrail: do not describe the competition weather as local field-station measurements merely because older G2F papers used stations. Our 2024 retrospective evaluation is not a competition entry, and our RMSE outcome is not the competition's correlation-based leaderboard criterion.

## Evaluation and uncertainty

### `roberts2017` — validation must match the transfer problem

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. DOI: [10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881).

- Verified: [publisher full text](https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881), abstract and structured-data discussion.
- Supports: random splits can misrepresent prediction error for structured data; the blocking design should reflect the intended prediction setting.
- Difference here: temporal forward testing is evaluated, not stringent spatial extrapolation. A temporal fold can contain sites observed in training, so it must not be called independent spatial validation.

### `leakage2023` — leakage and reproducibility safeguards

Kapoor, S., and Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. DOI: [10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804).

- Verified: [publisher record](https://doi.org/10.1016/j.patter.2023.100804) via indexed publisher content and [author preprint](https://arxiv.org/abs/2207.07048); bibliographic cross-check [PubMed](https://pubmed.ncbi.nlm.nih.gov/37720327/).
- Supports: leakage can produce overoptimistic scientific conclusions; evaluation boundaries and preprocessing provenance should be explicit.
- Guardrail: a clean split and many automated checks do not prove that all bias has been eliminated. Feature-construction chronology, model selection, retrospective choices, and unmeasured selection effects still matter.

### `owen2007` — optional background for crossed resampling

Owen, A. B. (2007). The pigeonhole bootstrap. *The Annals of Applied Statistics*, 1(2), 386–411. DOI: [10.1214/07-AOAS122](https://doi.org/10.1214/07-AOAS122).

- Verified: [author manuscript](https://artowen.su.domains/reports/pbs.pdf) and [author arXiv record](https://arxiv.org/abs/0712.1111).
- Supports: independent resampling of crossed row and column units is an established alternative to treating dependent observations as independent.
- Guardrail: describe our year/site-code procedure as a crossed-resampling **sensitivity analysis**. Five test years, overlapping training sets, imperfect site-code geography, and no refitting are not covered by a claim of exact confidence-interval calibration. Cite only if the implemented resampling actually matches the described mechanism.

## Manuscript wording to avoid

- "A novel weather-aware maize prediction framework" without a specific new methodological contribution.
- "Phenological stages" for fixed 0–30, 30–60, 60–90, and 90–120-day calendar blocks.
- "Weather has no effect on maize yield" or "segmentation is ineffective in general."
- "Prospective independent confirmation" for an extension chosen after seeing the cumulative results.
- "Generalizes to unseen farms/regions" based only on temporal test years or a very small naturally unseen-site subgroup.
- "Outperforms the literature" by comparing our RMSE with unrelated correlation-based studies.
- "SCI Q1 paper" as a publication status or a guarantee of novelty/acceptance. The current artifact is an unpublished research manuscript.

## Primary algorithm references added for the Methods

### `hoerl1970` — Ridge regression

Hoerl, A. E., and Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55–67. DOI: [10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634).

- Verified: [publisher metadata](https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634) and [public copy of the original paper](https://homepages.math.uic.edu/~lreyzin/papers/ridge.pdf).
- Supports: Ridge provides a regularized linear reference through a squared-coefficient penalty. Cite with the model-family description, not as evidence that the chosen penalties are optimal for these maize trials.
- Date note: cite the original 1970 issue, not the publisher's 2012 online-digitization date.

### `geurts2006` — Extremely randomized trees

Geurts, P., Ernst, D., and Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63, 3–42. DOI: [10.1007/s10994-006-6226-1](https://doi.org/10.1007/s10994-006-6226-1).

- Verified: [publisher article](https://link.springer.com/article/10.1007/s10994-006-6226-1), abstract and bibliographic record.
- Supports: the tree-ensemble family randomizes candidate features and split thresholds. Our depth, leaf-size and feature-fraction settings remain implementation choices, not results established by the original paper.

### `prokhorenkova2018` — CatBoost

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, 31, 6638–6648. [Official proceedings](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html).

- Verified: official proceedings, paper, and [page-range metadata](https://proceedings.neurips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Metadata.json). No DOI is invented.
- Supports: the CatBoost algorithm-family citation. Do not claim that this experiment used ordered boosting or native categorical encoding unless the saved model parameters establish it. In particular, citing CatBoost does not replace auditing upstream feature encoding and temporal boundaries.

## Introduction and Discussion review

Reviewed the current `manuscript.md` on 18 October 2026, without editing it.

- The claims attributed to Rogers and Holland (2022) and Westhues et al. (2021) match the inspected primary sources. No claim of first weather inclusion, first temporal segmentation, superior predictive accuracy over prior studies, or mechanistically identified weather effects was found.
- The discussion correctly distinguishes fixed calendar blocks from observed phenological stages and distinguishes the present RMSE analysis from prior correlation-based comparisons.
- Rogers-related material in Introduction plus Discussion is below 200 words: even counting both entire citation-containing paragraphs, including their original commentary and the separate Westhues sentence, gives 188 whitespace-delimited words. The directly attributable Rogers description is substantially shorter. The corresponding conservative count for Westhues is 187 words. Do not expand these summaries unnecessarily.
- The proposed future restriction on stage information is this manuscript's inference about its prediction-time information set, not an accusation that Westhues et al. used an invalid validation protocol.
- Add the three algorithm citations to the existing Methods model-family paragraph; their bibliographic records are already present in `references.json`.
- Avoid an uncited bibliography: `jarquin2020`, `jarquin2014`, `roberts2017`, `leakage2023`, and `owen2007` should be cited at their relevant existing discussion/method locations or omitted from the rendered bibliography if not used. They should not be inserted merely to increase the reference count.
- The rendered citation strings now consistently use the first six authors plus “et al.” when there are more than six authors. The `authors_complete` field indicates whether the metadata author list itself is complete; it is not a claim that a truncated rendered citation lists every author.
