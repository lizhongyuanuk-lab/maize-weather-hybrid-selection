# Application-ready research summary

## Short description

I investigated whether adding observed weather to a maize yield-prediction
pipeline actually improves the breeding decision it is meant to support:
selecting high-yielding hybrids **within a trial environment**. Using public
Genomes to Fields data, I evaluated five forward-year test folds (2020–2024;
57,673 scored environment–hybrid observations in 129 environment-years) and
compared genetic/static baselines with weather-aware models. In the matched
CatBoost comparison, 120 days of observed weather increased top-20% selection
recall from 28.72% to 29.77%, but a crossed year/site-code sensitivity interval
for the 1.05-point gain included zero. Most of the point-estimated reduction in
squared yield error came from better prediction of environment means, rather
than markedly better separation of hybrids within an environment. An
exploratory VPD/heat extension slightly reduced RMSE while lowering selection
recall. The central finding is methodological: **a lower yield RMSE is not, by
itself, evidence of a better hybrid-selection model**.

## What the project demonstrates

- A leakage-conscious, chronological evaluation with preceding-year model
  selection and matched held-out predictions.
- Comparisons across Ridge, Extra Trees and CatBoost, multiple weather windows,
  and an additional stress-feature test; both mixed and negative results remain
  visible in the repository.
- A selection-oriented metric and an exact decomposition of prediction error
  into between-environment calibration and within-environment discrimination.
- Environment/year-aware sensitivity analyses and separate computational
  checks. SHAP is used to inspect fitted-model reliance, not to infer causality.

## Boundaries to state honestly

This is an independent, unpublished retrospective study, not an SCI paper or
an institutional collaboration. The selection-focused question was developed
after the original RMSE benchmark, so the result is exploratory. The weather
features were observed after planting; pre-planting utility, a genuinely new
physical-site test, and reliable future-selection improvement are unproven.
Source data, row-level predictions and fitted models are not redistributed in
the lightweight public code repository. Describe personal and AI-assisted
contributions accurately rather than implying sole unaided authorship.

Read the [current manuscript](selection_paper/manuscript.md), the
[PDF draft](../output/pdf/MAIZE_SELECTION_MANUSCRIPT_DRAFT.pdf), and the
[reproduction guide](../docs/DATA_AND_REPRODUCIBILITY.md) for the detailed
methods, evidence and limitations. Data: G2F, DOI
[10.25739/78mn-4394](https://doi.org/10.25739/78mn-4394).
