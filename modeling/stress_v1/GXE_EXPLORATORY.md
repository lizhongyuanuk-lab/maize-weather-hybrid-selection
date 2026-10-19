# Explicit genetic x stress follow-up (exploratory)

Specified on 2026-10-19 after the five-year stress-feature CatBoost results were
inspected. This is a post-hoc follow-up and **not** an independent confirmation.
The primary stress_v1 result remains unchanged: its small RMSE gain was
uncertain and its Top-20% recall did not improve.

Question: can a regularized, explicit genetic-PC x stress interaction rescue
within-environment hybrid selection? Two matched Ridge pipelines receive the
same W120 plus three stress features. The additive pipeline has main effects
only. The interaction pipeline additionally receives all 100 genomic PCs
multiplied by train-standardized mean VPD proxy and heat-excess35. Stress
standardization is fitted separately within each training fold. Both pipelines
search the same original alpha candidates (100 and 10000), selecting by the
preceding-year environment-equal RMSE; final test years remain 2020-2024.
Top-20% recall and RMSE are both reported, with every year retained. No genetic
PC is assigned a biological gene interpretation. This design assesses whether
an interaction *model* improves prediction, not whether VPD or heat caused a
yield response. The previously viewed outcomes make all positive findings
hypothesis-generating until tested on untouched external data.
