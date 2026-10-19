# Selection-focused manuscript QA

Reviewed 19 October 2026. This is an unpublished scientific draft, not a submitted or accepted article.

- The manuscript preserves the earlier weather-window paper in `reports/manuscript/` and creates a separate selection-focused article.
- The initial PDF had eleven pages, five numbered figures, two numbered tables and ten references.
- The builder checked year labels, cohort sizes, annual top-20% values, the exact MSE-component identity, W0/W120 RMSE reconciliation, and the presence of the principal numerical claims in PDF-extracted text.
- `modeling/stress_v1/validate_results.py` passed: saved prediction hashes matched, 2024 RMSE and top-20% recall independently reconciled to floating-point precision, and five-year summaries matched the frozen records.
- All eleven pages of that initial PDF were rendered at 90 dpi and visually reviewed. No clipped text, overlapping objects, missing glyphs or broken figure references were observed. Tables are unsplit, with captions following them.
- No model was refitted to write this article. The selection-centered thesis and error decomposition are explicitly retrospective/post-hoc; site-code intervals are conditional sensitivity checks, not definitive prospective confidence intervals.
- Before journal submission, confirm author identity/contributions, affiliation, funding and competing interests; select a specific journal and apply its article template; publicly archive the code and reproducibility materials if permissible; obtain scientific coauthor review. Independent new-year/new-site validation and stronger breeding baselines would materially improve publishability.

## Journal-style figure and abstract revision, 19 October 2026

- The abstract was reduced to 204 words and six keywords. It now identifies the selection analysis as retrospective/post-hoc, emphasizes the matched baseline and the crossed sensitivity interval, and removes SHAP rank and the fragile 92.9% ratio from the abstract.
- Figures 1-5 were regenerated from frozen analysis tables by `scripts/build_selection_paper.py`; no model was refitted. Each figure has a 600-dpi PNG for the review PDF plus standalone vector PDF and SVG files for submission preparation.
- Figure 2 now shows paired annual top-20% recall differences rather than nearly overlapping absolute-recall curves. Figure 3 uses separate, clearly labeled scales so the within-environment component remains visible. Figure 4 shows both paired estimates and crossed year/site-code sensitivity intervals. Figure 5 uses a consistent font and external caption, with a hatched stress-proxy bar rather than color-only identification.
- In-text figure references were added in numerical order. Captions state the population, comparison, units, interpretation and important caveats; no figure has an in-image title.
- The 11-page figure-revision PDF was rendered and visually checked. Table 1 and its caption were kept on the same page; no clipping, overlap, missing glyphs or figure-reference defects were seen. Figures remained legible at the full-width A4 review layout.
- This is a journal-style draft, not a guarantee of compliance with a particular journal. Before submission, follow the selected journal's required dimensions, file format, caption/table placement, disclosure policy and publication-ready declarations.

## Abstract and conclusion revision, 19 October 2026

- Replaced only the Abstract and Conclusion prose with a decision-target framing: paired weather increments, their uncertainty, the environment-mean calibration distinction and limits on operational selection claims. The abstract is 188 words with six keywords. No results, model fits or figures were changed.
- Removed the forced page break before References. The final review PDF has ten pages; declarations and references now share the last page.
- Builder text checks passed. All ten pages were rendered and visually inspected after the prose edit; pages 9-10 were re-rendered at 120 dpi after the pagination change. No clipping, overlap, orphaned table caption, broken figure reference or missing glyph was observed.
