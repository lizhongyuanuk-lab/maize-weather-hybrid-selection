"""Create a one-page English project brief from verified final artifacts only.

No model fitting, raw holdout-label reading, external downloads, or fictional scores.
The PDF skill create marker was invoked once when this generator was authored.
After generation, render the PDF and inspect the latest page PNG before delivery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from matplotlib import get_data_path
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PROJECT = Path(__file__).resolve().parents[1]
FINAL = PROJECT / "modeling/results/final"
DEFAULT_AUDIT = PROJECT / "modeling/final_validation_report.json"
DEFAULT_OUTPUT = PROJECT / "output/pdf/MAIZE_RESEARCH_BRIEF.pdf"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def iso_time(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Audit/completion timestamps must include timezone information")
    return value


def verified_inputs(audit_path):
    manifest_path = FINAL / "final_evaluation_manifest.json"
    freeze_path = PROJECT / "modeling/results/development/frozen_selection.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS" or audit.get("stage") != "final" or audit.get("failure") is not None:
        raise ValueError("Final independent audit must report status=PASS, stage=final, failure=null")
    if int(audit.get("checks_passed", 0)) <= 0 or audit.get("holdout_read_by_this_audit") is not True:
        raise ValueError("Independent final-label audit has not completed")
    if iso_time(audit["checked_at_utc"]) < iso_time(manifest["completed_at_utc"]):
        raise ValueError("Independent audit predates the completed final experiment")
    if audit.get("audited_manifest_sha256") != sha256(manifest_path):
        raise ValueError("Independent audit is not bound to the current final manifest hash")
    for relative, expected in manifest["artifacts"].items():
        if sha256(PROJECT / relative) != expected:
            raise ValueError(f"Final artifact hash mismatch: {relative}")
    if sha256(freeze_path) != manifest["frozen_selection_sha256"]:
        raise ValueError("Frozen selection hash mismatch")
    freeze = json.loads(freeze_path.read_text())
    if freeze["recommended_by_validation"] != manifest["recommended_by_validation"]:
        raise ValueError("Final report does not preserve validation-selected configuration")
    provenance_path = PROJECT / "preparation/provenance_followup/verification_report.json"
    provenance = json.loads(provenance_path.read_text())
    if not provenance["summary"]["all_followup_files_semantically_verified"]:
        raise ValueError("Complete official-source verification required for this brief")
    for item in provenance["files"]:
        if sha256(PROJECT / item["local_path"]) != item["local_sha256"]:
            raise ValueError("Current raw source differs from verified provenance")
    summary = pd.read_csv(FINAL / "summary_metrics.csv")
    deltas = pd.read_csv(FINAL / "weather_increment_comparisons.csv")
    sufficient = json.loads((FINAL / "earliest_sufficient_windows.json").read_text())
    cohorts = pd.read_csv(PROJECT / "preparation/outputs/audit/split_summary.csv")
    expected = {(m, w) for m in ["Ridge", "ExtraTrees", "CatBoost"] for w in [0, 30, 60, 90, 120]}
    expected |= {("GlobalMean", 0), ("HistoricalSiteMean", 0)}
    if len(summary) != 17 or set(zip(summary.model, summary.window)) != expected:
        raise ValueError("Exactly 15 model-window and two baseline scores required")
    sources = [manifest_path, freeze_path, audit_path, provenance_path,
               FINAL / "summary_metrics.csv", FINAL / "weather_increment_comparisons.csv",
               FINAL / "earliest_sufficient_windows.json"]
    return {"manifest": manifest, "audit": audit, "freeze": freeze, "summary": summary,
            "deltas": deltas, "sufficient": sufficient, "cohorts": cohorts,
            "input_hashes": {str(p.relative_to(PROJECT)): sha256(p) for p in sources}}


def unique_row(frame, model, window):
    found = frame.loc[(frame.model == model) & (frame.window == window)]
    if len(found) != 1:
        raise ValueError(f"Expected one {model} W{window} row")
    return found.iloc[0]


def metric_text(value):
    return f"{float(value):.3f}" if math.isfinite(float(value)) else "undefined"


def register_embedded_fonts():
    """Embed both weights; do not rely on viewer-specific base-14 substitutions."""
    directory = Path(get_data_path()) / "fonts/ttf"
    files = {"MaizeSans": directory / "DejaVuSans.ttf",
             "MaizeSans-Bold": directory / "DejaVuSans-Bold.ttf"}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required embedded font unavailable: {path.name}")
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily("MaizeSans", normal="MaizeSans", bold="MaizeSans-Bold",
                                  italic="MaizeSans", boldItalic="MaizeSans-Bold")
    return {path.name: sha256(path) for path in files.values()}


def build(output, audit_path):
    source = verified_inputs(audit_path)
    font_hashes = register_embedded_fonts()
    selected = source["freeze"]["recommended_by_validation"]
    family, horizon = selected["model"], selected["window"]
    summary = source["summary"]
    chosen, zero, full = [unique_row(summary, family, w) for w in [horizon, 0, 120]]
    delta = unique_row(source["deltas"], family, 120)
    change = 100 * (zero.macro_rmse-full.macro_rmse)/zero.macro_rmse
    direction = "lower" if change >= 0 else "higher"
    earliest = source["sufficient"]["models"][family]["earliest_sufficient_window"]
    sufficient_text = (f"The prespecified sufficiency rule was met at W{earliest} in this test year."
                       if earliest is not None else "No sufficient window was declared under the prespecified evidence rule.")
    navy, blue, gray, pale = [colors.HexColor(c) for c in ["#18334C", "#2467A3", "#55616C", "#EFF4F8"]]
    styles = {
        "eyebrow": ParagraphStyle("eyebrow", fontName="MaizeSans-Bold", fontSize=8, leading=10, textColor=blue, spaceAfter=7),
        "title": ParagraphStyle("title", fontName="MaizeSans-Bold", fontSize=23, leading=25, textColor=navy, spaceAfter=7),
        "subtitle": ParagraphStyle("subtitle", fontName="MaizeSans", fontSize=10.2, leading=14, textColor=gray, spaceAfter=10),
        "heading": ParagraphStyle("heading", fontName="MaizeSans-Bold", fontSize=10, leading=13, textColor=navy, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", fontName="MaizeSans", fontSize=9.1, leading=12.3, textColor=navy, spaceAfter=3),
        "small": ParagraphStyle("small", fontName="MaizeSans", fontSize=7.4, leading=10, textColor=gray, spaceAfter=3),
        "table": ParagraphStyle("table", fontName="MaizeSans", fontSize=8.5, leading=11, textColor=navy, alignment=TA_LEFT),
    }
    story = []
    plain_text = []

    def p(text, style="body"):
        if not text.isascii():
            raise ValueError("Brief uses ASCII text and punctuation only")
        plain_text.append(text)
        story.append(Paragraph(text, styles[style]))

    p("INDEPENDENT RESEARCH PROJECT | PUBLIC G2F DATA | 2014-2024", "eyebrow")
    p("How Early Can Maize<br/>Yield Be Predicted?", "title")
    p("A multi-model weather-window study with out-of-year evaluation", "subtitle")
    p("RESEARCH QUESTION", "heading")
    p("How much does weather observed within 30, 60, 90 or 120 days after planting improve yield prediction in a held-out year over genotype and static information?")
    p("DESIGN AND DATA", "heading")
    p("Environment-hybrid mean yield; 94,377 training combinations (2014-2022), 11,491 validation combinations (2023). Parameters were selected using 2023 environment-equal RMSE, then frozen and refitted on 2014-2023 before evaluating 2024.")
    p("Ridge, Extra Trees and CatBoost; five paired windows (W0/W30/W60/W90/W120); two preset candidates per model-window, plus global-mean and historical-site baselines. Genotype imputation and 100-component PCA use training materials only. Weather is clipped before imputation; all final predictions are saved before scoring.")
    p("VALIDATION-SELECTED RESULT", "heading")
    p(f"<b>{escape(family)} W{horizon}</b>: 2024 environment-equal RMSE <b>{chosen.macro_rmse:.3f} Mg/ha</b> "
      f"(95% environment-bootstrap interval {chosen.macro_rmse_ci95_low:.3f} to {chosen.macro_rmse_ci95_high:.3f}); "
      f"{int(chosen.n):,} labeled combinations in {int(chosen.n_environments)} environments. "
      f"Macro Spearman: {metric_text(chosen.macro_spearman)}; top-20% recall: {metric_text(chosen.macro_top20_recall)}.")
    p(f"<b>Same-family weather increment:</b> W120 RMSE was {abs(change):.3f}% {direction} than W0. "
      f"W120 minus W0: {delta.vs_W0_delta_rmse:+.3f} Mg/ha "
      f"(paired 95% interval {delta.vs_W0_ci95_low:+.3f} to {delta.vs_W0_ci95_high:+.3f}). "
      + sufficient_text)
    p("COMPLETE CORE MATRIX | 2024 MACRO RMSE (Mg/ha; LOWER IS BETTER)", "heading")
    windows = [0, 30, 60, 90, 120]
    data = [["Model", *[f"W{w}" for w in windows]]]
    for model in ["Ridge", "ExtraTrees", "CatBoost"]:
        data.append([model, *[f"{unique_row(summary, model, w).macro_rmse:.3f}" for w in windows]])
    table = Table(data, colWidths=[102, 79, 79, 79, 79, 79], hAlign="LEFT")
    commands = [("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "MaizeSans-Bold"), ("FONTNAME", (0, 1), (-1, -1), "MaizeSans"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.4), ("LEADING", (0, 0), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("LINEBELOW", (0, 1), (-1, -1), .35, colors.HexColor("#D7DFE5")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")])]
    highlighted_row = ["Ridge", "ExtraTrees", "CatBoost"].index(family)+1
    highlighted_col = windows.index(horizon)+1
    commands += [("BACKGROUND", (highlighted_col, highlighted_row), (highlighted_col, highlighted_row), pale),
                 ("FONTNAME", (highlighted_col, highlighted_row), (highlighted_col, highlighted_row), "MaizeSans-Bold")]
    table.setStyle(TableStyle(commands))
    story.append(table)
    p("Highlight = configuration chosen on 2023, not the smallest 2024 error. W0 includes genotype and static inputs; it is not an uninformed baseline.", "small")
    p("CONTRIBUTION", "heading")
    p("A reproducible temporal benchmark with leakage-aware feature assembly, paired weather ablations, environment-level uncertainty, and separate checks of yield calibration and hybrid ranking. Established models are compared; no new architecture or publication is claimed.")
    p("LIMITATIONS", "heading")
    p(f"One final year and a limited parameter search; {int(chosen.n_without_observed_target):,} test inputs lack observed yield. "
      "Weather is retrospective and real-time availability is unverified. Bootstrap intervals (2,000 environment resamples) are conditional on fitted models, do not include selection uncertainty, and are not multiplicity-adjusted. Predictive associations are not causal effects or deployment evidence.")
    story.append(Spacer(1, 7))
    p("DATA AND AUDIT TRAIL", "heading")
    p('G2F DOI: <link href="https://doi.org/10.25739/78mn-4394" color="#2467A3">10.25739/78mn-4394</link>. Official-source equivalence checks are complete; documented format conversions and the official metadata override are retained.', "small")
    p("Code: modeling/run_experiments.py | Frozen choices: modeling/results/development/frozen_selection.json | Evidence: modeling/results/final/summary_metrics.csv and weather_increment_comparisons.csv | Independent audit: modeling/final_validation_report.json.", "small")
    p("Implementation was AI-assisted. Personal contribution statements must reflect verified individual involvement; no institutional partnership is implied.", "small")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = PROJECT / "tmp/pdfs"
    temporary_directory.mkdir(parents=True, exist_ok=True)
    draft = temporary_directory / "MAIZE_RESEARCH_BRIEF_candidate.pdf"

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7DFE5"))
        canvas.line(40, 28, A4[0]-40, 28)
        canvas.setFont("MaizeSans", 7)
        canvas.setFillColor(gray)
        canvas.drawString(40, 17, "Research brief | Predictions locked before final evaluation")
        canvas.drawRightString(A4[0]-40, 17, "1 / 1")
        canvas.restoreState()

    document = SimpleDocTemplate(str(draft), pagesize=A4, leftMargin=40, rightMargin=40,
                                 topMargin=31, bottomMargin=37, title="How Early Can Maize Yield Be Predicted?",
                                 author="", subject="Independent, AI-assisted maize yield research project")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(draft)
    if len(reader.pages) != 1:
        raise ValueError(f"Brief must be one page, got {len(reader.pages)}; draft retained for layout repair")
    embedded = []
    for resource in reader.pages[0]["/Resources"]["/Font"].values():
        font = resource.get_object()
        if "DejaVuSans" in str(font.get("/BaseFont", "")):
            descriptor = font.get("/FontDescriptor")
            if descriptor is None or "/FontFile2" not in descriptor.get_object():
                raise ValueError("A required DejaVu weight is not embedded")
            embedded.append(str(font["/BaseFont"]))
    if len(embedded) != 2:
        raise ValueError("Expected embedded regular and bold DejaVu Sans fonts")
    extracted = reader.pages[0].extract_text()
    for required in ["How Early", "VALIDATION-SELECTED RESULT", "LIMITATIONS", "10.25739/78mn-4394",
                     f"{chosen.macro_rmse:.3f}"]:
        if required not in extracted:
            raise ValueError(f"Expected PDF text missing: {required}")
    draft.replace(output)
    receipt = {"output": str(output.relative_to(PROJECT)), "pages": 1, "text_checks": "PASS",
               "independent_final_audit": "PASS", "input_hashes": source["input_hashes"],
               "embedded_fonts": embedded, "font_source_sha256": font_hashes,
               "pdf_sha256": sha256(output), "visual_inspection": "REQUIRED: render and inspect page PNG before delivery"}
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    build(arguments.output, arguments.audit)
