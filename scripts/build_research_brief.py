"""Render a two-page brief from hash-bound, audited aggregate evidence only.

No model fitting, analysis rerun, individual-target access or experiment changes.
The PDF edit marker was invoked once before authoring this generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from matplotlib import get_data_path
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/"reports/RESEARCH_BRIEF.md"
OUTPUT = PROJECT/"output/pdf/MAIZE_RESEARCH_BRIEF.pdf"
MODELS = ("Ridge", "ExtraTrees", "CatBoost")
LABELS = {"Ridge": "Ridge", "ExtraTrees": "Extra Trees", "CatBoost": "CatBoost"}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def reviewed_evidence():
    evidence, hashes = {}, {}
    for name, table in (("rolling_v1", "cross_year_stability.csv"),
                        ("segmented_v1", "cross_year_comparisons.csv")):
        root = PROJECT/"modeling"/name
        manifest_path, audit_path = root/"analysis/analysis_manifest.json", root/"validation_checks.json"
        manifest, audit = (json.loads(p.read_text()) for p in (manifest_path, audit_path))
        if (audit.get("status") != "PASS" or audit.get("failure") is not None
                or Path(audit["audited_manifest_path"]).resolve() != manifest_path
                or audit["audited_manifest_sha256"] != sha(manifest_path)):
            raise ValueError(f"{name} requires a current hash-bound independent PASS audit")
        base = Path(manifest["artifact_path_base"]).resolve()
        if base != root:
            raise ValueError("Unexpected analysis artifact base")
        for relative, expected in manifest["artifacts"].items():
            path = (base/relative).resolve()
            if not path.is_relative_to(root) or sha(path) != expected:
                raise ValueError(f"Changed analytical artifact: {relative}")
        for name_or_path, expected in manifest["source_hashes"].items():
            path = Path(name_or_path)
            path = path if path.is_absolute() else PROJECT/path
            if sha(path) != expected:
                raise ValueError(f"Changed source binding: {path}")
        paths = [manifest_path, audit_path, root/"PROTOCOL.md", root/"analysis"/table,
                 root/"analysis/coverage.csv", root/"analysis/yearly_sufficient_windows.csv"]
        evidence[name] = {"audit": audit, "table": pd.read_csv(root/"analysis"/table),
                          "coverage": pd.read_csv(root/"analysis/coverage.csv")}
        coverage = evidence[name]["coverage"]
        if list(coverage.Year) != [2020, 2021, 2022, 2023, 2024] or int(coverage.n_scored.sum()) != 57673 or int(coverage.n_environments.sum()) != 129:
            raise ValueError("Five-year scoring cohort differs from brief")
        sufficient = pd.read_csv(root/"analysis/yearly_sufficient_windows.csv")
        if sufficient.earliest_sufficient_window.notna().any():
            raise ValueError("No-sufficient-window claim no longer matches evidence")
        hashes.update({str(p.relative_to(PROJECT)): sha(p) for p in paths})
    assessment_path = PROJECT/"modeling/segmented_v1/analysis/model_upgrade_assessment.csv"
    assessment = pd.read_csv(assessment_path)
    if set(assessment.model) != set(MODELS) or assessment.both_reference_screens_pass.any():
        raise ValueError("Upgrade-screen conclusion differs")
    audit = evidence["segmented_v1"]["audit"]
    if audit["n_new_candidate_fits"] != 90 or audit["n_new_final_fits"] != 45 or len(audit["new_final_model_replays"]) != 45:
        raise ValueError("New-fit/replay counts differ from brief")
    hashes[str(assessment_path.relative_to(PROJECT))] = sha(assessment_path)
    return evidence, hashes


def select(table, model, reference=None):
    mask = (table.model == model) & (table.window == 120)
    if reference is not None:
        mask &= table.reference == reference
    found = table.loc[mask]
    if len(found) != 1:
        raise ValueError("Expected one primary model/window/reference row")
    row = found.iloc[0]
    if (row.n_years, row.n_environment_years, row.n_site_codes) != (5, 129, 31):
        raise ValueError("Primary comparison support differs")
    return row


def validate_markdown(text, evidence):
    if not text.startswith("# Summary\n") or not text.isascii() or text.count("<!-- PAGEBREAK -->") != 1:
        raise ValueError("Brief must start Summary, use ASCII punctuation and define two pages")
    for model in MODELS:
        cumulative = select(evidence["rolling_v1"]["table"], model)
        segmented = select(evidence["segmented_v1"]["table"], model, "cumulative")
        absolute = (f"| {LABELS[model]} | {cumulative.year_equal_rmse_W0:.4f} | "
                    f"{cumulative.year_equal_rmse_weather:.4f} | {segmented.year_equal_rmse_segmented:.4f} |")
        difference = (f"| {LABELS[model]} | {cumulative.year_equal_delta_rmse:+.3f} "
                      f"[{cumulative.crossed_year_site_ci95_low:+.3f}, {cumulative.crossed_year_site_ci95_high:+.3f}] | "
                      f"{segmented.year_equal_delta_rmse:+.3f} "
                      f"[{segmented.crossed_year_site_ci95_low:+.3f}, {segmented.crossed_year_site_ci95_high:+.3f}] |")
        if absolute not in text or difference not in text:
            raise ValueError(f"Brief table does not match audited primary values: {model}")
        gain = 100*(cumulative.year_equal_rmse_W0-cumulative.year_equal_rmse_weather)/cumulative.year_equal_rmse_W0
        if not np.isclose(gain, cumulative.relative_gain_of_year_equal_rmse_percent, atol=1e-10):
            raise ValueError("Percentage uses a different estimand")
    cat = select(evidence["rolling_v1"]["table"], "CatBoost")
    seg = select(evidence["segmented_v1"]["table"], "CatBoost", "cumulative")
    if f"{cat.relative_gain_of_year_equal_rmse_percent:.2f}%" not in text or cat.years_improved != 3 or seg.years_improved != 0:
        raise ValueError("Narrative CatBoost claim differs")


def register_fonts():
    root = Path(get_data_path())/"fonts/ttf"
    paths = {"BriefSans": root/"DejaVuSans.ttf", "BriefSans-Bold": root/"DejaVuSans-Bold.ttf"}
    for name, path in paths.items():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily("BriefSans", normal="BriefSans", bold="BriefSans-Bold",
                                italic="BriefSans", boldItalic="BriefSans-Bold")
    return {p.name: sha(p) for p in paths.values()}


def inline(text):
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#255a77">\1</link>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`([^`]+)`", r"\1", text)


def build(output=OUTPUT):
    evidence, hashes = reviewed_evidence()
    text = SOURCE.read_text()
    validate_markdown(text, evidence)
    fonts = register_fonts()
    ink, muted, rule = (colors.HexColor(c) for c in ("#162c3a", "#47545d", "#c6cfd4"))
    styles = {
        "title": ParagraphStyle("title", fontName="BriefSans-Bold", fontSize=24, leading=29, textColor=ink, spaceAfter=6),
        "heading": ParagraphStyle("heading", fontName="BriefSans-Bold", fontSize=11.2, leading=14.5, textColor=ink, spaceBefore=11, spaceAfter=6, keepWithNext=True),
        "body": ParagraphStyle("body", fontName="BriefSans", fontSize=9.5, leading=13.3, textColor=muted, spaceAfter=7),
        "bullet": ParagraphStyle("bullet", fontName="BriefSans", fontSize=9.5, leading=13.3, textColor=ink, leftIndent=10, firstLineIndent=-9, spaceAfter=6),
        "cell": ParagraphStyle("cell", fontName="BriefSans", fontSize=8.7, leading=11.8, textColor=ink),
        "header": ParagraphStyle("header", fontName="BriefSans-Bold", fontSize=8.5, leading=11.5, textColor=ink),
        "footnote": ParagraphStyle("footnote", fontName="BriefSans", fontSize=7.6, leading=10.2, textColor=muted, spaceBefore=3),
    }
    story = []
    lines, index = text.splitlines(), 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line:
            continue
        if line == "<!-- PAGEBREAK -->":
            story.append(PageBreak())
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["title"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["heading"]))
        elif line.startswith("- "):
            story.append(Paragraph("- "+inline(line[2:]), styles["bullet"]))
        elif line.startswith("|"):
            rows = [line]
            while index < len(lines) and lines[index].startswith("|"):
                rows.append(lines[index])
                index += 1
            cells = [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]
            cells = [row for row in cells if not all(re.fullmatch(r":?-+:?", cell) for cell in row)]
            data = [[Paragraph(inline(cell), styles["header" if r == 0 else "cell"]) for cell in row] for r, row in enumerate(cells)]
            widths = [96, 113, 147, 151] if len(cells[0]) == 4 else [96, 195, 216]
            table = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
            table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f4")),
                ("LINEABOVE", (0, 0), (-1, 0), .7, ink),
                ("LINEBELOW", (0, 0), (-1, 0), .5, rule),
                ("LINEBELOW", (0, -1), (-1, -1), .5, rule),
                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ]))
            story.extend([table, Spacer(1, 8)])
        else:
            story.append(Paragraph(inline(line), styles["footnote" if line.startswith("Data:") else "body"]))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(rule)
        canvas.line(44, 33, A4[0]-44, 33)
        canvas.setFont("BriefSans", 7.2)
        canvas.setFillColor(muted)
        canvas.drawString(44, 21, "Research brief | Public G2F data | Retrospective evidence, 2020-2024")
        canvas.drawRightString(A4[0]-44, 21, f"{document.page} / 2")
        canvas.restoreState()

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = PROJECT/"tmp/pdfs"
    temporary.mkdir(parents=True, exist_ok=True)
    draft = temporary/"RESEARCH_BRIEF_candidate.pdf"
    document = SimpleDocTemplate(str(draft), pagesize=A4, leftMargin=44, rightMargin=44,
                                 topMargin=35, bottomMargin=44,
                                 title="Weather information in maize yield prediction - Research brief",
                                 author="", subject="Two-page summary of audited temporal and weather-representation comparisons")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(draft)
    if len(reader.pages) != 2:
        raise ValueError(f"Expected two A4 pages, got {len(reader.pages)}; inspect draft layout")
    extracted = "\n".join(page.extract_text() for page in reader.pages)
    required = ["Summary", "57,673", "129 environment-years", "31 released site codes", "0.72%",
                "2.5332", "2.5149", "2.6281", "-0.213", "+0.165", "AI-assisted", "10.25739/78mn-4394"]
    if any(value not in extracted for value in required):
        raise ValueError("Expected summary/metric/source text missing from PDF")
    embedded = set()
    for page in reader.pages:
        if not np.allclose([float(page.mediabox.width), float(page.mediabox.height)], A4, atol=.01):
            raise ValueError("PDF page is not A4")
        for reference in page["/Resources"]["/Font"].values():
            font = reference.get_object()
            if "DejaVuSans" in str(font.get("/BaseFont", "")):
                descriptor = font.get("/FontDescriptor")
                if descriptor is None or "/FontFile2" not in descriptor.get_object():
                    raise ValueError("Required regular/bold font is not embedded")
                embedded.add(str(font["/BaseFont"]))
    if len(embedded) != 2:
        raise ValueError("Both regular and bold embedded fonts are required")
    draft.replace(output)
    receipt = {"output": str(output.relative_to(PROJECT)), "pages": 2, "page_size": "A4",
               "text_checks": "PASS", "numeric_table_checks": "PASS", "source_audits": {name: "PASS" for name in evidence},
               "input_hashes": hashes, "source_markdown_sha256": sha(SOURCE), "builder_sha256": sha(Path(__file__)),
               "font_source_sha256": fonts, "embedded_fonts": sorted(embedded), "pdf_sha256": sha(output),
               "visual_inspection": "PENDING: render both pages and inspect latest PNGs"}
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k: receipt[k] for k in ("output", "pages", "text_checks", "numeric_table_checks", "pdf_sha256", "visual_inspection")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.output)
