"""Render the retrospective selection-focused manuscript from saved evidence.

This program reads frozen model outputs and makes figures/PDF only. It does not
fit models, recalculate intervals, or alter any experiment in modeling/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "reports/selection_paper"
FIG = PAPER / "figures"
OUT = ROOT / "output/pdf/MAIZE_SELECTION_MANUSCRIPT_DRAFT.pdf"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

BLUE = "#255f85"
ORANGE = "#c86c35"
INK = "#202c35"
GREY = "#687780"
PAPER_WIDTH = A4[0] - 100
CONTRASTS = ("W120_vs_W0", "stress_vs_W120")
CONTRAST_LABELS = ("Weather added", "Stress proxies added")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence() -> tuple[pd.DataFrame, dict, dict, pd.Series]:
    annual_path = ROOT / "modeling/stress_v1/objective_analysis/annual_components.csv"
    diagnostic_path = ROOT / "modeling/stress_v1/objective_analysis/summary.json"
    stress_path = ROOT / "modeling/stress_v1/results/summary.json"
    rolling_path = ROOT / "modeling/rolling_v1/analysis/cross_year_stability.csv"
    annual = pd.read_csv(annual_path).sort_values("Year")
    diagnostic = json.loads(diagnostic_path.read_text())
    stress = json.loads(stress_path.read_text())
    rolling = pd.read_csv(rolling_path)
    row = rolling.loc[rolling.model.eq("CatBoost") & rolling.window.eq(120)].squeeze()
    assert annual.Year.tolist() == [2020, 2021, 2022, 2023, 2024]
    assert diagnostic["n_scored_rows"] == stress["n_scored_rows"] == 57673
    assert diagnostic["n_environment_years"] == stress["n_environment_years"] == 129
    assert np.isclose(row.year_equal_rmse_W0, 2.53316801665226)
    assert np.isclose(row.year_equal_rmse_weather, stress["rmse"]["year_equal_reference"])
    for role, expected in (("W0", .2871972449881213),
                           ("W120", stress["top20_recall"]["year_equal_reference"]),
                           ("stress", stress["top20_recall"]["year_equal_candidate"])):
        assert np.isclose(annual[f"{role}_top20_recall"].mean(), expected, atol=2e-5)
    for contrast in diagnostic["contrasts"].values():
        parts = contrast["bias2"]["candidate_minus_base"] + contrast["within_mse"]["candidate_minus_base"]
        assert np.isclose(parts, contrast["mse"]["candidate_minus_base"], atol=1e-12)
    return annual, diagnostic, stress, row


def save_figure(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    # The PNG is for the review PDF. PDF/SVG are editable vector submission assets.
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=600, bbox_inches="tight",
                    pad_inches=.08, facecolor="white")
    plt.close(fig)


def make_figures(annual: pd.DataFrame, diagnostic: dict, stress: dict, rolling: pd.Series) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#a9b5bc", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    # Figure 1: regenerate the temporal design at final-size typography rather
    # than scaling the smaller legacy figure from the earlier manuscript.
    fig, ax = plt.subplots(figsize=(7.2, 2.7), layout="constrained")
    years = np.arange(2014, 2025)
    for row, test_year in enumerate(range(2020, 2025)):
        for year in years:
            if year <= test_year - 2:
                face, hatch = BLUE, None
            elif year == test_year - 1:
                face, hatch = "white", "///"
            elif year == test_year:
                face, hatch = INK, None
            else:
                continue
            ax.barh(row, .90, left=year - .45, height=.69, color=face,
                    edgecolor=INK, linewidth=.45, hatch=hatch)
    ax.set(xlim=(2013.45, 2024.55), ylim=(-.56, 4.56), xticks=years,
           yticks=np.arange(5), yticklabels=[f"Test {y}" for y in range(2020, 2025)],
           xlabel="Calendar year")
    ax.invert_yaxis()
    ax.tick_params(axis="both", length=0, labelsize=9)
    ax.spines[["left", "bottom"]].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=BLUE, edgecolor=INK, label="Development training"),
                       Patch(facecolor="white", edgecolor=INK, hatch="///", label="Validation / final refit"),
                       Patch(facecolor=INK, edgecolor=INK, label="Held-out test")],
              frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.15), fontsize=9)
    save_figure(fig, "figure1_temporal_design")

    # Figure 2: effect-focused annual view, avoiding lines that imply a smooth
    # temporal process and hide one-percentage-point paired differences.
    fig, ax = plt.subplots(figsize=(7.2, 3.25), layout="constrained")
    y = np.arange(5)
    first = 100 * annual.W120_top20_recall.sub(annual.W0_top20_recall).to_numpy()
    second = 100 * annual.stress_top20_recall.sub(annual.W120_top20_recall).to_numpy()
    expected_first = 100 * np.asarray(
        diagnostic["contrasts"]["W120_vs_W0"]["top20_recall"]["annual_differences"])
    assert np.allclose(first, expected_first, atol=1e-4)
    ax.axvline(0, color=INK, linewidth=.9)
    ax.scatter(first, y - .14, s=43, marker="o", facecolor=BLUE,
               edgecolor=INK, linewidth=.5, label="W120 - W0", zorder=3)
    ax.scatter(second, y + .14, s=48, marker="s", facecolor="white",
               edgecolor=INK, linewidth=1.2, label="Stress - W120", zorder=3)
    ax.set(xlim=(-2.35, 3.15), ylim=(-.55, 4.55), yticks=y,
           yticklabels=annual.Year.astype(str),
           xlabel="Change in top-20% recall (percentage points; positive is better)")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e6ebee", linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.18), fontsize=9)
    save_figure(fig, "figure2_annual_recall")

    contrasts = diagnostic["contrasts"]
    bias = [contrasts[c]["bias2"]["candidate_minus_base"] for c in CONTRASTS]
    within = [contrasts[c]["within_mse"]["candidate_minus_base"] for c in CONTRASTS]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), layout="constrained")
    for ax, values, panel, limits, face, hatch in (
        (axes[0], bias, "A", (-.305, .075), BLUE, None),
        (axes[1], within, "B", (-.025, .011), "white", "///"),
    ):
        y = np.arange(2)
        ax.barh(y, values, height=.47, color=face, edgecolor=INK,
                linewidth=.65, hatch=hatch)
        ax.axvline(0, color=INK, linewidth=.9)
        ax.set(xlim=limits, ylim=(-.55, 1.55), yticks=y,
               yticklabels=CONTRAST_LABELS,
               xlabel="Change in MSE component, (Mg/ha)$^2$")
        ax.invert_yaxis()
        ax.grid(axis="x", color="#e6ebee", linewidth=.6)
        ax.set_axisbelow(True)
        ax.text(.01, 1.02, panel, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom")
        if panel == "B":
            ax.set_yticklabels([])
    save_figure(fig, "figure3_mse_decomposition")

    # Figure 4: display interval uncertainty beside both decision outcomes.
    rmse = np.array([rolling.year_equal_delta_rmse, stress["rmse"]["year_equal_delta"]])
    rmse_low = np.array([rolling.crossed_year_site_ci95_low,
                         stress["rmse"]["crossed_year_site_ci95"][0]])
    rmse_high = np.array([rolling.crossed_year_site_ci95_high,
                          stress["rmse"]["crossed_year_site_ci95"][1]])
    recall = 100 * np.array([contrasts[c]["top20_recall"]["candidate_minus_base"]
                             for c in CONTRASTS])
    recall_low = 100 * np.array([contrasts[c]["top20_recall"]["crossed_year_site_ci95"][0]
                                 for c in CONTRASTS])
    recall_high = 100 * np.array([contrasts[c]["top20_recall"]["crossed_year_site_ci95"][1]
                                  for c in CONTRASTS])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), layout="constrained")
    for ax, point, low, high, panel, limits, label in (
        (axes[0], rmse, rmse_low, rmse_high, "A", (-.24, .19), "RMSE difference (Mg/ha)"),
        (axes[1], recall, recall_low, recall_high, "B", (-2.25, 3.25),
         "Top-20% recall difference (percentage points)"),
    ):
        y = np.arange(2)
        ax.axvline(0, color=INK, linewidth=.9, linestyle="--")
        for i in range(2):
            ax.plot([low[i], high[i]], [y[i], y[i]], color=INK, linewidth=1.4)
            ax.plot([low[i], low[i]], [y[i]-.08, y[i]+.08], color=INK, linewidth=1.1)
            ax.plot([high[i], high[i]], [y[i]-.08, y[i]+.08], color=INK, linewidth=1.1)
            ax.scatter(point[i], y[i], marker="o" if i == 0 else "s",
                       s=55, facecolor=BLUE if i == 0 else "white",
                       edgecolor=INK, linewidth=1.1, zorder=3)
        ax.set(xlim=limits, ylim=(-.55, 1.55), yticks=y,
               yticklabels=CONTRAST_LABELS, xlabel=label)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#e6ebee", linewidth=.6)
        ax.set_axisbelow(True)
        ax.text(.01, 1.02, panel, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom")
    save_figure(fig, "figure4_objective_tradeoff")

    # Figure 5: use frozen sampled SHAP summaries, not a recomputation.
    rank_path = ROOT / "modeling/stress_v1/explain/top_feature_importance.csv"
    ranked = pd.read_csv(rank_path).head(12).iloc[::-1]
    labels = {
        "wind_mean": "Mean wind speed", "wet_day_intensity": "Wet-day rain intensity",
        "hot30_days": "Days >30°C", "Longitude": "Longitude",
        "tmax_max": "Maximum Tmax", "PC002": "Genetic PC 2",
        "Latitude": "Latitude", "tmin_mean": "Mean Tmin",
        "precipitation_sum": "Cumulative precipitation",
        "Planting_DOY": "Planting day of year",
        "vpd_proxy_p90_kpa": "90th-percentile VPD proxy",
        "humidity_sd": "Humidity variability",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.45), layout="constrained")
    positions = np.arange(len(ranked))
    for i, row in enumerate(ranked.itertuples()):
        stress_feature = row.feature == "vpd_proxy_p90_kpa"
        ax.barh(i, row.year_equal_mean_abs_shap_mgha, height=.64,
                color="white" if stress_feature else BLUE, edgecolor=INK,
                linewidth=.65, hatch="///" if stress_feature else None)
    ax.set(yticks=positions, yticklabels=[labels[x] for x in ranked.feature],
           xlim=(0, .38), xlabel="Mean absolute SHAP attribution (Mg/ha)")
    ax.grid(axis="x", color="#e6ebee", linewidth=.6)
    ax.set_axisbelow(True)
    save_figure(fig, "figure5_shap_rank")


def register_fonts() -> None:
    base = Path(matplotlib.get_data_path()) / "fonts/ttf"
    fonts = {
        "Text": ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf", "DejaVuSerif-Italic.ttf"),
        "Sans": ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf"),
    }
    for family, filenames in fonts.items():
        for suffix, filename in zip(("", "-Bold", "-Italic"), filenames):
            pdfmetrics.registerFont(TTFont(family + suffix, str(base / filename)))
        pdfmetrics.registerFontFamily(family, normal=family, bold=family + "-Bold", italic=family + "-Italic")


def styles() -> dict[str, ParagraphStyle]:
    ink = colors.HexColor(INK)
    common = {"fontName": "Text", "fontSize": 9.6, "leading": 14.4,
              "textColor": ink, "allowWidows": 0, "allowOrphans": 0}
    return {
        "title": ParagraphStyle("title", fontName="Text-Bold", fontSize=18.2, leading=24, textColor=ink, spaceAfter=12),
        "h2": ParagraphStyle("h2", **{**common, "fontName": "Sans-Bold", "fontSize": 11.4, "leading": 15.2}, spaceBefore=14, spaceAfter=7, keepWithNext=True),
        "h3": ParagraphStyle("h3", **{**common, "fontName": "Sans-Bold", "fontSize": 9.8, "leading": 13.5}, spaceBefore=10, spaceAfter=5, keepWithNext=True),
        "body": ParagraphStyle("body", **common, alignment=TA_JUSTIFY, spaceAfter=8),
        "caption": ParagraphStyle("caption", fontName="Sans", fontSize=7.8, leading=10.7, textColor=ink, spaceAfter=10),
        "cell": ParagraphStyle("cell", fontName="Sans", fontSize=7.2, leading=9.5, textColor=ink, alignment=TA_LEFT),
        "head": ParagraphStyle("head", fontName="Sans-Bold", fontSize=7.1, leading=9.3, textColor=ink),
        "reference": ParagraphStyle("reference", **{**common, "fontSize": 8.3, "leading": 11.5}, leftIndent=15, firstLineIndent=-15, spaceAfter=6, wordWrap="CJK"),
        "meta": ParagraphStyle("meta", fontName="Sans", fontSize=8.1, leading=11.5, textColor=colors.HexColor(GREY), spaceAfter=12),
    }


def inline(value: str) -> str:
    value = value.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    value = escape(value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    value = re.sub(r"(https?://[^\s<]+)", r'<link href="\1" color="#255f85">\1</link>', value)
    return value


def make_table(lines: list[str], st: dict[str, ParagraphStyle]):
    rows = [[c.strip() for c in row.strip().strip("|").split("|")] for row in lines]
    rows = [rows[0]] + rows[2:]
    count = len(rows[0])
    if any(len(row) != count for row in rows):
        raise ValueError("Inconsistent manuscript table width")
    weights = np.array([max(9, min(24, len(header))) for header in rows[0]], float)
    weights /= weights.sum()
    table = Table([[Paragraph(inline(cell), st["head"] if i == 0 else st["cell"])
                    for cell in row] for i, row in enumerate(rows)],
                  colWidths=list(weights * PAPER_WIDTH), repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEABOVE", (0, 0), (-1, 0), .7, colors.HexColor(INK)),
        ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor(GREY)),
        ("LINEBELOW", (0, -1), (-1, -1), .6, colors.HexColor(INK)),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def pdf_story(st: dict[str, ParagraphStyle]):
    content = (PAPER / "manuscript.md").read_text()
    if "**Table 2.**" not in content or "## 5. Conclusion" not in content:
        raise ValueError("Manuscript appears incomplete")
    blocks = re.split(r"\n\s*\n", content.strip())
    story = [Paragraph("RESEARCH MANUSCRIPT  |  UNPUBLISHED DRAFT  |  19 OCTOBER 2026", st["meta"])]
    pending_table = None
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith("| "):
            if pending_table is not None:
                raise ValueError("Table is missing its caption")
            pending_table = make_table(block.splitlines(), st)
            continue
        if block.startswith("**Table ") and pending_table is not None:
            story.append(KeepTogether([Spacer(1, 5), pending_table, Spacer(1, 5),
                                       Paragraph(inline(" ".join(block.splitlines())), st["caption"])]))
            pending_table = None
            continue
        if pending_table is not None:
            raise ValueError("Table caption must immediately follow the table")
        if block.startswith("# "):
            story.append(Paragraph(inline(block[2:]), st["title"]))
        elif block.startswith("## "):
            story.append(Paragraph(inline(block[3:]), st["h2"]))
        elif block.startswith("### "):
            story.append(Paragraph(inline(block[4:]), st["h3"]))
        elif block.startswith("!["):
            match = re.fullmatch(r"!\[(.*)\]\(([^)]+)\)", block, re.S)
            if not match:
                raise ValueError("Invalid figure reference")
            caption, relative = match.groups()
            path = (PAPER / relative).resolve()
            if not path.exists() or not path.is_relative_to(ROOT):
                raise ValueError(f"Missing or out-of-scope figure: {relative}")
            im = Image(str(path))
            scale = min(PAPER_WIDTH / im.imageWidth, 320 / im.imageHeight)
            im.drawWidth, im.drawHeight = im.imageWidth * scale, im.imageHeight * scale
            im.hAlign = "CENTER"
            story.append(KeepTogether([Spacer(1, 6), im, Spacer(1, 6),
                                       Paragraph(inline(caption), st["caption"])]))
        elif re.match(r"^1\. ", block):
            for entry in block.splitlines():
                story.append(Paragraph(inline(entry), st["reference"]))
        elif block.startswith("*Unpublished"):
            story.append(Paragraph(inline(block), st["meta"]))
        else:
            story.append(Paragraph(inline(" ".join(block.splitlines())),
                                   st["caption"] if block.startswith("**Table ") else st["body"]))
    if pending_table is not None:
        raise ValueError("Final table is missing its caption")
    return story


def build() -> None:
    annual, diagnostic, stress, rolling = evidence()
    make_figures(annual, diagnostic, stress, rolling)
    register_fonts()
    st = styles()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#c9d1d5"))
        canvas.line(50, 37, A4[0] - 50, 37)
        canvas.setFont("Sans", 7)
        canvas.setFillColor(colors.HexColor(GREY))
        canvas.drawString(50, 25, "Maize yield prediction and hybrid selection | Unpublished draft")
        canvas.drawRightString(A4[0] - 50, 25, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=50, rightMargin=50,
                            topMargin=48, bottomMargin=52,
                            title="Weather-aware maize yield prediction and within-environment hybrid selection can diverge",
                            author="", subject="Retrospective G2F manuscript draft")
    doc.build(pdf_story(st), onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(OUT)
    extracted = "\n".join(page.extract_text() for page in reader.pages)
    required = ["57,673", "0.2977", "0.2872", "92.9%", "Figure 5.", "Table 2.", "References"]
    missing = [term for term in required if term not in extracted]
    if missing:
        raise ValueError(f"Rendered PDF text check failed: {missing}")
    receipt = {
        "status": "DRAFT_RENDERED_NOT_SUBMITTED", "output": str(OUT.relative_to(ROOT)),
        "pages": len(reader.pages), "pdf_sha256": sha(OUT),
        "manuscript_sha256": sha(PAPER / "manuscript.md"),
        "source_sha256": {
            "objective_summary": sha(ROOT / "modeling/stress_v1/objective_analysis/summary.json"),
            "annual_components": sha(ROOT / "modeling/stress_v1/objective_analysis/annual_components.csv"),
            "stress_summary": sha(ROOT / "modeling/stress_v1/results/summary.json"),
            "rolling_stability": sha(ROOT / "modeling/rolling_v1/analysis/cross_year_stability.csv"),
        },
        "text_checks": "PASS", "visual_inspection": "PENDING",
    }
    (PAPER / "render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    build()
