"""Build the manuscript PDF from reviewed prose and hash-bound result tables.

This renderer does not fit models, recompute bootstrap intervals, or alter any
frozen experiment. Run the evidence collector separately if sources change.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "reports/manuscript"
TMP = ROOT / "tmp/pdfs/manuscript"
TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP / "mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                              TableStyle, Image, KeepTogether, PageBreak)

MODELS = ["Ridge", "ExtraTrees", "CatBoost"]
LABELS = {"Ridge": "Ridge", "ExtraTrees": "Extra Trees", "CatBoost": "CatBoost",
          "GlobalMean": "Global mean", "HistoricalSiteMean": "Historical site mean"}
BLUE, GOLD, INK, GREY = "#255f85", "#b87530", "#1f2a32", "#78838c"
OUT = ROOT / "output/pdf/MAIZE_WEATHER_MANUSCRIPT.pdf"
WIDTH = A4[0] - 100


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_evidence():
    receipt = json.loads((DIR / "evidence.receipt.json").read_text())
    if receipt["status"] != "PASS" or digest(DIR / "evidence.json") != receipt["evidence_sha256"]:
        raise ValueError("Evidence export differs from independently reconciled receipt")
    data = json.loads((DIR / "evidence.json").read_text())
    for path, expected in data["source_hashes"].items():
        source = (ROOT / path).resolve()
        if not source.is_relative_to(ROOT) or digest(source) != expected:
            raise ValueError(f"Changed evidence source: {path}")
    if data["verification"]["status"] != "PASS":
        raise ValueError("Evidence reconciliation must pass")
    return data


def finish_figure(fig, name):
    target = DIR / "figures"
    target.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(target / f"{name}.{ext}", dpi=360, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def prepare_figures(data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 9, "axes.titlesize": 10,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#b9c1c6", "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.linewidth": .7, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "figure.constrained_layout.use": True})
    # Actual protocol years; validation is incorporated only at final refit.
    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    for i, fold in enumerate(data["design"]["folds"]):
        test = fold["test_year"]
        for year in range(2014, test + 1):
            c = BLUE if year < test - 1 else GOLD if year == test - 1 else INK
            ax.add_patch(Rectangle((year - .45, i - .31), .9, .62, facecolor=c, edgecolor="none"))
    ax.set_xlim(2013.4, 2024.6)
    ax.set_ylim(4.7, -.7)
    ax.set_xticks(range(2014, 2025))
    ax.set_yticks(range(5), [f"Test {y}" for y in range(2020, 2025)])
    ax.set_xlabel("Calendar year")
    ax.tick_params(axis="both", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.legend(handles=[Patch(color=BLUE, label="Development training"),
                       Patch(color=GOLD, label="Validation; included in final refit"),
                       Patch(color=INK, label="Test")],
              loc="upper center", bbox_to_anchor=(.5, 1.25), ncol=3,
              frameon=False, fontsize=8)
    finish_figure(fig, "figure1_temporal_design")

    primary = {r["model"]: r for r in data["primary_results"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharex=True, sharey=True)
    specifications = [("cumulative_minus_W0", "a  Cumulative W120 minus W0", BLUE),
                      ("segmented_minus_cumulative", "b  Segmented minus cumulative W120", GOLD)]
    for ax, (key, title, color) in zip(axes, specifications):
        for i, model in enumerate(MODELS):
            row = primary[model][key]
            x = row["year_equal_delta_rmse"]
            lo, hi = row["crossed_year_site_ci95_low"], row["crossed_year_site_ci95_high"]
            ax.errorbar(x, i, xerr=[[x-lo], [hi-x]], fmt="o", color=color,
                        markersize=6, capsize=3, elinewidth=1.4)
        ax.axvline(0, color=INK, linewidth=.9, linestyle="--", zorder=0)
        ax.set_xlim(-.26, .30)
        ax.set_xticks([-.2, -.1, 0, .1, .2, .3])
        ax.set_yticks(range(3), [LABELS[m] for m in MODELS])
        ax.set_ylim(2.55, -.55)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
        ax.set_xlabel("Paired RMSE difference (Mg/ha)")
        ax.grid(axis="x", color="#edf0f2", linewidth=.6)
        ax.set_axisbelow(True)
    finish_figure(fig, "figure2_primary_contrasts")

    yearly = pd.DataFrame(data["yearly_metrics"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), sharey=True)
    delta_values = []
    for model in MODELS:
        subset = yearly[yearly.model.eq(model)]
        base = subset[subset.representation.eq("cumulative") & subset.window.eq(0)].set_index("Year").macro_rmse
        for rep in ("cumulative", "segmented"):
            current = subset[subset.representation.eq(rep) & subset.window.eq(120)].set_index("Year").macro_rmse
            delta_values.extend((current-base).values.tolist())
    lo, hi = min(delta_values)-.035, max(delta_values)+.035
    for ax, model, letter in zip(axes, MODELS, "abc"):
        subset = yearly[yearly.model.eq(model)]
        base = subset[subset.representation.eq("cumulative") & subset.window.eq(0)].set_index("Year").macro_rmse
        for rep, color, marker, offset in (("cumulative", BLUE, "o", -.10), ("segmented", GOLD, "s", .10)):
            current = subset[subset.representation.eq(rep) & subset.window.eq(120)].set_index("Year").macro_rmse
            delta = current-base
            xs = np.arange(5)+offset
            ax.vlines(xs, 0, delta.values, colors=color, linewidth=1.2, alpha=.8)
            ax.scatter(xs, delta.values, color=color, marker=marker, s=26, label=rep.capitalize(), zorder=3)
        ax.axhline(0, color=INK, linewidth=.9)
        ax.set_xticks(range(5), [str(y)[2:] for y in range(2020, 2025)])
        ax.set_xlabel("Test year (20xx)")
        ax.set_title(f"{letter}  {LABELS[model]}", loc="left", fontweight="bold")
        ax.set_ylim(lo, hi)
        ax.grid(axis="y", color="#edf0f2", linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("W120 minus W0 RMSE (Mg/ha)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, 1.40), ncol=2, frameon=False, fontsize=9)
    finish_figure(fig, "figure3_annual_differences")

    cum = pd.DataFrame(data["cumulative_cross_year_comparisons"])
    seg = pd.DataFrame(data["segmented_cross_year_comparisons"])
    seg = seg[seg.reference.eq("no_weather")]
    both = pd.concat([cum, seg])
    lo = float(both.crossed_year_site_ci95_low.min()) - .025
    hi = float(both.crossed_year_site_ci95_high.max()) + .025
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.35), sharex=True, sharey=True)
    for ax, model, letter in zip(axes, MODELS, "abc"):
        for frame, color, marker, offset, label in ((cum, BLUE, "o", -2, "Cumulative"),
                                                  (seg, GOLD, "s", 2, "Segmented")):
            rows = frame[frame.model.eq(model)].sort_values("window")
            y = rows.year_equal_delta_rmse.to_numpy()
            lower = rows.crossed_year_site_ci95_low.to_numpy()
            upper = rows.crossed_year_site_ci95_high.to_numpy()
            ax.errorbar(rows.window+offset, y, yerr=np.vstack([y-lower, upper-y]),
                        fmt=marker, color=color, markersize=4.5, capsize=2.5,
                        elinewidth=1, label=label)
        ax.axhline(0, color=INK, linewidth=.9, linestyle="--")
        ax.set_xticks([30, 60, 90, 120])
        ax.set_xlim(21, 129)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{letter}  {LABELS[model]}", loc="left", fontweight="bold")
        ax.set_xlabel("Days after planting")
        ax.grid(axis="y", color="#edf0f2", linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Weather minus W0 RMSE (Mg/ha)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, 1.38), ncol=2, frameon=False)
    finish_figure(fig, "figure4_weather_horizons")


def register_fonts():
    root = Path(matplotlib.get_data_path()) / "fonts/ttf"
    families = {"Paper": ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf", "DejaVuSerif-Italic.ttf", "DejaVuSerif-BoldItalic.ttf"),
                "PaperSans": ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf")}
    for family, files in families.items():
        names = [family, family+"-Bold", family+"-Italic", family+"-BoldItalic"]
        for name, filename in zip(names, files):
            pdfmetrics.registerFont(TTFont(name, str(root / filename)))
        pdfmetrics.registerFontFamily(family, normal=names[0], bold=names[1], italic=names[2], boldItalic=names[3])


def styles():
    common = {"fontName": "Paper", "fontSize": 10.3, "leading": 15.2,
              "textColor": colors.HexColor(INK), "allowWidows": 0, "allowOrphans": 0}
    return {
        "title": ParagraphStyle("title", fontName="Paper-Bold", fontSize=20, leading=26, textColor=colors.HexColor(INK), spaceAfter=16),
        "h2": ParagraphStyle("h2", **{**common, "fontName":"PaperSans-Bold", "fontSize":12.2, "leading":17}, spaceBefore=17, spaceAfter=8, keepWithNext=True),
        "h3": ParagraphStyle("h3", **{**common, "fontName":"PaperSans-Bold", "fontSize":10, "leading":14}, spaceBefore=12, spaceAfter=6, keepWithNext=True),
        "body": ParagraphStyle("body", **common, alignment=TA_JUSTIFY, spaceAfter=9),
        "bullet": ParagraphStyle("bullet", **common, leftIndent=12, firstLineIndent=-10, spaceAfter=6),
        "caption": ParagraphStyle("caption", fontName="PaperSans", fontSize=8.3, leading=11.6, textColor=colors.HexColor(INK), spaceAfter=12),
        "cell": ParagraphStyle("cell", fontName="PaperSans", fontSize=8, leading=11, textColor=colors.HexColor(INK)),
        "number": ParagraphStyle("number", fontName="PaperSans", fontSize=8, leading=11, textColor=colors.HexColor(INK), alignment=TA_RIGHT),
        "thead": ParagraphStyle("thead", fontName="PaperSans-Bold", fontSize=7.9, leading=10.8, textColor=colors.HexColor(INK)),
        "reference": ParagraphStyle("reference", **{**common,"fontSize":9,"leading":12.5}, leftIndent=14, firstLineIndent=-14, spaceAfter=7, wordWrap="CJK"),
        "meta": ParagraphStyle("meta", fontName="PaperSans", fontSize=8, leading=11, textColor=colors.HexColor(GREY), spaceAfter=10),
    }


def inline(text):
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2011", "-")
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#255f85">\1</link>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("Mg ha−1", "Mg/ha")
    for source, rendered in {"n_et": "n<sub>e,t</sub>", "E_t": "E<sub>t</sub>",
                             "R_t": "R<sub>t</sub>", "R_reference": "R<sub>reference</sub>",
                             "R_candidate": "R<sub>candidate</sub>"}.items():
        text = re.sub(r"\b" + source + r"\b", rendered, text)
    return text


def number(value, digits=3):
    return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def interval(row):
    return (f"{row['year_equal_delta_rmse']:+.3f} "
            f"[{row['crossed_year_site_ci95_low']:+.3f}, {row['crossed_year_site_ci95_high']:+.3f}]")


def table_data(name, data):
    primary = data["primary_results"]
    if name == "coverage":
        rows = [[str(r["Year"]), f"{r['n_training_samples']:,}", f"{r['n_scored']:,}", str(r["n_environments"]), str(r["n_site_codes"])] for r in data["coverage"]]
        return "Table 1. Training and scored-test coverage by year.", ["Test year", "Final-fit observations", "Scored observations", "Environment-years", "Site codes"], rows, [65,110,110,105,105], "Final-fit observations include only years preceding the test year. Site codes are not necessarily independent physical locations; their unique count across all test years is 31, not the sum of this column. Scored observations total 57,673; environment-years total 129."
    if name == "primary":
        rows = [[LABELS[r["model"]], number(r["W0_year_equal_rmse"],4), number(r["cumulative_W120_year_equal_rmse"],4), number(r["segmented_W120_year_equal_rmse"],4)] for r in primary]
        return "Table 2. Primary five-year mean environment RMSE (Mg/ha).", ["Model", "W0", "Cumulative W120", "Segmented W120"], rows, [120,105,135,135], "Each environment receives equal weight within its test year; each of the five test years receives equal weight. Lower is better. W0 contains genetic and static information, not post-planting weather."
    if name == "contrasts":
        rows=[]
        for r in primary:
            for key, label in [("cumulative_minus_W0","Cumulative - W0"), ("segmented_minus_cumulative","Segmented - cumulative"), ("segmented_minus_W0","Segmented - W0")]:
                c=r[key]
                rows.append([LABELS[r["model"]],label,interval(c),f"{c['years_improved']}/5"])
        return "Table 3. W120 paired differences and conditional sensitivity intervals.", ["Model", "Contrast", "RMSE difference [95% interval]", "Years improved"], rows, [83,140,205,67], "Units are Mg/ha. Negative differences favor the first pipeline. Intervals use the crossed year/site-code bootstrap (2,000 draws); they are conditional on fitted models, approximate and not multiplicity-adjusted. An interval is not a prospective performance guarantee."
    if name == "secondary":
        means = data["year_equal_metrics"]
        rows=[]
        for model in MODELS:
            for rep,w,label in [("cumulative",0,"W0"),("cumulative",120,"Cumulative W120"),("segmented",120,"Segmented W120")]:
                r=next(r for r in means if r["model"]==model and r["representation"]==rep and r["window"]==w)
                rows.append([LABELS[model],label,number(r["year_equal_mae"]),number(r["year_equal_spearman"]),number(r["year_equal_top20_recall"])])
        return "Table 4. Secondary error and within-environment ranking measures.", ["Model", "Pipeline", "MAE (Mg/ha)", "Spearman", "Top-20% recall"], rows, [80,137,93,85,100], "MAE is lower-is-better; Spearman and recall are higher-is-better. Recall is a fraction, not a percentage. For these nine model summaries, Spearman is defined in all 129 environment-years. No secondary-metric uncertainty or confirmatory superiority claim is implied."
    if name == "subgroups":
        rows=[]
        for year in range(2020,2025):
            h=next(r for r in data["subgroup_support"] if r["Year"]==year and r["group_type"]=="hybrid" and r["group"]=="unseen")
            s=next(r for r in data["subgroup_support"] if r["Year"]==year and r["group_type"]=="site_code" and r["group"]=="unseen")
            rows.append([str(year),f"{h['n']:,}",str(h['n_environments']),f"{s['n']:,}",str(s['n_environments'])])
        return "Table 5. Scored support for previously unseen hybrids and site codes.", ["Test year", "Unseen-hybrid observations", "Environments", "Unseen-site observations", "Environments"], rows, [65,125,85,125,95], "Unseen status is relative to the corresponding final-fit training data. Zero indicates an empty evaluated subgroup, not zero prediction error. These cuts are descriptive and are not independent spatial holdouts."
    if name == "all_means":
        rows=[]
        means=data["year_equal_metrics"]
        for model in ["GlobalMean","HistoricalSiteMean",*MODELS]:
            for w in ([0] if model not in MODELS else [0,30,60,90,120]):
                c=next(r for r in means if r["model"]==model and r["window"]==w and r["representation"]=="cumulative")
                s=next(r for r in means if r["model"]==model and r["window"]==w and r["representation"]=="segmented")
                rows.append([LABELS[model],f"W{w}",number(c["year_equal_rmse"],4),"Shared" if model not in MODELS else number(s["year_equal_rmse"],4)])
        return "", ["Model", "Horizon", "Cumulative RMSE", "Segmented RMSE"], rows, [170,65,130,130], "RMSE units: Mg/ha. The full yearly and subgroup tables are available in the companion analytical outputs; no configurations are omitted from the means above."
    if name == "hyperparameters":
        return "", ["Family", "Candidate values", "Fixed configuration"], [["Ridge","alpha = 100 or 10,000","Training-only standardization"],["Extra Trees","min_samples_leaf = 5 or 20","160 trees; maximum depth 18; max_features = 0.8"],["CatBoost","depth = 6 or 8","600 iterations; learning rate 0.05; L2 = 5; RMSE loss"]], [100,160,235], "Model versions recorded in the frozen runs: scikit-learn 1.7.2 and CatBoost 1.2.8. The candidates have equal search counts, not equal statistical complexity across representations."
    if name == "provenance":
        return "", ["Evidence", "Companion project path"], [["Cohort and source verification","preparation/outputs/; preparation/provenance_followup/verification_report.json"],["Original model protocol","modeling/PROTOCOL_LOCK.md"],["Rolling design and results","modeling/rolling_v1/PROTOCOL.md; modeling/rolling_v1/analysis/"],["Segmented design and results","modeling/segmented_v1/PROTOCOL.md; modeling/segmented_v1/analysis/"],["Independent numerical audits","modeling/rolling_v1/validation_checks.json; modeling/segmented_v1/validation_checks.json"],["Manuscript evidence export","reports/manuscript/evidence.json; evidence.receipt.json"],["Reproduction of this PDF","scripts/collect_manuscript_evidence.py; scripts/build_manuscript.py"]], [125,370], "The manuscript build verifies the source hashes before rendering. It does not rerun model fitting or replace the frozen experimental records. A public code-release URL is not yet available."
    raise KeyError(name)


FIGURES = {
    "design": ("figure1_temporal_design", "Figure 1. Temporally ordered development, validation and test splits. Each row is one test-year evaluation. Model parameters and preprocessing are fitted using development training data, candidates are selected using the immediately preceding year, and final refits include both earlier portions before test prediction. The original 2024 evaluation was reused unchanged for cumulative weather. Historical backtests and the representation extension are retrospective, not prospectively independent confirmation."),
    "primary": ("figure2_primary_contrasts", "Figure 2. Primary W120 paired comparisons. Points show year-equal means of environment-level RMSE differences; whiskers show 95% crossed year/site-code bootstrap sensitivity intervals from 2,000 draws. (a) Cumulative weather versus the same-family W0 baseline. (b) Segmented versus cumulative weather. Negative values favor the first pipeline. Each comparison contains five years and 129 environment-years. The intervals are conditional on fitted models and do not include selection/refitting uncertainty or multiplicity adjustment."),
    "annual": ("figure3_annual_differences", "Figure 3. Annual W120 weather increments relative to the same-family W0 model. Each mark is one test year's environment-equal RMSE difference; negative values indicate improvement. Circles denote cumulative weather and squares denote segmented weather. Vertical stems connect descriptive point estimates to zero and are not uncertainty intervals. All five test years are displayed; aggregate uncertainty is provided in Figure 2 and Table 3."),
    "windows": ("figure4_weather_horizons", "Figure 4. Weather increments at all evaluated observation horizons. Points show year-equal means of paired environment RMSE differences from W0; whiskers show 95% crossed year/site-code bootstrap sensitivity intervals. Circles denote cumulative weather and squares denote segmented weather. Horizontal offsets separate the two representations visually without changing their true horizons of 30, 60, 90 and 120 days. W30 is identical across representations. The three panels use common scales; no intermediate observation days were evaluated. All intervals are conditional and unadjusted for multiple comparisons."),
}


def render_table(name, data, style):
    title, headers, rows, widths, note = table_data(name, data)
    widths=[x*WIDTH/sum(widths) for x in widths]
    content=[[Paragraph(inline(h),style["thead"]) for h in headers]]
    for row in rows:
        content.append([Paragraph(inline(str(cell)),style["number"] if re.fullmatch(r"[+\-\d., /]+",str(cell)) else style["cell"]) for cell in row])
    table=Table(content,colWidths=widths,repeatRows=1,hAlign="LEFT")
    table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                              ("LINEABOVE",(0,0),(-1,0),.7,colors.HexColor(INK)),
                              ("LINEBELOW",(0,0),(-1,0),.45,colors.HexColor(GREY)),
                              ("LINEBELOW",(0,-1),(-1,-1),.6,colors.HexColor(INK)),
                              ("TOPPADDING",(0,0),(-1,-1),6),
                              ("BOTTOMPADDING",(0,0),(-1,-1),6),
                              ("LEFTPADDING",(0,0),(-1,-1),5),
                              ("RIGHTPADDING",(0,0),(-1,-1),5)]))
    items=[]
    if title:
        items.append(Paragraph("<b>"+inline(title)+"</b>",style["caption"]))
    items.extend([table,Spacer(1,5),Paragraph(inline(note),style["caption"])])
    return KeepTogether(items) if len(rows)<12 else items


def reference_entries():
    raw=json.loads((DIR/"references.json").read_text())
    # Accept the literature collector's explicit records; no bibliography invented.
    return raw["references"] if isinstance(raw,dict) else raw


def reference_text(row):
    if "formatted_reference" in row:
        return row["formatted_reference"]
    if "citation" in row:
        return row["citation"]
    authors=row["authors"]
    authors="; ".join(authors) if isinstance(authors,list) else authors
    return f"{authors} ({row['year']}). {row['title']}. {row.get('journal',row.get('publisher',''))}. DOI: {row['doi']}."


def expand_manuscript():
    text=(DIR/"manuscript.md").read_text()
    for match in list(re.finditer(r"\{\{include:([^}]+)\}\}",text)):
        path=(DIR/match.group(1)).resolve()
        if not path.is_relative_to(DIR):
            raise ValueError("Include outside manuscript directory")
        text=text.replace(match.group(),path.read_text().strip())
    return text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2011", "-")


def build():
    data=load_evidence()
    register_fonts()
    prepare_figures(data)
    st=styles()
    text=expand_manuscript()
    # Body must remain a review draft, never a fabricated published article.
    required=["## Abstract","## 1. Introduction","## 2.","## 3. Results","## 4. Discussion",
              "## 5. Conclusions","57,673","0.72%","retrospective","571"]
    if any(term not in text for term in required):
        raise ValueError("Required manuscript section/evidence missing")
    if any(c in text for c in ("\u2011","\ufffd")):
        raise ValueError("Unsupported or malformed manuscript character")
    story=[Paragraph("RESEARCH MANUSCRIPT  |  REVIEW DRAFT  |  18 OCTOBER 2026",st["meta"])]
    paragraphs=re.split(r"\n\s*\n",text.strip())
    full_text=[]
    for block in paragraphs:
        block=block.strip()
        if not block or block.startswith("<!--"):
            continue
        table_match=re.fullmatch(r"\{\{table:(\w+)\}\}",block)
        figure_match=re.fullmatch(r"\{\{figure:(\w+)\}\}",block)
        if block == "{{equation:metrics}}":
            equations = [
                "RMSE<sub>e,t</sub> = [(1/n<sub>e,t</sub>) Σ<sub>i</sub> (ŷ<sub>i,e,t</sub> - y<sub>i,e,t</sub>)<super>2</super>]<super>1/2</super>",
                "R<sub>t</sub> = (1/E<sub>t</sub>) Σ<sub>e</sub> RMSE<sub>e,t</sub>;     R = (1/5) Σ<sub>t</sub> R<sub>t</sub>",
                "ΔR = R<sub>candidate</sub> - R<sub>reference</sub>;     Gain (%) = 100 (R<sub>reference</sub> - R<sub>candidate</sub>)/R<sub>reference</sub>",
            ]
            formula = Table([[Paragraph(eq, st["body"]), Paragraph(f"({i+1})",st["number"])] for i,eq in enumerate(equations)], colWidths=[WIDTH-25,25])
            formula.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
            story.extend([formula,Spacer(1,8)])
            full_text.extend([eq.replace("<super>", "<sup>").replace("</super>", "</sup>") for eq in equations])
        elif table_match:
            name=table_match.group(1)
            items=render_table(name,data,st)
            story.extend(items if isinstance(items,list) else [items])
            title,headers,rows,_,note=table_data(name,data)
            full_text.extend([title,"\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |",*["| "+" | ".join(str(c) for c in r)+" |" for r in rows]]),note])
        elif figure_match:
            filename,caption=FIGURES[figure_match.group(1)]
            im=Image(str(DIR/"figures"/f"{filename}.png"))
            height=im.imageHeight*WIDTH/im.imageWidth
            im.drawWidth,im.drawHeight=WIDTH,height
            story.append(KeepTogether([Spacer(1,8),im,Spacer(1,8),Paragraph(inline(caption),st["caption"])]))
            full_text.extend([f"![{caption}](figures/{filename}.png)"])
        elif block=="{{references}}":
            for ref in reference_entries():
                citation=reference_text(ref)
                doi=ref.get("doi")
                if doi:
                    citation=re.sub(r"https://doi\.org/\S+", "", citation).strip()
                    citation+=f" [https://doi.org/{doi}](https://doi.org/{doi})"
                elif ref.get("url"):
                    citation=citation.replace(ref["url"], f"[Publisher record]({ref['url']})")
                story.append(Paragraph(inline(citation),st["reference"]))
                full_text.append(citation)
        elif block.startswith("### "):
            if block.startswith("### Supplementary Table S2."):
                story.append(PageBreak())
            story.append(Paragraph(inline(block[4:]),st["h3"]))
            full_text.append(block)
        elif block.startswith("## "):
            if block=="## Supplementary material":
                story.append(PageBreak())
            story.append(Paragraph(inline(block[3:]),st["h2"]))
            full_text.append(block)
        elif block.startswith("# "):
            story.append(Paragraph(inline(block[2:]),st["title"]))
            full_text.append(block)
        elif block.startswith("- "):
            for item in block.splitlines():
                story.append(Paragraph("- "+inline(item.removeprefix("- ")),st["bullet"]))
            full_text.append(block)
        else:
            story.append(Paragraph(inline(" ".join(block.splitlines())),st["body"]))
            full_text.append(block)

    def footer(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd1d6"))
        canvas.setLineWidth(.4)
        canvas.line(50,38,A4[0]-50,38)
        canvas.setFont("PaperSans",7)
        canvas.setFillColor(colors.HexColor(GREY))
        canvas.drawString(50,26,"Within-season weather and maize yield | Unpublished research manuscript")
        canvas.drawRightString(A4[0]-50,26,str(doc.page))
        if doc.page>1:
            canvas.setFont("PaperSans",7)
            canvas.drawString(50,A4[1]-30,"FIVE-YEAR RETROSPECTIVE BENCHMARK")
        canvas.restoreState()

    OUT.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=50,rightMargin=50,
                          topMargin=48,bottomMargin=52,
                          title="Incremental value of within-season weather information for maize yield prediction: a five-year retrospective benchmark",
                          author="",subject="Unpublished research manuscript; verified retrospective G2F results")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(OUT)
    extracted="\n".join(p.extract_text() for p in reader.pages)
    checks=[term for term in required if not term.startswith("##")]
    checks += ["Figure 1.","Figure 2.","Figure 3.","Figure 4.","Table 5.","2.5332","2.5149","2.6281"]
    missing=[term for term in checks if term not in extracted]
    if missing:
        raise ValueError(f"PDF extraction missing expected content: {missing}")
    (DIR/"manuscript_full.md").write_text("\n\n".join(full_text)+"\n")
    receipt={"output":str(OUT.relative_to(ROOT)),"pages":len(reader.pages),
             "pdf_sha256":digest(OUT),"text_checks":"PASS",
             "evidence_sha256":digest(DIR/"evidence.json"),
             "source_hashes_rechecked":len(data["source_hashes"]),
             "source_hashes":data["source_hashes"],
             "manuscript_sha256":digest(DIR/"manuscript.md"),
             "methods_sha256":digest(DIR/"methods_draft.md"),
             "references_sha256":digest(DIR/"references.json"),
             "builder_sha256":digest(Path(__file__)),
             "figures":{name:digest(DIR/"figures"/(name+".png")) for name,_ in FIGURES.values()},
             "visual_inspection":"PENDING - render and inspect every page"}
    OUT.with_suffix(".receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    build()
