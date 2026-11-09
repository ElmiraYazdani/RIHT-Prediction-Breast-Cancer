"""Build the manuscript (.docx) from scripts/manuscript_content.py, outputs/manuscript tables and figures.

Usage: python scripts/build_manuscript.py   ->  <project>/Manuscript_Thyroid_Dosiomics_V2.docx
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

import manuscript_content as C
from thyroid_rihp.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "manuscript"
FONT = "Times New Roman"
TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|\{\{.+?\}\}|\[@[^\]]+\])", re.S)

# ---------------------------------------------------------------- citation numbering (order of first appearance)
ORDER = []


def scan(text):
    for m in re.finditer(r"\[@([^\]]+)\]", text):
        for k in re.split(r"\s*[,;]\s*@?", m.group(1)):
            k = k.strip().lstrip("@")
            if k and k not in ORDER:
                ORDER.append(k)


texts = [t for _, t in C.ABSTRACT] + [C.TITLE]
for kind, payload in C.B:
    if kind in ("h1", "h2", "p", "bullet"):
        texts.append(payload)
    elif kind in ("fig", "table"):
        texts.append(payload[1])
for t in texts:
    scan(t)
missing = [k for k in ORDER if k not in C.REFS]
assert not missing, f"unknown reference keys: {missing}"
NUM = {k: i + 1 for i, k in enumerate(ORDER)}


def cite_text(m):
    keys = [k.strip().lstrip("@") for k in re.split(r"\s*[,;]\s*@?", m.group(1)) if k.strip()]
    nums = sorted(NUM[k] for k in keys)
    out, i = [], 0
    while i < len(nums):          # collapse runs 3,4,5 -> 3–5
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if j == i else (f"{nums[i]},{nums[j]}" if j == i + 1 else f"{nums[i]}–{nums[j]}"))
        i = j + 1
    return "[" + ",".join(out) + "]"


# ---------------------------------------------------------------- docx helpers
def set_cell_bg(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def add_runs(par, text, size=None, bold=None, italic=None):
    for tok in TOKEN.split(text):
        if not tok:
            continue
        r = None
        if tok.startswith("**") and tok.endswith("**"):
            r = par.add_run(tok[2:-2])
            r.bold = True
        elif tok.startswith("{{"):
            r = par.add_run("[" + tok[2:-2] + "]")
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif tok.startswith("[@"):
            r = par.add_run(re.sub(r"\[@([^\]]+)\]", cite_text, tok))
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = par.add_run(tok[1:-1])
            r.italic = True
        else:
            r = par.add_run(tok)
        r.font.name = FONT
        r._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        if size:
            r.font.size = Pt(size)
        if bold is not None and not r.bold:
            r.bold = bold
        if italic is not None and not r.italic:
            r.italic = italic
    return par


def para(doc, text, size=11, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=6, bold=None, italic=None, indent=None):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_after, pf.space_before, pf.line_spacing = Pt(space_after), Pt(0), 1.25
    if indent:
        pf.left_indent = Inches(indent)
    add_runs(p, text, size=size, bold=bold, italic=italic)
    return p


def heading(doc, text, level):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before, pf.space_after, pf.keep_with_next = Pt(14 if level == 1 else 9), Pt(5), True
    r = p.add_run(text)
    r.bold = True
    r.font.name = FONT
    r.font.size = Pt(14 if level == 1 else 12)
    if level == 2:
        r.italic = True
    # outline level so Word shows it in the navigation pane
    pPr = p._p.get_or_add_pPr()
    ol = OxmlElement("w:outlineLvl")
    ol.set(qn("w:val"), str(level - 1))
    pPr.append(ol)
    return p


def add_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if t:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), t)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = txt
        r._r.append(el)
    r.font.name, r.font.size = FONT, Pt(10)


def add_table(doc, df, caption, col_widths=None, font=8, header_bg="DCE8F5", landscape_note=False):
    cp = para(doc, caption, size=10, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=3, bold=None)
    cp.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(c))
        r.bold, r.font.size, r.font.name = True, Pt(font), FONT
        set_cell_bg(cell, header_bg)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run("" if pd.isna(v) else str(v))
            r.font.size, r.font.name = Pt(font), FONT
            if i > 0:
                cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if col_widths:
        for row in t.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def fmt3(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = df[c].map(lambda x: f"{x:.3f}")
    return df


def build():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.left_margin = sec.right_margin = Inches(1.0)
    sec.top_margin = sec.bottom_margin = Inches(0.9)
    add_page_number(sec)
    st = doc.styles["Normal"]
    st.font.name, st.font.size = FONT, Pt(11)

    p = para(doc, C.TITLE, size=16, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8, bold=True)
    para(doc, C.AUTHORS, size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para(doc, C.CORRESP, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para(doc, "Running title: " + C.RUNNING, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10, italic=True)

    heading(doc, "Abstract", 1)
    for k, v in C.ABSTRACT:
        para(doc, f"**{k}:** {v}", size=10.5, space_after=4)
    para(doc, f"**Keywords:** {C.KEYWORDS}", size=10.5, space_after=6)

    figno, tabno = 0, 0
    t1 = pd.read_csv(OUT / "table1_characteristics.csv")
    t2 = pd.read_csv(OUT / "table2_primary_performance.csv", dtype=str)
    t2 = t2.drop(columns=["Calibration slope", "SD of AUC across repeats"]).rename(
        columns={"Feature configuration": "Configuration", "ROC-AUC (95% CI)": "ROC-AUC (95% CI)",
                 "Balanced accuracy": "Bal. acc."})
    tables = {"table1": (t1, [1.7, 1.45, 1.45, 0.9, 1.4], 8.5),
              "table2": (t2, [1.35, 1.25, 0.5, 0.5, 0.5, 0.5, 0.45, 0.5, 0.5], 7.5)}
    for kind, payload in C.B:
        if kind == "h1":
            heading(doc, payload, 1)
        elif kind == "h2":
            heading(doc, payload, 2)
        elif kind == "p":
            para(doc, payload)
        elif kind == "bullet":
            p = doc.add_paragraph(style="List Bullet")
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.line_spacing = 1.2
            add_runs(p, payload, size=11)
        elif kind == "fig":
            name, cap, width = payload
            doc.add_paragraph().add_run().add_picture(str(OUT / f"{name}.png"), width=Inches(width))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.paragraphs[-1].paragraph_format.keep_with_next = True
            para(doc, cap, size=9.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=10)
        elif kind == "table":
            key, cap = payload
            df, widths, font = tables[key]
            add_table(doc, df, cap, widths, font)

    for k, v in C.DECLARATIONS:
        pass
    heading(doc, "Declarations", 1)
    for k, v in C.DECLARATIONS:
        para(doc, f"**{k}.** {v}", size=10.5, space_after=4)
    heading(doc, "List of abbreviations", 1)
    para(doc, C.ABBREV, size=10)

    heading(doc, "References", 1)
    for k in ORDER:
        p = para(doc, f"{NUM[k]}.\t{C.REFS[k]}", size=10, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=3)
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)

    # ---------------- supplementary material
    doc.add_page_break()
    heading(doc, "Supplementary material", 1)
    s1 = pd.read_csv(OUT / "table3_dosiomics_vs_dvh.csv")
    s1 = s1[["model", "strategy", "AUC_DVH", "AUC_DFs", "AUC_DVH+DFs", "AUC_CBs", "AUC_CT RFs", "dAUC_DFs_minus_DVH",
             "dAUC_DVH+DFs_minus_DVH"]].sort_values("AUC_DVH", ascending=False)
    s1 = fmt3(s1, [c for c in s1.columns if c.startswith(("AUC", "dAUC"))])
    s1.columns = ["Classifier", "Imbalance", "DVH", "DFs", "DVH+DFs", "CBs", "CT RFs", "Δ DFs−DVH", "Δ DVH+DFs−DVH"]
    add_table(doc, s1, "Supplementary Table S1. Out-of-fold ROC-AUC of single groups and of DVH + DFs for all 20 "
              "classifier–imbalance combinations (mean of 3 repeats; Δ = difference in AUC). Main set: L2 logistic regression "
              "(logreg_l2), elastic-net logistic regression (logreg_enet), linear SVM and random forest with none/class weight/SMOTE; "
              "extended set without resampling.", [1.3, 0.9, 0.5, 0.5, 0.6, 0.5, 0.55, 0.6, 0.7], 7.5)
    s2 = pd.read_csv(OUT / "table_s1_univariate_auc_apparent.csv")
    s2["apparent_univariate_auc"] = s2["apparent_univariate_auc"].map("{:.3f}".format)
    s2["group"] = s2["group"].map({"CB": "CBs", "DVH": "DVH", "RF": "CT RFs", "DF": "DFs"})
    s2.columns = ["Group", "Feature", "Apparent univariate AUC"]
    add_table(doc, s2, "Supplementary Table S2. Best three single features per group by apparent (non-cross-validated, "
              "full-data) univariate AUC, max(AUC, 1−AUC). Descriptive only; these values are optimistic and were not used for selection.",
              [1.0, 3.6, 1.6], 8.5)
    s3 = pd.read_csv(OUT / "table_s3_selected_features.csv")
    s3 = s3[s3.times_selected >= 3]
    s3["selection_proportion"] = s3["selection_proportion"].map("{:.2f}".format)
    s3.columns = ["Configuration", "Feature", "Times selected", "Outer fits", "Proportion"]
    add_table(doc, s3, "Supplementary Table S3. Features selected in at least 3 of 12 outer training fits (4 folds × 3 repeats) for "
              "selected configurations (primary pipeline).", [1.6, 2.7, 0.8, 0.7, 0.8], 8.5)
    para(doc, "Additional files in the repository: complete metric tables for all 15 configurations × 20 classifier–imbalance "
              "combinations, per-fold metrics, selected features and optimised hyper-parameters per fold, all Optuna trials, "
              "out-of-fold predictions (not public), and publication-quality figures (PNG and PDF).", size=10)

    out = PROJECT_ROOT / "Manuscript_Thyroid_Dosiomics_V2.docx"
    doc.save(out)
    print("saved", out, "| references:", len(ORDER))
    return out


if __name__ == "__main__":
    build()
