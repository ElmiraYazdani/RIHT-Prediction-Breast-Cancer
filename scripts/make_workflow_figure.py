"""Study workflow figure (Figure 1 of the manuscript) -> outputs/manuscript/figure1_workflow.(png|pdf)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from thyroid_rihp.config import PROJECT_ROOT

NAVY, TEAL, ORANGE, GREY, RED = "#1f4e79", "#2a9d8f", "#e76f51", "#6c757d", "#b5179e"
LIGHT = {"navy": "#dce8f5", "teal": "#d8f0ec", "orange": "#fbe3da", "grey": "#eceff1", "red": "#f6dcee"}

fig, ax = plt.subplots(figsize=(8.2, 10.6))
ax.set_xlim(0, 100)
ax.set_ylim(-13, 130)
ax.axis("off")


def box(x, y, w, h, title, body="", fc="navy", tsize=8.6, bsize=7.3, ec=None):
    col = {"navy": NAVY, "teal": TEAL, "orange": ORANGE, "grey": GREY, "red": RED}[fc]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=LIGHT[fc],
                                ec=ec or col, lw=1.4))
    ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=tsize, fontweight="bold", color=col)
    if body:
        ax.text(x + w / 2, y + h - 4.6, body, ha="center", va="top", fontsize=bsize, color="#212529", linespacing=1.28)


def arrow(x1, y1, x2, y2, color="#343a40", style="-|>", lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def band(y, h, label):
    ax.add_patch(plt.Rectangle((0.5, y), 3.2, h, fc="#343a40", ec="none"))
    ax.text(2.1, y + h / 2, label, rotation=90, ha="center", va="center", color="white", fontsize=8.2, fontweight="bold")


# ---------------- Stage A: cohort & data ----------------
band(98.5, 29, "A  Data")
box(6, 112, 90, 15.5, "Prospective breast-cancer radiotherapy cohort",
    "N = 115 patients analysed (27 hypothyroidism, 88 none)  |  planning CT + 3-D dose distribution\n"
    "Outcome: hypothyroidism at 1-year follow-up (serum TSH, before RT and 1 year after RT)\n"
    "Follow-up TSH defines the label only; it is never a predictor", "navy")
arrow(51, 112, 51, 108.5)
box(6, 98.5, 21.5, 10, "CBs (n = 5)", "age, diabetes,\nhypertension, smoking,\nthyroid volume", "grey", bsize=6.6)
box(29.2, 98.5, 21.5, 10, "DVH (n = 12)", "Dmean, Dmax,\nV5 ... V50\n(5-Gy steps)", "teal", bsize=6.6)
box(52.4, 98.5, 21.5, 10, "CT RFs (n = 107)", "thyroid ROI on planning CT\n3D Slicer, IBSI, 1 mm voxel,\nbin width 25", "orange", bsize=6.2)
box(75.6, 98.5, 20.4, 10, "DFs (n = 107)", "same 107 features,\ncomputed on the 3-D\ndose map in the ROI", "red", bsize=6.4)

# ---------------- Stage B: validation & design ----------------
band(82, 13.5, "B  Design")
arrow(51, 98.5, 51, 96.3)
box(6, 82, 43.5, 13.5, "Data validation (ID-based merge)",
    "identical IDs/labels across 4 files; no missing or\nnon-finite values; identifiers, names and any\npost-outcome column excluded by rule", "navy", bsize=6.9)
box(52.5, 82, 43.5, 13.5, "15 feature-group configurations",
    "all non-empty combinations of CBs, DVH,\nCT RFs and DFs (4 single, 6 pairs,\n4 triples, 1 quadruple)", "navy", bsize=6.9)
arrow(49.5, 88.7, 52.5, 88.7)

# ---------------- Stage C: nested CV ----------------
band(23.5, 55.5, "C  Nested CV")
arrow(51, 82, 51, 79.2)
box(6, 70, 90, 8.5, "OUTER loop: stratified, shuffled 4-fold CV x 3 repeats (identical splits for all configurations)",
    "", "navy", tsize=7.6)
ax.text(51, 72.4, "outer training set (~86 patients, 20-21 events)  ->  everything below is fitted here only",
        ha="center", va="center", fontsize=7.2, color="#212529")
arrow(30, 70, 30, 67.6)
arrow(74, 70, 74, 67.6, color=RED)

box(6, 25.0, 56, 42.0, "Training-fold pipeline (no test data)", "", "teal", tsize=8.4)
steps = [
    ("1  Pre-processing", "median imputation, zero-variance filter, z-score"),
    ("2  Feature selection (per group)", "volume filter |rho| > 0.5 (RF, DF) > redundancy |r| > 0.6\n> mRMR > stability selection (75 %, max 5 features)"),
    ("3  Inner 3-fold CV + Optuna (30 TPE trials)", "selection re-run inside each inner split;\nclassifier hyper-parameters tuned on inner AUC"),
    ("4  Class imbalance & classifier", "none | class weights | SMOTE (training rows only)\nL2-LR | elastic-net LR | linear SVM | random forest"),
    ("5  Operating threshold", "Youden cut-off from inner out-of-fold probabilities"),
    ("6  Refit on full outer-training set", "selected features + best hyper-parameters"),
]
heights = [5.0, 7.2, 7.2, 7.2, 5.0, 5.0]
y = 62.6
for (t, b), h in zip(steps, heights):
    ax.add_patch(FancyBboxPatch((8, y - h + 0.5), 52, h - 1.2, boxstyle="round,pad=0.1,rounding_size=0.6", fc="white",
                                ec=TEAL, lw=0.9))
    ax.text(9, y - 0.9, t, fontsize=7.0, fontweight="bold", va="top", color=TEAL)
    ax.text(9, y - 3.0, b, fontsize=6.3, va="top", color="#212529", linespacing=1.25)
    y -= h
box(66, 44, 30, 23, "Held-out outer test fold", "untouched until the end:\nnever used for feature\nselection, tuning, threshold\nor scaling\n\nstep 6 model predicts it once\n-> out-of-fold probabilities",
    "red", tsize=8.2, bsize=6.9)
arrow(62, 50, 66, 50, color=RED)
ax.text(6, 22.2, "Leakage guards: unit tests scramble test labels / features and confirm that selected features,\nhyper-parameters and thresholds are unchanged (see Methods 2.9).", ha="left", va="center", fontsize=6.0, style="italic", color=GREY)

# ---------------- Stage D: evaluation ----------------
arrow(79, 44, 79, 20.3, color=RED)
box(6, 9.5, 43.5, 10.5, "Pooled out-of-fold performance",
    "ROC-AUC, PR-AUC, sensitivity, specificity,\nprecision, F1, balanced accuracy, Brier,\ncalibration slope; mean over repeats", "navy", bsize=6.8)
box(52.5, 9.5, 43.5, 10.5, "Uncertainty & comparison",
    "patient-level bootstrap 95 % CI; paired\nbootstrap dAUC (DFs vs DVH ...);\nlabel-permutation null (sanity check)", "navy", bsize=6.8)
box(6, -8.5, 43.5, 14.5, "Secondary analyses",
    "imbalance strategies (none/weights/SMOTE);\n12 further classifiers; relaxed stability\nselection; calibration; decision curves", "grey", bsize=6.8)
box(52.5, -8.5, 43.5, 14.5, "Primary question",
    "Do dosiomic features improve prediction\nover DVH parameters, CT radiomics\nand clinical variables?\n(internal validation only)", "orange", bsize=6.8)
arrow(49.5, 14.7, 52.5, 14.7)
band(-10, 31, "D  Evaluation")

for ext in ("png", "pdf"):
    out = PROJECT_ROOT / "outputs" / "manuscript" / f"figure1_workflow.{ext}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("saved")
