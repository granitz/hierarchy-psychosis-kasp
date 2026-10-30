"""
Clinical and demographic associations.

Builds the demographic table and relates each participant's gradient-hierarchy
alignment to PANSS positive symptoms.

Reported in: Table 1 and Figure 4.

Original filename: Correlate-With-Behaviour.py

Note on inputs
--------------
The original script read the clinical tables interactively with
``pandas.read_clipboard()``. They are now loaded from a CSV through
``kasp.data.load_clinical()``; see the docstring there for the expected
columns and ``data/clinical/clinical_data_TEMPLATE.csv`` for a template.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kasp import config
from kasp.config import BASE_PATH as basepath

import numpy as np
import pandas as pd 
import scipy.io as sio
import pingouin as pg 
from scipy import stats as sts
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
plt.ion(); plt.rc("axes.spines", top=False, right=False); sns.set_palette("pastel")

# ---------------------------------------------------------------------------
# Helpers inherited from the interactive session.
# ---------------------------------------------------------------------------
from kasp import data as kdata
from kasp.parcellation import fc_gradient
from kasp.plotting import scatterplot2, set_style
from kasp.stats import permtest_pearsonr  # noqa: F401  (also defined below)

set_style()
fc_grad_parc, parcellation = fc_gradient()

basepath = config.BASE_PATH

#############################
### LOAD BEHAVIOURAL DATA ###
#############################

# ---------------------------------------------------------------------------
# Clinical and demographic table.
#
# The original script pasted seven tables in from a spreadsheet with
# pandas.read_clipboard() and merged them on an internal identifier. That is
# not reproducible and the identifier linked scans to clinical records, so the
# table is now read from a single de-identified CSV keyed on the scan id.
# See kasp.data.load_clinical() for the expected columns and
# data/clinical/clinical_data_TEMPLATE.csv for a template.
#
# Downstream code is unchanged and still expects DF / DF_SZ / DF_CTRL with
# Group in {"Control", "FEP"}.
# ---------------------------------------------------------------------------

clinical = kdata.load_clinical()

DF_SZ = clinical[clinical["Group"] != "Control"].copy()
DF_CTRL = clinical[clinical["Group"] == "Control"].copy()
DF_SZ["Group"] = "FEP"
DF_CTRL["Group"] = "Control"

DF = pd.concat([DF_SZ, DF_CTRL], axis=0)
DF.reset_index(drop=True, inplace=True)

# 0. Load your data
# df = pd.read_csv("your_data.csv")

df = DF.copy() 

# 1. Define groups
hc  = df[df["Group"] == "Control"].copy()
fep_unmed = df[(df["Group"] == "FEP") & (df["Is_Medicated"] == 0)].copy()
fep_med   = df[(df["Group"] == "FEP") & (df["Is_Medicated"] == 1)].copy()

groups = {"HC": hc, "FEP_unmed": fep_unmed, "FEP_med": fep_med}

n_hc  = len(hc)
n_um  = len(fep_unmed)
n_med = len(fep_med)

# 1b. Ensure numeric dtypes
numeric_cols = [
    "Age", "Is_Medicated", "Gender",
    "Symptoms", "Functioning",
    "Positive", "Negative", "General", "Total", "CGI_Score",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Re-define groups AFTER the conversion
hc        = df[df["Group"] == "Control"].copy()
fep_unmed = df[(df["Group"] == "FEP") & (df["Is_Medicated"] == 0)].copy()
fep_med   = df[(df["Group"] == "FEP") & (df["Is_Medicated"] == 1)].copy()

# 2. Helper functions
def mean_sd(series):
    """Return 'mean (sd)' string, rounded to 2 decimals."""
    return f"{series.mean():.2f} ({series.std():.2f})"

def gender_count(series):
    """Return 'n_male / n_female' string (0 = male, 1 = female)."""
    n_male   = int((series == 0).sum())
    n_female = int((series == 1).sum())
    return f"{n_male} / {n_female}"

EMPTY = ""        # placeholder for cells that don't apply
DASH  = "—"       # placeholder shown in the table for N/A cells

# 3. Build Table 1
rows = []

# Age
rows.append({
    "Characteristic": "Age, mean (SD)",
    f"Healthy Controls (n={n_hc})": mean_sd(hc["Age"]),
    f"FEP_unmed (n={n_um})":        mean_sd(fep_unmed["Age"]),
    f"FEP_med (n={n_med})":         mean_sd(fep_med["Age"]),
})

# Gender
rows.append({
    "Characteristic": "Gender (male / female)",
    f"Healthy Controls (n={n_hc})": gender_count(hc["Gender"]),
    f"FEP_unmed (n={n_um})":        gender_count(fep_unmed["Gender"]),
    f"FEP_med (n={n_med})":         gender_count(fep_med["Gender"]),
})

# PANSS header row (empty)
rows.append({
    "Characteristic": "PANSS, mean (SD)",
    f"Healthy Controls (n={n_hc})": EMPTY,
    f"FEP_unmed (n={n_um})":        EMPTY,
    f"FEP_med (n={n_med})":         EMPTY,
})

# PANSS sub-scales + Total
for label, col in [("  Positive", "Positive"),
                   ("  Negative", "Negative"),
                   ("  General",  "General"),
                   ("  Total",    "Total")]:
    rows.append({
        "Characteristic": label,
        f"Healthy Controls (n={n_hc})": DASH,
        f"FEP_unmed (n={n_um})":        mean_sd(fep_unmed[col]),
        f"FEP_med (n={n_med})":         mean_sd(fep_med[col]),
    })

# Functioning header row (empty)
rows.append({
    "Characteristic": "Functioning, mean (SD)",
    f"Healthy Controls (n={n_hc})": EMPTY,
    f"FEP_unmed (n={n_um})":        EMPTY,
    f"FEP_med (n={n_med})":         EMPTY,
})

# GAF Symptoms & Functioning
rows.append({
    "Characteristic": "  GAF Symptoms",
    f"Healthy Controls (n={n_hc})": DASH,
    f"FEP_unmed (n={n_um})":        mean_sd(fep_unmed["Symptoms"]),
    f"FEP_med (n={n_med})":         mean_sd(fep_med["Symptoms"]),
})
rows.append({
    "Characteristic": "  GAF Functioning",
    f"Healthy Controls (n={n_hc})": DASH,
    f"FEP_unmed (n={n_um})":        mean_sd(fep_unmed["Functioning"]),
    f"FEP_med (n={n_med})":         mean_sd(fep_med["Functioning"]),
})

# CGI
rows.append({
    "Characteristic": "CGI Score, mean (SD)",
    f"Healthy Controls (n={n_hc})": DASH,
    f"FEP_unmed (n={n_um})":        mean_sd(fep_unmed["CGI_Score"]),
    f"FEP_med (n={n_med})":         mean_sd(fep_med["CGI_Score"]),
})

table1 = pd.DataFrame(rows)
table1.index = range(1, len(table1) + 1)
table1.index.name = None

print("=" * 90)
print("TABLE 1 — Demographic and Clinical Characteristics")
print("=" * 90)
print(table1.to_string(index=False))
print()

# Optional: export to LaTeX / CSV
# table1.to_csv("table1.csv", index=False)
# print(table1.to_latex(index=False, caption="Demographic and Clinical Characteristics"))

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.table import WD_TABLE_ALIGNMENT

# Create Word document
doc = Document()
doc.add_heading("Table 1 — Demographic and Clinical Characteristics", level=2)

# Add table (rows = header + data rows, cols = number of DataFrame columns)
n_rows, n_cols = table1.shape
word_table = doc.add_table(rows=n_rows + 1, cols=n_cols, style="Table Grid")
word_table.alignment = WD_TABLE_ALIGNMENT.CENTER

# Write header row
for j, col_name in enumerate(table1.columns):
    cell = word_table.rows[0].cells[j]
    cell.text = col_name
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.bold = True
            run.font.size = Pt(10)

# Write data rows
for i in range(n_rows):
    for j in range(n_cols):
        cell = word_table.rows[i + 1].cells[j]
        cell.text = str(table1.iloc[i, j])
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(10)

# Save
output_path = "Table1_Demographics.docx"
doc.save(output_path)
print(f"Saved to {output_path}")


# 4. Statistical comparisons
print("=" * 90)
print("STATISTICAL COMPARISONS")
print("=" * 90)

# 4a. Age: three-group comparison (one-way ANOVA or Kruskal-Wallis)
# Normality check (Shapiro-Wilk per group)
age_normal = all(
    sts.shapiro(g["Age"].dropna()).pvalue > 0.05 for g in [hc, fep_unmed, fep_med]
)

if age_normal:
    stat_age, p_age = sts.f_oneway(
        hc["Age"].dropna(), fep_unmed["Age"].dropna(), fep_med["Age"].dropna()
    )
    test_name_age = "One-way ANOVA"
else:
    stat_age, p_age = sts.kruskal(
        hc["Age"].dropna(), fep_unmed["Age"].dropna(), fep_med["Age"].dropna()
    )
    test_name_age = "Kruskal–Wallis H"

print(f"\nAge across 3 groups — {test_name_age}: "
      f"statistic = {stat_age:.4f}, p = {p_age:.4f}")

# ---- 4b. Gender - Chi-square test across three groups ----------------------
gender_ct = pd.crosstab(
    pd.concat([
        hc["Gender"].map({0: "Male", 1: "Female"}),
        fep_unmed["Gender"].map({0: "Male", 1: "Female"}),
        fep_med["Gender"].map({0: "Male", 1: "Female"}),
    ]),
    pd.concat([
        pd.Series(["HC"] * n_hc, index=hc.index),
        pd.Series(["FEP_unmed"] * n_um, index=fep_unmed.index),
        pd.Series(["FEP_med"] * n_med, index=fep_med.index),
    ]),
)
chi2, p_gender, dof, _ = sts.chi2_contingency(gender_ct)
print(f"Gender across 3 groups — Chi-square: "
      f"χ² = {chi2:.4f}, df = {dof}, p = {p_gender:.4f}")

# ---- 4c. Patient-only comparisons (FEP_unmed vs FEP_med) -------------------
clinical_vars = {
    "Positive":    "Positive",
    "Negative":    "Negative",
    "General":     "General",
    "Total":       "Total",
    "GAF Symptoms":    "Symptoms",
    "GAF Functioning": "Functioning",
    "CGI Score":       "CGI_Score",
}

print(f"\n{'Variable':<20} {'Test':<25} {'Statistic':>10} {'p-value':>10}")
print("-" * 70)

for label, col in clinical_vars.items():
    a = fep_unmed[col].dropna()
    b = fep_med[col].dropna()

    # Choose test based on normality
    norm_a = sts.shapiro(a).pvalue > 0.05 if len(a) >= 3 else False
    norm_b = sts.shapiro(b).pvalue > 0.05 if len(b) >= 3 else False

    if norm_a and norm_b:
        stat_val, p_val = sts.ttest_ind(a, b, equal_var=False)  # Welch's t
        test_used = "Welch's t-test"
    else:
        stat_val, p_val = sts.mannwhitneyu(a, b, alternative="two-sided")
        test_used = "Mann–Whitney U"

    print(f"{label:<20} {test_used:<25} {stat_val:>10.4f} {p_val:>10.4f}")

# Also compare Age & Gender between the two patient groups
a_age = fep_unmed["Age"].dropna()
b_age = fep_med["Age"].dropna()
norm_a = sts.shapiro(a_age).pvalue > 0.05 if len(a_age) >= 3 else False
norm_b = sts.shapiro(b_age).pvalue > 0.05 if len(b_age) >= 3 else False
if norm_a and norm_b:
    stat_val, p_val = sts.ttest_ind(a_age, b_age, equal_var=False)
    test_used = "Welch's t-test"
else:
    stat_val, p_val = sts.mannwhitneyu(a_age, b_age, alternative="two-sided")
    test_used = "Mann–Whitney U"
print(f"{'Age':<20} {test_used:<25} {stat_val:>10.4f} {p_val:>10.4f}")

# Gender between patient groups
ct_pat = pd.crosstab(
    pd.concat([fep_unmed["Gender"], fep_med["Gender"]]),
    pd.concat([
        pd.Series(["FEP_unmed"] * n_um, index=fep_unmed.index),
        pd.Series(["FEP_med"] * n_med, index=fep_med.index),
    ]),
)
chi2_p, p_g_pat, dof_p, _ = sts.chi2_contingency(ct_pat)
print(f"{'Gender':<20} {'Chi-square':<25} {chi2_p:>10.4f} {p_g_pat:>10.4f}")



#####################################
### Table with statistics and all ###
#####################################

# Helper: format p-value
def fmt_p(p):
    """Journal-style p-value formatting."""
    if p < 0.001:
        return "<.001"
    else:
        return f"{p:.3f}"

# Helper: 3-group age test
def compare_age_3group(hc, fep_unmed, fep_med):
    a, b, c = hc["Age"].dropna(), fep_unmed["Age"].dropna(), fep_med["Age"].dropna()
    normality = all(sts.shapiro(g).pvalue > 0.05 for g in [a, b, c])
    if normality:
        stat, p = sts.f_oneway(a, b, c)
        test = "F"
    else:
        stat, p = sts.kruskal(a, b, c)
        test = "H"
    return f"{test}={stat:.2f}", fmt_p(p)

# Helper: 3-group gender test (chi-square)
def compare_gender_3group(hc, fep_unmed, fep_med):
    ct = pd.crosstab(
        pd.concat([hc["Gender"], fep_unmed["Gender"], fep_med["Gender"]]),
        pd.concat([
            pd.Series("HC", index=hc.index),
            pd.Series("FEP_unmed", index=fep_unmed.index),
            pd.Series("FEP_med", index=fep_med.index),
        ]),
    )
    chi2, p, dof, _ = sts.chi2_contingency(ct)
    return f"χ²={chi2:.2f}", fmt_p(p)

# Helper: 2-group patient comparison
def compare_patient_groups(fep_unmed, fep_med, col):
    a = fep_unmed[col].dropna()
    b = fep_med[col].dropna()
    norm = all(
        sts.shapiro(g).pvalue > 0.05 for g in [a, b] if len(g) >= 3
    )
    if norm:
        stat, p = sts.ttest_ind(a, b, equal_var=False)
        test = "t"
    else:
        stat, p = sts.mannwhitneyu(a, b, alternative="two-sided")
        test = "U"
    return f"{test}={stat:.2f}", fmt_p(p)


# Build Table 1 with statistics
col_hc  = f"Healthy Controls (n={n_hc})"
col_um  = f"FEP_unmed (n={n_um})"
col_med = f"FEP_med (n={n_med})"

rows = []

# Age (3-group comparison)
stat_str, p_str = compare_age_3group(hc, fep_unmed, fep_med)
rows.append({
    "Characteristic": "Age, mean (SD)",
    col_hc:  mean_sd(hc["Age"]),
    col_um:  mean_sd(fep_unmed["Age"]),
    col_med: mean_sd(fep_med["Age"]),
    "Statistic": stat_str,
    "p": p_str,
})

# Gender (3-group comparison)
stat_str, p_str = compare_gender_3group(hc, fep_unmed, fep_med)
rows.append({
    "Characteristic": "Gender (male / female)",
    col_hc:  gender_count(hc["Gender"]),
    col_um:  gender_count(fep_unmed["Gender"]),
    col_med: gender_count(fep_med["Gender"]),
    "Statistic": stat_str,
    "p": p_str,
})

# PANSS header
rows.append({
    "Characteristic": "PANSS, mean (SD)",
    col_hc: "", col_um: "", col_med: "",
    "Statistic": "", "p": "",
})

# PANSS sub-scales (2-group: FEP_unmed vs FEP_med)
for label, col in [("  Positive", "Positive"),
                   ("  Negative", "Negative"),
                   ("  General",  "General"),
                   ("  Total",    "Total")]:
    stat_str, p_str = compare_patient_groups(fep_unmed, fep_med, col)
    rows.append({
        "Characteristic": label,
        col_hc:  "—",
        col_um:  mean_sd(fep_unmed[col]),
        col_med: mean_sd(fep_med[col]),
        "Statistic": stat_str,
        "p": p_str,
    })

# Functioning header
rows.append({
    "Characteristic": "Functioning, mean (SD)",
    col_hc: "", col_um: "", col_med: "",
    "Statistic": "", "p": "",
})

# GAF & CGI (2-group)
for label, col in [("  GAF Symptoms",    "Symptoms"),
                   ("  GAF Functioning",  "Functioning"),
                   ("  CGI Score",        "CGI_Score")]:
    stat_str, p_str = compare_patient_groups(fep_unmed, fep_med, col)
    rows.append({
        "Characteristic": label,
        col_hc:  "—",
        col_um:  mean_sd(fep_unmed[col]),
        col_med: mean_sd(fep_med[col]),
        "Statistic": stat_str,
        "p": p_str,
    })

table1 = pd.DataFrame(rows)
print(table1.to_string(index=False))

# F = one-way ANOVA F-statistic; H = Kruskal-Wallis H; t = Welch's t-test; U = Mann-Whitney U; 
# χ² = Chi-square test. Age and Gender were compared across all three groups; 
# clinical measures were compared between FEP subgroups only. Significance threshold: p < .05.

### Save to doc: 
print(f"Saved to {output_path}")


#################################
### |corr(HL_i, fc_gradient)| ###
#################################
### Correlate hierarchical levels (individual level) with fc_gradient (cortical only) and then associate with behaviour, i.e., only FEP
CTRLHL = sio.loadmat(basepath + '/results/results_subject_refinement/no_grid_subjectRefinement_group-ctrl_tau-1_med-0.mat')
SZ0HL = sio.loadmat(basepath + '/results/results_subject_refinement/no_grid_subjectRefinement_group-sz_tau-1_med-0.mat')
SZ1HL = sio.loadmat(basepath + '/results/results_subject_refinement/no_grid_subjectRefinement_group-sz_tau-1_med-1.mat')

SZ0HL.keys() #  'hierarchicallevels_full', 'trophiccoherence'])

SZ0HL_ = SZ0HL['hierarchicallevels_full'] # shape (n_subjects, n_regions)
SZ1HL_ = SZ1HL['hierarchicallevels_full'] # shape (n_subjects, n_regions)
CTRLHL_ = CTRLHL['hierarchicallevels_full'] # shape (n_subjects, n_regions)

# Contains NaNs
# use only indexN
IDX = (~np.isnan(SZ0HL_[0]))[16:] # cortical regions only

grad_parc = fc_grad_parc[IDX]

CORR0 = []
P0 = []
for x in range(SZ0HL_.shape[0]):
    h = SZ0HL_[x][16:][IDX]
    corr_sz0 = sts.pearsonr(h, grad_parc) 
    CORR0.append(corr_sz0[0])
    P0.append(corr_sz0[1])

CORR0 = np.array(CORR0)
P0 = -np.log10(np.array(P0))

CORR1 = []
P1 = []
for x in range(SZ1HL_.shape[0]):
    h = SZ1HL_[x][16:][IDX]
    corr_sz1 = sts.pearsonr(h, grad_parc)
    CORR1.append(corr_sz1[0])
    P1.append(corr_sz1[1])

CORR1 = np.array(CORR1)
P1 = -np.log10(np.array(P1))

CORRC = []
PC = []
for x in range(CTRLHL_.shape[0]):
    h = CTRLHL_[x][16:][IDX]
    corr_ctrl = sts.pearsonr(h, grad_parc)
    CORRC.append(corr_ctrl[0])
    PC.append(corr_ctrl[1])

CORRC = np.array(CORRC)
PC = -np.log10(np.array(PC))

plt.boxplot([abs(CORR0), abs(CORR1),abs(CORRC)],labels=['FEP Unmed', 'FEP Med', 'Control'])
plt.title('Correlation between hierarchical levels and FC gradient (cortical only)')
plt.ylabel('Absolute Pearson r')
plt.show()

#######################################
### Statistical test between groups ###
#######################################

### Non-parametric ttest:
u_stat_0c, p_val_0c = sts.mannwhitneyu(abs(CORR0), abs(CORRC), alternative='two-sided')
u_stat_1c, p_val_1c = sts.mannwhitneyu(abs(CORR1), abs(CORRC), alternative='two-sided')
u_stat_01, p_val_01 = sts.mannwhitneyu(abs(CORR0), abs(CORR1), alternative='two-sided')
print(f"FEP Unmed vs Control: U = {u_stat_0c:.4f}, p = {p_val_0c:.4f}")
print(f"FEP Med vs Control: U = {u_stat_1c:.4f}, p = {p_val_1c:.4f}")
print(f"FEP Unmed vs FEP Med: U = {u_stat_01:.4f}, p = {p_val_01:.4f}")

### Correlate with behaviour. 
SZ0SUB = SZ0HL['subjects'][:,0] 
SZ0SUB = np.array([x[0] for x in SZ0SUB]) 
SZ1SUB = SZ1HL['subjects'][:,0] 
SZ1SUB = np.array([x[0] for x in SZ1SUB]) 

DF_SZ0 = DF.query('ID1 in @SZ0SUB')
DF_SZ1 = DF.query('ID1 in @SZ1SUB')

# Is the DATAFRAME ID1 sorted in the same order as SZ0SUB and SZ1SUB?  
DF_SZ0 = DF_SZ0.set_index('ID1').loc[SZ0SUB].reset_index()
DF_SZ1 = DF_SZ1.set_index('ID1').loc[SZ1SUB].reset_index()

# integrate with CORR. 
DF_SZ0['HL_FC_Corr'] = CORR0
DF_SZ1['HL_FC_Corr'] = CORR1

DF_SZ0['absHFC'] = np.abs(CORR0)
DF_SZ1['absHFC'] = np.abs(CORR1)

# Drop the row with NaN in DF_SZ0
ISNA = ~DF_SZ0.Symptoms.isna()
DF_SZ0 = DF_SZ0[ISNA]

### Now concatenate the two dataframes and correlate with behaviour
D = pd.concat([DF_SZ0, DF_SZ1], axis=0)
D.reset_index(drop=True, inplace=True)
D = D[~D.Symptoms.isna()]

###############################################################
# Correlation between abs(HL-FC) corr and Positive symptoms ###
###############################################################
# permutation test 
stat = permtest_pearsonr(D['absHFC'], D['Positive'], n_perm=10000, seed=42)
print(f"Correlation between abs(HL-FC) corr and Positive symptoms:")
print(f"FEP: Pearson r = {stat[0]:.4f}, p = {stat[1]:.4f}")

#
# Correlation between abs(HL-FC) corr and Positive:
# FEP: Pearson r = -0.2736, p = 0.0450
#

# Quick figure:
plt.figure(figsize=(8,6))
sns.regplot(data=D,x='Positive',y='absHFC')
plt.rcParams['font.family'] = 'Helvetica Neue'
plt.rcParams['font.size'] = 12
plt.ylabel('|corr(HL, PC1)|')
plt.xlabel('PANSS Positive symptoms')
plt.title('Correlation between hierarchical levels-FC correlation and Positive symptoms')

# Publication ready 

# -----------------------------------------------------------------------------
# Global style settings (set BEFORE creating the figure)
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.rcParams.update({
    'font.family'      : 'sans-serif',
    'font.sans-serif'  : ['Helvetica Neue', 'Helvetica', 'Arial'],
    'font.size'        : 12,
    'axes.titlesize'   : 12,
    'axes.labelsize'   : 13,
    'xtick.labelsize'  : 11,
    'ytick.labelsize'  : 11,
    'axes.linewidth'   : 1.0,
    'pdf.fonttype'     : 42,
    'ps.fonttype'      : 42,
    'svg.fonttype'     : 'none',
})

fig, ax = plt.subplots(figsize=(4.5, 4.0))

sns.regplot(
    data=D,
    x='Positive',
    y='absHFC',
    ax=ax,
    ci=95,
    scatter_kws={'s': 35, 'color': '#4C72B0',
                 'edgecolor': 'white', 'linewidths': 0.5, 'alpha': 0.85},
    line_kws={'color': '#C44E52', 'linewidth': 2.0},
)

ax.set_xlabel('PANSS positive symptom score')
ax.set_ylabel('|corr(hierarchical level, PC1)|')

stats_text = r'$r = -0.27$' + '\n' + r'$p_{perm} = 0.045$' + '\n' + r'$n = 53$'
ax.text(0.97, 0.97, stats_text,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.35',
                  facecolor='white', edgecolor='0.7', alpha=0.9))

sns.despine(ax=ax, trim=False)
plt.tight_layout()

config.ensure_output_dirs()
fig.savefig(config.FIGURE_DIR / 'figure_4.pdf', dpi=300, bbox_inches='tight')
fig.savefig(config.FIGURE_DIR / 'figure_4.tiff', dpi=600, bbox_inches='tight')
plt.show()


##########################
### From netneurotools ###
##########################
import scipy.special as sspecial
from sklearn.utils.validation import check_random_state

def _chk2_asarray(a, b, axis):
    # Removed from scipy in https://github.com/scipy/scipy/pull/23088
    if axis is None:
        a = np.ravel(a)
        b = np.ravel(b)
        outaxis = 0
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)
    if b.ndim == 0:
        b = np.atleast_1d(b)

    return a, b, outaxis

def efficient_pearsonr(a, b, ddof=1, nan_policy="propagate"):
    """
    Compute correlation of matching columns in `a` and `b`.

    Parameters
    ----------
    a,b : array_like
        Sample observations. These arrays must have the same length and either
        an equivalent number of columns or be broadcastable
    ddof : int, optional
        Degrees of freedom correction in the calculation of the standard
        deviation. Default: 1
    nan_policy : bool, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default: 'propagate'

    Returns
    -------
    corr : float or numpy.ndarray
        Pearson's correlation coefficient between matching columns of inputs
    pval : float or numpy.ndarray
        Two-tailed p-values

    Notes
    -----
    If either input contains nan and nan_policy is set to 'omit', both arrays
    will be masked to omit the nan entries.

    Examples
    --------
    >>> from netneurotools import stats as nnstats

    Generate some not-very-correlated and some highly-correlated data:

    >>> np.random.seed(12345678)  # set random seed for reproducible results
    >>> x1, y1 = nnstats.make_correlated_xy(corr=0.1, size=100)
    >>> x2, y2 = nnstats.make_correlated_xy(corr=0.8, size=100)

    Calculate both correlations simultaneously:

    >>> nnstats.efficient_pearsonr(np.c_[x1, x2], np.c_[y1, y2])
    (array([0.10032565, 0.79961189]), array([3.20636135e-01, 1.97429944e-23]))
    """
    a, b, _ = _chk2_asarray(a, b, 0)
    if len(a) != len(b):
        raise ValueError("Provided arrays do not have same length")

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan

    if nan_policy not in ("propagate", "raise", "omit"):
        raise ValueError(f'Value for nan_policy "{nan_policy}" not allowed')

    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    if a.shape[1] != b.shape[1]:
        a, b = np.broadcast_arrays(a, b)

    mask = np.logical_or(np.isnan(a), np.isnan(b))
    if nan_policy == "raise" and np.any(mask):
        raise ValueError('Input cannot contain NaN when nan_policy is "omit"')
    elif nan_policy == "omit":
        # avoid making copies of the data, if possible
        a = np.ma.masked_array(a, mask, copy=False, fill_value=np.nan)
        b = np.ma.masked_array(b, mask, copy=False, fill_value=np.nan)

    with np.errstate(invalid="ignore"):
        corr = sts.zscore(a, ddof=ddof, nan_policy=nan_policy) * sts.zscore(
            b, ddof=ddof, nan_policy=nan_policy
        )

    sumfunc, n_obs = np.sum, len(a)
    if nan_policy == "omit":
        corr = corr.filled(np.nan)
        sumfunc = np.nansum
        n_obs = np.squeeze(np.sum(np.logical_not(np.isnan(corr)), axis=0))

    corr = sumfunc(corr, axis=0) / (n_obs - 1)
    corr = np.squeeze(np.clip(corr, -1, 1)) / 1

    # taken from scipy.stats
    ab = (n_obs / 2) - 1
    prob = 2 * sspecial.betainc(ab, ab, 0.5 * (1 - np.abs(corr)))

    return corr, prob

def permtest_pearsonr(a, b, axis=0, n_perm=1000, resamples=None, seed=0):
    """
    Non-parametric equivalent of :py:func:`scipy.stats.pearsonr`.

    Generates two-tailed p-value for hypothesis of whether samples `a` and `b`
    are correlated using permutation tests

    Parameters
    ----------
    a,b : (N[, M]) array_like
        Sample observations. These arrays must have the same length and either
        an equivalent number of columns or be broadcastable
    axis : int or None, optional
        Axis along which to compute test. If None, compute over whole arrays
        of `a` and `b`. Default: 0
    n_perm : int, optional
        Number of permutations to assess. Unless `a` and `b` are very small
        along `axis` this will approximate a randomization test via Monte
        Carlo simulations. Default: 1000
    resamples : (N, P) array_like, optional
        Resampling array used to shuffle `a` when generating null distribution
        of correlations. This array must have the same length as `a` and `b`
        and should have at least the same number of columns as `n_perm` (if it
        has more then only `n_perm` columns will be used. When not specified a
        standard permutation is used to shuffle `a`. Default: None
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Set to None for "randomness".
        Default: 0

    Returns
    -------
    corr : float or numpyndarray
        Correlations
    pvalue : float or numpy.ndarray
        Non-parametric p-value

    Notes
    -----
    The lowest p-value that can be returned by this function is equal to 1 /
    (`n_perm` + 1).

    Examples
    --------
    >>> from netneurotools import stats as nnstats

    >>> np.random.seed(12345678)  # set random seed for reproducible results
    >>> x, y = nnstats.make_correlated_xy(corr=0.1, size=100)
    >>> nnstats.permtest_pearsonr(x, y)  # doctest: +SKIP
    (0.10032564626876286, 0.3046953046953047)

    >>> x, y = nnstats.make_correlated_xy(corr=0.5, size=100)
    >>> nnstats.permtest_pearsonr(x, y)  # doctest: +SKIP
    (0.500040365781984, 0.000999000999000999)

    Also works with multiple columns by either broadcasting the smaller array
    to the larger:

    >>> z = x + np.random.normal(loc=1, size=100)
    >>> nnstats.permtest_pearsonr(x, np.column_stack([y, z]))
    (array([0.50004037, 0.25843187]), array([0.000999  , 0.01098901]))

    or by using matching columns in the two arrays (e.g., `x` and `y` vs
    `a` and `b`):

    >>> a, b = nnstats.make_correlated_xy(corr=0.9, size=100)
    >>> nnstats.permtest_pearsonr(np.column_stack([x, a]), np.column_stack([y, b]))
    (array([0.50004037, 0.89927523]), array([0.000999, 0.000999]))
    """  # noqa
    a, b, axis = _chk2_asarray(a, b, axis)
    rs = check_random_state(seed)

    if len(a) != len(b):
        raise ValueError('Provided arrays do not have same length')

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan

    if resamples is not None:
        if n_perm > resamples.shape[-1]:
            raise ValueError('Number of permutations requested exceeds size '
                             'of resampling array.')

    # divide by one forces coercion to float if ndim = 0
    true_corr = efficient_pearsonr(a, b)[0] / 1
    abs_true = np.abs(true_corr)

    permutations = np.ones(true_corr.shape)
    for perm in range(n_perm):
        # permute `a` and determine whether correlations exceed original
        if resamples is None:
            ap = a[rs.permutation(len(a))]
        else:
            ap = a[resamples[:, perm]]
        permutations += np.abs(efficient_pearsonr(ap, b)[0]) >= abs_true

    pvals = permutations / (n_perm + 1)  # + 1 in denom accounts for true_corr

    return true_corr, pvals