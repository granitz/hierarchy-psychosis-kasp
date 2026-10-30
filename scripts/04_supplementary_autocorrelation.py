"""
Supplementary spatial and temporal autocorrelation control analyses.

Compares spatial autocorrelation (SA-lambda, SA-infinity) and lag-1 temporal
autocorrelation (TA-delta-1) across groups, and tests whether the group
differences in these simpler measures track the functional gradient in the
same way the hierarchical levels do.

Reported in: Supplementary Figure 2 and the corresponding Results paragraph.

The SA-lambda / SA-infinity estimators follow Shinn et al. (2023),
Nat Neurosci 26:867-878.

Original filename: Supplementary_spatial_temporal_autocorrelation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kasp import config
from kasp.config import BASE_PATH as basepath

"""
Spatial and temporal autocorrelation analysis across participant groups.

Compares spatial autocorrelation (SA-λ, SA-∞) globally across groups (Control,
Unmedicated SZ, Medicated SZ) — no significant group differences found.

Then compares temporal autocorrelation (TA-Δ₁) across groups to examine whether
group differences relate to the FC gradient and follow a similar hierarchical pattern.

SA-∞: average correlation between distant brain regions (fitted exponential asymptote).
SA-λ: rate of spatial correlation decay with distance.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas
import json 
import scipy.io as sio
import scipy.spatial
import scipy.optimize
import scipy.stats as sts
import matplotlib.pyplot as plt
import pingouin as pg
from neuromaps.datasets import fetch_annotation
from neuromaps.images import dlabel_to_gifti
from neuromaps import nulls, parcellate, stats
from netneurotools import datasets as nntdata

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("RedWhiteBlue", ["blue", "white", "red"])
cmap = LinearSegmentedColormap.from_list("Gray_Interp", ["black", "white"])


# ──────────────────────────────────────────────────────────────────────────────
# Function definitions
# ──────────────────────────────────────────────────────────────────────────────

def spatial_autocorrelation(cm, dist, discretization=1):
    """Calculate the SA-λ and SA-∞ measures of spatial autocorrelation, defined in [Shinn et al (2022)](https://www.biorxiv.org/content/10.1101/2021.06.01.446561v1)

    Args:
      cm (NxN numpy array): NxN correlation matrix of timeseries, where N is the number of
          timeseries
      dist (NxN numpy array): the NxN distance matrix, representing the spatial distance
          between location of each of the timeseries.  This should usually be the
          output of the `distance_matrix_euclidean` function.
      discretization (int): The size of the bins to use when computing the SA parameters.
          The size of the discretization should ensure that there are a sufficient number of
          observations in each bin, but also enough total bins to make a meaningful estimation.
          Try increasing it or decreasing it according to the scale of your data.  Data that has values
          up to around 100 should be fine with the default.  Decrease or increase as necessary
          to get an appropriate estimation.

    Returns:
      tuple of (SA-λ, SA-∞)
    """
    cm_flat = cm.flatten()
    dist_flat = dist.flatten()
    df = pandas.DataFrame(np.asarray([dist_flat, cm_flat]).T, columns=["dist", "corr"])
    df['dist_bin'] = np.round(df['dist']/discretization)*discretization
    df_binned = df.groupby('dist_bin').mean().reset_index().sort_values('dist_bin')
    binned_dist_flat = df_binned['dist_bin']
    binned_cm_flat = df_binned['corr']
    binned_dist_flat[0] = 1
    spatialfunc = lambda v : np.exp(-binned_dist_flat/v[0])*(1-v[1])+v[1]
    with np.errstate(all='warn'):
        res = scipy.optimize.minimize(lambda v : np.sum((binned_cm_flat-spatialfunc(v))**2), [10, .3], bounds=[(.1, 100), (-1, 1)])
    return (res.x[0], res.x[1])


def temporal_autocorrelation(x):
    """Compute TA-Δ₁ (lag-1 temporal autocorrelation) from the timeseries.

    Args:
      x: the timeseries of which to compute TA-Δ₁

        If `x` is a single list or one-dimensional numpy array, return the
        TA-Δ₁ estimate.  If `x` contains nested lists or is a NxT numpy, return
        a numpy array of length N, giving the TA-Δ₁ of each row of x.

    Returns:
      list of floats: The temporal autocorrelation of each timeseries in
          `x` is multidimensional, return nested lists in the same shape as the
           leading dimensions of `x`.

    Note:
        This is the biased estimator, but it is the default in numpy.  This
        is what we use throughout the manuscript.  The purpose of this function
        is to standardize computing TA-Δ₁.

    """
    if isinstance(x[0], (list, np.ndarray)):
        return np.asarray([temporal_autocorrelation(xe) for xe in x])
    return np.corrcoef(x[0:-1], x[1:])[0,1]


def distance_matrix_euclidean(distances):
    """Returns a Euclidean distance matrix.

    `distances` should be a Nx3 numpy matrix, providing the xyz coordinates for
    N brain regions.

    """
    return scipy.spatial.distance.cdist(distances, distances)


def compute_sa_per_group(ceff_array, dist_matrix, discretization=1):
    """Compute SA-λ and SA-∞ for each subject's Ceff in a group."""
    sa_inf_list, sa_lambda_list = [], []
    for ceff in ceff_array:
        sa_lam, sa_inf = spatial_autocorrelation(ceff, dist_matrix, discretization)
        sa_inf_list.append(sa_inf)
        sa_lambda_list.append(sa_lam)
    return np.asarray(sa_lambda_list), np.asarray(sa_inf_list)


def load_timeseries(subjects, tspath, suffix):
    """Load timeseries .mat files for a list of subjects."""
    ts_list = []
    for s in subjects:
        ts = sio.loadmat(f'{tspath}{s}{suffix}')
        ts_list.append(ts['ts'])
    return np.asarray(ts_list)


def extract_subjects(mat_data):
    """Extract subject IDs from a loaded .mat struct."""
    subjects = mat_data['subjects'][:, 0]
    return np.array([s[0] for s in subjects])


def plot_sa_inf_trophic(sa_inf, mat_data, group_label):
    """Scatter SA-∞ vs trophic coherence for one group."""
    dc = mat_data['trophiccoherence'][:, 0]
    print(pg.corr(sa_inf, dc, method='spearman'))
    plt.figure(figsize=(10, 5))
    plt.scatter(sa_inf, dc)
    plt.xlabel('SA-∞')
    plt.ylabel('Trophic Coherence')
    plt.title(group_label)


# ──────────────────────────────────────────────────────────────────────────────
# Constants & paths
# ──────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Helpers inherited from the interactive session: the scatter helper, the
# per-parcel colours, the parcellated gradient and the parcel distance matrix.
# ---------------------------------------------------------------------------
import pandas as pd  # the module is imported as `pandas` above but used as `pd`

from kasp.parcellation import (
    fc_gradient,
    network_assignment,
    parcel_distance_matrix,
    valid_cortical_index,
)
from kasp.plotting import network_colors, scatterplot2, set_style

set_style()
com = network_assignment()
colors = network_colors(com)
fc_grad_parc, parcellation = fc_gradient()
fc_grad_parc2, _ = fc_gradient(desc='fcgradient02')  # second gradient
dist2 = parcel_distance_matrix()   # requires data/atlas/parcel_centroids.csv

RESULTS_DIR = str(config.RESULTS_DIR) + '/'
TS_DIR = str(config.TS_DIR) + '/'
TS_SUFFIX = '_task-rest_run-1_feature-corrMatrixICA24Motion_atlas-SchaeferTian416_timeseries.mat'

# Valid (non-NaN) parcel indices. These were previously hard-coded here as a
# literal array; they are now derived from the data itself, immediately after
# the model results are loaded, so that the mask can never drift out of sync
# with the parcels the model actually fitted.

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────────────────────────

# coords = ['']
# dist = distance_matrix_euclidean(coords)

Control = sio.loadmat(f'{RESULTS_DIR}no_grid_subjectRefinement_group-ctrl_tau-1_med-0.mat')
Unmed   = sio.loadmat(f'{RESULTS_DIR}no_grid_subjectRefinement_group-sz_tau-1_med-0.mat')
Med     = sio.loadmat(f'{RESULTS_DIR}no_grid_subjectRefinement_group-sz_tau-1_med-1.mat')

# Parcels the model could be fitted in: everything not excluded for low coverage.
# Identical across the three groups (verified), so Control defines the mask.
IDX = np.flatnonzero(~np.isnan(np.nanmean(Control['hierarchicallevels_full'], 0)))

# Cortex-only boolean version of the same mask, for plotting.
VIDX = valid_cortical_index(Control)

Control_full = Control['Ceff_Subj']
Unmed_full   = Unmed['Ceff_Subj']
Med_full     = Med['Ceff_Subj']

# ──────────────────────────────────────────────────────────────────────────────
# 2. Spatial autocorrelation — compute per group
# ──────────────────────────────────────────────────────────────────────────────

SA_lambda_control, SA_inf_control = compute_sa_per_group(Control_full, dist2)
SA_lambda_unmed,   SA_inf_unmed   = compute_sa_per_group(Unmed_full, dist2)
SA_lambda_med,     SA_inf_med     = compute_sa_per_group(Med_full, dist2)

plt.figure(figsize=(10, 5))
plt.boxplot((SA_inf_control, SA_inf_unmed, SA_inf_med), labels=['Control', 'Unmed', 'Med'])
plt.title('SA-∞')
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 3. Spatial autocorrelation — group comparisons (Mann–Whitney U)
# ──────────────────────────────────────────────────────────────────────────────

print(pg.mwu(SA_inf_control, SA_inf_unmed))
print(pg.mwu(SA_inf_control, SA_inf_med))
print(pg.mwu(SA_inf_unmed, SA_inf_med))

# With MAD outlier removal
keep_control = ~pg.madmedianrule(SA_inf_control)
keep_unmed   = ~pg.madmedianrule(SA_inf_unmed)
keep_med     = ~pg.madmedianrule(SA_inf_med)

print(pg.mwu(SA_inf_control[keep_control], SA_inf_unmed[keep_unmed]))
print(pg.mwu(SA_inf_control[keep_control], SA_inf_med[keep_med]))
print(pg.mwu(SA_inf_unmed[keep_unmed], SA_inf_med[keep_med]))

plt.figure(figsize=(10, 5))
plt.boxplot((SA_inf_control[keep_control], SA_inf_unmed[keep_unmed], SA_inf_med[keep_med]),
            labels=['Control', 'Unmed', 'Med'])
plt.title('SA-∞ (MAD outliers removed)')
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 4. SA-∞ vs trophic coherence
# ──────────────────────────────────────────────────────────────────────────────

plot_sa_inf_trophic(SA_inf_control, Control, 'Control')
plot_sa_inf_trophic(SA_inf_unmed, Unmed, 'Unmed')
plot_sa_inf_trophic(SA_inf_med, Med, 'Med')

# ──────────────────────────────────────────────────────────────────────────────
# 5. Temporal autocorrelation — load timeseries & compute TA-Δ₁
# ──────────────────────────────────────────────────────────────────────────────

Control_subjects = extract_subjects(Control)
Unmed_subjects   = extract_subjects(Unmed)
Med_subjects     = extract_subjects(Med)

TS_control = load_timeseries(Control_subjects, TS_DIR, TS_SUFFIX)
TS_unmed   = load_timeseries(Unmed_subjects, TS_DIR, TS_SUFFIX)
TS_med     = load_timeseries(Med_subjects, TS_DIR, TS_SUFFIX)

TA_control = temporal_autocorrelation(TS_control)
TA_unmed   = temporal_autocorrelation(TS_unmed)
TA_med     = temporal_autocorrelation(TS_med)

# Zero out ROIs not in IDX
for ta in [TA_control, TA_unmed, TA_med]:
    ta[:, ~np.isin(np.arange(ta.shape[1]), IDX)] = 0

TA_control_mean = TA_control.mean(0)
TA_unmed_mean   = TA_unmed.mean(0)
TA_med_mean     = TA_med.mean(0)

# ──────────────────────────────────────────────────────────────────────────────
# 6. TA-Δ₁ group differences vs FC gradient
# ──────────────────────────────────────────────────────────────────────────────

# Cortical parcels in the 400-parcel gradient space.
#
# The y-values below are indexed as X[IDX][16:] — the valid parcels, subcortex
# dropped — so the x-values must be the *same* parcels expressed in 400-parcel
# space. That is simply the cortical part of IDX shifted down by the 16
# subcortical parcels.
#
# The original script instead built this from a hand-typed exclusion list:
#
#     Z = np.arange(0, 400)
#     IgnoreNodes = np.array([28, 31, 85, 130, ...])   # 45 entries
#     IDX2 = np.setdiff1d(Z, IgnoreNodes - 16)
#
# Every entry of that list is exactly +1 relative to the parcels actually
# excluded in the data, i.e. it was written in MATLAB's 1-based indexing and
# used as 0-based Python indices. The result masked the wrong parcels, so the
# x and y of these scatterplots referred to different parcels at 16 of the 355
# positions. The statistics were unaffected — they never used IDX2 — but the
# plotted points were.
Z = np.arange(0, 400)
IDX2 = IDX[IDX >= 16] - 16

scatterplot2(fc_grad_parc[IDX2], (TA_unmed_mean - TA_control_mean)[IDX][16:],
             xlabel='FC Gradient', ylabel='TA-Δ₁ Unmed - Control', regpl=True, cmap=cmap)
scatterplot2(fc_grad_parc[IDX2], (TA_med_mean - TA_control_mean)[IDX][16:],
             xlabel='FC Gradient', ylabel='TA-Δ₁ Med - Control', regpl=True, cmap=cmap)
scatterplot2(fc_grad_parc[IDX2], (TA_unmed_mean - TA_med_mean)[IDX][16:],
             xlabel='FC Gradient', ylabel='TA-Δ₁ Unmed - Med', regpl=True, cmap=cmap)

# ──────────────────────────────────────────────────────────────────────────────
# 7. Spin-test null comparisons (neuromaps)
# ──────────────────────────────────────────────────────────────────────────────

mapix = fetch_annotation(source='margulies2016', desc='fcgradient01', space='fsLR', den='32k')

schaefer = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
parcellation = dlabel_to_gifti(schaefer)
schaefer = parcellate.Parcellater(parcellation, 'fsLR').fit()

grad_parc = schaefer.transform(mapix, 'fsLR')

rotated = nulls.alexander_bloch(grad_parc, atlas='fsLR', density='32k',
                                n_perm=1000, seed=1234, parcellation=parcellation)

for label, ta_diff in [('Unmed vs Control', TA_unmed_mean - TA_control_mean),
                        ('Med vs Control',   TA_med_mean - TA_control_mean),
                        ('Unmed vs Med',     TA_unmed_mean - TA_med_mean)]:
    corr, pval = stats.compare_images(grad_parc, ta_diff[16:], nulls=rotated)
    print(f'{label}: r = {corr:.3f}, p = {pval:.3f}')


# ──────────────────────────────────────────────────────────────────────────────
# 8. Compare hierarchical levels vs TA-Δ₁
# ──────────────────────────────────────────────────────────────────────────────

### control 
h = Control['hierarchicallevels_full'] # nsub x nodes. bad regions contains nans. 
corr = []
pval = []
for sub in range(h.shape[0]):
    r = pg.corr(h[sub, :], TA_control[sub, :], method='spearman')
    corr.append(r.r.values[0])
    pval.append(r['p-val'].values[0])

corr = np.asarray(corr)
pval = -np.log10(np.asarray(pval))

h_mean = np.nanmean(h, 0)
scatterplot2(TA_control.mean(0), h_mean, xlabel='TA-Δ₁', ylabel='Hierarchical Level', regpl=True)
plt.title('Control')
print(pg.corr(TA_control.mean(0), h_mean, method='spearman'))

rotated = nulls.alexander_bloch(TA_control.mean(0), atlas='fsLR', density='32k',
                                n_perm=1000, seed=1234, parcellation=parcellation)

corr, pval = stats.compare_images(TA_control.mean(0)[16:], h_mean[16:], nulls=rotated)
print(f'Control: r = {corr:.3f}, p = {pval:.3f}')

### unmed 
h = Unmed['hierarchicallevels_full'] # nsub x nodes. bad regions contains nans.
corr = []
pval = []
for sub in range(h.shape[0]):
    r = pg.corr(h[sub, :], TA_unmed[sub, :], method='spearman')
    corr.append(r.r.values[0])
    pval.append(r['p-val'].values[0])
corr = np.asarray(corr)
pval = -np.log10(np.asarray(pval))

h_mean = np.nanmean(h, 0)
scatterplot2(TA_unmed.mean(0), h_mean, xlabel='TA-Δ₁', ylabel='Hierarchical Level', regpl=True)
plt.title('Unmed')
print(pg.corr(TA_unmed.mean(0), h_mean, method='spearman'))

rotated = nulls.alexander_bloch(TA_unmed.mean(0), atlas='fsLR', density='32k',
                                n_perm=1000, seed=1234, parcellation=parcellation)

corr, pval = stats.compare_images(TA_unmed.mean(0)[16:], h_mean[16:], nulls=rotated)
print(f'Unmed: r = {corr:.3f}, p = {pval:.3f}')

### med
h = Med['hierarchicallevels_full'] # nsub x nodes. bad regions contains nans.
corr = []
pval = []
for sub in range(h.shape[0]):
    r = pg.corr(h[sub, :], TA_med[sub, :], method='spearman')
    corr.append(r.r.values[0])
    pval.append(r['p-val'].values[0])
corr = np.asarray(corr)
pval = -np.log10(np.asarray(pval))

h_mean = np.nanmean(h, 0)
scatterplot2(TA_med.mean(0), h_mean, xlabel='TA-Δ₁', ylabel='Hierarchical Level', regpl=True)
plt.title('Med')
print(pg.corr(TA_med.mean(0), h_mean, method='spearman'))   

rotated = nulls.alexander_bloch(TA_med.mean(0), atlas='fsLR', density='32k',
                                n_perm=1000, seed=1234, parcellation=parcellation)

corr, pval = stats.compare_images(TA_med.mean(0)[16:], h_mean[16:], nulls=rotated)
print(f'Med: r = {corr:.3f}, p = {pval:.3f}')

# ──────────────────────────────────────────────────────────────────────────────
# 9. Compare Mean TSNR
# ──────────────────────────────────────────────────────────────────────────────
# For each subject and group get regional 'MeanAtlasTSNR'. Compare to hierarchical levels.

tspath = str(config.TS_DIR) + '/'

Control_tsnr = []
for s in Control_subjects:
    tsnr = json.load(open(f'{tspath}{s}_task-rest_run-1_feature-corrMatrixICA24Motion_atlas-SchaeferTian416_timeseries.json'))['MeanAtlasTSNR']
    Control_tsnr.append(tsnr)
Control_tsnr = np.asarray(Control_tsnr)

Unmed_tsnr = []
for s in Unmed_subjects:
    tsnr = json.load(open(f'{tspath}{s}_task-rest_run-1_feature-corrMatrixICA24Motion_atlas-SchaeferTian416_timeseries.json'))['MeanAtlasTSNR']
    Unmed_tsnr.append(tsnr)
Unmed_tsnr = np.asarray(Unmed_tsnr)

Med_tsnr = []
for s in Med_subjects:
    tsnr = json.load(open(f'{tspath}{s}_task-rest_run-1_feature-corrMatrixICA24Motion_atlas-SchaeferTian416_timeseries.json'))['MeanAtlasTSNR']
    Med_tsnr.append(tsnr)
Med_tsnr = np.asarray(Med_tsnr)


Control_tsnr_mean = Control_tsnr.mean(0)
Control_tsnr_mean[np.isnan(Control_tsnr_mean)] = 0
Unmed_tsnr_mean = Unmed_tsnr.mean(0)
Unmed_tsnr_mean[np.isnan(Unmed_tsnr_mean)] = 0
Med_tsnr_mean = Med_tsnr.mean(0)
Med_tsnr_mean[np.isnan(Med_tsnr_mean)] = 0

### Scatterplot and statistical test (spin) of delta TSNR vs Gradient
scatterplot2(fc_grad_parc[IDX2], (Unmed_tsnr_mean - Control_tsnr_mean)[IDX][16:],
             xlabel='FC Gradient', ylabel='TSNR Unmed - Control', regpl=True, cmap=cmap)
scatterplot2(fc_grad_parc[IDX2], (Med_tsnr_mean - Control_tsnr_mean)[IDX][16:],
             xlabel='FC Gradient', ylabel='TSNR Med - Control', regpl=True, cmap=cmap)
scatterplot2(fc_grad_parc[IDX2], (Unmed_tsnr_mean - Med_tsnr_mean)[IDX][16:],
             xlabel='FC Gradient', ylabel='TSNR Unmed - Med', regpl=True, cmap=cmap)


for label, ta_diff in [('Unmed vs Control', Unmed_tsnr_mean - Control_tsnr_mean),
                        ('Med vs Control',   Med_tsnr_mean - Control_tsnr_mean),
                        ('Unmed vs Med',     Unmed_tsnr_mean - Med_tsnr_mean)]:
    corr, pval = stats.compare_images(grad_parc, ta_diff[16:], nulls=rotated)
    print(f'{label}: r = {corr:.3f}, p = {pval:.3f}')


### Scatterplot of hierarchical levels vs TSNR for each group

### control 
h = Control['hierarchicallevels_full'] # nsub x nodes. bad regions contains nans.
h_mean = np.nanmean(h, 0)
h_mean[np.isnan(h_mean)] = 0
Control_tsnr_mean = Control_tsnr.mean(0) 
Control_tsnr_mean[np.isnan(Control_tsnr_mean)] = 0

scatterplot2(Control_tsnr_mean[IDX], h_mean[IDX], xlabel='Mean Atlas TSNR', ylabel='Hierarchical Level', regpl=True, cmap=cmap)
plt.title('Control')
print(pg.corr(Control_tsnr_mean[IDX], h_mean[IDX], method='spearman'))

rotated = nulls.alexander_bloch(Control_tsnr_mean, atlas='fsLR', density='32k',
                                n_perm=1000, seed=1234, parcellation=parcellation)

corr, pval = stats.compare_images(Control_tsnr_mean[16:], h_mean[16:], nulls=rotated)
print(f'Control: r = {corr:.3f}, p = {pval:.3f}')

### unmed 
h = Unmed['hierarchicallevels_full'] # nsub x nodes. bad regions contains nans.
h_mean = np.nanmean(h, 0)
h_mean[np.isnan(h_mean)] = 0
Unmed_tsnr_mean = Unmed_tsnr.mean(0)
Unmed_tsnr_mean[np.isnan(Unmed_tsnr_mean)] = 0

scatterplot2(Unmed_tsnr_mean[IDX], h_mean[IDX], xlabel='Mean Atlas TSNR', ylabel='Hierarchical Level', regpl=True, cmap=cmap)
plt.title('Unmed')
print(pg.corr(Unmed_tsnr_mean[IDX], h_mean[IDX], method='spearman'))

rotated = nulls.alexander_bloch(Unmed_tsnr_mean, atlas='fsLR', density='32k',
                                n_perm=1000, seed=1234, parcellation=parcellation)
corr, pval = stats.compare_images(Unmed_tsnr_mean[16:], h_mean[16:], nulls=rotated)
print(f'Unmed: r = {corr:.3f}, p = {pval:.3f}')

### med 
h = Med['hierarchicallevels_full'] # nsub x nodes. bad regions contains nans.
h_mean = np.nanmean(h, 0)
h_mean[np.isnan(h_mean)] = 0
Med_tsnr_mean = Med_tsnr.mean(0)
Med_tsnr_mean[np.isnan(Med_tsnr_mean)] = 0

scatterplot2(Med_tsnr_mean[IDX], h_mean[IDX], xlabel='Mean Atlas TSNR', ylabel='Hierarchical Level', regpl=True, cmap=cmap)
plt.title('Med')
print(pg.corr(Med_tsnr_mean[IDX], h_mean[IDX], method='spearman'))

rotated = nulls.alexander_bloch(Med_tsnr_mean, atlas='fsLR', density='32k',
                                n_perm=1000, seed=1234, parcellation=parcellation)
corr, pval = stats.compare_images(Med_tsnr_mean[16:], h_mean[16:], nulls=rotated)
print(f'Med: r = {corr:.3f}, p = {pval:.3f}')

### subject-level correlation for each group between hierarchical levels and tsnr 

control_corr = []
h = Control['hierarchicallevels_full'] # nsub x nodes. bad regions contains nans.
for sub in range(len(Control_subjects)):
    r = pg.corr(Control_tsnr[sub, :], h[sub, :], method='spearman')
    control_corr.append(r.r.values[0])
control_corr = np.asarray(control_corr)

unmed_corr = []
h = Unmed['hierarchicallevels_full'] # nsub x nodes. bad regions contains
for sub in range(len(Unmed_subjects)):
    r = pg.corr(Unmed_tsnr[sub, :], h[sub, :], method='spearman')
    unmed_corr.append(r.r.values[0])
unmed_corr = np.asarray(unmed_corr)

med_corr = []
h = Med['hierarchicallevels_full'] # nsub x nodes. bad regions contains
for sub in range(len(Med_subjects)):
    r = pg.corr(Med_tsnr[sub, :], h[sub, :], method='spearman')
    med_corr.append(r.r.values[0])
med_corr = np.asarray(med_corr)

scatterplot2(fc_grad_parc[IDX2], Control_tsnr_mean[IDX][16:], ylabel='Mean TSNR', xlabel='FC gradient', regpl=True, cmap=cmap)
plt.title('Control')

scatterplot2(fc_grad_parc[IDX2], Unmed_tsnr_mean[IDX][16:], ylabel='Mean TSNR', xlabel='FC gradient', regpl=True, cmap=cmap)
plt.title('Unmed')

scatterplot2(fc_grad_parc[IDX2], Med_tsnr_mean[IDX][16:], ylabel='Mean TSNR', xlabel='FC gradient', regpl=True, cmap=cmap)
plt.title('Med')



# ──────────────────────────────────────────────────────────────────────────────
# 10. Compare hierarchical levels vs global FC
# ──────────────────────────────────────────────────────────────────────────────
C = sio.loadmat(str(config.BASE_DIR / 'data/derived_data/FC_ctrl.mat'))
FC_C = C['FCPB']
COV_C = C['COVtauPB']

SZ0 = sio.loadmat(str(config.BASE_DIR / 'data/derived_data/FC_sz-0.mat'))
FC_sz0 = SZ0['FCPB']
COV_sz0 = SZ0['COVtauPB']

SZ1 = sio.loadmat(str(config.BASE_DIR / 'data/derived_data/FC_sz-1.mat'))
FC_sz1 = SZ1['FCPB']
COV_sz1 = SZ1['COVtauPB']

FCc_mean = np.mean(FC_C, 0).mean(0)
FCsz0_mean = np.mean(FC_sz0, 0).mean(0)
FCsz1_mean = np.mean(FC_sz1, 0).mean(0)

tmp = np.zeros(416)
tmp[IDX] = FCc_mean
FCc_mean = tmp

tmp = np.zeros(416)
tmp[IDX] = FCsz0_mean
FCsz0_mean = tmp

tmp = np.zeros(416)
tmp[IDX] = FCsz1_mean
FCsz1_mean = tmp

scatterplot2(fc_grad_parc[IDX2], (FCsz0_mean - FCc_mean)[IDX][16:], xlabel='FC gradient', ylabel='Regional FC', regpl=True, cmap=cmap)
plt.title('Unmed vs Control')
scatterplot2(fc_grad_parc[IDX2], (FCsz1_mean - FCc_mean)[IDX][16:], xlabel='FC gradient', ylabel='Regional FC', regpl=True, cmap=cmap)
plt.title('Med vs Control')
scatterplot2(fc_grad_parc[IDX2], (FCsz0_mean - FCsz1_mean)[IDX][16:], xlabel='FC gradient', ylabel='Regional FC', regpl=True, cmap=cmap)
plt.title('Unmed vs Med')

### Compare vs gradient
for label, ta_diff in [('Unmed vs Control', FCsz0_mean - FCc_mean),
                        ('Med vs Control',   FCsz1_mean - FCc_mean),
                        ('Unmed vs Med',     FCsz0_mean - FCsz1_mean)]:
    corr, pval = stats.compare_images(grad_parc, ta_diff[16:], nulls=rotated)
    print(f'{label}: r = {corr:.3f}, p = {pval:.3f}')

### Now compare global FC vs hierarchical levels
### Just curious about this, no need to report on this. 

h = Control['hierarchicallevels_full'] # nsub x nodes. bad regions contains nans. 
TMP = np.zeros((h.shape[0], 416))
TMP[:, IDX] = FC_C.mean(1)

corr = []
pval = []
for sub in range(h.shape[0]):
    r = pg.corr(h[sub, :], TMP[sub, :], method='spearman')
    corr.append(r.r.values[0])
    pval.append(r['p-val'].values[0])

corr = np.asarray(corr)
pval = -np.log10(np.asarray(pval))

h_mean = np.nanmean(h, 0)
scatterplot2(TMP.mean(0), h_mean, xlabel='Global FC', ylabel='Hierarchical Level', regpl=True)
plt.title('Control')
print(pg.corr(TMP.mean(0), h_mean, method='spearman'))

rotated = nulls.alexander_bloch(TMP.mean(0), atlas='fsLR', density='32k',
                                n_perm=1000, seed=1234, parcellation=parcellation)

corr, pval = stats.compare_images(TMP.mean(0)[16:], h_mean[16:], nulls=rotated)
print(f'Control: r = {corr:.3f}, p = {pval:.3f}')

### unmed 
h = Unmed['hierarchicallevels_full'] # nsub x nodes. bad regions contains nans.
TMP = np.zeros((h.shape[0], 416))
TMP[:, IDX] = FC_sz0.mean(1)

corr = []
pval = []
for sub in range(h.shape[0]):
    r = pg.corr(h[sub, :], TMP[sub, :], method='spearman')
    corr.append(r.r.values[0])
    pval.append(r['p-val'].values[0])
corr = np.asarray(corr)
pval = -np.log10(np.asarray(pval))

h_mean = np.nanmean(h, 0)
scatterplot2(TMP.mean(0), h_mean, xlabel='Global FC', ylabel='Hierarchical Level', regpl=True)
plt.title('Unmed')
print(pg.corr(TMP.mean(0), h_mean, method='spearman'))

rotated = nulls.alexander_bloch(TMP.mean(0), atlas='fsLR', density='32k',
                                n_perm=1000, seed=1234, parcellation=parcellation)

corr, pval = stats.compare_images(TMP.mean(0)[16:], h_mean[16:], nulls=rotated)
print(f'Unmed: r = {corr:.3f}, p = {pval:.3f}')

### med
h = Med['hierarchicallevels_full'] # nsub x nodes. bad regions contains nans.
TMP = np.zeros((h.shape[0], 416))
TMP[:, IDX] = FC_sz1.mean(1)

corr = []
pval = []
for sub in range(h.shape[0]):
    r = pg.corr(h[sub, :], TMP[sub, :], method='spearman')
    corr.append(r.r.values[0])
    pval.append(r['p-val'].values[0])
corr = np.asarray(corr)
pval = -np.log10(np.asarray(pval))

h_mean = np.nanmean(h, 0)
scatterplot2(TMP.mean(0), h_mean, xlabel='Global FC', ylabel='Hierarchical Level', regpl=True)
plt.title('Med')
print(pg.corr(TMP.mean(0), h_mean, method='spearman'))   

rotated = nulls.alexander_bloch(TMP.mean(0), atlas='fsLR', density='32k',
                                n_perm=1000, seed=1234, parcellation=parcellation)

corr, pval = stats.compare_images(TMP.mean(0)[16:], h_mean[16:], nulls=rotated)
print(f'Med: r = {corr:.3f}, p = {pval:.3f}')


### Compute gradient and measure the compression

from brainspace.gradient import GradientMaps

gm = GradientMaps(n_components=10, approach='dm', kernel='normalized_angle', random_state=0, alignment='procrustes')

# control group
gm.fit(list(FC_C))
# gradFCc = np.array(gm.gradients_)
gradFCc = np.array(gm.aligned_)

# unmedicated
gm.fit(list(FC_sz0))
# gradFCsz0 = np.array(gm.gradients_)
gradFCsz0 = np.array(gm.aligned_)

# medicated
gm.fit(list(FC_sz1))
# gradFCsz1 = np.array(gm.gradients_)
gradFCsz1 = np.array(gm.aligned_)

# Compute standard deviation across regions for each gradient, as a measure of compression
std_gradFCc = np.std(gradFCc, axis=1)
std_gradFCsz0 = np.std(gradFCsz0, axis=1)
std_gradFCsz1 = np.std(gradFCsz1, axis=1)

### boxplot std for all gradients for each group. build df and use sns. Remember each std_grad is n_subs x n_components. groups are unqueal length. 
df1 = pd.DataFrame({
    'std': np.concatenate((std_gradFCc.flatten(), std_gradFCsz0.flatten(), std_gradFCsz1.flatten())),
    'group': ['Control'] * std_gradFCc.size + ['Unmed'] * std_gradFCsz0.size + ['Med'] * std_gradFCsz1.size,
    'gradient': np.tile(np.arange(1, 11), std_gradFCc.shape[0] + std_gradFCsz0.shape[0] + std_gradFCsz1.shape[0])
})

tmp = np.zeros((gradFCc.shape[0], 416, 10))
tmp[:, IDX, :] = gradFCc
gradFCc = tmp

tmp = np.zeros((gradFCsz0.shape[0], 416, 10))
tmp[:, IDX, :] = gradFCsz0
gradFCsz0 = tmp

tmp = np.zeros((gradFCsz1.shape[0], 416, 10))
tmp[:, IDX, :] = gradFCsz1
gradFCsz1 = tmp

# Principal FC gradient from data vs independent principal FC gradient
scatterplot2(fc_grad_parc[VIDX],gradFCsz0.mean(0)[:,0][16:][VIDX],colors[16:][VIDX], xlabel='FC Gradient', ylabel='Gradient 1', regpl=True)
plt.title('PC1: Antipsychotic-naïve FEP')
scatterplot2(fc_grad_parc[VIDX],gradFCsz1.mean(0)[:,0][16:][VIDX],colors[16:][VIDX], xlabel='FC Gradient', ylabel='Gradient 1', regpl=True)
plt.title('PC1: Medicated FEP')
scatterplot2(fc_grad_parc[VIDX],gradFCc.mean(0)[:,0][16:][VIDX],colors[16:][VIDX], xlabel='FC Gradient', ylabel='Gradient 1', regpl=True)
plt.title('PC1: Control')

# PC2 from data vs independent principal FC gradient
scatterplot2(fc_grad_parc[VIDX],gradFCsz0.mean(0)[:,1][16:][VIDX],colors[16:][VIDX], xlabel='FC Gradient', ylabel='Gradient 2', regpl=True)
plt.title('Antipsychotic-naïve FEP')
scatterplot2(fc_grad_parc[VIDX],gradFCsz1.mean(0)[:,1][16:][VIDX],colors[16:][VIDX], xlabel='FC Gradient', ylabel='Gradient 2', regpl=True)
plt.title('Medicated FEP')
scatterplot2(fc_grad_parc[VIDX],gradFCc.mean(0)[:,1][16:][VIDX],colors[16:][VIDX], xlabel='FC Gradient', ylabel='Gradient 2', regpl=True)
plt.title('Control')

# Principal FC gradient from data vs independent FC gradient (PC2)
scatterplot2(fc_grad_parc2[VIDX],gradFCsz0.mean(0)[:,0][16:][VIDX],colors[16:][VIDX], xlabel='FC Gradient', ylabel='Gradient 1', regpl=True)
plt.title('PC1: Antipsychotic-naïve FEP')
scatterplot2(fc_grad_parc2[VIDX],gradFCsz1.mean(0)[:,0][16:][VIDX],colors[16:][VIDX], xlabel='FC Gradient', ylabel='Gradient 1', regpl=True)
plt.title('PC1: Medicated FEP')
scatterplot2(fc_grad_parc2[VIDX],gradFCc.mean(0)[:,0][16:][VIDX],colors[16:][VIDX], xlabel='FC Gradient', ylabel='Gradient 1', regpl=True)
plt.title('PC1: Control')

# PC2 from data vs independent FC gradient (PC2)
scatterplot2(fc_grad_parc2[VIDX],gradFCsz0.mean(0)[:,1][16:][VIDX],colors[16:][VIDX], xlabel='FC Gradient', ylabel='Gradient 2', regpl=True)
plt.title('Antipsychotic-naïve FEP')
scatterplot2(fc_grad_parc2[VIDX],gradFCsz1.mean(0)[:,1][16:][VIDX],colors[16:][VIDX], xlabel='FC Gradient', ylabel='Gradient 2', regpl=True)
plt.title('Medicated FEP')
scatterplot2(fc_grad_parc2[VIDX],gradFCc.mean(0)[:,1][16:][VIDX],colors[16:][VIDX], xlabel='FC Gradient', ylabel='Gradient 2', regpl=True)
plt.title('Control')


