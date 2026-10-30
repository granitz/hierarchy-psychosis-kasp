"""
Main hierarchical organization analysis.

Loads the generative effective connectivity model fits, compares global
directedness (trophic coherence) between groups, relates regional
hierarchical levels to the principal functional connectivity gradient with
spin tests, and produces the network-level summaries.

Reported in: Figures 2 and 3, and the directedness statistics in Results.

Original filename: ANALYSIS.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kasp import config
from kasp.config import BASE_PATH as basepath

import os
import numpy as np
import pandas as pd 
import pingouin as pg 
import scipy.io as sio
import scipy.stats as sts 
from scipy.stats import zscore, mannwhitneyu, ttest_ind, PermutationMethod
from statsmodels.stats.multitest import fdrcorrection
from docx import Document

from neuromaps import nulls, stats as nnstats
from neuromaps.datasets import fetch_annotation
from netneurotools import datasets as nntdata
from neuromaps.parcellate import Parcellater
from neuromaps.images import dlabel_to_gifti

import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
import ptitprince as pt
import matplotlib

plt.ion(); plt.rc("axes.spines", top=False, right=False); sns.set_palette("pastel")
font = {'family':'Helvetica','size':18}; matplotlib.rc('font', **font)
SEED = 42 

def get_fc_grad(nroi=400):
    grad = 'fcgradient01'
    fc_grad = fetch_annotation(source='margulies2016', desc=grad, space='fsLR', den='32k')
    schaefer = nntdata.fetch_schaefer2018('fslr32k')[f'{nroi}Parcels7Networks']
    parcellation = dlabel_to_gifti(schaefer)
    parc = Parcellater(parcellation, 'fsLR')
    fc_grad_parc = parc.fit_transform(fc_grad, 'fsLR')
    return fc_grad_parc, parcellation

fc_grad_parc, parcellation = get_fc_grad(nroi=400)

# ---------------------------------------------------------------------------
# Helpers that the original script inherited from the interactive session:
# the scatter helper `scatterplot2`, the per-parcel colour array `colors` and
# the network assignment `com`. See kasp/plotting.py and kasp/parcellation.py.
# ---------------------------------------------------------------------------
from kasp.parcellation import network_assignment
from kasp.plotting import network_colors, scatterplot2, set_style

set_style()
com = network_assignment()      # integer network code per parcel, length 416
colors = network_colors(com)    # per-parcel colour, length 416

basepath = config.BASE_PATH

res_path = os.path.join(basepath, 'results/results_subject_refinement/')

### Load subject list 
subsc = sio.loadmat(os.path.join(basepath, 'subjects_ctrl_clean_good_coverage.mat'))['subjects']
subsz = sio.loadmat(os.path.join(basepath, 'subjects_sz_clean_good_coverage.mat'))['subjects']
subsz = np.array([x[0] for x in subsz[0]])
subsc = np.array([x[0] for x in subsc[0]])

### Load subjects results
Control_res = sio.loadmat(os.path.join(res_path, 'no_grid_subjectRefinement_group-ctrl_tau-1_med-0.mat'))
Unmed_res = sio.loadmat(os.path.join(res_path, 'no_grid_subjectRefinement_group-sz_tau-1_med-0.mat'))
Med_res = sio.loadmat(os.path.join(res_path, 'no_grid_subjectRefinement_group-sz_tau-1_med-1.mat'))

####################
### Directedness ###
####################
D_control = Control_res['trophiccoherence'][:,0]
D_unmed = Unmed_res['trophiccoherence'][:,0]
D_med = Med_res['trophiccoherence'][:,0]
# plt.boxplot((D_unmed, D_med, D_control), labels=['Unmed', 'Med', 'Control'])

#######################
### Remove outliers ###
#######################
Unmed_keep = ~pg.madmedianrule(D_unmed)
Med_keep = ~pg.madmedianrule(D_med)
Control_keep = ~pg.madmedianrule(D_control)

D_unmed_clean = D_unmed[Unmed_keep]
D_med_clean = D_med[Med_keep]
D_control_clean = D_control[Control_keep]
# plt.boxplot((D_unmed_clean, D_med_clean, D_control_clean), labels=['Unmed', 'Med', 'Control'])

##################
### Statistics ###
##################
pm = PermutationMethod(n_resamples=100_000, random_state=SEED)

statistic = pg.mwu(D_unmed_clean, D_control_clean, alternative='two-sided', method=pm)
print(f'Unmed vs Control: \n {statistic}')

statistic = pg.mwu(D_med_clean, D_control_clean, alternative='two-sided', method=pm)
print(f'Med vs Control: \n {statistic}')

statistic = pg.mwu(D_unmed_clean, D_med_clean, alternative='two-sided', method=pm)
print(f'Unmed vs Med: \n {statistic}')

### Hierarchical levels group level # nsub x nodes 
# NOTE: `GROUP` must be True. Everything below compares group-average maps
# parcel by parcel, so `process()` has to return a length-416 vector. With
# GROUP = False it returns one row per subject, and `H_unmed - H_control`
# then fails to broadcast (26, 416) against (41, 416); `[16:]` would also
# slice subjects rather than parcels. The committed value was False.
GROUP = True
ZSCORE = False

def process(res, keep):
    x = res['hierarchicallevels_full'][keep]
    if ZSCORE:
        x = zscore(x, axis=1, nan_policy='omit')
    if GROUP:
        x = x.mean(0)
    return x

H_control = process(Control_res, Control_keep)
H_unmed   = process(Unmed_res,   Unmed_keep)
H_med     = process(Med_res,     Med_keep)

###############################################
### Scatterplot delta group HL vs. gradient ###
###############################################

VIDX = ~np.isnan(np.nanmean(Control_res['hierarchicallevels_full'],0))[16:]

min_val = np.nanmin(((H_unmed-H_control)[16:][VIDX].min(), (H_med-H_control)[16:][VIDX].min(), (H_unmed-H_med)[16:][VIDX].min()))
max_val = np.nanmax(((H_unmed-H_control)[16:][VIDX].max(), (H_med-H_control)[16:][VIDX].max(), (H_unmed-H_med)[16:][VIDX].max()))
min_val = np.round(min_val,2)
max_val = np.round(max_val,2)

# cmap = LinearSegmentedColormap.from_list("RedWhiteBlue", ["blue", "white", "red"])
# cmap = LinearSegmentedColormap.from_list("Gray_Interp", ["black", "white"])

scatterplot2(fc_grad_parc[VIDX],(H_unmed-H_control)[16:][VIDX],colors[16:][VIDX],ylim=(min_val, max_val),
            xlabel='Functional Gradient', ylabel='ΔH (Unmed - Control)', regpl=True)
scatterplot2(fc_grad_parc[VIDX],(H_med-H_control)[16:][VIDX],colors[16:][VIDX],ylim=(min_val, max_val),
            xlabel='Functional Gradient', ylabel='ΔH  (Med - Control)', regpl=True)
scatterplot2(fc_grad_parc[VIDX],(H_unmed-H_med)[16:][VIDX],colors[16:][VIDX],ylim=(min_val, max_val),
            xlabel='Functional Gradient', ylabel='ΔH (Unmed - Med)', regpl=True)

#########################################
### Scatterplot group HL vs. gradient ###
#########################################

VIDX = ~np.isnan(np.nanmean(Control_res['hierarchicallevels_full'],0))[16:]

min_val = np.nanmin((H_unmed[16:][VIDX].min(), H_med[16:][VIDX].min(), H_control[16:][VIDX].min()))
max_val = np.nanmax((H_unmed[16:][VIDX].max(), H_med[16:][VIDX].max(), H_control[16:][VIDX].max()))
min_val = np.round(min_val,2)
max_val = np.round(max_val,2)

scatterplot2(fc_grad_parc[VIDX],H_unmed[16:][VIDX],colors[16:][VIDX],ylim=(min_val, max_val),
            xlabel='Functional Gradient', ylabel='HL Unmed', regpl=True) 

scatterplot2(fc_grad_parc[VIDX],H_med[16:][VIDX],colors[16:][VIDX],ylim=(min_val, max_val),
            xlabel='Functional Gradient', ylabel='HL Med', regpl=True) 

scatterplot2(fc_grad_parc[VIDX],H_control[16:][VIDX],colors[16:][VIDX],ylim=(min_val, max_val),
            xlabel='Functional Gradient', ylabel='HL Control', regpl=True)

#############################
### spin permutation test ###
#############################

rotated = nulls.alexander_bloch(fc_grad_parc, atlas='fsLR', density='32k',
                                n_perm=1000, seed=1234, parcellation=parcellation)

####
### DELTA HL vs. gradient
###
corr, pval = nnstats.compare_images(fc_grad_parc, (H_unmed-H_control)[16:], nulls=rotated)
print(f'Unmed vs Control: r = {corr:.3f}, p = {pval:.3f}')

corr, pval = nnstats.compare_images(fc_grad_parc, (H_med-H_control)[16:], nulls=rotated)
print(f'Med vs Control: r = {corr:.3f}, p = {pval:.3f}')

corr, pval = nnstats.compare_images(fc_grad_parc, (H_unmed-H_med)[16:], nulls=rotated)
print(f'Unmed vs Med: r = {corr:.3f}, p = {pval:.3f}')

###
### HL vs. gradient
###
corr, pval = nnstats.compare_images(fc_grad_parc, H_unmed[16:], nulls=rotated)
print(f'Unmed: r = {corr:.3f}, p = {pval:.3f}')

corr, pval = nnstats.compare_images(fc_grad_parc, H_med[16:], nulls=rotated)
print(f'Med: r = {corr:.3f}, p = {pval:.3f}')

corr, pval = nnstats.compare_images(fc_grad_parc, H_control[16:], nulls=rotated)
print(f'Control: r = {corr:.3f}, p = {pval:.3f}')


############################
#### NETWORK LEVEL PLOT ####
############################ 
from scipy.stats import sem
from itertools import combinations
from statsmodels.stats.multitest import fdrcorrection

def network_summary(HL_group, com, exclude_net=4, agg_fn=np.nanmean):
    # HL_group: (nsub, nreg)
    nreg = HL_group.shape[1]
    nets = np.unique(com)
    nets = np.setdiff1d(nets, [exclude_net])
    net_ids = []
    subj_vals = {net: [] for net in nets}
    for net in nets:
        idx = np.where(com == net)[0]
        if idx.size == 0:
            continue
        # compute per-subject network summary (agg across regions in that net)
        vals = np.array([agg_fn(HL_group[s, idx]) for s in range(HL_group.shape[0])])
        subj_vals[net] = vals
        net_ids.append(net)
    return np.array(net_ids), subj_vals

# compute summaries for each group
net_ids, subj_A = network_summary(H_unmed, com, exclude_net=4, agg_fn=np.nanmean)
_, subj_B = network_summary(H_med, com, exclude_net=4, agg_fn=np.nanmean)
_, subj_C = network_summary(H_control, com, exclude_net=4, agg_fn=np.nanmean)


def per_subject_network_vals(HL, com, exclude_net=4, agg_fn=np.nanmean):
    # HL: (nsub, nreg); com: (nreg,) network ids
    nets = np.unique(com)
    nets = np.setdiff1d(nets, [exclude_net])
    nsub = HL.shape[0]
    rows = []
    for s in range(nsub):
        for net in nets:
            idx = np.where(com == net)[0]
            if idx.size == 0:
                continue
            val = agg_fn(HL[s, idx])
            rows.append({'subj': s, 'network': int(net), 'value': val})
    return pd.DataFrame(rows)

def permutation_pvals_by_network(df, groups, pairs=None, n_perm=5000, seed=0):
    # df: DataFrame with columns ['subj','network','value']; groups: dict subj->group label
    rng = np.random.default_rng(seed)
    df = df.copy()
    df['group'] = df['subj'].map(groups)
    nets = sorted(df['network'].unique())
    if pairs is None:
        pairs = [('Unmed','Med'), ('Unmed','Control'), ('Med','Control')]
    pvals = {pair: [] for pair in pairs}
    for net in nets:
        sub = df[df['network'] == net]
        for a,b in pairs:
            xa = sub.loc[sub['group']==a, 'value'].dropna().values
            xb = sub.loc[sub['group']==b, 'value'].dropna().values
            if xa.size==0 or xb.size==0:
                pvals[(a,b)].append(np.nan); continue
            pooled = np.concatenate([xa, xb])
            nx = xa.size
            obs = xa.mean() - xb.mean()
            nulls = np.empty(n_perm)
            for i in range(n_perm):
                p = rng.permutation(pooled)
                nulls[i] = p[:nx].mean() - p[nx:].mean()
            pvals[(a,b)].append(np.mean(np.abs(nulls) >= np.abs(obs)))
    # FDR correction per pair across networks
    pvals_fdr = {}
    for pair, pv in pvals.items():
        pv = np.array(pv, dtype=float)
        valid = ~np.isnan(pv)
        adj = np.full_like(pv, np.nan)
        sig = np.full_like(pv, False, dtype=bool)
        if valid.sum()>0:
            rej, p_adj = fdrcorrection(pv[valid], alpha=0.05)
            adj[valid] = p_adj
            sig[valid] = rej
        pvals_fdr[pair] = {'p_raw': pv, 'p_fdr': adj, 'sig': sig}
    return nets, pvals_fdr

# ---------- Usage ----------
# HL_*: group HL arrays (nsub, nodes)
# com: network assignment (nodes,) 
# groups_map: dict mapping subj index -> group label string ('Unmed','Med','Control')

# Compute per-subject network values
df_A = per_subject_network_vals(H_unmed, com, exclude_net=4, agg_fn=np.nanmean)
df_B = per_subject_network_vals(H_med,  com, exclude_net=4, agg_fn=np.nanmean)
df_C = per_subject_network_vals(H_control, com, exclude_net=4, agg_fn=np.nanmean)
df = pd.concat([df_A, df_B, df_C], ignore_index=True)

# Build groups mapping 
# If subjects indices overlap across groups, provide a custom mapping (unique subj ids required)
# Here we assume subj indices are local per group; to avoid overlap assign new subj ids:
def relabel_df(df, start_idx):
    df2 = df.copy()
    df2['subj'] = df2['subj'] + start_idx
    return df2

nA = H_unmed.shape[0]
nB = H_med.shape[0]
nC = H_control.shape[0]
dfA = relabel_df(df_A, 0)
dfB = relabel_df(df_B, nA)
dfC = relabel_df(df_C, nA + nB)
df = pd.concat([dfA, dfB, dfC], ignore_index=True)

groups_map = {}
for s in dfA['subj'].unique(): groups_map[s] = 'Unmed'
for s in dfB['subj'].unique(): groups_map[s] = 'Med'
for s in dfC['subj'].unique(): groups_map[s] = 'Control'
df['group'] = df['subj'].map(groups_map)

# network name mapping
name_map = {7:'Sub', 0:'Vis', 1:'SM', 2:'DA', 3:'SA', 5:'FP', 6:'DMN'}
# only keep networks present (and excluded limbic earlier)
df['network_name'] = df['network'].map(name_map)

# permutation p-values
nets, pvals_fdr = permutation_pvals_by_network(df, groups_map, n_perm=5000)

# Plot: violin + box + swarm per network
groups = ['Unmed','Med','Control']
order = sorted(df['network_name'].unique(), key=lambda s: list(name_map.values()).index(s))
nnet = len(order) 
base_x = np.arange(nnet)
width = 0.28
offsets = np.linspace(-width, width, len(groups))

plt.figure(figsize=(12,6))
# Draw violins (dodge=True for side-by-side)
sns.violinplot(x='network_name', y='value', hue='group', data=df, order=order,
               split=False, inner=None, palette='Set2', cut=0, dodge=True)
ax = plt.gca()

# Prepare boxplot data and positions
box_data = []
positions = []
for i, net in enumerate(order):
    for j, g in enumerate(groups):
        vals = df[(df['network_name'] == net) & (df['group'] == g)]['value'].dropna().values
        box_data.append(vals)
        positions.append(base_x[i] + offsets[j])

# Draw matplotlib boxplots at the specified positions
bp = ax.boxplot(box_data, positions=positions, widths=0.10, patch_artist=True,
                showfliers=False, manage_ticks=False)

######### SET THE MIDDILE COLOR #########
for med in bp['medians']:
    med.set_color('black')     # color
    med.set_linewidth(2.0)     # thickness

###################################################################################################
#### MAKE DOTS APPEAR IN THE FRONT ####
for elem in ('boxes','whiskers','caps','medians','fliers','means'):
    for art in bp.get(elem, []):
        art.set_zorder(1)
# make violin bodies behind
for coll in ax.collections:          # violin parts are Collections
    coll.set_zorder(0)

# now draw points on top with high zorder
for i, net in enumerate(order):
    for j, g in enumerate(groups):
        vals = df[(df['network_name']==net) & (df['group']==g)]['value'].dropna().values
        if vals.size==0: continue
        xj = np.random.normal(loc=base_x[i] + offsets[j], scale=0.03, size=len(vals))
        ax.scatter(xj, vals, color='k', s=8, alpha=0.9, zorder=100, edgecolor='none')
###################################################################################################

# Color boxes to match seaborn palette
palette = sns.color_palette('Set2', n_colors=len(groups))
for k, patch in enumerate(bp['boxes']):
    grp_idx = k % len(groups)
    patch.set_facecolor(palette[grp_idx])
    patch.set_alpha(0.9)

# Overlay individual points jittered around each box position
for i, net in enumerate(order):
    for j, g in enumerate(groups):
        vals = df[(df['network_name'] == net) & (df['group'] == g)]['value'].dropna().values
        if vals.size == 0:
            continue
        xj = np.random.normal(loc=base_x[i] + offsets[j], scale=0.03, size=len(vals))
        ax.scatter(xj, vals, color='k', s=8, alpha=0.6)

# Tidy axes and legend
ax.set_xticks(base_x)
ax.set_xticklabels(order, rotation=45, ha='right')
# manual legend matching palette
for j, g in enumerate(groups):
    ax.plot([], c=palette[j], label=g)
ax.legend(title='Group', loc='upper right')
ax.set_ylabel('Hierarchical level (z-score)')
plt.tight_layout()
plt.show()


# Annotate significant pairs: place a star above the groups if any pair significant after FDR
xpos = {name:i for i,name in enumerate(order)}
y_max_per_net = df.groupby('network_name')['value'].max()
y_off = (df['value'].max() - df['value'].min()) * 0.08
for i, net in enumerate(nets):
    netname = name_map[net]
    y = y_max_per_net[netname] + y_off
    # check pairs
    for j, pair in enumerate([('Unmed','Med'), ('Unmed','Control'), ('Med','Control')]):
        sig = pvals_fdr[pair]['sig'][i]
        if sig:
            # horizontal offset between groups for jittered plotting
            dx = (-0.25, 0.0, 0.25)[j]
            plt.text(xpos[netname] + dx, y + j*0.03, '*', ha='center', va='bottom', fontsize=12, color='k')

plt.tight_layout()
plt.show()


#########################################
### STATISTICAL TEST BETWEEN NETWORKS ###
#########################################
def cohens_d(x, y):
    x = np.asarray(x)[~np.isnan(x)]
    y = np.asarray(y)[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx<2 or ny<2: return np.nan
    dof = nx + ny - 2
    pooled_sd = np.sqrt(((nx-1)*x.std(ddof=1)**2 + (ny-1)*y.std(ddof=1)**2) / dof)
    return (np.nanmean(x) - np.nanmean(y)) / pooled_sd if pooled_sd>0 else np.nan

def rank_biserial(x,y):
    # common nonparametric effect size (Mann-Whitney)
    x = np.asarray(x)[~np.isnan(x)]
    y = np.asarray(y)[~np.isnan(y)]
    if len(x)==0 or len(y)==0: return np.nan
    U = mannwhitneyu(x,y, alternative='two-sided').statistic
    nx, ny = len(x), len(y)
    return 1 - (2*U)/(nx*ny)

rows = []
pairs = [('Unmed','Med', subj_A, subj_B),
         ('Unmed','Control', subj_A, subj_C),
         ('Med','Control', subj_B, subj_C)]
n_perm = 5000
for net_i, net in enumerate(net_ids):
    for nameA, nameB, GA, GB in pairs:
        xa = np.asarray(GA[net]); xb = np.asarray(GB[net])
        xa = xa[~np.isnan(xa)]; xb = xb[~np.isnan(xb)]
        nA, nB = len(xa), len(xb)
        obs = np.nanmean(xa) - np.nanmean(xb) if (nA>0 and nB>0) else np.nan
        d = cohens_d(xa, xb)
        rb = rank_biserial(xa, xb)
        p_raw = pvals_fdr[(nameA,nameB)]['p_raw'][net_i]
        p_fdr = pvals_fdr[(nameA,nameB)]['p_fdr'][net_i]
        rows.append({'network': int(net), 'pair': f'{nameA} vs {nameB}',
                     'nA': nA, 'nB': nB, 'obs_diff': obs,
                     'cohens_d': d, 'rank_biserial': rb,
                     'p_raw': p_raw, 'p_fdr': p_fdr, 'n_perm': n_perm})
# convert to DataFrame for display/export
df_stats = pd.DataFrame(rows)
df_stats = df_stats.sort_values(['network','pair'])
print(df_stats.to_string(index=False))


#### PRINT NETWORK-LEVEL STATSTO WORD DOCUMENT:

# assume df_stats exists and is sorted
fmt = df_stats.copy()

# numeric rounding (keeps numeric dtype)
fmt[['obs_diff','cohens_d','rank_biserial']] = df_stats[['obs_diff','cohens_d','rank_biserial']].round(3)

# format p-values as strings with 4 decimal places, preserving NaN
for c in ['p_raw','p_fdr']:
    fmt[c] = df_stats[c].map(lambda x: '' if pd.isna(x) else f"{x:.4f}")

# now write fmt to Word as before
doc = Document()
cols = list(fmt.columns)
table = doc.add_table(rows=1, cols=len(cols))
for i, col in enumerate(cols):
    table.rows[0].cells[i].text = str(col)

for _, row in fmt.iterrows():
    cells = table.add_row().cells
    for i, col in enumerate(cols):
        cells[i].text = '' if pd.isna(row[col]) else str(row[col])

# doc.save(str(config.BASE_DIR / 'slurm/tables/df_stats.docx'))



###########################################################
### Compare individual hierarchical levels vs. gradient ###
###########################################################

### CONTROL GROUP 
CORR_CONTROL = []
PVAL_CONTROL = []

print('Control group')
for sub in range(H_control.shape[0]):
    h = H_control[sub, :]
    corr, pval = sts.spearmanr(fc_grad_parc, h[16:], nan_policy='omit')
    CORR_CONTROL.append(corr)
    PVAL_CONTROL.append(pval)

CORR_CONTROL = np.array(CORR_CONTROL)
PVAL_CONTROL = np.array(PVAL_CONTROL)

### UNMED GROUP
CORR_UNMED = []
PVAL_UNMED = []

print('Unmed group')
for sub in range(H_unmed.shape[0]):
    h = H_unmed[sub, :]
    corr, pval = sts.spearmanr(fc_grad_parc, h[16:], nan_policy='omit')
    CORR_UNMED.append(corr)
    PVAL_UNMED.append(pval)

CORR_UNMED = np.array(CORR_UNMED)
PVAL_UNMED = np.array(PVAL_UNMED)

### MED GROUP
CORR_MED = []
PVAL_MED = []

print('Med group')
for sub in range(H_med.shape[0]):
    h = H_med[sub, :]
    corr, pval = sts.spearmanr(fc_grad_parc, h[16:], nan_policy='omit')
    CORR_MED.append(corr)
    PVAL_MED.append(pval)

CORR_MED = np.array(CORR_MED)
PVAL_MED = np.array(PVAL_MED)

data = pd.DataFrame({
    'Correlation': np.concatenate((CORR_UNMED, CORR_MED, CORR_CONTROL)),
    'Group': ['Unmed']*len(CORR_UNMED) + ['Med']*len(CORR_MED) + ['Control']*len(CORR_CONTROL)
})

plt.figure(figsize=(8, 6))
sns.violinplot(x='Group', y='Correlation', data=data, palette='pastel')    

####
### plot abs correlation values
data['AbsCorrelation'] = np.abs(data['Correlation'])

# statistical test of the abs. correlation 
CORR_UNMED_ABS = data[data['Group']=='Unmed']['AbsCorrelation']
CORR_MED_ABS = data[data['Group']=='Med']['AbsCorrelation']
CORR_CONTROL_ABS = data[data['Group']=='Control']['AbsCorrelation']

stat, p = mannwhitneyu(CORR_UNMED_ABS, CORR_CONTROL_ABS, alternative='two-sided')
print(f'Unmed vs Control (abs): stat={stat:.3f}, p={p:.3f}')
stat, p = mannwhitneyu(CORR_MED_ABS, CORR_CONTROL_ABS, alternative='two-sided')
print(f'Med vs Control (abs): stat={stat:.3f}, p={p:.3f}')
stat, p = mannwhitneyu(CORR_UNMED_ABS, CORR_MED_ABS, alternative='two-sided')
print(f'Unmed vs Med (abs): stat={stat:.3f}, p={p:.3f}')

sns.violinplot(x='Group', y='AbsCorrelation', data=data, palette='pastel')

# plot only significant values as boxplot of absolute correlation values
data['Significant'] = np.concatenate((PVAL_UNMED, PVAL_MED, PVAL_CONTROL)) < 0.05  

print(f'Number of significant individuals in each group:')
print(f'Unmed: {data[(data["Group"]=="Unmed") & (data["Significant"])].shape[0]}/{len(CORR_UNMED)}')
print(f'Med: {data[(data["Group"]=="Med") & (data["Significant"])].shape[0]}/{len(CORR_MED)}')
print(f'Control: {data[(data["Group"]=="Control") & (data["Significant"])].shape[0]}/{len(CORR_CONTROL)}')




#####################################
### PLOT MODEL ERRROR 
#####################################

SZ = sio.loadmat(f'{basepath}/results/results_group_only/no_grid_GroupResults_group-sz_tau-1_med-0.mat')
FC = SZ['errFC'][0]
COV = SZ['errCOVtau'][0]

# Create a tidy DataFrame
df = pd.DataFrame({
    "value": np.concatenate([FC, COV]),
    "variable": ["FC"]*len(FC) + ["COVtau"]*
    len(COV)
})  


fig, ax = plt.subplots(figsize=(8, 5))

groups = list(df.groupby("variable"))
n = len(groups)

# Generate evenly spaced grey tones
greys = np.linspace(0.25, 0.75, n)

for (grey, (name, g)) in zip(greys, groups):
    ax.plot(
        range(len(g)),
        g["value"],
        color=str(grey),   # matplotlib greyscale
        linewidth=2,
        alpha=0.9,
        label=name
    )

# ax.set_xlim(0, 200)
ax.set_xlabel("Index")
ax.set_ylabel("Value")

ax.legend(frameon=False)
ax.grid(True, alpha=0.0)
plt.tight_layout()
plt.show()


######################
### PLOT MODEL FIT ###
######################
FEP0 = sio.loadmat(res_path + f'no_grid_subjectRefinement_group-sz_tau-1_med-1.mat')
FEP1 = sio.loadmat(res_path + f'no_grid_subjectRefinement_group-sz_tau-1_med-0.mat')
CTRL = sio.loadmat(res_path + f'no_grid_subjectRefinement_group-ctrl_tau-1_med-0.mat')

FCz0 = FEP0['fittFC_subj'][:,0]
COVz0 = FEP0['fittCOVtau_subj'][:,0]
dfz0 = pd.DataFrame(FCz0)
dfz0['COVtau'] = COVz0
dfz0['Group'] = np.repeat('FEP0',len(FCz0))
dfz0.columns = ['FC','COVtau','Group']
dfz0['nonzero'] = ~(dfz0.FC==0)
FCz1 = FEP1['fittFC_subj'][:,0]
COVz1 = FEP1['fittCOVtau_subj'][:,0]
dfz1 = pd.DataFrame(FCz1)
dfz1['COVtau'] = COVz1
dfz1['Group'] = np.repeat('FEP1',len(FCz1))
dfz1.columns = ['FC','COVtau','Group']
dfz1['nonzero'] = ~(dfz1.FC==0)
FCc = CTRL['fittFC_subj'][:,0]
COVc = CTRL['fittCOVtau_subj'][:,0]
dfc = pd.DataFrame(FCc)
dfc['COVtau'] = COVc
dfc['Group'] = np.repeat('Control',len(FCc))
dfc.columns = ['FC','COVtau','Group']
dfc['nonzero'] = ~(dfc.FC==0)   

df = pd.concat((dfz0,dfz1,dfc))
dff=df.melt(('Group','nonzero'))
dff = dff[dff.nonzero] # keep only non-zero 

# Custom gray palette (adjust number of shades to your groups)
unique_groups = dff['Group'].unique()
gray_palette = sns.color_palette("Greys", n_colors=len(unique_groups) + 2)[2:]  # skip very light shades
palette = dict(zip(unique_groups, gray_palette))

plt.figure(figsize=(8, 5))

# Boxplot (semi-transparent)
sns.boxplot(
    data=dff,
    x='variable',
    y='value',
    hue='Group',
    palette=sns.color_palette("light:b", n_colors=len(unique_groups)),
    showfliers=False,
    boxprops={'alpha': 0.6}
)

# Dots (overlay)
sns.stripplot(
    data=dff,
    x='variable',
    y='value',
    hue='Group',
    dodge=True,
    jitter=True,
    palette=palette,
    size=3,
    alpha=0.6,
    linewidth=0.3,
    edgecolor='gray'
)

# Clean up legend (avoid duplication)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:len(unique_groups)], labels[:len(unique_groups)],
           title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title("Boxplot with Dots — Grouped by Variable (Gray Palette)")
plt.tight_layout()
plt.show()

### Statistical test comparing the three
### Simple paired ttest between groups for FC and COVtau
for var in ['FC','COVtau']:
    print(f'Variable: {var}')
    data0 = dfz0[var][dfz0.nonzero]
    data1 = dfz1[var][dfz1.nonzero]
    datac = dfc[var][dfc.nonzero]
    tstat, pval = ttest_ind(data0, data1, equal_var=False)
    print(f'FEP0 vs FEP1: t = {tstat:.3f}, p = {pval:.3f}')
    tstat, pval = ttest_ind(data0, datac, equal_var=False)
    print(f'FEP0 vs Control: t = {tstat:.3f}, p = {pval:.3f}')
    tstat, pval = ttest_ind(data1, datac, equal_var=False)
    print(f'FEP1 vs Control: t = {tstat:.3f}, p = {pval:.3f}')


Variable: FC
# FEP0 vs FEP1: t = -0.442, p = 0.661
# FEP0 vs Control: t = 1.366, p = 0.179
# FEP1 vs Control: t = 1.922, p = 0.060
Variable: COVTAU
# FEP0 vs FEP1: t = -0.283, p = 0.778
# FEP0 vs Control: t = 1.488, p = 0.144
# FEP1 vs Control: t = 1.906, p = 0.062


#########################
### PLOT DIRECTEDNESS ###
#########################

# Create a tidy DataFrame
df = pd.DataFrame({
    "value": np.concatenate([D_unmed_clean, D_med_clean, D_control_clean]),
    "variable": ["a"]*len(D_unmed_clean) + ["b"]*len(D_med_clean) + ["c"]*len(D_control_clean)
})

# Create a figure
plt.figure(figsize=(8, 6))

# Plot a raincloud plot without colors (all in grayscale)
pt.RainCloud(x="variable", y="value", data=df, 
             palette=["0.8"],  # grayscale
             bw=.2, width_viol=.6, ax=plt.gca(),
             orient="v",
             alpha=.65,
             move=.2)

plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.show()

