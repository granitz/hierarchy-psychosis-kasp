"""
Head motion quality control.

Computes maximum and mean framewise displacement per participant from the
preprocessing confound files and compares the three groups with
Kruskal-Wallis tests.

Reported in: Results, final paragraph ("Groups did not differ in head
motion") and the Limitations section.

Original filename: MOTION.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kasp import config
from kasp.config import BASE_PATH as basepath

import os
import pandas as pd 
import numpy as np
import scipy.io as sio
import scipy.stats as sts 
import pingouin as pg 
import matplotlib.pyplot as plt
import seaborn as sns

plt.ion(); plt.rc("axes.spines", top=False, right=False); sns.set_palette("pastel")

# ---------------------------------------------------------------------------
# Directedness per group.
#
# The original script used the variables D_unmed / D_med / D_control, which it
# inherited from ANALYSIS.py having been run earlier in the same interactive
# session, so it could not run on its own. They are loaded here from the same
# source ANALYSIS.py used, leaving the outlier rule and the tests unchanged.
# ---------------------------------------------------------------------------
from kasp import data as kdata

_res = kdata.load_subject_results()
D_control = _res['ctrl']['trophiccoherence'][:, 0]
D_unmed = _res['sz0']['trophiccoherence'][:, 0]
D_med = _res['sz1']['trophiccoherence'][:, 0]

base_path = config.BASE_PATH
motion_path = f'{base_path}/data/MOTION/'

suff = '_task-rest_run-1_setting-preproc_desc-confounds_regressors.tsv'

subjects_ctrl = f'{base_path}/subjects_ctrl_clean_good_coverage.mat' 
subjects_ctrl = np.array([x[0] for x in sio.loadmat(subjects_ctrl)['subjects'][0]])

subjects_sz = f'{base_path}/subjects_sz_clean_good_coverage.mat'
subjects_sz = np.array([x[0] for x in sio.loadmat(subjects_sz)['subjects'][0]])

is_med = sio.loadmat(f'{base_path}/is_med.mat')['is_med'][0]

subjects_sz0 = subjects_sz[is_med == 0]
subjects_sz1 = subjects_sz[is_med == 1]

maxFDC = []
maxFDC_sz0 = []
maxFDC_sz1 = []
meanFDC = []
meanFDC_sz0 = []
meanFDC_sz1 = []

for subject in subjects_ctrl:
    path = os.path.join(motion_path, subject + suff)
    data = pd.read_csv(path, sep='\t')
    maxfd = data['framewise_displacement'].max()
    meanfd = data['framewise_displacement'].mean()
    maxFDC.append(maxfd)
    meanFDC.append(meanfd)

maxFDC = np.asarray(maxFDC)
meanFDC = np.asarray(meanFDC)

for subject in subjects_sz0:
    path = os.path.join(motion_path, subject + suff)
    data = pd.read_csv(path, sep='\t')
    maxfd = data['framewise_displacement'].max()
    meanfd = data['framewise_displacement'].mean()
    maxFDC_sz0.append(maxfd)
    meanFDC_sz0.append(meanfd)

maxFDC_sz0 = np.asarray(maxFDC_sz0)
meanFDC_sz0 = np.asarray(meanFDC_sz0)

for subject in subjects_sz1:
    path = os.path.join(motion_path, subject + suff)
    data = pd.read_csv(path, sep='\t')
    maxfd = data['framewise_displacement'].max()
    meanfd = data['framewise_displacement'].mean()
    maxFDC_sz1.append(maxfd)
    meanFDC_sz1.append(meanfd)

maxFDC_sz1 = np.asarray(maxFDC_sz1)
meanFDC_sz1 = np.asarray(meanFDC_sz1)


### Remove outliers ###
Unmed_keep = ~pg.madmedianrule(D_unmed)
Med_keep = ~pg.madmedianrule(D_med)
Control_keep = ~pg.madmedianrule(D_control)

D_unmed_clean = D_unmed[Unmed_keep]
D_med_clean = D_med[Med_keep]
D_control_clean = D_control[Control_keep]


### Nice clean boxplot for supplementary results 
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=[maxFDC[Control_keep], maxFDC_sz0[Unmed_keep], maxFDC_sz1[Med_keep]], palette='pastel')
plt.xticks([0, 1, 2], ['Control', 'Sz (unmed)', 'Sz (med)'])
plt.ylabel('Max Framewise Displacement (mm)')
plt.title('Max Framewise Displacement by Group')    

plt.subplot(1, 2, 2)
sns.boxplot(data=[meanFDC[Control_keep], meanFDC_sz0[Unmed_keep], meanFDC_sz1[Med_keep]], palette='pastel')
plt.xticks([0, 1, 2], ['Control', 'Sz (unmed)', 'Sz (med)'])
plt.ylabel('Mean Framewise Displacement (mm)')
plt.title('Mean Framewise Displacement by Group')

### Statistical testing ### 
# Max FDC
stat_max, p_max = sts.kruskal(maxFDC[Control_keep], maxFDC_sz0[Unmed_keep], maxFDC_sz1[Med_keep])
print(f"Kruskal-Wallis test for Max FDC: H={stat_max:.3f}, p={p_max:.3f}")

# Mean FDC
stat_mean, p_mean = sts.kruskal(meanFDC[Control_keep], meanFDC_sz0[Unmed_keep], meanFDC_sz1[Med_keep])
print(f"Kruskal-Wallis test for Mean FDC: H={stat_mean:.3f}, p={p_mean:.3f}")



