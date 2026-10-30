# Regional hierarchical reorganization in antipsychotic-naive first-episode psychosis

Analysis code for the paper of the same name.

Resting-state fMRI and diffusion MRI from patients with first-episode
psychosis (N=54; 31 antipsychotic-naive, 23 with brief antipsychotic
exposure) and healthy controls (N=45) are used to fit a linearized Hopf
whole-brain model constrained by structural connectivity. The resulting
generative effective connectivity is submitted to trophic analysis, giving a
global measure of hierarchy (directedness) and a regional one (hierarchical
level). These are compared against the principal functional connectivity
gradient spanning unimodal to transmodal cortex.

## Pipeline

```
 BOLD + diffusion MRI
         │
         ▼
 ┌───────────────────┐   MATLAB    empirical FC and time-shifted covariance,
 │  matlab/          │  ─────────▶ group-level fit, per-subject refinement,
 └───────────────────┘             trophic levels and coherence
         │
         ▼   subjectRefinement_group-*.mat
 ┌───────────────────┐   Python    group comparisons, gradient alignment,
 │  notebooks/,      │  ─────────▶ spin tests, symptom correlations, figures
 │  kasp/, scripts/  │
 └───────────────────┘
```

The MATLAB stage is the expensive one and is normally run on a cluster; see
[`matlab/README.md`](matlab/README.md), which also documents the one external
dependency (`HopfModel_LNA.m`, by Ponce-Alvarez and Deco, not redistributed
here).

## Results

Every run of the notebook is named by `KASP_RUN_TAG` and writes to
`runs/<tag>/`: `stats.csv` holds every reported statistic at full precision
and `manifest.json` fingerprints the model fits it read; both are committed,
figures and tables are not. `runs/main/` is the run reported in the paper.

**To see the results without running anything**, open
[`runs/main/reproduce_figures.executed.ipynb`](runs/main/reproduce_figures.executed.ipynb):
the notebook as run, with every number and figure. If GitHub does not render
it, use [nbviewer](https://nbviewer.org/github/granitz/hierarchy-psychosis-kasp/blob/main/runs/main/reproduce_figures.executed.ipynb),
or download `runs/main/reproduce_figures.executed.html` and open it in a
browser. `notebooks/reproduce_figures.ipynb` is the same notebook without
outputs, for running it yourself.

## What produces what

**Start with `notebooks/reproduce_figures.ipynb`.** It computes every
statistic and every figure in the paper and is self-contained:

| Notebook section | Reproduces |
|---|---|
| 1 | Setup: model fits, gradient, parcel masks |
| 2 | Table 1 — demographic and clinical characteristics, with group comparisons |
| 3 | Global directedness, group comparisons |
| 4 | Model fit (Supplementary Figure 1A) |
| 5 | Figure 2 — hierarchical levels vs the principal gradient |
| 6 | Figure 3 — group differences vs the principal gradient |
| 7 | Figure 4 — gradient-hierarchy alignment vs positive symptoms |
| 8 | Supplementary Figure 2 — regional FC, lag-1 temporal autocorrelation and tSNR, each against the gradient |
| 9 | Head motion |
| 10 | Figures 2-3 and Supplementary Figures 1-2, composed (PDF and SVG, editable text) |
| 11 | Writes this run's `stats.csv` and `manifest.json` |

Figures and tables are written to `runs/<KASP_RUN_TAG>/figures/` and
`.../tables/` (git-ignored); the statistics to `runs/<KASP_RUN_TAG>/stats.csv`.

The same analyses are also available as plain scripts:

| Script | Purpose |
|---|---|
| `scripts/run_notebook.sh` | runs the notebook headless for the tag in `.env` and keeps an executed copy (`.ipynb` + `.html`) |
| `scripts/00_build_clinical_table.py` | prerequisite — builds the de-identified clinical table |
| `scripts/01_motion_qc.py` | head-motion comparison |
| `scripts/02_hierarchy_analysis.py` | Figures 2 and 3 |
| `scripts/03_behaviour_correlations.py` | Table 1; Figure 4 |
| `scripts/04_supplementary_autocorrelation.py` | Supplementary Figure 2 |
| `scripts/redact_html.py` | removes local paths from an executed notebook before it is shared |

Shared code lives in `kasp/`: `config` (paths and constants), `data`
(loaders), `parcellation` (atlas, gradient, geometry), `plotting`, `stats`,
`rendering` (surface and subcortical maps), `figures` (the composed figures)
and `record` (the per-run log).

## Setup

```bash
git clone https://github.com/granitz/hierarchy-psychosis-kasp.git
cd hierarchy-psychosis-kasp

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then edit the paths in .env
python config.py         # prints every resolved path, flags the missing ones
```

`.env` holds one essential setting, `KASP_DATA`, pointing at the project root
— the folder holding the subject-list `.mat` files plus the `results/` and
`data/` directories — and the run's name, `KASP_RUN_TAG`. Everything else is
derived from them. The file is git-ignored, so local paths never get
committed.

**Use `.env` rather than shell exports if you work in notebooks.** A variable
exported in a terminal is only inherited by processes started from that same
terminal, so a Jupyter kernel launched from VS Code, JupyterLab or Anaconda
Navigator will not see it. `.env` is read every time `kasp` is imported, so it
works regardless of how the kernel was started.

Shell exports still work if you prefer them, and must say `export`:

```bash
export KASP_DATA="/path/to/tmp/"
```

Individual directories can be redirected too, under either the canonical
`KASP_`-prefixed name or the bare attribute name:

| Setting | Canonical | Alias | Default, relative to `KASP_DATA` |
|---|---|---|---|
| per-subject fits | `KASP_RESULTS` | `RESULTS_DIR` | `results/results_subject_refinement` |
| group fits | `KASP_GROUP_RESULTS` | `GROUP_RESULTS_DIR` | `results/results_group_only` |
| timeseries `.mat` | `KASP_TIMESERIES` | `TS_DIR` | `data/tseries_2/mat_files` |
| tSNR | `KASP_TSNR` | `TSNR_DIR` | `data/tseries_2` |
| confound `.tsv` | `KASP_MOTION` | `MOTION_DIR` | `data/MOTION` |
| empirical FC | `KASP_DERIVED` | `DERIVED_DIR` | `data/derived_data` |
| clinical table | `KASP_CLINICAL` | `CLINICAL_DIR` | `data/clinical` |

Then open `notebooks/reproduce_figures.ipynb` and run it top to bottom, or
run `scripts/run_notebook.sh`. The notebook is committed with its outputs
stripped; the numbers a run produced live in `runs/<tag>/stats.csv`.

No absolute paths appear anywhere in the code: everything resolves through
`kasp/config.py`. The MATLAB code reads the same variables.

## Data availability

**No participant data is included in this repository**, and `.gitignore` is
written to keep it that way — `data/` is excluded wholesale, along with
`*.mat`, `*.tsv`, and anything matching clinical or demographic filename
patterns.

Deidentified individual participant data underlying the results may be made
available from the corresponding author on reasonable request, provided no
legal, regulatory or ethical barriers to sharing apply. The clinical table
the code expects is keyed on scan identifier only; its format is documented
in `data/clinical/clinical_data_TEMPLATE.csv` and in
`kasp.data.load_clinical`.

The data were collected as part of the Karolinska Schizophrenia Project
(KaSP) and approved by the Regional Ethics Committee in Stockholm
(2010/879-31-1). All participants gave written informed consent.

## Software

Analyses were run in Python (see `requirements.txt`) and MATLAB. Key
dependencies: `neuromaps` and `netneurotools` for the gradient, parcellation
and spin tests; `pingouin` and `statsmodels` for statistics; `brainspace` for
gradient decomposition; `surfplot` and `subcortex-visualization` for the
brain renderings.

`kasp/stats.py` vendors `efficient_pearsonr` and `permtest_pearsonr` from
netneurotools (BSD-3-Clause), because the private scipy helper they relied on
was removed in scipy 1.15.

## Citing

See `CITATION.cff`. Please cite the paper; the code is released under the MIT
licence (`LICENSE`).

## Key references

- Ponce-Alvarez A, Deco G. The Hopf whole-brain model and its linear approximation. *Sci Rep* 2024;14:2615.
- MacKay RS, Johnson S, Sansom B. How directed is a directed network? *R Soc Open Sci* 2020;7:201138.
- Margulies DS, et al. Situating the default-mode network along a principal gradient of macroscale cortical organization. *PNAS* 2016;113:12574-9.
- Schaefer A, et al. Local-global parcellation of the human cerebral cortex. *Cereb Cortex* 2018;28:3095-114.
- Tian Y, et al. Topographic organization of the human subcortex. *Nat Neurosci* 2020;23:1421-32.
- Alexander-Bloch AF, et al. On testing for spatial correspondence between maps of human brain structure and function. *NeuroImage* 2018;178:540-51.
- Shinn M, et al. Functional brain networks reflect spatial and temporal autocorrelation. *Nat Neurosci* 2023;26:867-78.
