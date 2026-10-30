# MATLAB pipeline

Fits the whole-brain model to the empirical data and derives the directed
connectivity and hierarchy measures that the Python scripts then analyse.

## Required external dependency: `HopfModel_LNA.m`

**This function is not distributed here.** The linear-noise approximation of
the Hopf whole-brain model is the work of Ponce-Alvarez and Deco, published as:

> Ponce-Alvarez A, Deco G. The Hopf whole-brain model and its linear
> approximation. *Scientific Reports* 2024;14:2615.
> https://doi.org/10.1038/s41598-024-53105-0

Obtain it from the authors' repository and place it on the MATLAB path before
running anything here:

    https://github.com/adrianponce/Linear-Hopf-model

`fit_group_level.m`, `run_hopf_grid_group.m` and `run_hopf_subject_refinement.m`
call it with the signature

```matlab
[FCsim, Cov0, Ct] = HopfModel_LNA(C, aVec, gVal, wo, sigma, lags_model);
```

`gaussfilt.m` is a small filtering utility that originates from the same
group's whole-brain modelling codebase and is included here only so the
pipeline runs unmodified.

## Configuration

One pipeline for every atlas: `run_hopf_grid_group.m` and
`run_hopf_subject_refinement.m` take the atlas from `ATLASNAME` and every path
from `kasp_paths.m`, which reads the environment. No path, atlas-specific
filename or participant identifier is written in the code.

| variable | meaning |
|---|---|
| `KASP_DATA` | project root: `subjects_*_clean_good_coverage.mat`, `is_med.mat`, `data/` with the SC matrix **(required)** |
| `KASP_RUN_TAG` | name of the run; set by `run_hopf_all.sh -TAG` **(required)** |
| `KASP_TIMESERIES` | timeseries and `IgnoreNodes.mat` (default `<KASP_DATA>/data/tseries/mat_files`) |
| `KASP_RESULTS_ROOT` | parent of all runs' results (default `<KASP_DATA>/results`) |
| `KASP_LNA_DIR` | folder holding `HopfModel_LNA.m`, if it is not already on the path |

Launch one group with

```bash
export KASP_DATA=/path/to/project
export KASP_TIMESERIES=$KASP_DATA/data/tseries_2/mat_files
./run_hopf_all.sh -TAG main -ATLASNAME 416 -group sz -MED 0 -GRIDON 0
```

`-GRIDON 0` fits at the fixed `g = 0.2` used in the paper; `-GRIDON 1`
instead searches `g` from 0.01 to 2.0 per group.

**Everything a run writes goes under `results/<TAG>/`** — group fits, subject
fits, and the derived FC and intrinsic frequencies. Nothing is written into
`data/`. Two runs cannot overwrite each other's files, and the launcher refuses
to start a run (tag, group, medication) that is already running. Logs are named
by run and start time.

**Exclusions come from `IgnoreNodes.mat` only.** Its `IgnoreNodes` field lists
the parcels to drop and its `ExcludeSubjects` field the participants; both are
written together. For the main analysis, `make_manuscript_ignore_nodes.m`
writes the 45-parcel list with no further participant excluded. A file
without `ExcludeSubjects` stops the run rather than silently including everyone. The reasons for each
exclusion belong in the decision log kept with the data, not in this repository.

**Every results file records the code that made it:** `code_commit`,
`code_dirty` and `run_tag` are saved alongside the fits.

## Model parameters

Set at the top of `run_hopf_grid_group.m`, matching the Methods section:
`N` from `ATLASNAME` (416 in the paper), `TR = 2 s`, noise amplitude `sigma = 0.01`,
learning rates `epsFC = 0.0004` and `epsFCtau = 0.0001`, maximum coupling
`maxC = 0.2`, at most 5000 iterations, global coupling `g = 0.2` (or a grid, `-GRIDON 1`),
bifurcation parameter `a = -0.02`, lag `tau = 1` TR.

## Trophic analysis

Implemented inline at the end of `run_hopf_subject_refinement.m`. Hierarchical levels are the solution
of the weighted graph Laplacian system and trophic coherence is `1 - F0`,
following:

> MacKay RS, Johnson S, Sansom B. How directed is a directed network?
> *Royal Society Open Science* 2020;7:201138.

> Levine S. Several measures of trophic structure applicable to complex food
> webs. *Journal of Theoretical Biology* 1980;83:195-207.
