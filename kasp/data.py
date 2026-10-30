"""
Loaders for the model outputs, timeseries and clinical tables.

Every path is resolved through :mod:`kasp.config`, so nothing here assumes a
particular machine. See ``data/README.md`` for the expected directory layout
and file contents.

Group labels used throughout the codebase:

=========  =======================================================
``ctrl``   healthy controls
``sz0``    antipsychotic-naive first-episode psychosis (unmedicated)
``sz1``    first-episode psychosis with brief antipsychotic exposure
=========  =======================================================
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from . import config

__all__ = [
    "TS_SUFFIX",
    "MOTION_SUFFIX",
    "extract_subjects",
    "load_subject_lists",
    "load_subject_results",
    "trophic_coherence",
    "load_group_results",
    "is_grid_run",
    "fit_curves",
    "grid_curves",
    "group_fc_from_timeseries",
    "load_timeseries",
    "load_empirical_fc",
    "regional_mean_fc",
    "expand_to_full_parcellation",
    "load_tsnr",
    "load_ignore_nodes",
    "load_clinical",
]

#: Filename suffix of the parcellated timeseries .mat files. The atlas token
#: comes from config, so switching KASP_ATLAS switches the filenames too.
TS_SUFFIX = (
    "_task-rest_run-1_feature-corrMatrixICA24Motion"
    f"_atlas-{config.ATLAS}_timeseries.mat"
)

#: Filename suffix of the confound regressor .tsv files.
MOTION_SUFFIX = "_task-rest_run-1_setting-preproc_desc-confounds_regressors.tsv"

#: Invariant middle of the per-subject fit filenames. The prefix
#: (``no_grid_`` / none) and any trailing atlas token vary by run.
_SUBJECT_STEM = "subjectRefinement_group-{group}_tau-{tau}_med-{med}"

#: Invariant middle of the group-level fit filenames. Prefixed with
#: ``no_grid_`` or ``grid`` depending on GRIDON.
_GROUP_STEM = "GroupResults_group-{group}_tau-{tau}_med-{med}"


def extract_subjects(mat_data, key="subjects"):
    """Pull the list of scan identifiers out of a loaded .mat struct."""
    subjects = mat_data[key]
    subjects = subjects[0] if subjects.shape[0] == 1 else subjects[:, 0]
    return np.array([s[0] for s in subjects])


def load_subject_lists():
    """Return ``(controls, patients, is_medicated)``.

    ``controls`` and ``patients`` are arrays of scan identifiers;
    ``is_medicated`` is a 0/1 array aligned with ``patients`` (1 = received
    antipsychotic medication before scanning).
    """
    controls = extract_subjects(sio.loadmat(config.SUBJECTS_CTRL))
    patients = extract_subjects(sio.loadmat(config.SUBJECTS_SZ))
    is_med = sio.loadmat(config.IS_MED)["is_med"][0]
    return controls, patients, is_med


def _find_results_file(directory, stem, what):
    """Locate one results file from the invariant middle of its name.

    The MATLAB pipeline varies both ends of these filenames. The prefix
    depends on whether the grid search ran::

        GRIDON = false  ->  no_grid_subjectRefinement_...   no_grid_GroupResults_...
        GRIDON = true   ->  subjectRefinement_...           gridGroupResults_...

    and an atlas token may be appended (``_atlas-416``, ``_atlas-116``). Only
    the middle — ``...group-ctrl_tau-1_med-0`` — is stable, so that is what is
    matched. An exact name is preferred when present; otherwise a single
    loose match is accepted, and several matches are reported rather than
    guessed between.
    """
    directory = Path(directory)

    for name in (f"no_grid_{stem}.mat", f"{stem}.mat"):
        path = directory / name
        if path.exists():
            return path

    matches = sorted(directory.glob(f"*{stem}*.mat"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No {what} matching *{stem}*.mat in {directory}.\n"
            "The prefix varies with GRIDON (no_grid_/grid) and an atlas token "
            "may be appended; both are handled, so check that the directory "
            "holds the .mat files themselves rather than their parent."
        )
    raise RuntimeError(
        f"Several files match *{stem}*.mat in {directory}: "
        f"{[m.name for m in matches]}.\nThis usually means grid and no-grid "
        "runs are mixed in one directory. Point at a directory holding one "
        "fit per group, or move the run you are not using."
    )


def _find_subject_results(directory, group, tau, med):
    """Locate one per-subject results file."""
    stem = _SUBJECT_STEM.format(group=group, tau=tau, med=med)
    return _find_results_file(directory, stem, "per-subject fit")


def load_subject_results(tau=None, results_dir=None):
    """Load the per-subject model fits for all three groups.

    Each returned struct contains at least ``Ceff_Subj`` (generative effective
    connectivity), ``hierarchicallevels_full`` (trophic levels, NaN-padded to
    the full parcellation) and ``trophiccoherence`` (directedness).

    Returns
    -------
    dict with keys ``ctrl``, ``sz0``, ``sz1``
    """
    tau = config.TAU if tau is None else tau
    directory = config.RESULTS_DIR if results_dir is None else Path(results_dir)
    spec = {"ctrl": ("ctrl", 0), "sz0": ("sz", 0), "sz1": ("sz", 1)}
    return {label: sio.loadmat(_find_subject_results(directory, group, tau, med))
            for label, (group, med) in spec.items()}


def trophic_coherence(result):
    """Directedness for every subject in a group, as a 1-D array.

    The pipeline has written this as ``(n_subjects, 1)`` in some runs and
    ``(1, n_subjects)`` in others, so it is flattened rather than indexed
    positionally. Indexing with ``[:, 0]`` silently returns a single value on
    the transposed layout.
    """
    return np.asarray(result["trophiccoherence"]).ravel()


def load_group_results(tau=None):
    """Load the group-level (pre-refinement) model fits."""
    tau = config.TAU if tau is None else tau
    spec = {"ctrl": ("ctrl", 0), "sz0": ("sz", 0), "sz1": ("sz", 1)}
    out = {}
    for label, (group, med) in spec.items():
        stem = _GROUP_STEM.format(group=group, tau=tau, med=med)
        try:
            out[label] = sio.loadmat(
                _find_results_file(config.GROUP_RESULTS_DIR, stem, "group fit"))
        except FileNotFoundError:
            pass        # group-level fits are optional
    return out


def n_fitted_parcels(result):
    """Number of parcels a group fit was run on.

    Newer pipeline versions save ``indexN``; older fits do not, so fall back to the size of ``Ceff_best``, which is
    defined on the same subset in every version.
    """
    if "indexN" in result:
        return int(np.asarray(result["indexN"]).size)
    return int(np.asarray(result["Ceff_best"]).shape[0])


def is_grid_run(result):
    """True when the group fit came from a grid search over global coupling."""
    gvals = np.asarray(result.get("Gvals", [])).ravel()
    return gvals.size > 1


def fit_curves(result):
    """Per-iteration error curves of the retained solution.

    ``fit_group_level.m`` records ``errFC(iter)`` and ``errCOVtau(iter)`` at
    every iteration, so these are curves rather than scalars. Their length is
    set by early stopping and so varies between runs and grid points.

    Returns
    -------
    dict with 1-D arrays ``errFC``, ``errCOVtau`` and their sum ``total``.
    """
    err_fc = np.asarray(result["errFC"], dtype=float).ravel()
    err_cov = np.asarray(result["errCOVtau"], dtype=float).ravel()
    n = min(err_fc.size, err_cov.size)
    return {"errFC": err_fc[:n], "errCOVtau": err_cov[:n],
            "total": err_fc[:n] + err_cov[:n]}


def _unwrap_struct_field(item, field):
    """Pull one field out of a MATLAB struct however scipy has nested it."""
    obj = item
    for _ in range(4):
        if isinstance(obj, np.ndarray) and obj.dtype.names and field in obj.dtype.names:
            obj = obj[field]
            while isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
                obj = obj.reshape(-1)[0]
            return np.asarray(obj, dtype=float).ravel()
        if isinstance(obj, np.ndarray) and obj.size:
            obj = obj.reshape(-1)[0]
        else:
            break
    raise KeyError(f"{field!r} not found in the FitStore entry")


def grid_curves(result, field="errFC"):
    """Error curves for every grid point, as a padded ``(n_g, n_iter)`` array.

    ``FitStore`` holds one struct per grid point, each carrying the full
    iteration history. Early stopping makes those histories different lengths,
    so shorter ones are padded with NaN to the longest.
    """
    store = np.asarray(result["FitStore"]).ravel()
    curves = [_unwrap_struct_field(item, field) for item in store]
    width = max(c.size for c in curves)
    out = np.full((len(curves), width), np.nan)
    for i, c in enumerate(curves):
        out[i, :c.size] = c
    return out


def group_fc_from_timeseries(subjects, index_n=None, trim=10, ts_dir=None,
                             suffix=TS_SUFFIX, n_parcels=None):
    """Group-average empirical FC matrix, computed from the timeseries.

    Same definition as ``matlab/compute_group_empirical.m`` (mean-remove,
    linear detrend, drop ``trim`` samples from each end, Pearson). Useful for
    comparing against ``FCsim`` when the precomputed ``FC_*.mat`` for this run
    does not exist yet.

    ``index_n`` restricts to the fitted parcels; pass the ``indexN`` stored in
    the group results (it is 1-based, and is converted here).
    """
    n_parcels = config.N_PARCELS if n_parcels is None else n_parcels
    ts_dir = config.TS_DIR if ts_dir is None else ts_dir

    if index_n is not None:
        idx = np.asarray(index_n, dtype=int).ravel()
        if idx.min() >= 1 and idx.max() >= n_parcels:
            idx = idx - 1                      # 1-based, as MATLAB writes it
        elif idx.min() >= 1 and idx.max() == n_parcels - 1:
            idx = idx - 1
        elif idx.min() == 1:
            idx = idx - 1
    else:
        idx = None

    acc, n_used = None, 0
    for subject in subjects:
        ts = np.asarray(sio.loadmat(f"{ts_dir}/{subject}{suffix}")["ts"], dtype=float)
        if ts.shape[0] != n_parcels and ts.shape[1] == n_parcels:
            ts = ts.T
        ts = _detrend_rows(ts)
        if trim:
            ts = ts[:, trim:-trim]
        if idx is not None:
            ts = ts[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            fc = np.corrcoef(ts)
        acc = fc if acc is None else acc + fc
        n_used += 1
    return acc / n_used


def load_timeseries(subjects, ts_dir=None, suffix=TS_SUFFIX):
    """Load parcellated BOLD timeseries for a list of scan identifiers.

    Returns an array of shape ``(n_subjects, n_timepoints, n_parcels)``.
    """
    ts_dir = config.TS_DIR if ts_dir is None else ts_dir
    return np.asarray([sio.loadmat(f"{ts_dir}/{s}{suffix}")["ts"] for s in subjects])


def load_empirical_fc():
    """Load the empirical functional connectivity and covariance per group.

    Each ``.mat`` written by ``matlab/compute_group_empirical.m`` contains

    ===================  ==========================  ==========================
    ``FCPB``             (n_subjects, 371, 371)      per-subject empirical FC
    ``COVtauPB``         (n_subjects, 371, 371)      per-subject shifted covariance
    ``FCemp_group``      (371, 371)                  group-average FC
    ``COVtauemp_group``  (371, 371)                  group-average shifted covariance
    ===================  ==========================  ==========================

    Note that these matrices cover only the 371 parcels the model was fitted
    in, not the full 416. Use :func:`expand_to_full_parcellation` to place
    them back into the full parcellation.

    Returns
    -------
    dict keyed by ``ctrl`` / ``sz0`` / ``sz1``, each a dict of the above.
    """
    tau = config.TAU
    directory = Path(config.DERIVED_DIR)
    spec = {"ctrl": ("ctrl", 0, "FC_ctrl.mat"),
            "sz0": ("sz", 0, "FC_sz-0.mat"),
            "sz1": ("sz", 1, "FC_sz-1.mat")}

    out = {}
    for label, (group, med, legacy) in spec.items():
        # Two naming conventions are in use: the short legacy form, and the
        # longer one the re-fit writes. Try both, then fall back to a glob.
        candidates = [legacy, f"FC_group-{group}_tau-{tau}_med-{med}.mat"]
        path = next((directory / c for c in candidates if (directory / c).exists()), None)
        if path is None:
            loose = sorted(directory.glob(f"FC_group-{group}_tau-*_med-{med}*.mat"))
            path = loose[0] if len(loose) == 1 else None
        if path is None:
            raise FileNotFoundError(
                f"No empirical connectivity for {label} in {directory}.\n"
                f"Looked for {candidates}. Produce it with "
                "matlab/compute_group_empirical.m, or point KASP_DERIVED at the "
                "directory matching your current fit."
            )
        mat = sio.loadmat(path)
        out[label] = {k: mat[k] for k in ("FCPB", "COVtauPB",
                                          "FCemp_group", "COVtauemp_group")
                      if k in mat}
    return out


def _detrend_rows(x):
    """Remove the mean and a linear trend from each row of ``(..., n_time)``."""
    t = np.arange(x.shape[-1], dtype=float)
    t -= t.mean()
    slope = (x * t).sum(-1, keepdims=True) / (t ** 2).sum()
    return x - x.mean(-1, keepdims=True) - slope * t


def regional_mean_fc(subjects, trim=10, ts_dir=None, suffix=TS_SUFFIX,
                     n_parcels=None):
    """Mean functional connectivity of each parcel, computed from timeseries.

    Reproduces ``matlab/compute_group_empirical.m``: every parcel's timeseries
    is mean-removed and linearly detrended, ``trim`` samples are dropped from
    each end, and FC is the Pearson correlation matrix. The regional value is
    the mean of each row of that matrix.

    Use this when the precomputed ``FC_*.mat`` files do not match the
    parcellation of the current fit — they are tied to the run that produced
    them, and a re-fit with different coverage exclusions invalidates them.

    Returns an array of shape ``(n_subjects, n_parcels)``. Parcels that are
    constant or missing give NaN, which the coverage mask removes downstream.
    """
    n_parcels = config.N_PARCELS if n_parcels is None else n_parcels
    ts_dir = config.TS_DIR if ts_dir is None else ts_dir

    out = np.full((len(subjects), n_parcels), np.nan, dtype=float)
    for i, subject in enumerate(subjects):
        ts = np.asarray(sio.loadmat(f"{ts_dir}/{subject}{suffix}")["ts"], dtype=float)
        if ts.shape[0] != n_parcels and ts.shape[1] == n_parcels:
            ts = ts.T                       # stored time-by-parcels
        if ts.shape[0] != n_parcels:
            raise RuntimeError(
                f"Timeseries for one scan has shape {ts.shape}; expected one "
                f"axis of length {n_parcels}."
            )
        ts = _detrend_rows(ts)
        if trim:
            ts = ts[:, trim:-trim]
        with np.errstate(invalid="ignore", divide="ignore"):
            fc = np.corrcoef(ts)
        out[i] = np.nanmean(fc, axis=1)
    return out


def expand_to_full_parcellation(values, idx, n_parcels=None, fill=0.0):
    """Place values defined on the fitted parcels back into the full parcellation.

    Parameters
    ----------
    values : (n_fitted,) array_like
    idx : (n_fitted,) array_like of int
        Positions of the fitted parcels within the full parcellation.
    fill : float
        Value given to parcels that were not fitted. The original analysis
        used 0 here, which matters downstream: ``neuromaps.stats.compare_images``
        drops exact zeros by default, which is how those parcels end up being
        excluded from the statistics.
    """
    n_parcels = config.N_PARCELS if n_parcels is None else n_parcels
    out = np.full(n_parcels, fill, dtype=float)
    out[np.asarray(idx)] = np.asarray(values)
    return out


def _load_tsnr_consolidated(path, subjects, n_parcels=None):
    """Select rows of a consolidated tSNR table by scan identifier.

    The re-fitted pipeline writes one ``parcel_tsnr_*.mat`` for the whole
    cohort — ``tsnr`` of shape ``(n_scans, n_parcels)`` alongside a
    ``subjects`` cell array — rather than a JSON sidecar per scan.
    """
    n_parcels = config.N_PARCELS if n_parcels is None else n_parcels
    mat = sio.loadmat(path)
    table = np.asarray(mat["tsnr"], dtype=float)

    ids = extract_subjects(mat, key="subjects")
    if table.shape[0] != len(ids) and table.shape[1] == len(ids):
        table = table.T          # stored parcels-by-scans
    if table.shape[0] != len(ids):
        raise RuntimeError(
            f"{path.name}: 'tsnr' has shape {table.shape} but there are "
            f"{len(ids)} entries in 'subjects'; cannot match rows to scans."
        )
    if table.shape[1] != n_parcels:
        raise RuntimeError(
            f"{path.name}: 'tsnr' has {table.shape[1]} parcels, expected "
            f"{n_parcels}. Check that this file matches the parcellation "
            "the model was fitted in."
        )

    index = {str(s): i for i, s in enumerate(ids)}
    missing = [s for s in subjects if str(s) not in index]
    if missing:
        raise KeyError(
            f"{len(missing)} of {len(subjects)} scans are absent from "
            f"{path.name} (it holds {len(ids)}). The tSNR file and the model "
            "fits appear to come from different runs."
        )
    return table[[index[str(s)] for s in subjects]]


def load_tsnr(subjects, tsnr_dir=None, suffix=TS_SUFFIX):
    """Regional temporal signal-to-noise ratio for a list of scan identifiers.

    Two layouts are supported, chosen automatically:

    * a consolidated ``parcel_tsnr*.mat`` covering the whole cohort, as the
      re-fitted pipeline writes;
    * one JSON sidecar per scan carrying ``MeanAtlasTSNR``, as the original
      pipeline wrote.

    Note that neither necessarily sits beside the ``.mat`` timeseries — in
    both projects they are one level above — so this uses ``config.TSNR_DIR``
    rather than ``config.TS_DIR``.

    Returns an array of shape ``(n_subjects, n_parcels)``, ordered to match
    ``subjects``.
    """
    tsnr_dir = Path(config.TSNR_DIR if tsnr_dir is None else tsnr_dir)

    consolidated = sorted(tsnr_dir.glob("parcel_tsnr*.mat"))
    if len(consolidated) > 1:
        raise RuntimeError(
            f"Several consolidated tSNR files in {tsnr_dir}: "
            f"{[c.name for c in consolidated]}. Leave only the one matching "
            "your fit, or pass tsnr_dir explicitly."
        )
    if consolidated:
        return _load_tsnr_consolidated(consolidated[0], subjects)

    json_suffix = suffix.replace(".mat", ".json")
    out = []
    for s in subjects:
        path = tsnr_dir / f"{s}{json_suffix}"
        if not path.exists():
            raise FileNotFoundError(
                f"No tSNR for one of the scans in {tsnr_dir}: neither a "
                f"consolidated parcel_tsnr*.mat nor a sidecar {path.name}."
            )
        with open(path) as fh:
            out.append(json.load(fh)["MeanAtlasTSNR"])
    return np.asarray(out, dtype=float)


def load_ignore_nodes(path=None, one_based="auto", n_parcels=None):
    """Excluded parcel indices from an ``IgnoreNodes*.mat``, returned 0-based.

    The MATLAB pipeline writes these in MATLAB's 1-based indexing. Using them
    unshifted as Python indices selects the wrong parcels — the error that
    affected the original supplementary figure — so the conversion happens
    here once, explicitly.

    Parameters
    ----------
    path : path-like, optional
        Defaults to the first ``IgnoreNodes.mat`` found under ``TS_DIR``.
    one_based : {"auto", True, False}
        ``"auto"`` infers the convention: an index equal to ``n_parcels``
        cannot be 0-based, and an index of 0 cannot be 1-based.

    Returns
    -------
    numpy.ndarray of int, sorted, 0-based
    """
    n_parcels = config.N_PARCELS if n_parcels is None else n_parcels

    if path is None:
        candidates = sorted(Path(config.TS_DIR).glob("IgnoreNodes.mat"))
        if not candidates:
            raise FileNotFoundError(
                f"No IgnoreNodes.mat in {config.TS_DIR}; pass an explicit path."
            )
        path = candidates[0]

    mat = sio.loadmat(path)
    key = "IgnoreNodes" if "IgnoreNodes" in mat else next(
        k for k in mat if not k.startswith("__"))
    idx = np.asarray(mat[key], dtype=float).ravel().astype(int)

    if one_based == "auto":
        if idx.min() == 0:
            one_based = False
        elif idx.max() >= n_parcels:
            one_based = True
        else:
            one_based = True     # the pipeline's convention
    return np.unique(idx - 1 if one_based else idx)


def load_clinical(filename="clinical_data.csv"):
    """Load the de-identified clinical and demographic table.

    The original analysis pasted these columns in from a spreadsheet at the
    console (``pandas.read_clipboard``). That is neither reproducible nor
    safe to distribute, so the table is now read from a CSV placed in
    ``CLINICAL_DIR``. The file is deliberately excluded from version control.

    Expected columns (see ``data/clinical/clinical_data_TEMPLATE.csv``):

    ==================  ====================================================
    ``ID1``             scan identifier, matching the subject list files
    ``Group``           ``Control`` or ``SZ``
    ``Is_Medicated``    0/1, patients only
    ``Gender``          0 = man, 1 = woman
    ``Age``             years
    ``Positive``        PANSS positive subscale
    ``Negative``        PANSS negative subscale
    ``General``         PANSS general subscale
    ``Total``           PANSS total
    ``CGI_Score``       Clinical Global Impression, integer
    ``Symptoms``        GAF symptom score
    ``Functioning``     GAF functioning score
    ==================  ====================================================

    No direct identifiers, dates or free-text fields belong in this file.
    """
    path = config.CLINICAL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Clinical table not found at {path}.\n"
            "This file contains participant data and is not distributed with "
            "the repository — see the 'Data availability' section of README.md. "
            "The expected column layout is documented in "
            "data/clinical/clinical_data_TEMPLATE.csv."
        )
    df = pd.read_csv(path)
    expected = {"ID1", "Group"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Clinical table is missing required columns: {sorted(missing)}")
    return df
