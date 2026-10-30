"""
Central path configuration.

All scripts and notebooks in this repository resolve their input and output
locations through this module, so no absolute paths appear anywhere else in
the codebase.

By default everything is read from the ``data/`` directory inside the
repository. Point the code at a different location by setting the
``KASP_DATA`` environment variable before running anything:

    export KASP_DATA=/path/to/your/data
    python scripts/02_hierarchy_analysis.py

The participant data itself is not distributed with this repository
(see README.md, "Data availability"). ``data/README.md`` documents the
directory layout and file formats the code expects.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Root locations
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path):
    """Read ``KEY=value`` lines from a ``.env`` file into the environment.

    This is what makes the settings reach a Jupyter kernel: environment
    variables exported in a terminal are only inherited by processes started
    from that same terminal, whereas this file is read whenever the package is
    imported, no matter how the kernel was launched.

    Variables already set in the real environment take precedence, so an
    explicit ``export`` still wins over the file.
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            # Unquoted value: a trailing " # comment" is a comment, as in a
            # shell. Inside quotes '#' is kept, since paths may contain it.
            value = value.split(" #", 1)[0].rstrip()
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(REPO_ROOT / ".env")


def _env(*names):
    """First of ``names`` that is set in the environment, else None."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


#: Root of the (non-distributed) data tree. Override with ``KASP_DATA``.
BASE_DIR = Path(_env("KASP_DATA") or REPO_ROOT / "data").expanduser().resolve()

#: String form with a trailing separator, for legacy ``basepath + 'file.mat'``
#: style concatenation in the analysis scripts.
BASE_PATH = str(BASE_DIR) + os.sep

# --------------------------------------------------------------------------
# Inputs
#
# The defaults follow the directory layout of the original project, so that
# pointing KASP_DATA at the project root is enough. Any individual directory
# can be redirected with its own environment variable if your layout differs.
# --------------------------------------------------------------------------

def _resolve(env_vars, *relative):
    """Directory from the first set environment variable, else ``BASE_DIR / relative``.

    ``env_vars`` may name several variables. The ``KASP_``-prefixed name is
    canonical; the bare name (``TS_DIR`` and friends) is accepted as a
    convenience alias, since it matches the attribute it sets.
    """
    override = _env(*env_vars)
    if override:
        return Path(override).expanduser().resolve()
    return BASE_DIR.joinpath(*relative)


#: Per-subject model fits produced by matlab/run_hopf_subject_refinement.m
RESULTS_DIR = _resolve(("KASP_RESULTS", "RESULTS_DIR"),
                       "results", "results_subject_refinement")

#: Group-level model fits produced by matlab/run_hopf_grid_group.m
GROUP_RESULTS_DIR = _resolve(("KASP_GROUP_RESULTS", "GROUP_RESULTS_DIR"),
                             "results", "results_group_only")

#: Parcellated BOLD timeseries, one .mat per subject.
TS_DIR = _resolve(("KASP_TIMESERIES", "TS_DIR"), "data", "tseries_2", "mat_files")

#: JSON sidecars carrying MeanAtlasTSNR. In the original project these sit one
#: level above the .mat files, so this is a separate setting.
TSNR_DIR = _resolve(("KASP_TSNR", "TSNR_DIR"), "data", "tseries_2")

#: fMRIPrep/HALFpipe confound regressor .tsv files, one per subject.
MOTION_DIR = _resolve(("KASP_MOTION", "MOTION_DIR"), "data", "MOTION")

#: Derived group-level matrices (empirical FC and covariance per group).
DERIVED_DIR = _resolve(("KASP_DERIVED", "DERIVED_DIR"), "data", "derived_data")

#: De-identified clinical table (never committed; see data/README.md).
CLINICAL_DIR = _resolve(("KASP_CLINICAL", "CLINICAL_DIR"), "data", "clinical")

# Subject-list files. These contain scan identifiers only, and are shared
# across atlases — when BASE_DIR points at an atlas-specific tree they usually
# live one level up, so each is individually overridable.
SUBJECTS_CTRL = _resolve(("KASP_SUBJECTS_CTRL", "SUBJECTS_CTRL"),
                         "subjects_ctrl_clean_good_coverage.mat")
SUBJECTS_SZ = _resolve(("KASP_SUBJECTS_SZ", "SUBJECTS_SZ"),
                       "subjects_sz_clean_good_coverage.mat")
IS_MED = _resolve(("KASP_IS_MED", "IS_MED"), "is_med.mat")

# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

#: Name for this run's outputs. Every run of the notebook writes into
#: ``runs/<RUN_TAG>/`` rather than into one shared folder, so running it on a
#: different atlas or a refit cannot overwrite the figures of another. Set it in
#: ``.env`` together with the inputs it describes, e.g. ``main``. The committed part of a run is its
#: ``stats.csv`` and ``manifest.json``; figures and tables stay local.
RUN_TAG = _env("KASP_RUN_TAG") or "untagged"

RUN_DIR = REPO_ROOT / "runs" / RUN_TAG
FIGURE_DIR = RUN_DIR / "figures"
TABLE_DIR = RUN_DIR / "tables"

# --------------------------------------------------------------------------
# Analysis constants (shared across scripts; see Methods in the manuscript)
# --------------------------------------------------------------------------

#: Random seed used for all permutation procedures.
SEED = 42

#: Number of resamples for permutation-based group comparisons.
N_PERM_GROUP = 100_000

#: Number of spatial (spin) permutations for cortical map comparisons.
N_PERM_SPIN = 1_000

# --------------------------------------------------------------------------
# Parcellation
#
# The analysis has been run with two atlases. Set KASP_ATLAS and the two
# counts together; everything downstream (filename suffixes, the gradient
# resolution, cortex/subcortex slicing) follows from them.
#
#   SchaeferTian416   400 cortical + 16 subcortical   (the published run)
#   SchaeferTian116   100 cortical + 16 subcortical
# --------------------------------------------------------------------------

#: Atlas name as it appears in the timeseries filenames.
ATLAS = _env("KASP_ATLAS") or "SchaeferTian416"

#: Number of cortical parcels (Schaefer 2018, 7-network solution).
N_CORTEX = int(_env("KASP_N_CORTEX") or 400)

#: Number of subcortical parcels (Melbourne Subcortical Atlas, S1).
N_SUBCORTEX = int(_env("KASP_N_SUBCORTEX") or 16)

#: Total parcels. Subcortical parcels occupy the first N_SUBCORTEX rows,
#: so cortical maps are indexed as ``x[N_SUBCORTEX:]``.
N_PARCELS = N_CORTEX + N_SUBCORTEX

if ATLAS.endswith(str(N_PARCELS)) is False and ATLAS[-3:].isdigit():
    raise ValueError(
        f"KASP_ATLAS={ATLAS!r} does not match KASP_N_CORTEX + KASP_N_SUBCORTEX "
        f"= {N_PARCELS}. Set all three consistently."
    )

#: Lag (in TR) used for the time-shifted covariance.
TAU = 1

#: Optional combined cortical+subcortical ``.dlabel.nii``, used only for the
#: surface renderings. Leave unset: ``kasp.rendering`` then takes the Schaefer
#: dlabel that netneurotools fetches, which is the same file the network
#: assignment is parsed from, so the parcel ordering is guaranteed to agree.
#: Set it only to render with a different combined atlas, whose labels must
#: run 1..N_PARCELS in the pipeline's own order.
ATLAS_DLABEL = _env("KASP_ATLAS_DLABEL") or None


def ensure_output_dirs():
    """Create the figure and table output directories if they do not exist."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def describe():
    """Print the resolved paths and flag any that are missing."""
    print(f"run tag: {RUN_TAG}   ->  runs/{RUN_TAG}/")
    if RUN_TAG == "untagged":
        print("         set KASP_RUN_TAG in .env to name this run's outputs")
    print(f"atlas: {ATLAS}  ({N_CORTEX} cortical + {N_SUBCORTEX} subcortical "
          f"= {N_PARCELS} parcels)\n")
    entries = [
        ("BASE_DIR", BASE_DIR, "root"),
        ("SUBJECTS_CTRL", SUBJECTS_CTRL, "sections 1-9"),
        ("SUBJECTS_SZ", SUBJECTS_SZ, "sections 1-9"),
        ("IS_MED", IS_MED, "sections 1-9"),
        ("RESULTS_DIR", RESULTS_DIR, "sections 3-8"),
        ("GROUP_RESULTS_DIR", GROUP_RESULTS_DIR, "optional"),
        ("DERIVED_DIR", DERIVED_DIR, "section 8, regional FC"),
        ("TS_DIR", TS_DIR, "section 8, autocorrelation"),
        ("TSNR_DIR", TSNR_DIR, "section 8, tSNR"),
        ("MOTION_DIR", MOTION_DIR, "section 9, head motion"),
        ("CLINICAL_DIR", CLINICAL_DIR, "sections 2 and 7"),
    ]
    width = max(len(name) for name, _, _ in entries)
    missing = 0
    for name, path, used_for in entries:
        ok = path.exists()
        missing += not ok
        print(f"{name:<{width}}  [{'ok' if ok else 'MISSING':>7}]  {path}"
              f"{'' if ok else f'   <- needed for {used_for}'}")
    if missing:
        print(f"\n{missing} path(s) missing. Set KASP_DATA to the project root, or "
              "redirect individual directories with KASP_RESULTS, KASP_TIMESERIES, "
              "KASP_TSNR, KASP_MOTION, KASP_DERIVED, KASP_CLINICAL.")
    else:
        print("\nAll inputs found.")
