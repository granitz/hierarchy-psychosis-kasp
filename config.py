"""
Convenience shim.

The actual configuration lives in ``kasp/config.py`` — edit that file, or set
the ``KASP_DATA`` environment variable, to point the code at your data:

    export KASP_DATA=/path/to/your/data

This module simply re-exports it so that ``import config`` also works when
running scripts from the repository root.
"""

from kasp.config import *  # noqa: F401,F403
from kasp.config import (  # noqa: F401  (explicit re-export for linters)
    BASE_DIR,
    BASE_PATH,
    CLINICAL_DIR,
    DERIVED_DIR,
    FIGURE_DIR,
    GROUP_RESULTS_DIR,
    IS_MED,
    MOTION_DIR,
    N_CORTEX,
    N_PARCELS,
    N_PERM_GROUP,
    N_PERM_SPIN,
    ATLAS,
    ATLAS_DLABEL,
    N_SUBCORTEX,
    REPO_ROOT,
    RESULTS_DIR,
    RUN_DIR,
    RUN_TAG,
    SEED,
    SUBJECTS_CTRL,
    SUBJECTS_SZ,
    TABLE_DIR,
    TAU,
    TSNR_DIR,
    TS_DIR,
    describe,
    ensure_output_dirs,
)

if __name__ == "__main__":
    describe()
