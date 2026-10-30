"""
Parcellation, gradient and geometry helpers.

The analysis uses a combined parcellation of 400 cortical parcels
(Schaefer et al., 2018, 7-network solution) and 16 subcortical parcels
(Tian et al., 2020, Melbourne Subcortical Atlas S1), giving 416 regions.
Subcortical parcels occupy the **first** 16 rows/columns of every matrix, so
cortex-only maps are obtained by slicing ``x[16:]``.

Note on provenance
------------------
``fc_gradient`` and ``valid_cortical_index`` reproduce code that was present
in the original analysis scripts. ``network_assignment`` and
``parcel_distance_matrix`` replace variables (``com``, ``dist2``) that the
original scripts inherited from an interactive session and that are not
present in the project's source files; they are reconstructed here from the
published atlas definitions.
"""

import numpy as np

from . import config

__all__ = [
    "YEO7_NETWORKS",
    "NETWORK_LABELS",
    "fc_gradient",
    "network_assignment",
    "valid_cortical_index",
    "distance_matrix_euclidean",
    "parcel_distance_matrix",
]

#: Yeo/Schaefer 7-network names, in the canonical order used by the atlas.
YEO7_NETWORKS = (
    "Vis",        # 0  visual
    "SomMot",     # 1  somatomotor
    "DorsAttn",   # 2  dorsal attention
    "SalVentAttn",  # 3  salience / ventral attention
    "Limbic",     # 4  limbic  (excluded from the analyses; see Limitations)
    "Cont",       # 5  frontoparietal control
    "Default",    # 6  default mode
)

#: Short labels used in the figures, keyed by the integer code in
#: ``network_assignment``. Index 7 denotes subcortex.
NETWORK_LABELS = {0: "Vis", 1: "SM", 2: "DA", 3: "SA", 4: "Lim", 5: "FP", 6: "DMN", 7: "Sub"}


def fc_gradient(nroi=None, desc="fcgradient01"):
    """Fetch and parcellate a functional connectivity gradient.

    Returns the Margulies et al. (2016) gradient, parcellated with the
    Schaefer 7-network solution, together with the parcellation object needed
    for spin tests. ``desc="fcgradient01"`` is the principal gradient used
    throughout the paper; ``"fcgradient02"`` is the second gradient.

    Returns
    -------
    fc_grad_parc : (nroi,) numpy.ndarray
    parcellation : tuple of GiftiImage
    """
    from neuromaps.datasets import fetch_annotation
    from neuromaps.images import dlabel_to_gifti
    from neuromaps.parcellate import Parcellater
    from netneurotools import datasets as nntdata

    nroi = config.N_CORTEX if nroi is None else nroi
    fc_grad = fetch_annotation(
        source="margulies2016", desc=desc, space="fsLR", den="32k"
    )
    schaefer = nntdata.fetch_schaefer2018("fslr32k")[f"{nroi}Parcels7Networks"]
    parcellation = dlabel_to_gifti(schaefer)
    parc = Parcellater(parcellation, "fsLR")
    fc_grad_parc = parc.fit_transform(fc_grad, "fsLR")
    return fc_grad_parc, parcellation


def _schaefer_label_names(schaefer, dlabel_to_gifti):
    """Parcel names from the Schaefer atlas, in parcel order.

    Tries the GIFTI label tables first, since ``dlabel_to_gifti`` is the same
    conversion :func:`fc_gradient` already relies on. Falls back to reading the
    CIFTI label axis directly if that conversion does not carry the table.
    """
    names = []
    try:
        for gifti in dlabel_to_gifti(schaefer):
            labels = gifti.labeltable.get_labels_as_dict()
            names.extend(labels[key] for key in sorted(labels) if key != 0)
    except Exception:  # noqa: BLE001 - fall through to the CIFTI reader
        names = []

    if names:
        return names

    import nibabel as nib

    paths = [schaefer] if isinstance(schaefer, (str, bytes)) else list(schaefer)
    for path in paths:
        img = nib.load(str(path))
        axis = img.header.get_axis(0)
        for label_map in np.atleast_1d(axis.label):
            for key in sorted(label_map):
                if key == 0:  # background / medial wall
                    continue
                entry = label_map[key]
                names.append(entry[0] if isinstance(entry, (tuple, list)) else entry)
    return names


def network_assignment(nroi=None, n_subcortex=None):
    """Integer network code for every parcel in the 416-region parcellation.

    Cortical parcels take values 0-6 following :data:`YEO7_NETWORKS`;
    subcortical parcels take the value 7. The ordering matches the matrices
    produced by the MATLAB pipeline (subcortex first, then cortex).

    RECONSTRUCTED: the original scripts used a variable ``com`` that is not
    defined in any surviving source file. This function rebuilds the same
    quantity by parsing network membership out of the Schaefer label table.
    """
    from neuromaps.images import dlabel_to_gifti
    from netneurotools import datasets as nntdata

    nroi = config.N_CORTEX if nroi is None else nroi
    n_subcortex = config.N_SUBCORTEX if n_subcortex is None else n_subcortex

    # NOTE: fetch_schaefer2018('fslr32k') returns the path of a single
    # .dlabel.nii covering both hemispheres — a string, not a pair of
    # per-hemisphere files. Iterating it directly walks its characters, which
    # is why an earlier version of this function tried to open "/".
    schaefer = nntdata.fetch_schaefer2018("fslr32k")[f"{nroi}Parcels7Networks"]

    names = _schaefer_label_names(schaefer, dlabel_to_gifti)

    # De-duplicate while preserving order (each hemisphere carries the same
    # label table, so names repeat).
    seen, ordered = set(), []
    for name in names:
        name = str(name).strip()
        if not name or name in {"???", "unknown", "Unknown"}:
            continue
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    if len(ordered) != nroi:
        raise RuntimeError(
            f"Expected {nroi} Schaefer labels, parsed {len(ordered)}. "
            "The atlas file layout may have changed; supply the network "
            "assignment manually instead."
        )

    cortex = np.full(nroi, -1, dtype=int)
    for i, name in enumerate(ordered):
        for code, network in enumerate(YEO7_NETWORKS):
            if network.lower() in name.lower():
                cortex[i] = code
                break
    if (cortex < 0).any():
        unmatched = [ordered[i] for i in np.flatnonzero(cortex < 0)][:5]
        raise RuntimeError(f"Could not assign a network to labels such as {unmatched}")

    subcortex = np.full(n_subcortex, 7, dtype=int)
    return np.concatenate([subcortex, cortex])


def valid_cortical_index(result, n_subcortex=None):
    """Boolean mask of cortical parcels that survived the coverage criterion.

    Parcels with less than 50% coverage were dropped during preprocessing and
    appear as NaN in ``hierarchicallevels_full``. This reproduces the ``VIDX``
    mask used in the original scripts.
    """
    n_subcortex = config.N_SUBCORTEX if n_subcortex is None else n_subcortex
    levels = result["hierarchicallevels_full"]
    return ~np.isnan(np.nanmean(levels, 0))[n_subcortex:]


def distance_matrix_euclidean(coords):
    """Euclidean distance matrix from an ``(n_parcels, 3)`` coordinate array."""
    import scipy.spatial

    return scipy.spatial.distance.cdist(coords, coords)


def parcel_distance_matrix(coords_file=None):
    """Distance matrix between parcel centroids.

    RECONSTRUCTED: the original spatial-autocorrelation script referenced a
    distance matrix ``dist2`` that was built in the interactive session and is
    not present in any surviving source file (the line that would have created
    it is commented out). This function rebuilds it from a CSV of parcel
    centroid coordinates.

    Parameters
    ----------
    coords_file : path-like, optional
        CSV with one row per parcel, in parcellation order, and columns
        ``x``, ``y``, ``z`` in millimetres. Defaults to
        ``<KASP_DATA>/atlas/parcel_centroids.csv``.

    Returns
    -------
    (n_parcels, n_parcels) numpy.ndarray
    """
    import pandas as pd

    coords_file = (
        config.BASE_DIR / "atlas" / "parcel_centroids.csv"
        if coords_file is None
        else coords_file
    )
    if not str(coords_file) or not __import__("os").path.exists(coords_file):
        raise FileNotFoundError(
            f"Parcel centroid coordinates not found at {coords_file}.\n"
            "The spatial-autocorrelation analysis needs an (n_parcels, 3) array "
            "of centroid coordinates in parcellation order. See 'Known gaps' in "
            "README.md."
        )
    coords = pd.read_csv(coords_file)
    missing = {"x", "y", "z"} - set(coords.columns)
    if missing:
        raise ValueError(f"Centroid file is missing columns: {sorted(missing)}")
    return distance_matrix_euclidean(coords[["x", "y", "z"]].to_numpy())
