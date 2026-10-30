"""
Surface and subcortical renderings of per-parcel maps.

Every function here takes a **full-parcellation** vector — length
``config.N_PARCELS``, subcortex in the first ``config.N_SUBCORTEX`` entries —
and splits it itself. Passing the whole map rather than pre-sliced pieces is
deliberate: slicing the wrong way round is the mistake this parcellation
invites, and doing it in one place means it is either right everywhere or
wrong everywhere.

Parcels the model was not fitted in should be ``NaN``. They render as the
surface's own background colour rather than as a value at the bottom of the
scale, so an excluded parcel never reads as a real measurement.

The three plotting dependencies are heavy and only needed here, so they are
imported inside the functions. The notebook runs end to end without them; the
rendering cells are the only ones that fail, and they say what to install.

    pip install surfplot neuromaps subcortex-visualization

Cortical geometry
-----------------
Vertex-wise labels come from the Schaefer fsLR-32k ``.dlabel.nii`` that
netneurotools fetches — the same file :func:`kasp.parcellation.network_assignment`
parses, so the parcel ordering is guaranteed to agree with the network
assignment and with the model output. Set ``KASP_ATLAS_DLABEL`` to use a
combined cortical+subcortical dlabel instead; then its labels are taken to
run ``1 .. N_PARCELS`` in the pipeline's own order.
"""

from functools import lru_cache

import numpy as np

from . import config

__all__ = [
    "SUBCORTEX_REORDER",
    "cortex_vertex_map",
    "plot_cortex",
    "plot_subcortex",
    "render_map",
    "symmetric_range",
]


#: Reordering from the pipeline's subcortical order (Tian et al. 2020, as
#: distributed at github.com/yetianmed/subcortex) to the order
#: ``subcortex_visualization`` expects for Melbourne S1. Two pairs are
#: transposed within each hemisphere; everything else is already aligned.
#:
#: This was established empirically against the rendered figures, not derived
#: from either atlas's documentation, so it is worth re-checking if either
#: package changes its label table.
SUBCORTEX_REORDER = np.array([0, 1, 2, 3, 5, 4, 6, 7,
                              8, 9, 10, 11, 13, 12, 14, 15])

#: Hemisphere of each subcortical parcel, in the pipeline's order.
SUBCORTEX_HEMISPHERE = np.array(["R"] * 8 + ["L"] * 8)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _split(values):
    """Subcortical and cortical halves of a full-parcellation map."""
    values = np.asarray(values, dtype=float).ravel()
    if values.size != config.N_PARCELS:
        raise ValueError(
            f"expected a full-parcellation map of {config.N_PARCELS} values "
            f"(subcortex first), got {values.size}. If this is a cortex-only "
            f"map, pad it with NaN for the {config.N_SUBCORTEX} subcortical "
            f"parcels."
        )
    return values[:config.N_SUBCORTEX], values[config.N_SUBCORTEX:]


def symmetric_range(*maps, quantile=None):
    """A diverging colour range centred on zero, ignoring NaN.

    With ``quantile`` set (say 0.98), the range covers that quantile of
    ``|value|`` rather than the maximum, so one extreme parcel cannot flatten
    the rest of the map. Pass several maps to get a range shared across them.
    """
    stacked = np.concatenate([np.asarray(m, dtype=float).ravel() for m in maps])
    finite = np.abs(stacked[np.isfinite(stacked)])
    if finite.size == 0:
        raise ValueError("no finite values to scale from")
    v = float(np.quantile(finite, quantile) if quantile else finite.max())
    return (-v, v)


def shared_range(*maps, quantile=None):
    """A colour range covering every given map, ignoring NaN."""
    stacked = np.concatenate([np.asarray(m, dtype=float).ravel() for m in maps])
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        raise ValueError("no finite values to scale from")
    if quantile:
        lo, hi = np.quantile(finite, [1 - quantile, quantile])
    else:
        lo, hi = finite.min(), finite.max()
    return (float(lo), float(hi))


# --------------------------------------------------------------------------
# cortex
# --------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _cortical_labels(dlabel_path=None, nroi=None):
    """Per-vertex parcel label for each fsLR-32k hemisphere.

    Returns two ``(32492,)`` integer arrays, left then right, holding the
    one-based parcel label at each vertex and 0 outside the parcellation
    (the medial wall).
    """
    import nibabel as nib

    if dlabel_path is None:
        from netneurotools import datasets as nntdata
        nroi = config.N_CORTEX if nroi is None else nroi
        dlabel_path = nntdata.fetch_schaefer2018("fslr32k")[f"{nroi}Parcels7Networks"]

    cifti = nib.load(str(dlabel_path))
    labels = cifti.get_fdata().squeeze().astype(int)
    axis = cifti.header.get_axis(1)

    lh = np.zeros(32492, dtype=int)
    rh = np.zeros(32492, dtype=int)
    for name, sl, model in axis.iter_structures():
        if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
            lh[model.vertex] = labels[sl]
        elif name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            rh[model.vertex] = labels[sl]
    return lh, rh


def cortex_vertex_map(values, dlabel_path=None):
    """Project a full-parcellation map onto fsLR-32k vertices.

    Returns a ``(64984,)`` array, left hemisphere then right, with NaN at the
    medial wall and at any parcel whose value is NaN.
    """
    _, cortex = _split(values)

    dlabel_path = dlabel_path or getattr(config, "ATLAS_DLABEL", None)
    lh, rh = _cortical_labels(str(dlabel_path) if dlabel_path else None)

    if dlabel_path:
        # A combined atlas: its labels run 1..N_PARCELS over the whole map.
        lookup = np.concatenate([[np.nan], np.asarray(values, dtype=float)])
    else:
        # Schaefer only: label k is the k-th cortical parcel.
        lookup = np.concatenate([[np.nan], cortex])

    n_labels = int(max(lh.max(), rh.max()))
    if n_labels > lookup.size - 1:
        raise ValueError(
            f"the atlas has {n_labels} labels but only {lookup.size - 1} values "
            f"were supplied. Check KASP_ATLAS_DLABEL against KASP_ATLAS."
        )

    return np.concatenate([lookup[lh], lookup[rh]])


def plot_cortex(values, cmap="viridis", color_range=None, diverging=False,
                title=None, cbar_label="", views=("lateral", "medial"),
                zoom=1.5, dlabel_path=None, save=None, dpi=330, cbar=True,
                layout="grid", mirror_views=False, alpha=1.0, size=None,
                zero_transparent=True):
    """Render a full-parcellation map on the inflated fsLR surface.

    Parameters
    ----------
    values : (N_PARCELS,) array
        Subcortex first; the subcortical entries are ignored here.
    color_range : (vmin, vmax), optional
        Explicit range. Pass one computed with :func:`shared_range` across
        several maps to make panels comparable. Without it the range is
        symmetric about zero when ``diverging``, and ``(0, max)`` otherwise.
    save : path, optional
        Written only if given; nothing is saved by default.
    cbar : bool
        Draw surfplot's own colour bar. The composed manuscript figures
        (:mod:`kasp.figures`) turn it off and draw a vector one instead, so that
        its text stays editable.
    layout, mirror_views, size
        Passed to ``surfplot.Plot``: ``layout="row"`` with
        ``mirror_views=True`` gives the four views in one row, lateral-medial-
        medial-lateral. ``mirror_views`` is dropped on surfplot versions that
        do not have it.
    alpha, zero_transparent
        Passed to ``Plot.add_layer``. A categorical map whose codes include 0
        needs ``zero_transparent=False`` (or codes shifted away from 0).

    Returns
    -------
    matplotlib Figure
    """
    import inspect

    from surfplot import Plot
    from neuromaps.datasets import fetch_fslr

    vertex_values = cortex_vertex_map(values, dlabel_path=dlabel_path)

    if color_range is None:
        _, cortex = _split(values)
        finite = cortex[np.isfinite(cortex)]
        if finite.size == 0:
            raise ValueError("the cortical map is entirely NaN")
        color_range = (symmetric_range(cortex) if diverging
                       else (0.0, float(np.abs(finite).max())))

    lh, rh = fetch_fslr()["inflated"]
    plot_kwargs = {"views": list(views), "zoom": zoom, "layout": layout}
    if size is not None:
        plot_kwargs["size"] = size
    if mirror_views and "mirror_views" in inspect.signature(Plot).parameters:
        plot_kwargs["mirror_views"] = True
    plot = Plot(lh, rh, **plot_kwargs)
    plot.add_layer(vertex_values, cbar=cbar, cmap=cmap, alpha=alpha,
                   color_range=color_range, cbar_label=cbar_label,
                   zero_transparent=zero_transparent)

    fig = plot.build(colorbar=cbar) if _build_takes_colorbar(plot) else plot.build()
    if title:
        fig.suptitle(title)
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=dpi)
    return fig


def _build_takes_colorbar(plot):
    import inspect
    return "colorbar" in inspect.signature(plot.build).parameters


# --------------------------------------------------------------------------
# subcortex
# --------------------------------------------------------------------------

def subcortex_frame(values, atlas="Melbourne_S1"):
    """The long-format table ``plot_subcortical_data`` expects.

    Reorders the pipeline's subcortical parcels into the package's own order
    (see :data:`SUBCORTEX_REORDER`) and attaches region names and hemispheres.
    """
    import pandas as pd
    from subcortex_visualization.utils import get_atlas_regions

    subcortex, _ = _split(values)
    if subcortex.size != SUBCORTEX_REORDER.size:
        raise ValueError(
            f"the reordering is defined for {SUBCORTEX_REORDER.size} "
            f"subcortical parcels, but this atlas has {subcortex.size}"
        )

    regions = np.asarray(get_atlas_regions(atlas))
    regions = np.concatenate([regions, regions])

    return pd.DataFrame({"value": subcortex[SUBCORTEX_REORDER],
                         "region": regions,
                         "Hemisphere": SUBCORTEX_HEMISPHERE})


def plot_subcortex(values, cmap="viridis", color_range=None, diverging=False,
                   atlas="Melbourne_S1", hemisphere="both", title=None,
                   save=None, dpi=330):
    """Render the subcortical part of a full-parcellation map.

    Takes the same full-length vector as :func:`plot_cortex`, so the two can
    be called on the same array without slicing at the call site.
    """
    import matplotlib.pyplot as plt
    from subcortex_visualization.plotting import plot_subcortical_data

    frame = subcortex_frame(values, atlas=atlas)

    if color_range is None:
        subcortex, _ = _split(values)
        finite = subcortex[np.isfinite(subcortex)]
        if finite.size == 0:
            raise ValueError("the subcortical map is entirely NaN")
        color_range = (symmetric_range(subcortex) if diverging
                       else (0.0, float(np.abs(finite).max())))

    fig = plot_subcortical_data(subcortex_data=frame, atlas=atlas,
                                hemisphere=hemisphere, show_figure=False,
                                cmap=cmap, vmin=color_range[0],
                                vmax=color_range[1])
    if title:
        (fig or plt.gcf()).suptitle(title)
    if save:
        plt.savefig(save, bbox_inches="tight", dpi=dpi)
    return fig


# --------------------------------------------------------------------------
# both at once
# --------------------------------------------------------------------------

def render_map(values, cmap="viridis", color_range=None, diverging=False,
               title=None, cbar_label="", save_prefix=None, **kwargs):
    """Render cortex and subcortex from one full-parcellation map.

    Both panels share ``color_range``, so the two halves of the same map are
    on the same scale — which they are not if each is scaled to its own
    extremes, and the subcortical values in this parcellation have a visibly
    different spread from the cortical ones.

    Returns
    -------
    (cortex_figure, subcortex_figure)
    """
    if color_range is None:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("the map is entirely NaN")
        color_range = (symmetric_range(values) if diverging
                       else (0.0, float(np.abs(finite).max())))

    cortex_kwargs = {k: v for k, v in kwargs.items()
                     if k in {"views", "zoom", "dlabel_path", "dpi"}}
    sub_kwargs = {k: v for k, v in kwargs.items()
                  if k in {"atlas", "hemisphere", "dpi"}}

    fig_ctx = plot_cortex(
        values, cmap=cmap, color_range=color_range, title=title,
        cbar_label=cbar_label,
        save=f"{save_prefix}_cortex.png" if save_prefix else None,
        **cortex_kwargs)

    fig_sub = plot_subcortex(
        values, cmap=cmap, color_range=color_range, title=title,
        save=f"{save_prefix}_subcortex.png" if save_prefix else None,
        **sub_kwargs)

    return fig_ctx, fig_sub
