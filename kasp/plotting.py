"""
Plotting helpers.

RECONSTRUCTED: the original analysis scripts called a helper named
``scatterplot2`` and used a per-parcel colour array named ``colors``. Neither
is defined in any surviving source file — they were carried in an interactive
session. They are reimplemented here from their call signatures and from the
published figures. They affect only the appearance of the figures, not any
reported statistic.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import parcellation

__all__ = [
    "YEO7_COLORS",
    "SUBCORTEX_COLOR",
    "set_style",
    "network_colors",
    "scatterplot2",
    "stat_text",
    "emptiest_corner",
]

#: Standard Yeo 7-network colours (Yeo et al., 2011), keyed by the integer
#: codes in :func:`kasp.parcellation.network_assignment`.
YEO7_COLORS = {
    0: "#781286",  # visual
    1: "#4682B4",  # somatomotor
    2: "#00760E",  # dorsal attention
    3: "#C43AFA",  # salience / ventral attention
    4: "#DCF8A4",  # limbic (excluded from analyses)
    5: "#E69422",  # frontoparietal control
    6: "#CD3E4E",  # default mode
}

#: Colour used for subcortical parcels.
SUBCORTEX_COLOR = "#8A8A8A"


def set_style(font_family="Helvetica", font_size=18):
    """Apply the figure style used throughout the manuscript."""
    import matplotlib

    plt.rc("axes.spines", top=False, right=False)
    sns.set_palette("pastel")
    matplotlib.rc("font", **{"family": font_family, "size": font_size})


def network_colors(com=None):
    """Per-parcel colour array for the 416-region parcellation.

    Parameters
    ----------
    com : array_like, optional
        Integer network assignment per parcel. Defaults to
        :func:`kasp.parcellation.network_assignment`.

    Returns
    -------
    (n_parcels,) numpy.ndarray of colour strings
    """
    com = parcellation.network_assignment() if com is None else np.asarray(com)
    return np.array([YEO7_COLORS.get(int(c), SUBCORTEX_COLOR) for c in com])


def scatterplot2(
    x,
    y,
    colors=None,
    ylim=None,
    xlim=None,
    xlabel="",
    ylabel="",
    regpl=False,
    ax=None,
    figsize=(6, 5),
    s=60,
    alpha=0.9,
    line_color="black",
    annotate=None,
    annotate_loc="upper left",
    edgecolor="white",
):
    """Scatter one parcel-wise map against another, optionally with a fit line.

    This is the helper behind the gradient scatter panels of Figures 2 and 3:
    one point per cortical parcel, coloured by network, with an ordinary least
    squares regression line.

    Parameters
    ----------
    x, y : array_like
        Parcel-wise values. Must be the same length.
    colors : array_like, optional
        Per-point colours, e.g. from :func:`network_colors`.
    ylim, xlim : tuple, optional
        Axis limits.
    xlabel, ylabel : str
        Axis labels.
    regpl : bool
        Draw an OLS regression line with a 95% confidence band.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. A new figure is created when omitted.
    annotate : str, optional
        Text placed in a corner, e.g. :func:`stat_text`.
    annotate_loc : str
        ``"upper left"``, ``"upper right"``, ``"lower left"``, ``"lower right"``,
        or ``"auto"`` for the corner with the fewest points.

    Returns
    -------
    matplotlib.axes.Axes
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    finite = np.isfinite(x) & np.isfinite(y)

    if regpl:
        sns.regplot(
            x=x[finite],
            y=y[finite],
            scatter=False,
            ax=ax,
            color=line_color,
            line_kws={"linewidth": 2},
        )

    point_colors = None if colors is None else np.asarray(colors)[finite]
    ax.scatter(
        x[finite],
        y[finite],
        c=point_colors,
        s=s,
        alpha=alpha,
        edgecolors=edgecolor,
        linewidths=0.5,
        zorder=3,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if annotate:
        loc = annotate_loc
        if loc == "auto":
            loc = emptiest_corner(ax, x[finite], y[finite])
        vert, horiz = loc.split()
        ax.text(
            0.97 if horiz == "right" else 0.03,
            0.97 if vert == "upper" else 0.03,
            annotate,
            transform=ax.transAxes,
            va="top" if vert == "upper" else "bottom",
            ha=horiz,
            multialignment=horiz,
            fontsize="small",
            zorder=5,
        )
    return ax


def stat_text(r, p, test="spin"):
    """``r = 0.468`` over ``p_spin = 0.003``, with a real subscript.

    No plus sign on positive r, and a typographic minus on negative ones. The
    subscript is mathtext, which stays text in PDF and SVG output.
    """
    r_str = f"{float(r):.3f}".replace("-", "\N{MINUS SIGN}")
    p_sym = f"$p_{{\\mathrm{{{test}}}}}$" if test else "$p$"
    return f"$r$ = {r_str}\n{p_sym} = {float(p):.3f}"


def emptiest_corner(ax, x, y, box=(0.38, 0.22)):
    """The axes corner holding the fewest points, for placing a label."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    u = (np.asarray(x) - x0) / (x1 - x0)
    v = (np.asarray(y) - y0) / (y1 - y0)
    counts = {}
    for vert in ("upper", "lower"):
        for horiz in ("right", "left"):
            in_u = u > 1 - box[0] if horiz == "right" else u < box[0]
            in_v = v > 1 - box[1] if vert == "upper" else v < box[1]
            counts[f"{vert} {horiz}"] = int((in_u & in_v).sum())
    return min(counts, key=counts.get)
