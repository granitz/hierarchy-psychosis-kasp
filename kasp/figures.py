"""
The manuscript's figures, composed panel by panel and saved editable.

Figures 2 and 3 and Supplementary Figures 1 and 2 are assembled here from the
same quantities the notebook tests, so a figure can never show a different
analysis from the one reported.

Editability. Every figure is written as PDF and SVG with its text kept as
text (``pdf.fonttype = 42``, ``svg.fonttype = "none"``), so the font, size
and wording can be changed in Illustrator or Inkscape afterwards. Brain
renderings are embedded images (they come from a 3-D renderer); everything
around them -- titles, colour bars, axes, statistics -- is vector.

The renderings need ``surfplot``, ``neuromaps`` and ``subcortex-visualization``
(see :mod:`kasp.rendering`); the scatter and box panels need only matplotlib.
"""

import io
from pathlib import Path

import matplotlib as mpl
import matplotlib.ticker  # noqa: F401  (mpl.ticker)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch

from . import config, plotting

__all__ = [
    "GROUP_ORDER", "CONTRASTS", "POINT_ALPHA",
    "editable_style", "save", "group_title", "contrast_title",
    "render_cortex", "render_subcortex", "render_networks",
    "network_legend_handles", "network_figure",
    "figure_maps_and_scatter", "supplementary_figure_1",
    "supplementary_figure_2",
]

#: Column order of the submitted figures.
GROUP_ORDER = ("sz0", "sz1", "ctrl")
CONTRASTS = (("sz0", "ctrl"), ("sz1", "ctrl"), ("sz0", "sz1"))

#: Opacity of the scatter points, and of the network map that keys their
#: colours, so that the two read as the same colours.
POINT_ALPHA = 0.55

_GROUP_TITLE = {"sz0": r"FEP$_{\mathrm{unmed}}$",
                "sz1": r"FEP$_{\mathrm{med}}$",
                "ctrl": "Control"}

MINUS = "\N{MINUS SIGN}"


def group_title(g):
    return _GROUP_TITLE[g]


def contrast_title(a, b):
    return f"{_GROUP_TITLE[a]} {MINUS} {_GROUP_TITLE[b]}"


# --------------------------------------------------------------------------
# style and saving
# --------------------------------------------------------------------------

def _available(family):
    try:
        font_manager.findfont(font_manager.FontProperties(family=family),
                              fallback_to_default=False)
        return True
    except ValueError:
        return False


def editable_style(family=None, size=8):
    """Font and output settings for editable figures. Returns the family used.

    Uses the first of Helvetica Neue, Helvetica, Arial that is installed (or
    ``family``). Subscripts and italics go through mathtext set to the same
    family, so "p" and "spin" stay ordinary text in the saved file.
    """
    candidates = [family] if family else []
    candidates += ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans",
                   "DejaVu Sans"]
    chosen = next(f for f in candidates if f and _available(f))
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": [chosen, "DejaVu Sans"],
        "font.size": size, "axes.titlesize": size + 1, "axes.labelsize": size,
        "xtick.labelsize": size - 1, "ytick.labelsize": size - 1,
        "legend.fontsize": size - 1,
        "mathtext.fontset": "custom", "mathtext.rm": chosen,
        "mathtext.it": f"{chosen}:italic", "mathtext.bf": f"{chosen}:bold",
        "mathtext.default": "it",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })
    return chosen


def save(fig, path_stem, dpi=600):
    """Write ``<stem>.pdf`` and ``<stem>.svg`` (text editable) and ``<stem>.png``.

    Some font files (macOS ``.ttc`` collections) cannot be embedded as
    TrueType in PDF by every matplotlib version; the PDF then falls back to
    Type 3, which most editors still read as text. The SVG is unaffected.
    """
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    try:
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", dpi=dpi)
    except Exception as err:                        # noqa: BLE001
        print(f"PDF with TrueType text failed ({err}); writing Type 3 text instead.")
        with mpl.rc_context({"pdf.fonttype": 3}):
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", dpi=dpi)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    return [stem.with_suffix(s) for s in (".pdf", ".svg", ".png")]


# --------------------------------------------------------------------------
# renderings as images
# --------------------------------------------------------------------------

def _trim(img, pad=4):
    """Crop an RGB(A) image to its content (non-white, non-transparent)."""
    scale = 255.0 if img.dtype == np.uint8 else 1.0
    rgb = img[..., :3] / scale
    alpha = img[..., 3] / scale if img.shape[-1] == 4 else np.ones(img.shape[:2])
    content = (alpha > 0.02) & (rgb.min(axis=-1) < 0.985)
    rows = np.flatnonzero(content.any(axis=1))
    cols = np.flatnonzero(content.any(axis=0))
    if rows.size == 0:
        return img
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, img.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1]


def _fig_to_image(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = _trim(plt.imread(buf))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _drop_colorbars(fig):
    for ax in list(fig.axes):
        if ax.get_label() == "<colorbar>":
            ax.remove()
    for txt in list(fig.texts):
        txt.remove()


def render_cortex(values, cmap, color_range, dpi=300, **kwargs):
    """Cortex in the 2 x 2 layout (lateral above medial), no title or colour bar."""
    from . import rendering
    fig = rendering.plot_cortex(values, cmap=cmap, color_range=color_range,
                                cbar=False, title=None, **kwargs)
    _drop_colorbars(fig)
    return _fig_to_image(fig, dpi)


def render_subcortex(values, cmap, color_range, dpi=300):
    """Subcortex, both hemispheres, no title or colour bar."""
    from . import rendering
    fig = rendering.plot_subcortex(values, cmap=cmap, color_range=color_range,
                                   title=None)
    fig = fig or plt.gcf()
    _drop_colorbars(fig)
    # subcortex_visualization may draw its colour bar into a plain axes rather
    # than through fig.colorbar; remove long, thin axes that hold no region
    # outlines (the brain views are drawn as patches, a colour bar is not).
    for ax in list(fig.axes):
        pos = ax.get_position()
        thin = pos.height > 0 and not 1 / 6 < pos.width / pos.height < 6
        if thin and not ax.patches:
            ax.remove()
    return _fig_to_image(fig, dpi)


def _network_codes(com, valid):
    """Network code per parcel, NaN where the parcel was not fitted."""
    codes = np.asarray(com, dtype=float).copy()
    codes[~np.asarray(valid, dtype=bool)] = np.nan
    codes[codes >= len(plotting.YEO7_COLORS)] = np.nan     # subcortex
    return codes


def render_networks(com, valid, alpha=POINT_ALPHA, dpi=300):
    """The Schaefer 7-network assignment, four views in one row.

    Coloured exactly as the scatter points (:data:`plotting.YEO7_COLORS`,
    same opacity), restricted to the fitted parcels; the rest stay grey, as in
    the hierarchy maps.
    """
    from . import rendering
    codes = _network_codes(com, valid)
    n = len(plotting.YEO7_COLORS)
    cmap = ListedColormap([plotting.YEO7_COLORS[k] for k in range(n)])
    # Codes shifted to 1..n so that code 0 (visual) is not made transparent.
    fig = rendering.plot_cortex(codes + 1, cmap=cmap, color_range=(0.5, n + 0.5),
                                cbar=False, layout="row", mirror_views=True,
                                size=(1600, 300), zoom=1.25, alpha=alpha)
    _drop_colorbars(fig)
    return _fig_to_image(fig, dpi)


def network_legend_handles(com, valid, alpha=POINT_ALPHA):
    codes = _network_codes(com, valid)
    present = sorted({int(c) for c in codes[np.isfinite(codes)]})
    from .parcellation import NETWORK_LABELS
    return [Patch(facecolor=plotting.YEO7_COLORS[c], alpha=alpha,
                  edgecolor="none", label=NETWORK_LABELS[c]) for c in present]


def network_figure(net_img, handles, width=6.0):
    """The network map on its own, with its legend."""
    h = width * net_img.shape[0] / net_img.shape[1]
    fig = plt.figure(figsize=(width, h + 0.35))
    ax = fig.add_axes([0, 0.35 / (h + 0.35), 1, h / (h + 0.35)])
    ax.imshow(net_img)
    ax.axis("off")
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, handlelength=1.0, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.0))
    return fig


# --------------------------------------------------------------------------
# layout helpers (all sizes in inches, stacked from the top)
# --------------------------------------------------------------------------

class _Stack:
    """Places axes top-down in inches; converts to figure fractions at the end."""

    def __init__(self, width):
        self.width = width
        self.y = 0.0
        self.items = []          # (kind, payload, x, y_top, w, h)

    def add(self, kind, payload, x, w, h, advance=True):
        self.items.append((kind, payload, x, self.y, w, h))
        if advance:
            self.y += h

    def skip(self, h):
        self.y += h

    def build(self):
        height = self.y
        fig = plt.figure(figsize=(self.width, height))
        made = []
        for kind, payload, x, top, w, h in self.items:
            rect = [x / self.width, 1 - (top + h) / height,
                    w / self.width, h / height]
            if kind == "text":
                text, kw = payload
                fig.text(rect[0], rect[1] + rect[3] / 2, text, va="center", **kw)
                made.append(None)
            else:
                ax = fig.add_axes(rect)
                made.append((kind, payload, ax))
        return fig, made


def _heading(stack, letter, text, x=0.0, h=0.28):
    stack.add("text", (letter, {"fontsize": 11, "fontweight": "bold"}), x, 0.3, h,
              advance=False)
    stack.add("text", (text, {"fontsize": 11}), x + 0.2, 1.0, h)


#: Resolution at which renderings are embedded in the composed figures.
IMAGE_DPI = 450


def _image(ax, img, dpi=IMAGE_DPI):
    """Show a rendering, resampled to ``dpi`` at its printed size.

    Keeps the files small and the embedded image exactly as large as it is
    printed, so nothing is resampled again when the figure is saved.
    """
    from PIL import Image

    fig_w, fig_h = ax.figure.get_size_inches()
    pos = ax.get_position()
    width_px = max(int(pos.width * fig_w * dpi), 1)
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if arr.shape[1] > width_px:
        h_px = max(int(round(arr.shape[0] * width_px / arr.shape[1])), 1)
        arr = np.asarray(Image.fromarray(arr).resize((width_px, h_px),
                                                     Image.LANCZOS))
    ax.imshow(arr, interpolation="none")
    ax.set_axis_off()


def _hcolorbar(ax, cmap, color_range, label, ticks=None, label_left=True):
    sm = ScalarMappable(norm=Normalize(*color_range), cmap=cmap)
    cb = plt.colorbar(sm, cax=ax, orientation="horizontal")
    lo, hi = color_range
    if ticks is None:
        ticks = [lo, 0.0, hi] if lo < 0 < hi else [lo, (lo + hi) / 2, hi]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.2f}".replace("-", MINUS) if t else "0" for t in ticks])
    cb.outline.set_linewidth(0.5)
    ax.tick_params(length=2, width=0.5, labelsize=mpl.rcParams["font.size"] - 1)
    if label:
        if label_left:
            ax.text(-0.04, 0.5, label, transform=ax.transAxes, ha="right",
                    va="center", fontsize=mpl.rcParams["font.size"] - 0.5)
        else:
            cb.set_label(label)
    return cb


# --------------------------------------------------------------------------
# Figures 2 and 3
# --------------------------------------------------------------------------

def figure_maps_and_scatter(panels, gradient, colors, cbar, headings, ylabel,
                            stats_loc, net_img, net_handles,
                            xlabel="Functional gradient", width=7.2,
                            alpha=POINT_ALPHA, ylim=None):
    """Figure 2 or 3: renderings (A), scatters against the gradient (B).

    Parameters
    ----------
    panels : list of dict
        One per column, with ``title``, ``cortex`` and ``subcortex`` (images
        from :func:`render_cortex` / :func:`render_subcortex`), ``y`` (the
        cortical map, NaN where not fitted, length N_CORTEX), ``r`` and ``p``.
    gradient : (N_CORTEX,) array
    colors : (N_CORTEX,) array of colours per cortical parcel
    cbar : dict or list of dict
        ``{"cmap", "range", "label"}``; one dict = one shared colour bar under
        the row, a list = one under each column.
    headings : (str, str)
        Text of panels A and B.
    stats_loc : str
        Corner of the r / p_spin label in the scatters (see
        :func:`plotting.scatterplot2`).
    """
    left, right, gap = 0.45, 0.1, 0.18
    colw = (width - left - right - 2 * gap) / 3
    xs = [left + i * (colw + gap) for i in range(3)]
    st = _Stack(width)

    _heading(st, "A", headings[0])
    st.skip(0.02)
    for x, p in zip(xs, panels):
        st.add("text", (p["title"], {"fontsize": 10, "ha": "center"}),
               x + colw / 2, 0.01, 0.24, advance=False)
    st.skip(0.26)

    ctx_h = max(colw * p["cortex"].shape[0] / p["cortex"].shape[1] for p in panels)
    for x, p in zip(xs, panels):
        st.add("image", p["cortex"], x, colw, ctx_h, advance=False)
    st.skip(ctx_h + 0.04)

    sub_w = colw * 0.92
    sub_h = max(sub_w * p["subcortex"].shape[0] / p["subcortex"].shape[1]
                for p in panels)
    for x, p in zip(xs, panels):
        st.add("image", p["subcortex"], x + (colw - sub_w) / 2, sub_w, sub_h,
               advance=False)
    st.skip(sub_h + 0.12)

    if isinstance(cbar, dict):
        cw = 1.8
        st.add("cbar", cbar, left + (width - left - right - cw) / 2 + 0.4,
               cw, 0.08, advance=False)
    else:
        cw = colw * 0.62
        for x, c in zip(xs, cbar):
            st.add("cbar_below", c, x + (colw - cw) / 2, cw, 0.08, advance=False)
    st.skip(0.08 + (0.32 if isinstance(cbar, dict) else 0.5))

    _heading(st, "B", headings[1])
    st.skip(0.05)
    for x, p in zip(xs, panels):
        st.add("text", (p["title"], {"fontsize": 10, "ha": "center"}),
               x + colw / 2, 0.01, 0.24, advance=False)
    st.skip(0.28)
    sc_h = 1.6
    for i, (x, p) in enumerate(zip(xs, panels)):
        st.add("scatter", (i, p), x, colw, sc_h, advance=False)
    st.skip(sc_h + 0.42)

    net_w = width * 0.62
    net_h = net_w * net_img.shape[0] / net_img.shape[1]
    st.add("image", net_img, (width - net_w) / 2, net_w, net_h)
    st.add("legend", None, (width - net_w) / 2, net_w, 0.22)

    fig, made = st.build()

    finite = np.isfinite(gradient)
    ys = [p["y"][finite & np.isfinite(p["y"])] for p in panels]
    if ylim is None:
        lo = min(v.min() for v in ys)
        hi = max(v.max() for v in ys)
        pad = 0.06 * (hi - lo)
        ylim = (lo - pad, hi + pad)

    scatter_axes = []
    for item in made:
        if item is None:
            continue
        kind, payload, ax = item
        if kind == "image":
            _image(ax, payload)
        elif kind == "cbar":
            _hcolorbar(ax, payload["cmap"], payload["range"], payload["label"])
        elif kind == "cbar_below":
            _hcolorbar(ax, payload["cmap"], payload["range"], payload.get("label"),
                       label_left=False)
        elif kind == "scatter":
            i, p = payload
            ok = finite & np.isfinite(p["y"])
            plotting.scatterplot2(
                gradient[ok], p["y"][ok], np.asarray(colors)[ok], ylim=ylim,
                regpl=True, ax=ax, s=14, alpha=alpha, edgecolor="none",
                annotate=plotting.stat_text(p["r"], p["p"]),
                annotate_loc=stats_loc)
            for line in ax.get_lines():
                line.set_linewidth(1.2)
            ax.set_ylabel(ylabel if i == 0 else "")
            ax.set_xlabel(xlabel if i == 1 else "")
            if i:
                ax.tick_params(labelleft=False)
            scatter_axes.append(ax)
        elif kind == "legend":
            ax.set_axis_off()
            ax.legend(handles=net_handles, loc="center", ncol=len(net_handles),
                      frameon=False, handlelength=1.0, columnspacing=1.1,
                      handletextpad=0.4)
    return fig


# --------------------------------------------------------------------------
# Supplementary Figure 1
# --------------------------------------------------------------------------

#: Box colours of the submitted Supplementary Figure 1.
SUPP1_COLORS = {"sz0": "#EEEDF8", "sz1": "#BDB8F0", "ctrl": "#7C74E6"}


def supplementary_figure_1(curves, fit_fc, fit_cov, directedness, width=7.2,
                           curve_group="sz0", seed=0):
    """Model fit (A) and global directedness (B), all participants.

    curves : dict with ``errFC`` and ``errCOVtau`` of one group-level fit
    fit_fc, fit_cov, directedness : dict group -> per-participant values
    """
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(width, 2.5))
    ax_mse = fig.add_axes([0.07, 0.17, 0.22, 0.62])
    ax_box = fig.add_axes([0.37, 0.17, 0.27, 0.62])
    ax_dir = fig.add_axes([0.74, 0.17, 0.24, 0.62])
    fig.text(0.0, 0.95, "A", fontsize=11, fontweight="bold", va="center")
    fig.text(0.028, 0.95, "Model fit", fontsize=11, va="center")
    fig.text(0.68, 0.95, "B", fontsize=11, fontweight="bold", va="center")
    fig.text(0.708, 0.95, "Global hierarchy", fontsize=11, va="center")

    it = np.arange(curves["errFC"].size)
    ax_mse.plot(it, curves["errFC"], color="#BDBDBD", lw=1.4, label="FC")
    ax_mse.plot(it, curves["errCOVtau"], color="#4D4D4D", lw=1.4,
                label=r"COV$_{\tau}$")
    ax_mse.set_xlabel("Iterations")
    ax_mse.set_ylabel("MSE (emp, sim)")
    ax_mse.set_title(f"Group fit, {group_title(curve_group)}", fontsize=8)
    ax_mse.legend(frameon=False, loc="upper right", handlelength=1.2)

    width_box = 0.24
    for k, (measure, vals) in enumerate([("FC", fit_fc),
                                         (r"COV$_{\tau}$", fit_cov)]):
        for j, g in enumerate(GROUP_ORDER):
            pos = k + (j - 1) * (width_box + 0.03)
            v = np.asarray(vals[g], dtype=float)
            v = v[np.isfinite(v)]
            ax_box.boxplot(v, positions=[pos], widths=width_box,
                           patch_artist=True, showfliers=False,
                           boxprops={"facecolor": SUPP1_COLORS[g],
                                     "edgecolor": "#555555", "linewidth": 0.6},
                           medianprops={"color": "#333333", "linewidth": 0.8},
                           whiskerprops={"color": "#555555", "linewidth": 0.6},
                           capprops={"color": "#555555", "linewidth": 0.6})
            ax_box.scatter(pos + rng.uniform(-0.07, 0.07, v.size), v, s=3,
                           color="#333333", alpha=0.6, linewidths=0, zorder=3)
    ax_box.set_xticks([0, 1], ["FC", r"COV$_{\tau}$"])
    ax_box.set_xlim(-0.55, 1.55)
    ax_box.set_ylabel("corr (emp, sim)")
    ax_box.legend(handles=[Patch(facecolor=SUPP1_COLORS[g], edgecolor="#555555",
                                 linewidth=0.6, label=group_title(g))
                           for g in GROUP_ORDER],
                  frameon=False, loc="lower left", handlelength=1.0)

    # Raincloud: half violin, box, jittered points.
    for j, g in enumerate(GROUP_ORDER):
        v = np.asarray(directedness[g], dtype=float)
        v = v[np.isfinite(v)]
        parts = ax_dir.violinplot(v, positions=[j - 0.18], widths=0.32,
                                  showextrema=False)
        for body in parts["bodies"]:
            verts = body.get_paths()[0].vertices
            verts[:, 0] = np.minimum(verts[:, 0], j - 0.18)
            body.set_facecolor("#D9D9D9")
            body.set_edgecolor("#7F7F7F")
            body.set_linewidth(0.5)
            body.set_alpha(1)
        ax_dir.boxplot(v, positions=[j], widths=0.14, showfliers=False,
                       boxprops={"linewidth": 0.6}, medianprops={"color": "#333333",
                                                                 "linewidth": 0.8},
                       whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6})
        ax_dir.scatter(j + 0.2 + rng.uniform(-0.05, 0.05, v.size), v, s=3,
                       color="#7F7F7F", alpha=0.8, linewidths=0)
    ax_dir.set_xticks(range(3), [group_title(g) for g in GROUP_ORDER])
    ax_dir.set_xlim(-0.6, 2.5)
    ax_dir.set_ylabel("Directedness")
    ax_dir.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
    return fig


# --------------------------------------------------------------------------
# Supplementary Figure 2
# --------------------------------------------------------------------------

def supplementary_figure_2(rows, gradient, colors, net_img, net_handles,
                           width=7.2, alpha=POINT_ALPHA, stats_loc="auto"):
    """Simpler measures against the gradient, one row per measure.

    rows : list of dict
        ``{"letter", "heading", "ylabel", "panels"}``; ``panels`` is a list of
        three ``{"title", "y", "r", "p"}`` in column order (y: cortical map,
        NaN where not fitted).
    """
    left, right, gap = 0.55, 0.1, 0.28
    colw = (width - left - right - 2 * gap) / 3
    xs = [left + i * (colw + gap) for i in range(3)]
    st = _Stack(width)
    sc_h = 1.35
    for r_i, row in enumerate(rows):
        if r_i:
            st.skip(0.12)
        _heading(st, row["letter"], row["heading"])
        for x, p in zip(xs, row["panels"]):
            st.add("text", (p["title"], {"fontsize": 9, "ha": "center"}),
                   x + colw / 2, 0.01, 0.22, advance=False)
        st.skip(0.24)
        for i, (x, p) in enumerate(zip(xs, row["panels"])):
            st.add("scatter", (r_i, i, row, p), x, colw, sc_h, advance=False)
        st.skip(sc_h + (0.42 if r_i == len(rows) - 1 else 0.22))

    net_w = width * 0.62
    net_h = net_w * net_img.shape[0] / net_img.shape[1]
    st.add("image", net_img, (width - net_w) / 2, net_w, net_h)
    st.add("legend", None, (width - net_w) / 2, net_w, 0.22)
    fig, made = st.build()

    finite = np.isfinite(gradient)
    for item in made:
        if item is None:
            continue
        kind, payload, ax = item
        if kind == "image":
            _image(ax, payload)
        elif kind == "legend":
            ax.set_axis_off()
            ax.legend(handles=net_handles, loc="center", ncol=len(net_handles),
                      frameon=False, handlelength=1.0, columnspacing=1.1,
                      handletextpad=0.4)
        elif kind == "scatter":
            r_i, i, row, p = payload
            ys = [q["y"][finite & np.isfinite(q["y"])] for q in row["panels"]]
            lo, hi = min(v.min() for v in ys), max(v.max() for v in ys)
            pad = 0.06 * (hi - lo)
            ok = finite & np.isfinite(p["y"])
            plotting.scatterplot2(
                gradient[ok], p["y"][ok], np.asarray(colors)[ok],
                ylim=(lo - pad, hi + pad), regpl=True, ax=ax, s=10, alpha=alpha,
                edgecolor="none", annotate=plotting.stat_text(p["r"], p["p"]),
                annotate_loc=stats_loc)
            for line in ax.get_lines():
                line.set_linewidth(1.1)
            ax.set_ylabel(row["ylabel"] if i == 0 else "")
            last = r_i == len(rows) - 1
            ax.set_xlabel("Functional gradient" if last and i == 1 else "")
            if i:
                ax.tick_params(labelleft=False)
    return fig
