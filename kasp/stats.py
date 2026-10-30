"""
Statistical helpers shared across the analysis scripts.

Contains three groups of functions:

1. Outlier handling (median absolute deviation rule) and the non-parametric
   group comparisons reported in the Results.
2. ``efficient_pearsonr`` / ``permtest_pearsonr``, vendored from
   ``netneurotools`` (BSD-3-Clause, https://github.com/netneurolab/netneurotools)
   because the private scipy helper they relied on was removed in
   scipy >= 1.15 (scipy PR #23088). Behaviour is unchanged from the
   netneurotools originals.
3. Spin-test convenience wrappers around ``neuromaps``.
"""

import numpy as np
import pingouin as pg
import scipy.stats as sts
import scipy.special as sspecial
from scipy.stats import PermutationMethod
from sklearn.utils.validation import check_random_state

from . import config

__all__ = [
    "outlier_mask",
    "outlier_masks",
    "compare_groups",
    "mad_keep",
    "rank_biserial",
    "permutation_test_trimmed",
    "parcelwise_effect_size",
    "temporal_autocorrelation",
    "efficient_pearsonr",
    "permtest_pearsonr",
    "spin_nulls",
]


# ---------------------------------------------------------------------------
# Outliers and group comparisons
# ---------------------------------------------------------------------------

def outlier_mask(x):
    """Boolean mask of observations to *keep* under the MAD-median rule.

    ``pingouin.madmedianrule`` (Leys et al., 2013): an observation is an
    outlier when its distance from the median exceeds 2.24 normalised median
    absolute deviations (the constant is sqrt(chi2.ppf(0.975, 1))).

    Note that trimming each group by its own median and MAD *before* an
    unpaired test is not a valid procedure -- see
    :func:`permutation_test_trimmed` -- so this mask is used only for the
    trimmed variants, not for the reported results.
    """
    return ~pg.madmedianrule(np.asarray(x))


def outlier_masks(*arrays):
    """Apply :func:`outlier_mask` to several arrays at once."""
    return tuple(outlier_mask(a) for a in arrays)


#: Maps the various names pingouin has used for its Mann-Whitney outputs onto
#: the canonical ones used here. Compared after lower-casing and stripping
#: hyphens and underscores.
_MWU_ALIASES = {
    "uval": "U", "u": "U", "ustat": "U", "ustatistic": "U",
    "pval": "p", "p": "p", "pvalue": "p", "punc": "p",
    "rbc": "RBC", "rankbiserial": "RBC", "r": "RBC",
    "cles": "CLES",
}


def _canonical_mwu(frame):
    """Rename a pingouin Mann-Whitney frame to canonical column names.

    pingouin has changed these labels between releases (``U-val`` versus
    ``U``, ``p-val`` versus ``p``), so they are matched by pattern instead of
    being assumed. Columns that are not recognised are left untouched.
    """
    renames = {}
    for col in frame.columns:
        key = str(col).strip().lower().replace("-", "").replace("_", "")
        target = _MWU_ALIASES.get(key)
        if target and target not in renames.values():
            renames[col] = target

    out = frame.rename(columns=renames)
    missing = {"U", "p"} - set(out.columns)
    if missing:
        raise RuntimeError(
            f"Could not find {sorted(missing)} in the Mann-Whitney output. "
            f"pingouin returned columns {list(frame.columns)}; add the "
            "appropriate alias to kasp.stats._MWU_ALIASES."
        )
    return out


def compare_groups(x, y, n_perm=None, seed=None, alternative="two-sided"):
    """Permutation-based Mann-Whitney U test between two groups.

    Returns a one-row frame with canonical columns ``U``, ``p``, ``RBC``
    (rank-biserial correlation, the sign convention used throughout; see below)
    and, where pingouin provides it, ``CLES``. U and p are pingouin's, as in
    the original analysis.
    """
    n_perm = config.N_PERM_GROUP if n_perm is None else n_perm
    seed = config.SEED if seed is None else seed
    x, y = np.asarray(x), np.asarray(y)

    method = PermutationMethod(n_resamples=n_perm, random_state=seed)
    result = _canonical_mwu(pg.mwu(x, y, alternative=alternative, method=method))

    # The rank-biserial correlation is computed here from U rather than taken
    # from pingouin, because pingouin's sign convention for it has changed
    # between versions (the same U has given +0.221 in one version and -0.221
    # in another). The convention used here is
    #     RBC = 1 - 2U / (n_x * n_y)
    # (positive: x tends to rank below y). pingouin's own value is kept as
    # RBC_pingouin for comparison.
    result = result.copy()
    if "RBC" in result.columns:
        result["RBC_pingouin"] = result["RBC"]
    result["RBC"] = 1 - (2 * result["U"]) / (len(x) * len(y))

    return result


# ---------------------------------------------------------------------------
# The corrected directedness tests
# ---------------------------------------------------------------------------

#: pingouin's MAD-median rule: threshold sqrt(chi2.ppf(0.975, 1)) on the
#: normalised MAD (scale 1 / norm.ppf(0.75)). Reproduced in numpy so the rule
#: can run inside a permutation loop; the notebook asserts the two agree.
MAD_THRESHOLD = 2.2414027276049473
MAD_SCALE = 1.482602218505602


def mad_keep(x):
    """numpy twin of :func:`outlier_mask`: True for observations to keep."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * MAD_SCALE
    if mad == 0:
        return np.ones(x.size, dtype=bool)
    return np.abs(x - med) / mad <= MAD_THRESHOLD


def _midranks(a):
    """Ranks 1..n with ties given their mean rank (as scipy's rankdata)."""
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    r = np.arange(1, a.size + 1, dtype=float)
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    mean_rank = np.bincount(inverse, weights=r) / counts
    out = np.empty(a.size)
    out[order] = mean_rank[inverse]
    return out


def rank_biserial(x, y):
    """Rank-biserial correlation, ``1 - 2U/(n_x n_y)``, U for ``x``.

    The convention of :func:`compare_groups`: positive means
    ``x`` tends to rank *below* ``y``. Unlike U it does not depend on the
    sample sizes, which is why it is the statistic used when those sizes
    change from one permutation to the next.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ranks = _midranks(np.concatenate([x, y]))
    u = ranks[:x.size].sum() - x.size * (x.size + 1) / 2
    return 1.0 - 2.0 * u / (x.size * y.size)


def permutation_test_trimmed(x, y, n_perm=None, seed=None):
    """A valid p-value for a trimmed statistic: trim, then rank test.

    Trimming each group by its own median and MAD and then running
    a Mann-Whitney test on what was left, treating the retained values as
    the samples. That is anti-conservative: under the null a group whose
    sample median happens to be low trims more of the common upper tail and
    keeps more of the lower one, so trimming *amplifies* chance differences
    before the test sees them; simulated null cohorts of these sizes give
    ~20% false positives at alpha = 0.05.

    The remedy for the trimmed statistic is to make the trimming part
    of the statistic: shuffle the group labels, trim each permuted group by
    the same rule, compute the rank-biserial correlation, and take the
    p-value from that distribution. The observed statistic is unchanged;
    only its p-value changes. (The other remedy is not to trim
    at all -- a rank test does not need it -- which is
    :func:`compare_groups` on the full samples.)

    Returns a one-row frame with ``RBC``, ``p``, and the trimmed sizes
    ``n_x`` and ``n_y`` of the observed comparison.
    """
    n_perm = config.N_PERM_GROUP if n_perm is None else n_perm
    seed = config.SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def statistic(a, b):
        a = a[mad_keep(a)]
        b = b[mad_keep(b)]
        return rank_biserial(a, b), a.size, b.size

    observed, n_x, n_y = statistic(x, y)
    pooled = np.concatenate([x, y])
    exceed = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(pooled)
        stat, _, _ = statistic(shuffled[:x.size], shuffled[x.size:])
        exceed += abs(stat) >= abs(observed)
    p = (exceed + 1) / (n_perm + 1)

    import pandas as pd
    return pd.DataFrame([{"RBC": observed, "p": p, "n_x": n_x, "n_y": n_y,
                          "n_perm": n_perm}])


def parcelwise_effect_size(x, y, min_n=3):
    """Rank-biserial correlation between two groups, parcel by parcel.

    A map-valued counterpart to :func:`compare_groups`, for rendering where
    the groups differ rather than by how much. Uses the same convention as the
    group-level tests, so the sign means the same thing in both::

        RBC = 1 - 2U / (n_x * n_y)

    **Mind the sign.** On this convention — pingouin's, and the one the
    reported group statistics use — a *positive* value means ``x`` tends to
    rank **below** ``y``. Verified at the extremes: with every value of ``x``
    above every value of ``y``, U is ``n_x * n_y`` and RBC is -1.

    The measure is bounded in [-1, 1] and is unaffected by the scale of the
    underlying values, which is what makes it comparable across parcels and
    across contrasts; a raw difference of means is neither.

    No multiple-comparison correction is applied and no p-values are returned.
    This is a descriptive map, and reading significance off 400 uncorrected
    parcel tests would be exactly the wrong thing to do with it.

    Parameters
    ----------
    x, y : (n_subjects, n_parcels) arrays
        NaN entries are dropped per parcel.
    min_n : int
        Parcels with fewer than this many observations in either group return
        NaN rather than an effect size computed from almost nothing.

    Returns
    -------
    (n_parcels,) array
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"the two groups have different parcel counts: "
            f"{x.shape[1]} and {y.shape[1]}"
        )

    out = np.full(x.shape[1], np.nan)
    for parcel in range(x.shape[1]):
        a = x[:, parcel][np.isfinite(x[:, parcel])]
        b = y[:, parcel][np.isfinite(y[:, parcel])]
        if a.size < min_n or b.size < min_n:
            continue
        u = sts.mannwhitneyu(a, b, alternative="two-sided").statistic
        out[parcel] = 1 - (2 * u) / (a.size * b.size)
    return out


# ---------------------------------------------------------------------------
# Temporal autocorrelation
# ---------------------------------------------------------------------------

def temporal_autocorrelation(x):
    """Lag-1 temporal autocorrelation (TA-delta-1).

    Follows Shinn et al. (2023), Nat Neurosci 26:867-878, and uses the biased
    estimator (numpy's default), as in that paper.

    Parameters
    ----------
    x : array_like
        A single timeseries, or an array whose **last** axis is time. For the
        parcellated data used here that means ``(n_parcels, n_timepoints)``
        for one participant, or ``(n_subjects, n_parcels, n_timepoints)`` for
        a group.

    Returns
    -------
    numpy.ndarray
        TA-delta-1 with the time axis collapsed, i.e. one value per parcel
        (per subject).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return np.corrcoef(x[:-1], x[1:])[0, 1]
    return np.apply_along_axis(
        lambda row: np.corrcoef(row[:-1], row[1:])[0, 1], -1, x
    )


# ---------------------------------------------------------------------------
# Vendored from netneurotools (BSD-3-Clause)
# ---------------------------------------------------------------------------

def _chk2_asarray(a, b, axis):
    # Removed from scipy in https://github.com/scipy/scipy/pull/23088
    if axis is None:
        a = np.ravel(a)
        b = np.ravel(b)
        outaxis = 0
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)
    if b.ndim == 0:
        b = np.atleast_1d(b)

    return a, b, outaxis


def efficient_pearsonr(a, b, ddof=1, nan_policy="propagate"):
    """Compute correlation of matching columns in `a` and `b`.

    Vendored from ``netneurotools.stats.efficient_pearsonr``.

    Returns
    -------
    corr : float or numpy.ndarray
        Pearson's correlation coefficient between matching columns of inputs
    pval : float or numpy.ndarray
        Two-tailed p-values
    """
    a, b, _ = _chk2_asarray(a, b, 0)
    if len(a) != len(b):
        raise ValueError("Provided arrays do not have same length")

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan

    if nan_policy not in ("propagate", "raise", "omit"):
        raise ValueError(f'Value for nan_policy "{nan_policy}" not allowed')

    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    if a.shape[1] != b.shape[1]:
        a, b = np.broadcast_arrays(a, b)

    mask = np.logical_or(np.isnan(a), np.isnan(b))
    if nan_policy == "raise" and np.any(mask):
        raise ValueError('Input cannot contain NaN when nan_policy is "omit"')
    elif nan_policy == "omit":
        # avoid making copies of the data, if possible
        a = np.ma.masked_array(a, mask, copy=False, fill_value=np.nan)
        b = np.ma.masked_array(b, mask, copy=False, fill_value=np.nan)

    with np.errstate(invalid="ignore"):
        corr = sts.zscore(a, ddof=ddof, nan_policy=nan_policy) * sts.zscore(
            b, ddof=ddof, nan_policy=nan_policy
        )

    sumfunc, n_obs = np.sum, len(a)
    if nan_policy == "omit":
        corr = corr.filled(np.nan)
        sumfunc = np.nansum
        n_obs = np.squeeze(np.sum(np.logical_not(np.isnan(corr)), axis=0))

    corr = sumfunc(corr, axis=0) / (n_obs - 1)
    corr = np.squeeze(np.clip(corr, -1, 1)) / 1

    # taken from scipy.stats
    ab = (n_obs / 2) - 1
    prob = 2 * sspecial.betainc(ab, ab, 0.5 * (1 - np.abs(corr)))

    return corr, prob


def permtest_pearsonr(a, b, axis=0, n_perm=1000, resamples=None, seed=0):
    """Non-parametric equivalent of :py:func:`scipy.stats.pearsonr`.

    Vendored from ``netneurotools.stats.permtest_pearsonr``. Generates a
    two-tailed p-value for the hypothesis that samples `a` and `b` are
    correlated, using permutation tests.

    Notes
    -----
    The lowest p-value that can be returned by this function is equal to
    1 / (`n_perm` + 1).
    """
    a, b, axis = _chk2_asarray(a, b, axis)
    rs = check_random_state(seed)

    if len(a) != len(b):
        raise ValueError('Provided arrays do not have same length')

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan

    if resamples is not None:
        if n_perm > resamples.shape[-1]:
            raise ValueError('Number of permutations requested exceeds size '
                             'of resampling array.')

    # divide by one forces coercion to float if ndim = 0
    true_corr = efficient_pearsonr(a, b)[0] / 1
    abs_true = np.abs(true_corr)

    permutations = np.ones(true_corr.shape)
    for perm in range(n_perm):
        # permute `a` and determine whether correlations exceed original
        if resamples is None:
            ap = a[rs.permutation(len(a))]
        else:
            ap = a[resamples[:, perm]]
        permutations += np.abs(efficient_pearsonr(ap, b)[0]) >= abs_true

    pvals = permutations / (n_perm + 1)  # + 1 in denom accounts for true_corr

    return true_corr, pvals


# ---------------------------------------------------------------------------
# Spatial (spin) permutation tests
# ---------------------------------------------------------------------------

def spin_nulls(cortical_map, parcellation, n_perm=None, seed=1234):
    """Generate spatial null maps with the Alexander-Bloch spin test.

    Thin wrapper around ``neuromaps.nulls.alexander_bloch`` that fixes the
    surface space and density used throughout the manuscript.
    """
    from neuromaps import nulls

    n_perm = config.N_PERM_SPIN if n_perm is None else n_perm
    return nulls.alexander_bloch(
        cortical_map,
        atlas="fsLR",
        density="32k",
        n_perm=n_perm,
        seed=seed,
        parcellation=parcellation,
    )
