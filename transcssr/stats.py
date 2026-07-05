"""Predictive-distribution comparison tests (chi-square and G-test).

These are pure functions on flat *morph* count vectors. A morph is a length
``n_x * n_y`` vector of emission counts, row-stacked by input symbol: entry
``k`` counts emission ``(x_ind, y_ind)`` where ``k = n_y * x_ind + y_ind`` (the
input index is the outer loop, the output index the inner loop). Row
``x_ind`` therefore occupies the contiguous slice
``morph[x_ind*n_y : x_ind*n_y + n_y]`` and, normalized, gives
``P(Y_t = y | X_t = x, state)``.

The tests reproduce the semantics of the legacy ``chisquared_test`` in
``transCSSR.py`` (see ``docs/SPEC-core.md`` §2), with one deliberate fix: the
per-input-symbol row stride is the mathematically correct ``n_y`` rather than
the legacy ``len(axs)`` (SPEC §2.1 documents the latent legacy stride bug).
The two agree whenever ``n_x == 1`` or ``n_x == n_y``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.stats

__all__ = ["compare_morphs", "morphs_differ", "morphs_pvalue"]


def compare_morphs(
    morph1: Sequence[float] | np.ndarray,
    morph2: Sequence[float] | np.ndarray,
    n_x: int,
    n_y: int,
    alpha: float = 0.001,
    test: str = "chi2",
) -> tuple[bool, float]:
    """Compare two morphs (predictive distributions) for equivalence.

    The comparison is done row-wise per input symbol (because the object being
    compared is the conditional stochastic matrix ``P(Y | X, past)``) and then
    summed to a single statistic. Rows unseen in *either* morph (zero total
    count) are skipped entirely -- they contribute neither to the statistic nor
    to the degrees of freedom.

    Parameters
    ----------
    morph1, morph2 : array_like of int
        Flat morph count vectors of length ``n_x * n_y``, indexed
        ``n_y * x_ind + y_ind``.
    n_x : int
        Number of input symbols (``len(axs)``).
    n_y : int
        Number of output symbols (``len(ays)``).
    alpha : float, optional
        Significance level for the test. Default ``0.001``.
    test : {'chi2', 'G'}, optional
        The statistic to use: Pearson chi-square (``'chi2'``) or the
        log-likelihood-ratio / G statistic (``'G'``). Default ``'chi2'``.

    Returns
    -------
    reject : bool
        ``True`` if the null hypothesis (the two morphs come from the same
        distribution) is rejected at level ``alpha`` -- i.e. the two histories
        are statistically *different* and should be split.
    pvalue : float
        The p-value of the test, ``1 - CDF(statistic; df)``.

    Raises
    ------
    ValueError
        If ``test`` is not ``'chi2'`` or ``'G'``, or if a morph length does
        not equal ``n_x * n_y``.
    """
    if test not in ("chi2", "G"):
        raise ValueError(f"Unknown test {test!r}; expected 'chi2' or 'G'.")

    a1 = np.asarray(morph1, dtype=float)
    a2 = np.asarray(morph2, dtype=float)
    expected_len = n_x * n_y
    if a1.shape != (expected_len,) or a2.shape != (expected_len,):
        raise ValueError(
            f"Morphs must be 1-d of length n_x*n_y = {expected_len}; got "
            f"{a1.shape} and {a2.shape}."
        )

    # Vectorized equivalent of the legacy per-cell loop (correct stride n_y):
    # reshape the flat morphs to ``(n_x, n_y)`` so each row is one input
    # symbol's conditional output counts, then accumulate the statistic over
    # all cells at once. Summation order differs from the scalar loop, so the
    # statistic can differ by a few ULP; this comparator backs
    # ``stats_mode='correct'`` only (golden parity uses the exact-order legacy
    # comparator in ``transcssr.cssr``), so that is acceptable here.
    r1 = a1.reshape(n_x, n_y)
    r2 = a2.reshape(n_x, n_y)
    n1 = r1.sum(axis=1)
    n2 = r2.sum(axis=1)

    # An (X_t, past) unseen in either morph is not held against a state: rows
    # with a zero total in either morph contribute to neither statistic nor df.
    row_mask = (n1 != 0) & (n2 != 0)
    df_x = int(row_mask.sum())

    if df_x == 0:
        statistic = 0.0
    else:
        r1 = r1[row_mask]
        r2 = r2[row_mask]
        n1 = n1[row_mask]
        n2 = n2[row_mask]
        theta0 = (r1 + r2) / (n1 + n2)[:, None]
        denom1 = n1[:, None] * theta0
        denom2 = n2[:, None] * theta0

        if test == "chi2":
            # denom == 0 (theta0 == 0) implies obs == 0 in that cell, so it
            # contributes nothing; guard the division to avoid 0/0.
            nz1 = denom1 != 0
            nz2 = denom2 != 0
            c1 = np.where(nz1, (r1 - denom1) ** 2 / np.where(nz1, denom1, 1.0), 0.0)
            c2 = np.where(nz2, (r2 - denom2) ** 2 / np.where(nz2, denom2, 1.0), 0.0)
            statistic = float(c1.sum() + c2.sum())
        else:  # G-test: obs * log(obs / denom), only where obs != 0.
            pos1 = r1 != 0
            pos2 = r2 != 0
            g1 = np.where(
                pos1, r1 * np.log(np.where(pos1, r1, 1.0) / np.where(pos1, denom1, 1.0)), 0.0
            )
            g2 = np.where(
                pos2, r2 * np.log(np.where(pos2, r2, 1.0) / np.where(pos2, denom2, 1.0)), 0.0
            )
            statistic = float(2.0 * (g1.sum() + g2.sum()))

    df = df_x * (n_y - 1)

    reject = bool(statistic > scipy.stats.chi2.ppf(1 - alpha, df))
    pvalue = float(1 - scipy.stats.chi2.cdf(statistic, df))
    return reject, pvalue


def morphs_differ(
    morph1: Sequence[float] | np.ndarray,
    morph2: Sequence[float] | np.ndarray,
    n_x: int,
    n_y: int,
    alpha: float = 0.001,
    test: str = "chi2",
) -> bool:
    """Return ``True`` if the two morphs are statistically different.

    Public convenience wrapper over :func:`compare_morphs` returning only the
    rejection decision (``True`` means "reject the null / split the histories").
    The CSSR engine calls :func:`compare_morphs` directly; this wrapper is
    provided for callers who want just the decision.

    See :func:`compare_morphs` for parameter descriptions.
    """
    reject, _ = compare_morphs(morph1, morph2, n_x, n_y, alpha=alpha, test=test)
    return reject


def morphs_pvalue(
    morph1: Sequence[float] | np.ndarray,
    morph2: Sequence[float] | np.ndarray,
    n_x: int,
    n_y: int,
    alpha: float = 0.001,
    test: str = "chi2",
) -> float:
    """Return the p-value of comparing the two morphs.

    Public convenience wrapper over :func:`compare_morphs` returning only the
    p-value. The CSSR engine calls :func:`compare_morphs` directly (its
    Stage-1 existing-state matching argmaxes over p-values internally); this
    wrapper is provided for callers who want just the p-value.

    See :func:`compare_morphs` for parameter descriptions.
    """
    _, pvalue = compare_morphs(morph1, morph2, n_x, n_y, alpha=alpha, test=test)
    return pvalue
