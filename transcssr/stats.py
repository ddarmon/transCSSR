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

    statistic = 0.0
    df_x = 0

    for row_ind in range(n_x):
        start = row_ind * n_y  # correct stride (SPEC §2.1)
        tmp1 = a1[start : start + n_y]
        tmp2 = a2[start : start + n_y]

        n1 = tmp1.sum()
        n2 = tmp2.sum()
        if n1 == 0 or n2 == 0:
            # An (X_t, past) unseen in either morph is not held against a state.
            continue

        df_x += 1
        theta0 = (tmp1 + tmp2) / (n1 + n2)
        ns = (n1, n2)
        table = (tmp1, tmp2)

        for array_ind in range(n_y):
            for sample_ind in (0, 1):
                denom = ns[sample_ind] * theta0[array_ind]
                if denom == 0:
                    continue
                obs = table[sample_ind][array_ind]
                if test == "chi2":
                    statistic += (obs - denom) ** 2 / denom
                else:  # G-test
                    if obs != 0:
                        statistic += obs * np.log(obs / denom)

    if test == "G":
        statistic *= 2

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

    Thin wrapper over :func:`compare_morphs` returning only the rejection
    decision. ``True`` means "reject the null / split the histories".

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

    Exposed for the CSSR Stage-1 existing-state matching step, which argmaxes
    over p-values among candidate states.

    See :func:`compare_morphs` for parameter descriptions.
    """
    _, pvalue = compare_morphs(morph1, morph2, n_x, n_y, alpha=alpha, test=test)
    return pvalue
