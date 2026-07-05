"""Parity tests for :mod:`transcssr.stats` against the legacy engine.

The new tests use the mathematically correct per-input-symbol row stride
``n_y``. The legacy ``chisquared_test`` uses ``len(axs)`` (SPEC §2.1 stride
bug), which is *untriggered* exactly when ``n_x == 1`` or ``n_x == n_y``. All
cases here are restricted to those shapes so that legacy and new must agree
bit-for-bit on the statistic.
"""

from __future__ import annotations

import numpy as np
import pytest

# Legacy oracle (tests run from repo root).
from transCSSR import chisquared_test

import scipy.stats

from transcssr.stats import compare_morphs, morphs_differ, morphs_pvalue


def _naive_compare(m1, m2, n_x, n_y, alpha=0.001, test="chi2"):
    """Scalar per-cell reference for the correct-stride (``n_y``) semantics.

    Independent of the vectorized :func:`compare_morphs` and of the legacy
    engine, so it validates the correct-stride behavior for shapes where the
    legacy stride bug is triggered (``n_x > 1`` and ``n_x != n_y``).
    """
    a1 = np.asarray(m1, dtype=float)
    a2 = np.asarray(m2, dtype=float)
    statistic = 0.0
    df_x = 0
    for row_ind in range(n_x):
        start = row_ind * n_y  # correct stride
        t1 = a1[start : start + n_y]
        t2 = a2[start : start + n_y]
        nn1 = t1.sum()
        nn2 = t2.sum()
        if nn1 == 0 or nn2 == 0:
            continue
        df_x += 1
        theta0 = (t1 + t2) / (nn1 + nn2)
        for i in range(n_y):
            for t, nn in ((t1, nn1), (t2, nn2)):
                denom = nn * theta0[i]
                if denom == 0:
                    continue
                obs = t[i]
                if test == "chi2":
                    statistic += (obs - denom) ** 2 / denom
                elif obs != 0:
                    statistic += obs * np.log(obs / denom)
    if test == "G":
        statistic *= 2
    df = df_x * (n_y - 1)
    reject = bool(statistic > scipy.stats.chi2.ppf(1 - alpha, df))
    pvalue = float(1 - scipy.stats.chi2.cdf(statistic, df))
    return reject, pvalue


def _dummy_symbols(n: int) -> list[str]:
    """A dummy symbol list of length ``n`` (legacy only uses ``len``)."""
    return [str(i) for i in range(n)]


# Shapes where the legacy stride bug is untriggered: n_x == 1 or n_x == n_y.
_SHAPES = [(1, 2), (1, 3), (1, 5), (2, 2), (3, 3), (4, 4)]


@pytest.mark.parametrize("test_type", ["chi2", "G"])
@pytest.mark.parametrize("n_x,n_y", _SHAPES)
def test_parity_random(n_x, n_y, test_type):
    rng = np.random.default_rng(12345 + n_x * 100 + n_y * 10)
    axs, ays = _dummy_symbols(n_x), _dummy_symbols(n_y)
    length = n_x * n_y

    for _ in range(40):
        m1 = rng.integers(0, 20, size=length).tolist()
        m2 = rng.integers(0, 20, size=length).tolist()

        legacy_reject, legacy_p = chisquared_test(
            m1, m2, axs, ays, alpha=0.001, test_type=test_type
        )
        new_reject, new_p = compare_morphs(
            m1, m2, n_x, n_y, alpha=0.001, test=test_type
        )

        assert new_reject == legacy_reject
        assert new_p == pytest.approx(legacy_p, abs=1e-12, rel=1e-9)


@pytest.mark.parametrize("test_type", ["chi2", "G"])
@pytest.mark.parametrize("n_x,n_y", _SHAPES)
def test_parity_zero_count_rows(n_x, n_y, test_type):
    """Rows unseen in either morph must be skipped identically."""
    rng = np.random.default_rng(999 + n_x + n_y)
    axs, ays = _dummy_symbols(n_x), _dummy_symbols(n_y)

    for _ in range(30):
        m1 = rng.integers(0, 15, size=n_x * n_y).reshape(n_x, n_y)
        m2 = rng.integers(0, 15, size=n_x * n_y).reshape(n_x, n_y)
        # Zero out a random input row in each morph so skipping is exercised.
        m1[rng.integers(0, n_x)] = 0
        m2[rng.integers(0, n_x)] = 0
        f1 = m1.reshape(-1).tolist()
        f2 = m2.reshape(-1).tolist()

        legacy_reject, legacy_p = chisquared_test(
            f1, f2, axs, ays, alpha=0.001, test_type=test_type
        )
        new_reject, new_p = compare_morphs(
            f1, f2, n_x, n_y, alpha=0.001, test=test_type
        )
        assert new_reject == legacy_reject
        # df_x == 0 (all shared rows skipped) yields nan on both sides.
        if np.isnan(legacy_p):
            assert np.isnan(new_p)
        else:
            assert new_p == pytest.approx(legacy_p, abs=1e-12, rel=1e-9)


@pytest.mark.parametrize("test_type", ["chi2", "G"])
def test_identical_morphs_accept(test_type):
    """Identical morphs never reject the null (should return False)."""
    m = [10, 5, 3, 8]  # 2x2
    reject, pvalue = compare_morphs(m, m, 2, 2, test=test_type)
    assert reject is False
    assert pvalue == pytest.approx(1.0)
    # Parity with legacy.
    legacy_reject, legacy_p = chisquared_test(
        m, m, ["0", "1"], ["0", "1"], test_type=test_type
    )
    assert reject == legacy_reject
    assert pvalue == pytest.approx(legacy_p)


@pytest.mark.parametrize("test_type", ["chi2", "G"])
def test_strongly_different_morphs_reject(test_type):
    """Sharply different large-count morphs reject the null (True)."""
    m1 = [1000, 5]     # 1x2 heavily weighted to output 0
    m2 = [5, 1000]     # 1x2 heavily weighted to output 1
    reject, _ = compare_morphs(m1, m2, 1, 2, test=test_type)
    assert reject is True
    legacy_reject, _ = chisquared_test(
        m1, m2, ["0"], ["0", "1"], test_type=test_type
    )
    assert reject == legacy_reject


def test_morphs_differ_and_pvalue_wrappers():
    m1 = [1000, 5]
    m2 = [5, 1000]
    assert morphs_differ(m1, m2, 1, 2) is True
    assert morphs_differ(m1, m1, 1, 2) is False
    p_same = morphs_pvalue(m1, m1, 1, 2)
    assert p_same == pytest.approx(1.0)


# Shapes where n_x > 1 and n_x != n_y (legacy stride bug regime); the
# vectorized comparator is validated against the scalar correct-stride
# reference rather than the legacy oracle here.
_RECT_SHAPES = [(2, 3), (3, 2), (2, 5), (5, 2), (3, 4), (4, 3)]


@pytest.mark.parametrize("test_type", ["chi2", "G"])
@pytest.mark.parametrize("n_x,n_y", _RECT_SHAPES)
def test_vectorized_matches_scalar_reference(n_x, n_y, test_type):
    rng = np.random.default_rng(2024 + n_x * 100 + n_y * 10)
    for _ in range(40):
        m1 = rng.integers(0, 20, size=n_x * n_y).tolist()
        m2 = rng.integers(0, 20, size=n_x * n_y).tolist()

        ref_reject, ref_p = _naive_compare(m1, m2, n_x, n_y, test=test_type)
        new_reject, new_p = compare_morphs(m1, m2, n_x, n_y, test=test_type)

        assert new_reject == ref_reject
        assert new_p == pytest.approx(ref_p, abs=1e-12, rel=1e-9)


@pytest.mark.parametrize("test_type", ["chi2", "G"])
@pytest.mark.parametrize("n_x,n_y", _RECT_SHAPES)
def test_vectorized_unseen_rows_rect(n_x, n_y, test_type):
    """n_x > 1 (n_x != n_y) with input rows unseen in either morph."""
    rng = np.random.default_rng(7000 + n_x * 13 + n_y)
    for _ in range(30):
        m1 = rng.integers(0, 15, size=n_x * n_y).reshape(n_x, n_y)
        m2 = rng.integers(0, 15, size=n_x * n_y).reshape(n_x, n_y)
        # Zero out a random input row in each morph so skipping is exercised.
        m1[rng.integers(0, n_x)] = 0
        m2[rng.integers(0, n_x)] = 0
        f1 = m1.reshape(-1).tolist()
        f2 = m2.reshape(-1).tolist()

        ref_reject, ref_p = _naive_compare(f1, f2, n_x, n_y, test=test_type)
        new_reject, new_p = compare_morphs(f1, f2, n_x, n_y, test=test_type)

        assert new_reject == ref_reject
        if np.isnan(ref_p):
            assert np.isnan(new_p)
        else:
            assert new_p == pytest.approx(ref_p, abs=1e-12, rel=1e-9)


def test_numpy_int64_input_matches_list():
    """int64 array input (as passed from the CSSR engine) matches list input."""
    m1 = [3, 7, 0, 5, 2, 9]
    m2 = [4, 1, 6, 0, 8, 2]
    for test_type in ("chi2", "G"):
        r_list = compare_morphs(m1, m2, 2, 3, test=test_type)
        r_arr = compare_morphs(
            np.array(m1, dtype=np.int64), np.array(m2, dtype=np.int64), 2, 3,
            test=test_type,
        )
        assert r_arr[0] == r_list[0]
        assert r_arr[1] == pytest.approx(r_list[1], abs=1e-12, rel=1e-12)


def test_invalid_test_name():
    with pytest.raises(ValueError, match="Unknown test"):
        compare_morphs([1, 2], [3, 4], 1, 2, test="ttest")


def test_wrong_length_raises():
    with pytest.raises(ValueError, match="length"):
        compare_morphs([1, 2, 3], [4, 5, 6], 2, 2)
