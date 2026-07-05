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

from transcssr.stats import compare_morphs, morphs_differ, morphs_pvalue


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


def test_invalid_test_name():
    with pytest.raises(ValueError, match="Unknown test"):
        compare_morphs([1, 2], [3, 4], 1, 2, test="ttest")


def test_wrong_length_raises():
    with pytest.raises(ValueError, match="length"):
        compare_morphs([1, 2, 3], [4, 5, 6], 2, 2)
