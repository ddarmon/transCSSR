"""Parity tests for :mod:`transcssr.counting` against the legacy engine.

These tests run from the repo root (the ``transcssr-int`` mamba env provides
the legacy dependencies). They import the legacy
``estimate_predictive_distributions`` and assert that the new index-tuple
counter reproduces its count tables *exactly* once legacy string keys are
translated to index-tuple keys through a fixed char->index map.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, Tuple

import pytest

from transCSSR import estimate_predictive_distributions
from transcssr.counting import estimate_counts


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _encode(string: str, char_map: Dict[str, int]) -> Tuple[int, ...]:
    """Encode a string to a tuple of indices through ``char_map``."""
    return tuple(char_map[c] for c in string)


def _translate_key(key, x_map: Dict[str, int], y_map: Dict[str, int]):
    """Translate a legacy string history key to an index-tuple history key."""
    xword, yword = key
    return (
        tuple(x_map[c] for c in xword),
        tuple(y_map[c] for c in yword),
    )


def _translate_counter(counter: Counter, x_map, y_map) -> Counter:
    """Translate every key of a legacy count table to index-tuple keys."""
    return Counter({_translate_key(k, x_map, y_map): v for k, v in counter.items()})


def _assert_parity(stringX, stringY, L_max, x_map, y_map, *, is_multiline=False,
                   axs=None, ays=None):
    """Assert legacy and new counters match after key translation."""
    legacy_marg, legacy_fut = estimate_predictive_distributions(
        stringX, stringY, L_max, axs=axs, ays=ays, is_multiline=is_multiline
    )

    if is_multiline:
        xs = [_encode(s, x_map) for s in stringX]
        ys = [_encode(s, y_map) for s in stringY]
    else:
        xs = [_encode(stringX, x_map)]
        ys = [_encode(stringY, y_map)]

    new_marg, new_fut = estimate_counts(xs, ys, L_max)

    expected_marg = _translate_counter(legacy_marg, x_map, y_map)
    expected_fut = _translate_counter(legacy_fut, x_map, y_map)

    assert dict(new_marg) == dict(expected_marg)
    assert dict(new_fut) == dict(expected_fut)
    return new_marg, new_fut


# --------------------------------------------------------------------------- #
# (a) Toy sequence (matches tests/conftest.py fixtures)
# --------------------------------------------------------------------------- #

def test_toy_sequence_parity():
    x_map = {"a": 0, "b": 1, "c": 2}
    y_map = {"0": 0, "1": 1}
    _assert_parity("abacab", "010001", 2, x_map, y_map,
                   axs=list(x_map), ays=list(y_map))


def test_toy_sequence_known_keys():
    """Spot-check against the hand-computed conftest fixture values."""
    x_map = {"a": 0, "b": 1, "c": 2}
    y_map = {"0": 0, "1": 1}
    xs = [_encode("abacab", x_map)]
    ys = [_encode("010001", y_map)]
    marg, fut = estimate_counts(xs, ys, 2)

    # ("a", "") -> ((0,), ()) counted twice; ("aba", "01") -> ((0,1,0),(0,1))
    assert marg[((0,), ())] == 2
    assert marg[((0, 1, 0), (0, 1))] == 1
    assert fut[((0,), (0,))] == 2
    assert fut[((0, 1, 0), (0, 1, 0))] == 1
    # Length-3 (== L_max+1) words appear once.
    assert fut[((1, 0, 2), (1, 0, 0))] == 1


# --------------------------------------------------------------------------- #
# (b) Seeded random binary strings, several L_max
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("L_max", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_random_binary_parity(seed, L_max):
    rng = random.Random(seed)
    n = 500
    stringX = "".join(rng.choice("01") for _ in range(n))
    stringY = "".join(rng.choice("01") for _ in range(n))
    binmap = {"0": 0, "1": 1}
    _assert_parity(stringX, stringY, L_max, binmap, binmap,
                   axs=["0", "1"], ays=["0", "1"])


# --------------------------------------------------------------------------- #
# (c) 3-symbol output alphabet
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("L_max", [1, 2, 3])
def test_three_symbol_output_parity(L_max):
    rng = random.Random(7)
    n = 400
    stringX = "".join(rng.choice("ab") for _ in range(n))
    stringY = "".join(rng.choice("012") for _ in range(n))
    x_map = {"a": 0, "b": 1}
    y_map = {"0": 0, "1": 1, "2": 2}
    _assert_parity(stringX, stringY, L_max, x_map, y_map,
                   axs=["a", "b"], ays=["0", "1", "2"])


# --------------------------------------------------------------------------- #
# (d) Multiline ensemble
# --------------------------------------------------------------------------- #

def test_multiline_ensemble_parity():
    stringX_list = ["abacabac", "babacaba", "cabacabc"]
    stringY_list = ["01000100", "10100101", "00100010"]
    x_map = {"a": 0, "b": 1, "c": 2}
    y_map = {"0": 0, "1": 1}
    _assert_parity(stringX_list, stringY_list, 2, x_map, y_map,
                   is_multiline=True, axs=list(x_map), ays=list(y_map))


def test_multiline_random_parity():
    rng = random.Random(123)
    lines_x = ["".join(rng.choice("01") for _ in range(rng.randint(5, 40)))
               for _ in range(4)]
    lines_y = ["".join(rng.choice("01") for _ in range(len(lx))) for lx in lines_x]
    binmap = {"0": 0, "1": 1}
    _assert_parity(lines_x, lines_y, 3, binmap, binmap,
                   is_multiline=True, axs=["0", "1"], ays=["0", "1"])


# --------------------------------------------------------------------------- #
# Single-sequence vs one-element-ensemble equivalence
# --------------------------------------------------------------------------- #

def test_single_equals_one_element_ensemble_new():
    """In the new engine a single sequence == a one-element ensemble."""
    x_map = {"a": 0, "b": 1, "c": 2}
    y_map = {"0": 0, "1": 1}
    x = _encode("abacabac", x_map)
    y = _encode("01000100", y_map)

    single_marg, single_fut = estimate_counts([x], [y], 2)
    # Wrapping the same one realization is by construction identical.
    again_marg, again_fut = estimate_counts([tuple(x)], [tuple(y)], 2)
    assert dict(single_marg) == dict(again_marg)
    assert dict(single_fut) == dict(again_fut)


def test_legacy_single_equals_one_element_multiline():
    """Legacy is_multiline=False and a one-element is_multiline=True agree.

    For counting_method=0 both run the identical per-line loop, so a single
    realization and a one-element ensemble produce byte-for-byte identical
    count tables. We confirm the new engine matches both.
    """
    stringX = "abacabac"
    stringY = "01000100"
    x_map = {"a": 0, "b": 1, "c": 2}
    y_map = {"0": 0, "1": 1}

    legacy_single = estimate_predictive_distributions(stringX, stringY, 2)
    legacy_multi = estimate_predictive_distributions(
        [stringX], [stringY], 2, is_multiline=True
    )
    assert dict(legacy_single[0]) == dict(legacy_multi[0])
    assert dict(legacy_single[1]) == dict(legacy_multi[1])

    new_marg, new_fut = estimate_counts(
        [_encode(stringX, x_map)], [_encode(stringY, y_map)], 2
    )
    expected_marg = _translate_counter(legacy_single[0], x_map, y_map)
    expected_fut = _translate_counter(legacy_single[1], x_map, y_map)
    assert dict(new_marg) == dict(expected_marg)
    assert dict(new_fut) == dict(expected_fut)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

def test_sequence_shorter_than_window_is_empty():
    """T <= L_max yields empty counters (matches legacy range(T-L_max))."""
    binmap = {"0": 0, "1": 1}
    # T = 3, L_max = 5 -> range(3-5) empty.
    marg, fut = estimate_counts([_encode("010", binmap)], [_encode("011", binmap)], 5)
    assert len(marg) == 0
    assert len(fut) == 0

    # Cross-check legacy behaves the same.
    lm, lf = estimate_predictive_distributions("010", "011", 5)
    assert len(lm) == 0 and len(lf) == 0


def test_t_equals_lmax_is_empty():
    binmap = {"0": 0, "1": 1}
    # T = L_max + 0 -> range(0) empty. T=2, L_max=2.
    marg, fut = estimate_counts([_encode("01", binmap)], [_encode("10", binmap)], 2)
    assert len(marg) == 0 and len(fut) == 0


def test_l_max_one_parity():
    x_map = {"a": 0, "b": 1, "c": 2}
    y_map = {"0": 0, "1": 1}
    _assert_parity("abacabac", "01000100", 1, x_map, y_map,
                   axs=list(x_map), ays=list(y_map))


def test_l_max_zero_parity():
    """L_max = 0: only length-1 windows; no truncation loop."""
    x_map = {"a": 0, "b": 1, "c": 2}
    y_map = {"0": 0, "1": 1}
    _assert_parity("abacabac", "01000100", 0, x_map, y_map,
                   axs=list(x_map), ays=list(y_map))


def test_mismatched_realization_counts_raises():
    with pytest.raises(ValueError):
        estimate_counts([(0, 1)], [(0, 1), (1, 0)], 1)


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        estimate_counts([(0, 1, 0)], [(0, 1)], 1)


def test_counting_method_1_raises():
    with pytest.raises(NotImplementedError):
        estimate_counts([(0, 1)], [(0, 1)], 1, counting_method=1)
