"""Parity tests for :class:`~transcssr.countstore.SuffixArrayCountStore`.

The suffix-array backend must answer the two :class:`CountStore` queries --
``children`` and ``extensions`` -- **identically** to the Counter backend built
from :func:`transcssr.counting.estimate_counts` on the same data, over a matrix
of processes chosen to exercise the tail-correction edge effect and the
``z``-order-vs-``(y, x)``-order re-sort. End-to-end, ``infer_transducer`` /
``infer_machine`` with ``count_backend="suffix"`` must reproduce the default
backend's model exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from transcssr import infer_machine, infer_transducer
from transcssr.counting import estimate_counts
from transcssr.countstore import CounterCountStore, SuffixArrayCountStore


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _histories_from_counts(marg, fut):
    """Every history reachable in the Counter store's indexes, plus the null."""
    hists = {((), ())}
    for (mx, my), count in marg.items():
        if count and len(mx) >= 2:
            hists.add((mx[1:-1], my[1:]))
    for (fx, fy), count in fut.items():
        if count and len(fx):
            hists.add((fx[:-1], fy[:-1]))
    return hists


def _assert_store_parity(xs, ys, L_max, n_x, n_y, extra_absent=()):
    """Assert the two stores agree on children/extensions for all histories."""
    marg, fut = estimate_counts(xs, ys, L_max)
    counter = CounterCountStore(marg, fut, n_y)
    suffix = SuffixArrayCountStore(xs, ys, L_max, n_x, n_y)

    assert suffix.n_y == n_y

    hists = _histories_from_counts(marg, fut)
    # A few histories that are absent from the data.
    absent = {
        ((n_x - 1,) * (L_max), (n_y - 1,) * (L_max)),
        ((0,) * (L_max + 2), (0,) * (L_max + 2)),  # too long to be a key
    }
    absent.update(extra_absent)

    for history in hists | absent:
        c_children = sorted(counter.children(history))
        s_children = sorted(suffix.children(history))
        assert s_children == c_children, ("children", history, c_children, s_children)
        # documented ascending (y_ind, x_ind) order (before we sorted it).
        assert list(suffix.children(history)) == sorted(suffix.children(history))

        c_ext = sorted(counter.extensions(history))
        s_ext = sorted(suffix.extensions(history))
        assert s_ext == c_ext, ("extensions", history, c_ext, s_ext)
        positions = [p for p, _ in suffix.extensions(history)]
        assert len(positions) == len(set(positions))  # flat_pos unique per history

    return counter, suffix


# --------------------------------------------------------------------------- #
# Store-level parity over a matrix of processes
# --------------------------------------------------------------------------- #

# (id, factory) where factory(L_max) -> (xs, ys, n_x, n_y).
def _binary_transducer(seed):
    def make(L_max):
        rng = np.random.default_rng(seed)
        x = rng.integers(0, 2, size=400).tolist()
        y = rng.integers(0, 2, size=400).tolist()
        return [x], [y], 2, 2
    return make


def _machine_path(seed):
    def make(L_max):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 3, size=400).tolist()
        return [[0] * 400], [y], 1, 3
    return make


def _asymmetric(seed):
    # n_x=2, n_y=3: z-order (input-major) differs from (y_ind, x_ind) order,
    # catching a missing re-sort in children.
    def make(L_max):
        rng = np.random.default_rng(seed)
        x = rng.integers(0, 2, size=500).tolist()
        y = rng.integers(0, 3, size=500).tolist()
        return [x], [y], 2, 3
    return make


def _multi_unequal(seed):
    # Ensemble with UNEQUAL lengths, at least one shorter than L_max + 1.
    def make(L_max):
        rng = np.random.default_rng(seed)
        lengths = [60, 25, L_max, L_max + 1, 120]  # lengths[2] < L_max + 1
        xs = [rng.integers(0, 2, size=n).tolist() for n in lengths]
        ys = [rng.integers(0, 3, size=n).tolist() for n in lengths]
        return xs, ys, 2, 3
    return make


_PROCESSES = [
    ("binary_transducer", _binary_transducer(11)),
    ("machine_path", _machine_path(22)),
    ("asymmetric_2x3", _asymmetric(33)),
    ("multi_unequal", _multi_unequal(44)),
]


@pytest.mark.parametrize("name,factory", _PROCESSES, ids=[p[0] for p in _PROCESSES])
@pytest.mark.parametrize("L_max", [1, 2, 3])
def test_store_parity(name, factory, L_max):
    xs, ys, n_x, n_y = factory(L_max)
    _assert_store_parity(xs, ys, L_max, n_x, n_y)


# --------------------------------------------------------------------------- #
# Tail-correction edge cases
# --------------------------------------------------------------------------- #


def test_word_only_in_forbidden_tail_vanishes():
    """A word occurring ONLY in the forbidden tail region has corrected 0."""
    # Single realization; L_max = 2, so windows start at offsets 0 .. T-3.
    # Build a stream whose z-word (as a whole) is unique and lands only in the
    # tail: put a distinctive symbol pair at the very end.
    # x, y over n_x=1, n_y=3 -> z == y. Use a symbol '2' appearing ONLY at the
    # final position (offset T-1, in the tail for any L_max >= 1).
    L_max = 2
    y = [0, 1, 0, 1, 0, 1, 0, 2]  # T = 8; '2' only at offset 7 (tail)
    x = [0] * len(y)
    n_x, n_y = 1, 3

    counter, suffix = _assert_store_parity([x], [y], L_max, n_x, n_y)

    # The length-1 word (2,) occurs once in the raw corpus but only in the tail
    # region, so its corrected count is 0: it must be absent from the null
    # history's extensions and children in BOTH stores.
    null = ((), ())
    flat_positions = [p for p, _ in suffix.extensions(null)]
    assert (n_y * 0 + 2) not in flat_positions
    assert (2, 0) not in list(suffix.children(null))  # (y_ind=2, x_ind=0)
    # And the raw suffix index does see it (sanity: the correction, not absence,
    # is what removes it).
    assert suffix._index.count((2,)) == 1


def test_all_realizations_shorter_than_window_yield_empty_store():
    """Every realization shorter than L_max + 1 -> both queries empty everywhere."""
    L_max = 3
    xs = [[0, 1, 0], [1, 1], [0]]  # all lengths <= L_max
    ys = [[2, 0, 1], [1, 2], [0]]
    n_x, n_y = 2, 3

    marg, fut = estimate_counts(xs, ys, L_max)
    assert len(marg) == 0 and len(fut) == 0  # legacy: no windows at all

    suffix = SuffixArrayCountStore(xs, ys, L_max, n_x, n_y)
    # Probe the null history and a spread of short histories: all empty.
    probe = [((), ()), ((0,), (2,)), ((0, 1), (2, 0)), ((1,), (1,))]
    for history in probe:
        assert list(suffix.children(history)) == []
        assert list(suffix.extensions(history)) == []


# --------------------------------------------------------------------------- #
# End-to-end parity: infer_* with count_backend="suffix"
# --------------------------------------------------------------------------- #


def _assert_model_equal(a, b):
    assert a.epsilon == b.epsilon
    assert a.n_states == b.n_states
    assert (a.morph_counts == b.morph_counts).all()


@pytest.mark.parametrize("L_max", [1, 2, 3])
@pytest.mark.parametrize("stats_mode", ["correct", "legacy"])
def test_infer_transducer_backend_parity(L_max, stats_mode):
    rng = np.random.default_rng(7 + L_max)
    x = rng.integers(0, 2, size=600).tolist()
    y = rng.integers(0, 3, size=600).tolist()  # n_x != n_y to exercise stride
    a = infer_transducer(x, y, L_max, stats_mode=stats_mode)
    b = infer_transducer(x, y, L_max, stats_mode=stats_mode, count_backend="suffix")
    _assert_model_equal(a, b)


@pytest.mark.parametrize("L_max", [1, 2, 3])
@pytest.mark.parametrize("stats_mode", ["correct", "legacy"])
def test_infer_machine_backend_parity(L_max, stats_mode):
    rng = np.random.default_rng(99 + L_max)
    y = rng.integers(0, 4, size=600).tolist()
    a = infer_machine(y, L_max, stats_mode=stats_mode)
    b = infer_machine(y, L_max, stats_mode=stats_mode, count_backend="suffix")
    _assert_model_equal(a, b)


def test_infer_transducer_ensemble_backend_parity():
    rng = np.random.default_rng(2024)
    xs = [rng.integers(0, 2, size=n).tolist() for n in (200, 5, 150)]
    ys = [rng.integers(0, 3, size=n).tolist() for n in (200, 5, 150)]
    a = infer_transducer(xs, ys, 3, ensemble=True)
    b = infer_transducer(xs, ys, 3, ensemble=True, count_backend="suffix")
    _assert_model_equal(a, b)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_bad_count_backend_raises():
    y = [0, 1, 0, 1, 1, 0, 1]
    with pytest.raises(ValueError, match="Unknown count_backend"):
        infer_machine(y, 2, count_backend="nope")
    with pytest.raises(ValueError, match="Unknown count_backend"):
        infer_transducer(y, y, 2, count_backend="nope")


def test_store_constructor_validates():
    with pytest.raises(ValueError):
        SuffixArrayCountStore([[0, 1]], [[0]], 2, 2, 2)  # length mismatch
    with pytest.raises(ValueError):
        SuffixArrayCountStore([[0, 5]], [[0, 1]], 2, 2, 2)  # x index out of range
    with pytest.raises(ValueError):
        SuffixArrayCountStore([[0, 1]], [[0, 9]], 2, 2, 2)  # y index out of range
    with pytest.raises(ValueError):
        SuffixArrayCountStore([[0]], [[0]], -1, 2, 2)  # negative L_max
