"""Unit tests for the :class:`~transcssr.countstore.CountStore` abstraction.

The :class:`CounterCountStore` is the default backend of :func:`run_cssr`; the
parity gate in ``tests/test_cssr.py`` already exercises it end-to-end. These
tests pin the store's two queries directly: for count tables produced by
:func:`estimate_counts` on small random / structured processes, the store's
``children`` / ``extensions`` must match an *independent* brute-force
recomputation from the raw ``marg`` / ``fut`` Counters, and the documented edge
cases (null history, absent histories, zero-count keys) must behave.
"""

from __future__ import annotations

from collections import Counter

import pytest

from transcssr import CountStore, CounterCountStore
from transcssr.counting import estimate_counts

from conftest import _delay_process, _random_binary


# ---------------------------------------------------------------------------
# Independent brute-force oracles (per-history pattern match over the tables)
# ---------------------------------------------------------------------------


def _brute_children(marg: Counter, history):
    """Sorted ``(y_ind, x_ind)`` child prepends of ``history`` from ``marg``."""
    hx, hy = history
    pairs = set()
    for (mx, my), count in marg.items():
        if count == 0 or len(mx) < 2:
            continue
        if mx[1:-1] == hx and my[1:] == hy:
            pairs.add((my[0], mx[0]))
    return sorted(pairs)


def _brute_extensions(fut: Counter, history, n_y):
    """Sorted ``(flat_pos, count)`` emission extensions of ``history``."""
    hx, hy = history
    out = []
    for (fx, fy), count in fut.items():
        if count == 0 or len(fx) == 0:
            continue
        if fx[:-1] == hx and fy[:-1] == hy:
            out.append((n_y * fx[-1] + fy[-1], count))
    return sorted(out)


def _all_histories(marg: Counter, fut: Counter):
    """Every history appearing on either side of the two inverted indexes."""
    hists = set()
    for (mx, my), count in marg.items():
        if count == 0 or len(mx) < 2:
            continue
        hists.add((mx[1:-1], my[1:]))
    for (fx, fy), count in fut.items():
        if count == 0 or len(fx) == 0:
            continue
        hists.add((fx[:-1], fy[:-1]))
    return hists


# ---------------------------------------------------------------------------
# Fixtures: (marg, fut, n_y) for a few small processes
# ---------------------------------------------------------------------------


def _counts_from_binary(seed, n, L_max):
    x = _random_binary(seed, n)
    y = _random_binary(seed + 1, n)
    xs = [[int(c) for c in x]]
    ys = [[int(c) for c in y]]
    marg, fut = estimate_counts(xs, ys, L_max)
    return marg, fut, 2


def _counts_from_delay(seed, n, L_max):
    x, y = _delay_process(seed, n)
    xs = [[int(c) for c in x]]
    ys = [[int(c) for c in y]]
    marg, fut = estimate_counts(xs, ys, L_max)
    return marg, fut, 2


COUNT_CASES = [
    ("indep_L2", lambda: _counts_from_binary(101, 400, 2)),
    ("indep_L3", lambda: _counts_from_binary(202, 400, 3)),
    ("delay_L2", lambda: _counts_from_delay(303, 400, 2)),
    ("delay_L3", lambda: _counts_from_delay(404, 400, 3)),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,factory", COUNT_CASES, ids=[c[0] for c in COUNT_CASES])
def test_children_match_bruteforce(name, factory):
    marg, fut, n_y = factory()
    store = CounterCountStore(marg, fut, n_y)
    hists = _all_histories(marg, fut)
    assert hists  # sanity: the process actually produced histories
    for history in hists:
        got = list(store.children(history))
        assert got == sorted(got)  # documented ascending (y_ind, x_ind) order
        assert got == _brute_children(marg, history)


@pytest.mark.parametrize("name,factory", COUNT_CASES, ids=[c[0] for c in COUNT_CASES])
def test_extensions_match_bruteforce(name, factory):
    marg, fut, n_y = factory()
    store = CounterCountStore(marg, fut, n_y)
    hists = _all_histories(marg, fut)
    for history in hists:
        got = list(store.extensions(history))
        # flat_pos unique per history; order is irrelevant, so compare sorted.
        positions = [pos for pos, _ in got]
        assert len(positions) == len(set(positions))
        assert sorted(got) == _brute_extensions(fut, history, n_y)


def test_null_history_is_queryable():
    marg, fut, n_y = _counts_from_delay(303, 400, 2)
    store = CounterCountStore(marg, fut, n_y)
    null_history = ((), ())
    # The null history has occurring children and observed extensions.
    assert list(store.children(null_history)) == _brute_children(marg, null_history)
    assert sorted(store.extensions(null_history)) == _brute_extensions(
        fut, null_history, n_y
    )
    assert len(store.children(null_history)) > 0
    assert len(store.extensions(null_history)) > 0


def test_absent_history_returns_empty():
    marg, fut, n_y = _counts_from_delay(303, 400, 2)
    store = CounterCountStore(marg, fut, n_y)
    # A history that cannot occur in a binary process (out-of-range symbol).
    absent = ((7, 7), (7, 7))
    assert list(store.children(absent)) == []
    assert list(store.extensions(absent)) == []


def test_zero_count_keys_are_skipped():
    # A Counter can carry explicit zero-count keys (e.g. after subtraction);
    # both index passes must skip them exactly as the engine did inline.
    n_y = 2
    marg = Counter()
    # mx=(1,0,0), my=(0,1): history=(mx[1:-1], my[1:])=((0,), (1,)),
    #                       child=(my[0], mx[0])=(0, 1).
    marg[((1, 0, 0), (0, 1))] = 3
    # mx=(1,1,0), my=(0,1): history=((1,), (1,)), but zero count -> skipped.
    marg[((1, 1, 0), (0, 1))] = 0
    fut = Counter()
    # fx=(0,1), fy=(1,0): history=(fx[:-1], fy[:-1])=((0,), (1,)),
    #                     flat_pos = n_y*fx[-1] + fy[-1] = 2*1 + 0 = 2.
    fut[((0, 1), (1, 0))] = 5
    # fx=(0,1), fy=(1,1): same history, zero count -> skipped.
    fut[((0, 1), (1, 1))] = 0
    store = CounterCountStore(marg, fut, n_y)

    assert list(store.children(((0,), (1,)))) == [(0, 1)]
    # ((1,), (1,)) came only from the zero-count marg key -> absent.
    assert list(store.children(((1,), (1,)))) == []

    assert list(store.extensions(((0,), (1,)))) == [(n_y * 1 + 0, 5)]


def test_n_y_property_and_flat_stride():
    fut = Counter()
    fut[((0,), (3,))] = 2   # ey=3 with n_y=5 -> flat_pos = 5*0 + 3 = 3
    fut[((4,), (1,))] = 7   # ex=4, ey=1 -> flat_pos = 5*4 + 1 = 21
    store = CounterCountStore(Counter(), fut, n_y=5)
    assert store.n_y == 5
    assert isinstance(store, CountStore)
    assert list(store.extensions(((), ()))) == [(3, 2), (21, 7)]


def test_run_cssr_accepts_a_prebuilt_store():
    # Passing a store (fut=None) must reproduce the Counter path exactly.
    from transcssr.cssr import run_cssr

    marg, fut, n_y = _counts_from_delay(303, 400, 2)
    n_x = 2
    eps_a, inv_a, morph_a = run_cssr(marg, fut, 2, n_x, n_y)

    store = CounterCountStore(marg, fut, n_y)
    eps_b, inv_b, morph_b = run_cssr(store, None, 2, n_x, n_y)

    assert eps_a == eps_b
    assert {s: dict(h) for s, h in inv_a.items()} == {
        s: dict(h) for s, h in inv_b.items()
    }
    assert morph_a.keys() == morph_b.keys()
    for s in morph_a:
        assert (morph_a[s] == morph_b[s]).all()


def test_run_cssr_store_with_fut_raises():
    from transcssr.cssr import run_cssr

    marg, fut, n_y = _counts_from_delay(303, 200, 2)
    store = CounterCountStore(marg, fut, n_y)
    with pytest.raises(ValueError, match="fut' must be None"):
        run_cssr(store, fut, 2, 2, n_y)


def test_run_cssr_store_n_y_mismatch_raises():
    from transcssr.cssr import run_cssr

    marg, fut, n_y = _counts_from_delay(303, 200, 2)
    store = CounterCountStore(marg, fut, n_y)
    with pytest.raises(ValueError, match="does not match n_y"):
        run_cssr(store, None, 2, 2, n_y + 1)
