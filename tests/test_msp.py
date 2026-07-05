"""Tests for the mixed-state-presentation module (``transcssr.msp``).

Two layers, mirroring ``tests/test_predict.py``:

1. **Parity vs the legacy oracle.** The legacy MSP predictor
   (``transCSSR.filter_and_pred_probs``) and stationary-distribution routine
   (``compute_channel_states_distribution``) are driven end-to-end through
   Graphviz ``.dot`` files written by ``run_transCSSR`` (``all_digits=True`` for
   full-precision probabilities). The token-first model is fitted on the same
   data with ``stats_mode='legacy'`` (identical partitions), and its
   ``msp_filter`` / ``stationary_distribution`` outputs are compared with
   ``np.allclose`` (float tolerance -- eigenvector and summation-order
   differences are expected in principle, though in practice the match is
   near-exact because the same probabilities and accumulation order are used).
2. **Unit tests** on tiny hand-built or small data-fit models: predictive rows
   sum to 1, belief filtering converges to the hard filter after
   synchronization, ``log_loss`` matches a hand-computed value, and the
   restart / zero-probability semantics behave as documented.

The datasets are the same channel/process files the existing golden fixtures
use (``data/Xt_delay-channel.dat``, ``data/even.dat``, ``data/barnettX.dat``,
...), truncated so the suite stays well under a minute.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from transcssr import (
    Alphabet,
    EpsilonMachine,
    infer_machine,
    infer_transducer,
    log_loss,
    machine_transition_matrix,
    mixed_transition_matrix,
    msp_filter,
    msp_filter_machine,
    stationary_distribution,
)

# Legacy engine (parity oracle) -- tests may import both engines.
from transCSSR import (
    compute_channel_states_distribution,
    compute_eM_transition_matrix,
    estimate_predictive_distributions,
    filter_and_pred_probs,
    make_null_input_dot,
    run_transCSSR,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "transCSSR_results"


# ---------------------------------------------------------------------------
# Helpers to drive the legacy MSP pipeline into transCSSR_results/*.dot
# ---------------------------------------------------------------------------


def _read(path: str, n: int) -> str:
    return (REPO_ROOT / path).read_text().strip()[:n]


def _legacy_transducer_msp(sx, sy, axs, ays, L_max, tag):
    """Run the legacy MSP filter for a transducer; return (pred_probs, support).

    Writes the transducer dot (``{tag}X+{tag}Y.dot``) and the input-process
    epsilon-machine dot (``+{tag}X.dot``) at full precision, then invokes
    ``filter_and_pred_probs``.
    """
    e = list(itertools.product(axs, ays))
    marg, fut = estimate_predictive_distributions(sx, sy, L_max, axs=axs, ays=ays)
    run_transCSSR(marg, fut, L_max, axs, ays, e, f"{tag}X", f"{tag}Y", all_digits=True)
    e_x = list(itertools.product(["0"], axs))
    marg_x, fut_x = estimate_predictive_distributions(
        "0" * len(sx), sx, L_max, axs=["0"], ays=axs
    )
    run_transCSSR(marg_x, fut_x, L_max, ["0"], axs, e_x, "", f"{tag}X", all_digits=True)
    machine_fname = str(RESULTS / f"+{tag}X.dot")
    transducer_fname = str(RESULTS / f"{tag}X+{tag}Y.dot")
    return filter_and_pred_probs(
        sx, sy, machine_fname, transducer_fname, axs, ays, "transCSSR"
    )


def _legacy_machine_msp(sy, ays, L_max, tag):
    """Run the legacy MSP filter for an output-only machine; return pred_probs."""
    sx = "0" * len(sy)
    axs = ["0"]
    e = list(itertools.product(axs, ays))
    marg, fut = estimate_predictive_distributions(sx, sy, L_max, axs=axs, ays=ays)
    run_transCSSR(marg, fut, L_max, axs, ays, e, "", f"{tag}Y", all_digits=True)
    make_null_input_dot()
    machine_fname = str(RESULTS / "+.dot")
    transducer_fname = str(RESULTS / f"+{tag}Y.dot")
    pred, _ = filter_and_pred_probs(
        sx, sy, machine_fname, transducer_fname, axs, ays, "transCSSR"
    )
    return pred


def _legacy_machine_stationary(process_str, symbols, L_max, tag):
    """Legacy stationary distribution over an epsilon-machine's states."""
    sx = "0" * len(process_str)
    e = list(itertools.product(["0"], symbols))
    marg, fut = estimate_predictive_distributions(
        sx, process_str, L_max, axs=["0"], ays=symbols
    )
    run_transCSSR(marg, fut, L_max, ["0"], symbols, e, "", f"{tag}M", all_digits=True)
    dot = str(RESULTS / f"+{tag}M.dot")
    P, M_idx, _ = compute_eM_transition_matrix(dot, symbols, "transCSSR")
    _, stat = compute_channel_states_distribution(P, {"A": 0}, M_idx)
    return np.asarray(stat, dtype=float)


# ---------------------------------------------------------------------------
# 1. Parity: transducer MSP filter
# ---------------------------------------------------------------------------

# (tag, X-file, Y-file, N, L_max) -- delay (IID input) and barnett (a genuine
# multi-state input machine x multi-state transducer -> multi-state mixed).
TRANSDUCER_CASES = [
    ("DLY", "data/Xt_delay-channel.dat", "data/Yt_delay-channel.dat", 1500, 3),
    ("BAR", "data/barnettX.dat", "data/barnettY.dat", 2000, 3),
]


@pytest.mark.parametrize("tag,xf,yf,n,L_max", TRANSDUCER_CASES)
def test_msp_filter_parity_transducer(tag, xf, yf, n, L_max):
    sx, sy = _read(xf, n), _read(yf, n)
    axs, ays = sorted(set(sx)), sorted(set(sy))

    legacy_pred, legacy_support = _legacy_transducer_msp(sx, sy, axs, ays, L_max, tag)

    transducer = infer_transducer(
        sx, sy, L_max=L_max, stats_mode="legacy",
        input_alphabet=axs, output_alphabet=ays,
    )
    input_machine = infer_machine(sx, L_max=L_max, stats_mode="legacy", alphabet=axs)
    result = msp_filter(transducer, sx, sy, input_machine)

    assert result.pred_probs.shape == (len(sx), len(ays))
    assert np.allclose(result.pred_probs, legacy_pred, atol=1e-8)
    # Belief-support size per step is permutation-invariant to state labeling.
    assert np.array_equal(result.n_possible, legacy_support.sum(axis=1))


# ---------------------------------------------------------------------------
# 2. Parity: output-only machine MSP filter
# ---------------------------------------------------------------------------

MACHINE_CASES = [
    ("EVN", "data/even.dat", 2000, 3),
    ("GM", "data/golden-mean.dat", 2000, 3),
]


@pytest.mark.parametrize("tag,yf,n,L_max", MACHINE_CASES)
def test_msp_filter_parity_machine(tag, yf, n, L_max):
    sy = _read(yf, n)
    ays = sorted(set(sy))
    legacy_pred = _legacy_machine_msp(sy, ays, L_max, tag)

    machine = infer_machine(sy, L_max=L_max, stats_mode="legacy", alphabet=ays)
    result = msp_filter_machine(machine, sy)

    assert result.pred_probs.shape == (len(sy), len(ays))
    assert np.allclose(result.pred_probs, legacy_pred, atol=1e-8)


# ---------------------------------------------------------------------------
# 3. Parity: stationary distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag,proc,n,L_max",
    [
        ("EVNS", "data/even.dat", 2000, 3),          # 2-state machine
        ("BARS", "data/barnettX.dat", 2000, 3),      # multi-state input machine
    ],
)
def test_stationary_distribution_parity(tag, proc, n, L_max):
    process_str = _read(proc, n)
    symbols = sorted(set(process_str))
    legacy_stat = _legacy_machine_stationary(process_str, symbols, L_max, tag)

    machine = infer_machine(
        process_str, L_max=L_max, stats_mode="legacy", alphabet=symbols
    )
    mine = stationary_distribution(machine)

    assert mine.shape == (machine.n_states,)
    assert np.isclose(mine.sum(), 1.0)
    assert np.all(mine >= -1e-12)
    # Compare up to state relabeling (sorted vectors).
    assert np.allclose(np.sort(mine), np.sort(legacy_stat), atol=1e-8)


# ---------------------------------------------------------------------------
# 4. Unit tests on small / hand-built models
# ---------------------------------------------------------------------------


def _period2_machine():
    """A deterministic period-2 machine (emits ...010101...), 2 states."""
    return infer_machine("01" * 500, L_max=2, stats_mode="legacy", alphabet=["0", "1"])


def test_pred_probs_rows_sum_to_one():
    machine = _period2_machine()
    result = msp_filter_machine(machine, "010101010101")
    assert np.allclose(result.pred_probs.sum(axis=1), 1.0)
    # Belief-support entries are 0/1.
    assert set(np.unique(result.possible_states)).issubset({0, 1})


def test_belief_filter_converges_to_hard_filter():
    """After synchronization the MSP belief collapses to the hard-filter state.

    The MSP belief state at step ``t`` is ``S_t`` (the state *after* emitting
    ``Y_t``); the hard filter records at index ``t`` the state used to *predict*
    ``Y_t`` (``S_{t-1}``). Hence MSP state ``t`` corresponds to hard-filter
    state ``t + 1``. Where both are synchronized the predictive distributions
    coincide exactly (both are the model's emission distribution).
    """
    sy = _read("data/even.dat", 2000)
    machine = infer_machine(sy, L_max=3, stats_mode="legacy", alphabet=["0", "1"])
    msp = msp_filter_machine(machine, sy)
    hard = machine.filter(sy)
    T = len(sy)

    # (a) Synchronized steps are singletons.
    for t in range(T):
        if msp.synchronized[t]:
            assert msp.n_possible[t] == 1

    # (b) Predictive distributions agree where both are synchronized.
    for t in range(T):
        if msp.synchronized[t] and hard.states[t] != -1:
            assert np.allclose(msp.pred_probs[t], hard.prediction_probs[t])

    # (c) MSP state at t == hard-filter state at t + 1 (the +1 offset above),
    #     once both are synchronized. Require the bulk to agree.
    agree = matched = 0
    for t in range(T - 1):
        if msp.synchronized[t] and hard.states[t + 1] != -1:
            matched += 1
            if int(np.argmax(msp.possible_states[t])) == hard.states[t + 1]:
                agree += 1
    assert matched > 0
    assert agree >= 0.99 * matched


def test_log_loss_hand_computed():
    """log_loss on a 1-state IID machine matches ``-sum log p(y_t)`` exactly."""
    machine = EpsilonMachine(
        output_alphabet=Alphabet([0, 1]),
        n_states=1,
        # Both output histories map to the single state, so both emissions have
        # a (self-loop) transition and neither is treated as forbidden.
        epsilon={((0,), (0,)): 0, ((0,), (1,)): 0},
        morph_counts=np.array([[1, 3]]),  # p(0)=1/4, p(1)=3/4
        L_max=1,
    )
    ys = [0, 1, 1, 0, 1]
    ll = log_loss(machine, ys)

    p = np.array([0.25, 0.75, 0.75, 0.25, 0.75])
    expected_nats = float(-np.log(p).sum())
    assert np.isclose(ll.total_nats, expected_nats)
    assert np.isclose(ll.total_bits, expected_nats / np.log(2))
    assert np.isclose(ll.per_symbol_nats, expected_nats / 5)
    assert np.isclose(ll.per_symbol_bits, expected_nats / np.log(2) / 5)
    assert ll.n_symbols == 5
    assert ll.n_zero_prob == 0
    # Rows are the (constant) emission distribution.
    result = msp_filter_machine(machine, ys)
    assert np.allclose(result.pred_probs, np.tile([0.25, 0.75], (5, 1)))


def test_log_loss_transducer_convention():
    """log_loss(transducer, xs, ys) scores the observed outputs, bits==nats/ln2."""
    sx, sy = _read("data/Xt_delay-channel.dat", 800), _read(
        "data/Yt_delay-channel.dat", 800
    )
    axs, ays = ["0", "1"], ["0", "1"]
    transducer = infer_transducer(
        sx, sy, L_max=3, stats_mode="legacy",
        input_alphabet=axs, output_alphabet=ays,
    )
    input_machine = infer_machine(sx, L_max=3, stats_mode="legacy", alphabet=axs)
    ll = log_loss(transducer, sx, sy, input_machine=input_machine)

    assert ll.n_symbols == len(sy)
    assert ll.per_step_nats.shape == (len(sy),)
    # Cross-check totals against a direct scoring of the msp predictive probs.
    result = msp_filter(transducer, sx, sy, input_machine)
    iy = Alphabet(ays).encode(sy)
    direct = -np.log([result.pred_probs[t, iy[t]] for t in range(len(sy))]).sum()
    assert np.isclose(ll.total_nats, direct)
    assert np.isclose(ll.total_bits, ll.total_nats / np.log(2))


def test_log_loss_zero_probability_event():
    """A globally forbidden symbol yields prob 0, inf loss, and a counted event.

    A 1-state machine that only ever emits ``0`` assigns the observed ``1``
    probability 0 (an event no fitted ergodic model produces, but a valid
    held-out surprise). ``log_loss`` reports ``inf`` totals and counts the
    zero-probability event so callers can smooth; the filter still terminates.
    """
    machine = EpsilonMachine(
        output_alphabet=Alphabet([0, 1]),
        n_states=1,
        epsilon={((0,), (0,)): 0},
        morph_counts=np.array([[5, 0]]),  # p(0)=1, p(1)=0
        L_max=1,
    )
    ys = [0, 0, 1, 0]
    result = msp_filter_machine(machine, ys)  # must terminate
    assert result.pred_probs.shape == (4, 2)
    assert result.pred_probs[2, 1] == 0.0

    ll = log_loss(machine, ys)
    assert ll.n_zero_prob == 1
    assert np.isinf(ll.total_nats)
    assert np.isinf(ll.total_bits)
    assert np.isinf(ll.per_step_nats[2])
    # The non-forbidden steps have finite per-step loss.
    finite = ll.per_step_nats[[0, 1, 3]]
    assert np.all(np.isfinite(finite))


# ---------------------------------------------------------------------------
# 5. Transition-matrix structure
# ---------------------------------------------------------------------------


def test_machine_transition_matrix_column_stochastic():
    machine = infer_machine(
        _read("data/even.dat", 2000), L_max=3, stats_mode="legacy", alphabet=["0", "1"]
    )
    P, trans = machine_transition_matrix(machine)
    assert P.shape == (machine.n_states, machine.n_states)
    # Every source state's outgoing mass sums to 1 (column-stochastic).
    assert np.allclose(P.sum(axis=0), 1.0)
    # trans probabilities per source state sum to 1.
    by_state: dict[int, float] = {}
    for (s, _sym), (_nxt, p) in trans.items():
        by_state[s] = by_state.get(s, 0.0) + p
    for s in range(machine.n_states):
        assert np.isclose(by_state.get(s, 0.0), 1.0)


def test_mixed_transition_matrix_is_column_stochastic_and_stationary():
    sx, sy = _read("data/barnettX.dat", 2000), _read("data/barnettY.dat", 2000)
    axs, ays = sorted(set(sx)), sorted(set(sy))
    transducer = infer_transducer(
        sx, sy, L_max=3, stats_mode="legacy",
        input_alphabet=axs, output_alphabet=ays,
    )
    input_machine = infer_machine(sx, L_max=3, stats_mode="legacy", alphabet=axs)
    mixed = mixed_transition_matrix(transducer, input_machine)

    n_mixed = transducer.n_states * input_machine.n_states
    assert mixed.P.shape == (n_mixed, n_mixed)
    assert np.allclose(mixed.P.sum(axis=0), 1.0)
    # Stationary distribution is a fixed point of P: P @ pi == pi.
    pi = mixed.stationary_dist_mixed
    assert np.isclose(pi.sum(), 1.0)
    assert np.allclose(mixed.P @ pi, pi, atol=1e-8)
    # Marginal over transducer states also sums to 1.
    assert np.isclose(mixed.stationary_dist_transducer().sum(), 1.0)


def test_mixed_transition_matrix_requires_input_machine_for_transducer():
    sx, sy = _read("data/Xt_delay-channel.dat", 500), _read(
        "data/Yt_delay-channel.dat", 500
    )
    transducer = infer_transducer(
        sx, sy, L_max=2, stats_mode="legacy",
        input_alphabet=["0", "1"], output_alphabet=["0", "1"],
    )
    if transducer.n_x > 1:
        with pytest.raises(ValueError, match="input_machine is required"):
            mixed_transition_matrix(transducer, None)


def test_msp_filter_rejects_ensemble():
    machine = _period2_machine()
    with pytest.raises(ValueError, match="ensemble"):
        msp_filter(machine, ["01", "01"], ["01", "01"])
