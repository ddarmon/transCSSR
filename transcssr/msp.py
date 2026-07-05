"""Mixed-state-presentation (MSP) filtering, prediction, and scoring.

This module ports the legacy mixed-state-presentation predictor of
``transCSSR.py`` to the token-first model classes
(:class:`~transcssr.machine.EpsilonTransducer` and
:class:`~transcssr.machine.EpsilonMachine`). Unlike the "hard" length-``L``
window filter of :mod:`transcssr.predict` (which emits a ``0.5`` placeholder
until it synchronizes), the MSP filter maintains a full Bayesian belief state
over *mixed states* -- pairs ``(transducer causal state, input-machine causal
state)`` -- and therefore produces a well-defined predictive distribution
``P(Y_t = y | X_t = x_t, past)`` at *every* step, including before
synchronization, and restarts from the stationary mixed-state prior whenever a
forbidden transition is observed.

Legacy-function correspondence
------------------------------
=====================================  ===================================
This module                            ``transCSSR.py``
=====================================  ===================================
:func:`machine_transition_matrix`      ``compute_eM_transition_matrix``
:func:`stationary_distribution`        ``compute_channel_states_distribution``
:func:`mixed_transition_matrix`        ``compute_mixed_transition_matrix``
:func:`msp_filter`                     ``filter_and_pred_probs``
:func:`msp_filter_machine`             ``filter_and_pred_probs`` (null input)
:func:`log_loss`                       (new held-out scorer)
=====================================  ===================================

The legacy routines communicate through Graphviz ``.dot`` files parsed by
``load_transition_matrix_{transducer,machine}``; the token-first port reads the
same information directly off the model objects' public surface
(``transitions()``, ``emission_probs()``, the alphabets), so no file round-trip
is involved and full-precision probabilities are used throughout.

Mixed-state indexing follows the legacy convention exactly: mixed state
``nM * t + m`` pairs transducer causal state ``t`` (``0 .. nT - 1``) with
input-machine causal state ``m`` (``0 .. nM - 1``), where ``nM`` is the number
of input-machine states. The stationary distribution over mixed states is the
eigenvalue-``1`` (right) eigenvector of the column-stochastic mixed transition
matrix ``P[i, j] = P(mixed_1 = i | mixed_0 = j)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .alphabet import _looks_like_ensemble
from .machine import EpsilonMachine, EpsilonTransducer

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = [
    "machine_transition_matrix",
    "stationary_distribution",
    "mixed_transition_matrix",
    "MixedModel",
    "MSPFilterResult",
    "msp_filter",
    "msp_filter_machine",
    "LogLoss",
    "log_loss",
]


# ---------------------------------------------------------------------------
# Transition matrices and stationary distributions
# ---------------------------------------------------------------------------


def _stationary_from_column_stochastic(P: np.ndarray) -> np.ndarray:
    """Return the stationary distribution of a column-stochastic matrix.

    ``P[i, j] = P(next = i | current = j)``, so the stationary distribution is
    the right eigenvector of ``P`` for eigenvalue ``1`` (legacy
    ``compute_channel_states_distribution``). The eigenvalue nearest ``1`` is
    selected via :func:`numpy.argmin` on ``|lambda - 1|`` (deterministic
    first-minimum tie-break, matching the legacy code), its real part is taken,
    and it is normalized to sum to ``1``.

    Parameters
    ----------
    P : numpy.ndarray
        A square ``(n, n)`` column-stochastic transition matrix.

    Returns
    -------
    numpy.ndarray
        The length-``n`` stationary distribution (real, sums to ``1``).
    """
    n = P.shape[0]
    if n == 1:
        return np.array([1.0])
    vals, vecs = np.linalg.eig(P)
    eig_one = int(np.argmin(np.abs(vals - 1.0)))
    vec = np.real(vecs[:, eig_one])
    total = vec.sum()
    if total == 0.0:
        # Degenerate sign cancellation; fall back to the magnitude, which is
        # still deterministic. (Not reached for ergodic chains.)
        vec = np.abs(vec)
        total = vec.sum()
    return vec / total


def machine_transition_matrix(
    machine: EpsilonMachine,
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, float]]]:
    """Labeled state-transition matrix of an epsilon-machine.

    Token-first analogue of ``transCSSR.compute_eM_transition_matrix``. The
    machine's emitted-symbol alphabet is its *output* alphabet
    (``machine.output_alphabet``); for an input process modeled by an
    :class:`~transcssr.machine.EpsilonMachine`, those output symbols are the
    input-process symbols.

    Parameters
    ----------
    machine : EpsilonMachine
        The epsilon-machine (a single-dummy-input transducer, ``n_x == 1``).

    Returns
    -------
    P : numpy.ndarray
        The ``(n_states, n_states)`` column-stochastic transition matrix,
        ``P[i, j] = P(state_1 = i | state_0 = j)`` summed over emitted symbols.
    trans : dict
        Mapping ``(state, symbol_index) -> (next_state, probability)`` where
        ``symbol_index`` indexes ``machine.output_alphabet`` and the
        probabilities are renormalized per source state (matching the legacy
        defensive renormalization).

    Notes
    -----
    Only symbols with a defined transition out of the state contribute to the
    per-state normalization, exactly as in the legacy routine.
    """
    n_states = machine.n_states
    n_sym = machine.n_y
    probs = machine.emission_probs()  # (n_states, n_x * n_y); n_x == 1
    raw_trans = machine.transitions()  # (state, 0, y_idx) -> next

    P = np.zeros((n_states, n_states))
    trans: dict[tuple[int, int], tuple[int, float]] = {}

    for s in range(n_states):
        row = np.zeros(n_sym)
        nexts: dict[int, int] = {}
        for y in range(n_sym):
            nxt = raw_trans.get((s, 0, y), -1)
            if nxt == -1:
                continue
            p = probs[s, y]
            if np.isnan(p):
                continue
            row[y] = p
            nexts[y] = nxt
        total = row.sum()
        if total == 0.0:
            continue
        row = row / total
        for y, nxt in nexts.items():
            p = float(row[y])
            trans[(s, y)] = (nxt, p)
            P[nxt, s] += p
    return P, trans


def stationary_distribution(machine: EpsilonMachine) -> np.ndarray:
    """Stationary distribution over an epsilon-machine's causal states.

    Token-first analogue of ``transCSSR.compute_channel_states_distribution``
    applied to the machine's own transition matrix
    (:func:`machine_transition_matrix`): the eigenvalue-``1`` right eigenvector
    of the column-stochastic transition matrix, real part, normalized to sum to
    ``1``.

    Parameters
    ----------
    machine : EpsilonMachine
        The epsilon-machine.

    Returns
    -------
    numpy.ndarray
        The length-``n_states`` stationary distribution.
    """
    P, _ = machine_transition_matrix(machine)
    return _stationary_from_column_stochastic(P)


def _transducer_transitions(
    transducer: EpsilonTransducer,
) -> dict[tuple[int, int, int], tuple[int, float]]:
    """Build the labeled transducer transition lookup.

    Returns ``(state, x_index, y_index) -> (next_state, P(Y = y | X = x, state))``
    for every defined transition whose emission probability is not ``nan``.
    Token-first analogue of the ``T_trans`` produced by
    ``load_transition_matrix_transducer``.
    """
    n_y = transducer.n_y
    probs = transducer.emission_probs()
    raw = transducer.transitions()
    trans: dict[tuple[int, int, int], tuple[int, float]] = {}
    for (s, x, y), nxt in raw.items():
        p = probs[s, x * n_y + y]
        if np.isnan(p):
            continue
        trans[(s, x, y)] = (nxt, float(p))
    return trans


@dataclass
class MixedModel:
    """The mixed-state presentation of a joint (input, output) process.

    Bundles the transducer/input-machine transition lookups, the mixed-state
    transition matrix, and its stationary distribution -- everything
    :func:`msp_filter` needs. Produced by :func:`mixed_transition_matrix`.

    Attributes
    ----------
    P : numpy.ndarray
        The ``(nT * nM, nT * nM)`` column-stochastic mixed transition matrix,
        ``P[i, j] = P(mixed_1 = i | mixed_0 = j)``. Mixed state ``nM * t + m``
        pairs transducer state ``t`` with input-machine state ``m``.
    stationary_dist_mixed : numpy.ndarray
        The stationary distribution over the ``nT * nM`` mixed states.
    n_transducer_states, n_machine_states : int
        ``nT`` and ``nM``.
    n_y : int
        Number of output symbols.
    t_trans : dict
        ``(transducer_state, x_index, y_index) -> (next_state, prob)``.
    m_trans : dict
        ``(machine_state, x_index) -> (next_state, prob)`` where ``x_index``
        indexes the *transducer* input alphabet.
    """

    P: np.ndarray
    stationary_dist_mixed: np.ndarray
    n_transducer_states: int
    n_machine_states: int
    n_y: int
    t_trans: dict[tuple[int, int, int], tuple[int, float]]
    m_trans: dict[tuple[int, int], tuple[int, float]]

    def stationary_dist_transducer(self) -> np.ndarray:
        """Marginal stationary distribution over the transducer causal states."""
        nM = self.n_machine_states
        return np.array(
            [
                self.stationary_dist_mixed[t * nM : (t + 1) * nM].sum()
                for t in range(self.n_transducer_states)
            ]
        )


def _trivial_input_machine_trans(
    n_x: int,
) -> dict[tuple[int, int], tuple[int, float]]:
    """Transition lookup of the single-state input machine over ``n_x`` inputs.

    The output-only (epsilon-machine) case uses a dummy constant input
    (``n_x == 1``); the legacy code models this with the null-input machine
    ``transCSSR_results/+.dot`` (one state ``A`` emitting the single input
    symbol with probability ``1``). This returns that machine's transition
    lookup: state ``0`` self-loops on input index ``0`` with probability ``1``.
    """
    return {(0, x): (0, 1.0) for x in range(n_x)}


def mixed_transition_matrix(
    transducer: EpsilonTransducer,
    input_machine: Optional[EpsilonMachine] = None,
) -> MixedModel:
    """Mixed-state transition matrix for a transducer and its input process.

    Token-first port of ``transCSSR.compute_mixed_transition_matrix`` (plus the
    stationary-distribution step of ``compute_channel_states_distribution``).
    The mixed states are the direct product of the transducer causal states and
    the input-machine causal states.

    Parameters
    ----------
    transducer : EpsilonTransducer
        The input/output transducer.
    input_machine : EpsilonMachine, optional
        An epsilon-machine for the input process, whose *output* alphabet is the
        transducer's *input* alphabet (token identity is used to align the two,
        so the alphabets need not share an index order). If ``None`` and the
        transducer is an output-only model (``n_x == 1``), the trivial
        single-state null-input machine is used, so the mixed states degenerate
        to the transducer's own causal states.

    Returns
    -------
    MixedModel
        The bundled mixed-state presentation.

    Raises
    ------
    ValueError
        If ``input_machine`` is ``None`` but the transducer has ``n_x > 1`` (a
        genuine input process, for which an input machine is required).
    KeyError
        If an input-machine output token is absent from the transducer's input
        alphabet.
    """
    nT = transducer.n_states
    n_x = transducer.n_x
    n_y = transducer.n_y
    t_trans = _transducer_transitions(transducer)

    if input_machine is None:
        if n_x != 1:
            raise ValueError(
                "input_machine is required for a transducer with n_x > 1 "
                f"(got n_x={n_x}); pass an EpsilonMachine for the input "
                "process. Only the output-only case (n_x == 1) may omit it."
            )
        nM = 1
        m_trans = _trivial_input_machine_trans(n_x)
    else:
        nM = input_machine.n_states
        _, mach_trans = machine_transition_matrix(input_machine)
        # Map input-machine output symbols to transducer input indices by token.
        out_to_in: dict[int, int] = {}
        for j in range(input_machine.n_y):
            token = input_machine.output_alphabet.token(j)
            out_to_in[j] = transducer.input_alphabet.index(token)
        m_trans = {
            (s, out_to_in[sym]): (nxt, p)
            for (s, sym), (nxt, p) in mach_trans.items()
        }

    n_mixed = nT * nM
    P = np.zeros((n_mixed, n_mixed))
    for t_from in range(nT):
        for m_from in range(nM):
            j_from = t_from * nM + m_from
            for x in range(n_x):
                m_res = m_trans.get((m_from, x))
                if m_res is None:
                    continue
                m_to, pM = m_res
                for y in range(n_y):
                    t_res = t_trans.get((t_from, x, y))
                    if t_res is None:
                        continue
                    t_to, pT = t_res
                    i_to = t_to * nM + m_to
                    P[i_to, j_from] += pT * pM

    stationary = _stationary_from_column_stochastic(P)
    return MixedModel(
        P=P,
        stationary_dist_mixed=stationary,
        n_transducer_states=nT,
        n_machine_states=nM,
        n_y=n_y,
        t_trans=t_trans,
        m_trans=m_trans,
    )


# ---------------------------------------------------------------------------
# The MSP belief-state filter
# ---------------------------------------------------------------------------


@dataclass
class MSPFilterResult:
    """Output of :func:`msp_filter` / :func:`msp_filter_machine`.

    Attributes
    ----------
    pred_probs : numpy.ndarray
        Array of shape ``(T, n_y)``. Row ``t`` is the predictive distribution
        ``P(Y_t = y | X_t = x_t, past)``, defined at *every* step (including
        before synchronization), and rows sum to ``1`` wherever the observed
        input history is allowed by the model. Legacy ``pred_probs_by_time``.
    possible_states : numpy.ndarray
        Integer array of shape ``(T, nT)``. ``possible_states[t, s] == 1`` when
        transducer causal state ``s`` is in the belief support at step ``t``
        (legacy ``cur_states_by_time``). ``possible_states[t].sum() == 1`` marks
        a synchronized step.
    n_possible : numpy.ndarray
        ``possible_states.sum(axis=1)``: the belief-support size per step.
    synchronized : numpy.ndarray
        Boolean array of shape ``(T,)``, ``True`` where ``n_possible == 1``.
    """

    pred_probs: np.ndarray
    possible_states: np.ndarray
    n_possible: np.ndarray
    synchronized: np.ndarray


def _encode_pair(
    transducer: EpsilonTransducer, xs: Any, ys: Any
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Encode a single (xs, ys) realization to aligned index tuples."""
    if _looks_like_ensemble(xs) or _looks_like_ensemble(ys):
        raise ValueError(
            "msp_filter is defined for a single input/output realization "
            "only; got what looks like an ensemble (list of sequences). "
            "Filter each realization separately."
        )
    ix = transducer.input_alphabet.encode(xs)
    iy = transducer.output_alphabet.encode(ys)
    if len(ix) != len(iy):
        raise ValueError(
            f"Input and output sequences must have the same length; got "
            f"{len(ix)} and {len(iy)}."
        )
    return ix, iy


def msp_filter(
    transducer: EpsilonTransducer,
    xs: Any,
    ys: Any,
    input_machine: Optional[EpsilonMachine] = None,
    mixed_model: Optional[MixedModel] = None,
) -> MSPFilterResult:
    """Mixed-state-presentation belief filter for one input/output realization.

    Token-first port of ``transCSSR.filter_and_pred_probs``. Starting from the
    stationary mixed-state prior, it accumulates per-mixed-state path
    probabilities, emits a predictive distribution ``P(Y_t | X_t, past)`` at
    every step (defined even before synchronization), and *restarts* from the
    stationary prior whenever a forbidden transition is observed (tracking the
    restart time ``t_unsynched``). The belief support over transducer causal
    states is reported per step.

    Parameters
    ----------
    transducer : EpsilonTransducer
        The fitted input/output transducer (or an
        :class:`~transcssr.machine.EpsilonMachine` for the output-only case).
    xs, ys : sequence
        A single input / output realization each (``str``, ``list``/``tuple``
        of tokens, or 1-d :class:`numpy.ndarray`). Ensembles are rejected.
    input_machine : EpsilonMachine, optional
        Epsilon-machine for the input process. Optional (and ignored) when the
        transducer is output-only (``n_x == 1``); required otherwise. Its output
        alphabet must be the transducer's input alphabet (matched by token).
    mixed_model : MixedModel, optional
        A precomputed :class:`MixedModel` (from :func:`mixed_transition_matrix`)
        to reuse across calls; recomputed from ``transducer`` / ``input_machine``
        when ``None``.

    Returns
    -------
    MSPFilterResult
        Per-step predictive distributions and belief support.

    Notes
    -----
    Restart-on-forbidden-transition semantics (legacy §``filter_and_pred_probs``):
    when the belief support collapses to zero states (a symbol pair allowed by
    no surviving mixed-state path) *or* a synchronized state observes a
    forbidden output, filtering restarts from the full stationary mixed-state
    prior at that time step, exactly reproducing the legacy behavior. The
    predictive probabilities at and after such a restart are the memoryless
    (stationary-prior) predictions until re-synchronization.
    """
    ix, iy = _encode_pair(transducer, xs, ys)
    if mixed_model is None:
        mixed_model = mixed_transition_matrix(transducer, input_machine)

    nT = mixed_model.n_transducer_states
    nM = mixed_model.n_machine_states
    n_y = mixed_model.n_y
    t_trans = mixed_model.t_trans
    m_trans = mixed_model.m_trans
    stationary = mixed_model.stationary_dist_mixed
    n_mixed = stationary.shape[0]

    T = len(ix)

    p_joint_L = np.zeros(T + 1)
    p_joint_Lp1 = np.zeros((T + 1, n_y))
    p_input_L = np.zeros(T + 1)
    p_input_Lp1 = np.zeros(T + 1)

    cur_states = np.zeros((T, nT), dtype=np.int16)
    pred_probs = np.zeros((T, n_y))

    mixed_states = np.arange(n_mixed)

    def _fresh_start():
        active = stationary > 0
        p_eT = np.ones(n_mixed)
        p_eM = np.ones(n_mixed)
        t_from = np.full(n_mixed, -1, dtype=int)
        m_from = np.full(n_mixed, -1, dtype=int)
        for idx in mixed_states[active]:
            t_from[idx] = idx // nM
            m_from[idx] = idx % nM
        return active, p_eT, p_eM, t_from, m_from

    active, p_eT, p_eM, T_from, M_from = _fresh_start()

    is_synched = False
    t_unsynched = 0
    t = 0

    while t < T:
        if not is_synched:
            # Whether this pass is the fresh (stationary-prior) restart for t.
            entered_fresh = t == t_unsynched
            for mixed in mixed_states[active]:
                t_state_from = int(T_from[mixed])
                m_state_from = int(M_from[mixed])
                x = ix[t]

                if t == t_unsynched:
                    # Restart: P(Y_t | X_t) from the (reset) stationary prior.
                    m_res = m_trans.get((m_state_from, x))
                    p_eM_new = 0.0 if m_res is None else p_eM[mixed] * m_res[1]

                    if p_eM_new != 0.0:
                        for y in range(n_y):
                            t_res = t_trans.get((t_state_from, x, y))
                            p_eT_new = 0.0 if t_res is None else p_eT[mixed] * t_res[1]
                            p_joint_Lp1[t, y] += p_eT_new * p_eM_new * stationary[mixed]
                        p_input_Lp1[t] += p_eM_new * stationary[mixed]
                        pred_probs[t, :] = p_joint_Lp1[t, :] / p_input_Lp1[t]

                    # Advance this mixed-state path on the observed (x, y).
                    y_obs = iy[t]
                    t_res = t_trans.get((t_state_from, x, y_obs))
                    if t_res is None:
                        active[mixed] = False
                    else:
                        t_state_to, pT = t_res
                        p_eT[mixed] = p_eT[mixed] * pT
                        m_res2 = m_trans.get((m_state_from, x))
                        pM = 0.0 if m_res2 is None else m_res2[1]
                        m_state_to = m_state_from if m_res2 is None else m_res2[0]
                        p_eM[mixed] = p_eM[mixed] * pM
                        T_from[mixed] = t_state_to
                        M_from[mixed] = m_state_to
                        if p_eT[mixed] != 0.0:
                            cur_states[t, t_state_to] = 1
                        p_joint_L[t + 1] += p_eT[mixed] * p_eM[mixed] * stationary[mixed]
                        p_input_L[t + 1] += p_eM[mixed] * stationary[mixed]
                else:
                    # Still desynchronized: accumulate the conditional predictive.
                    m_res = m_trans.get((m_state_from, x))
                    p_eM_new = 0.0 if m_res is None else p_eM[mixed] * m_res[1]
                    p_input_Lp1[t] += p_eM_new * stationary[mixed]
                    for y in range(n_y):
                        t_res = t_trans.get((t_state_from, x, y))
                        p_eT_new = 0.0 if t_res is None else p_eT[mixed] * t_res[1]
                        p_joint_Lp1[t, y] += p_eT_new * p_eM_new * stationary[mixed]

                    if (
                        p_input_Lp1[t] == 0
                        or p_input_L[t] == 0
                        or p_joint_L[t] / p_input_L[t] == 0
                    ):
                        pass
                    else:
                        pred_probs[t, :] = (p_joint_Lp1[t, :] / p_input_Lp1[t]) / (
                            p_joint_L[t] / p_input_L[t]
                        )

                    y_obs = iy[t]
                    t_res = t_trans.get((t_state_from, x, y_obs))
                    if t_res is None:
                        active[mixed] = False
                    else:
                        t_state_to, pT = t_res
                        p_eT[mixed] = p_eT[mixed] * pT
                        m_res2 = m_trans.get((m_state_from, x))
                        pM = 0.0 if m_res2 is None else m_res2[1]
                        m_state_to = m_state_from if m_res2 is None else m_res2[0]
                        p_eM[mixed] = p_eM[mixed] * pM
                        T_from[mixed] = t_state_to
                        M_from[mixed] = m_state_to
                        if p_eT[mixed] != 0.0:
                            cur_states[t, t_state_to] = 1
                        p_joint_L[t + 1] += p_eT[mixed] * p_eM[mixed] * stationary[mixed]
                        p_input_L[t + 1] += p_eM[mixed] * stationary[mixed]

            num_active = int(cur_states[t, :].sum())
            if num_active == 1:
                is_synched = True
                t += 1
            elif num_active == 0:
                if entered_fresh:
                    # This step was already processed from the fresh stationary
                    # prior and *still* has no surviving path: the observed
                    # symbol is forbidden even memorylessly (globally
                    # impossible). Legacy ``filter_and_pred_probs`` would loop
                    # forever here; we instead advance, leaving the memoryless
                    # ``pred_probs[t]`` (which assigns the observed symbol
                    # probability 0) in place and restarting at ``t + 1``. This
                    # branch is unreachable for any symbol a fitted, ergodic
                    # model emits, so it never affects legacy parity on real
                    # data; it only guarantees termination and a defined
                    # zero-probability event for held-out scoring.
                    t += 1
                    t_unsynched = t
                    is_synched = False
                    active, p_eT, p_eM, T_from, M_from = _fresh_start()
                else:
                    # Followed a path allowed by no mixed state: force a restart
                    # from the stationary prior at this same step (legacy).
                    is_synched = False
                    t_unsynched = t
                    active, p_eT, p_eM, T_from, M_from = _fresh_start()
            else:
                t += 1
        else:  # is_synched
            which_state = int(np.argmax(cur_states[t - 1, :]))
            t_state_from = which_state
            x = ix[t]
            for y in range(n_y):
                t_res = t_trans.get((t_state_from, x, y))
                pred_probs[t, y] = 0.0 if t_res is None else t_res[1]

            y_obs = iy[t]
            t_res = t_trans.get((t_state_from, x, y_obs))
            if t_res is None:
                # Forbidden output at a synchronized state: restart.
                is_synched = False
                t_unsynched = t
                active, p_eT, p_eM, T_from, M_from = _fresh_start()
            else:
                t_state_to = t_res[0]
                cur_states[t, t_state_to] = 1
                t += 1

    n_possible = cur_states.sum(axis=1)
    synchronized = n_possible == 1
    return MSPFilterResult(
        pred_probs=pred_probs,
        possible_states=cur_states,
        n_possible=n_possible,
        synchronized=synchronized,
    )


def msp_filter_machine(
    machine: EpsilonMachine, ys: Any
) -> MSPFilterResult:
    """MSP belief filter for an output-only epsilon-machine.

    Convenience wrapper around :func:`msp_filter` for the output-only case:
    the constant dummy input stream (single input token, index ``0``) is
    supplied internally (matching the legacy ``stringX = '0' * len(stringY)``
    convention and the null-input machine ``+.dot``), so the mixed states
    degenerate to the machine's own causal states.

    Parameters
    ----------
    machine : EpsilonMachine
        The fitted output-only model.
    ys : sequence
        A single output realization.

    Returns
    -------
    MSPFilterResult
        Per-step predictive distributions and belief support over the machine's
        causal states.
    """
    dummy_token = machine.input_alphabet.token(0)
    xs = [dummy_token] * len(ys)
    return msp_filter(machine, xs, ys, input_machine=None)


# ---------------------------------------------------------------------------
# Held-out scoring
# ---------------------------------------------------------------------------


@dataclass
class LogLoss:
    """Held-out log-loss of a model on one realization.

    Attributes
    ----------
    total_nats, total_bits : float
        Summed negative log predictive probability of the observed outputs, in
        nats (natural log) and bits (log base 2). ``inf`` when any observed
        symbol had predictive probability ``0`` (see ``n_zero_prob``).
    per_symbol_nats, per_symbol_bits : float
        The totals divided by ``n_symbols`` (mean cross-entropy; "bits per
        token"). ``inf`` when ``n_zero_prob > 0``.
    n_symbols : int
        Number of scored output symbols (``T``).
    n_zero_prob : int
        Count of steps whose observed output had predictive probability ``0``
        (forbidden events). Reported separately so callers may apply their own
        smoothing; when ``> 0`` the totals are ``inf``.
    per_step_nats : numpy.ndarray
        The per-step negative log (nats) of the observed symbol's predictive
        probability, length ``T`` (``inf`` at zero-probability steps). Lets
        callers inspect or smooth individual events.
    """

    total_nats: float
    total_bits: float
    per_symbol_nats: float
    per_symbol_bits: float
    n_symbols: int
    n_zero_prob: int
    per_step_nats: np.ndarray


_LOG2 = np.log(2.0)


def log_loss(
    model: EpsilonTransducer,
    xs: Any,
    ys: Any = None,
    input_machine: Optional[EpsilonMachine] = None,
    mixed_model: Optional[MixedModel] = None,
) -> LogLoss:
    """Held-out log-loss (cross-entropy) of a fitted model on one realization.

    Runs the MSP filter (:func:`msp_filter` / :func:`msp_filter_machine`) to
    obtain the always-defined predictive distributions ``P(Y_t | X_t, past)``,
    then scores the *observed* output symbols. This is the evaluation entry
    point for language-model-style experiments (bits per token).

    Two calling conventions:

    * **Output-only** ``log_loss(machine, ys)`` -- ``model`` is an
      :class:`~transcssr.machine.EpsilonMachine` and ``ys`` is omitted; the
      single sequence argument is the output stream.
    * **Transducer** ``log_loss(transducer, xs, ys, input_machine=...)`` --
      both input and output streams are given.

    Parameters
    ----------
    model : EpsilonTransducer or EpsilonMachine
        The fitted model.
    xs : sequence
        The input stream (transducer) or, in the output-only convention, the
        output stream (``ys`` left ``None``).
    ys : sequence, optional
        The output stream. Omit only for the output-only convention.
    input_machine : EpsilonMachine, optional
        Input-process machine for a genuine transducer (``n_x > 1``); see
        :func:`msp_filter`.
    mixed_model : MixedModel, optional
        Precomputed mixed-state presentation to reuse.

    Returns
    -------
    LogLoss
        Total and per-symbol log-loss in nats and bits, the scored-symbol
        count, the zero-probability event count, and the per-step nats array.

    Notes
    -----
    Zero-probability handling: when an observed symbol has predictive
    probability ``0`` (an event the model forbids), its contribution is
    ``+inf`` and ``n_zero_prob`` is incremented; the totals therefore become
    ``inf``. The count is reported so callers can apply their own smoothing
    (e.g. add-``k`` or a fallback distribution) rather than having a single
    forbidden event silently dominate.
    """
    if isinstance(model, EpsilonMachine) and ys is None:
        result = msp_filter_machine(model, xs)
        iy = model.output_alphabet.encode(xs)
    else:
        if ys is None:
            raise ValueError(
                "log_loss on a transducer requires both xs and ys; got ys=None. "
                "For an output-only EpsilonMachine call log_loss(machine, ys)."
            )
        result = msp_filter(model, xs, ys, input_machine, mixed_model)
        iy = model.output_alphabet.encode(ys)

    pred = result.pred_probs
    T = len(iy)
    per_step = np.empty(T)
    n_zero = 0
    for t in range(T):
        p = pred[t, iy[t]]
        if p <= 0.0:
            per_step[t] = np.inf
            n_zero += 1
        else:
            per_step[t] = -np.log(p)

    total_nats = float(per_step.sum())
    total_bits = total_nats / _LOG2
    n = T if T > 0 else 1
    return LogLoss(
        total_nats=total_nats,
        total_bits=total_bits,
        per_symbol_nats=total_nats / n,
        per_symbol_bits=total_bits / n,
        n_symbols=T,
        n_zero_prob=n_zero,
        per_step_nats=per_step,
    )
