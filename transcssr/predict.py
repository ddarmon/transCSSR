"""Filtering and prediction with a fitted token-first model.

This module reimplements the semantics of the legacy
:func:`transCSSR.filter_and_predict` (``docs/SPEC-core.md`` §5) against the
token-first :class:`~transcssr.machine.EpsilonTransducer`. It operates on
**index sequences** produced by the model's alphabets; tokens are encoded on
the way in and predicted output indices are decoded back to tokens on the way
out.

The legacy routine returns three parallel objects
(``filtered_states``, ``filtered_probs``, ``stringY_pred``); this module packs
the same information into a :class:`FilterResult`:

* ``states`` -- one integer causal-state label per time step, ``-1`` where the
  filter is not yet synchronized (legacy ``filtered_states``).
* ``predictions`` -- the MAP-predicted output *token* per step, ``None`` where
  the legacy engine emitted the literal placeholder ``'N'`` (legacy
  ``stringY_pred`` with each character a token).
* ``prediction_probs`` -- a dense ``(T, n_y)`` array of per-step emission
  probabilities ``P(Y_t = y | X_t, past)`` for the state and input used at that
  step. See the class docstring for exactly what unsynchronized rows contain
  and how they correspond to the legacy ``filtered_probs`` vector.

**Legacy ``filtered_probs`` correspondence.** The legacy ``filtered_probs``
is hardwired to output index ``1`` (``prob_by_state[state][n_y*x + 1]``; a
binary-``Y`` assumption, ``docs/SPEC-core.md`` §6.3.6). For a binary output
alphabet, ``result.prediction_probs[:, 1]`` reproduces legacy
``filtered_probs`` **exactly**, including the ``0.5`` fallback at
unsynchronized steps.

**Unsynchronized rows.** The legacy engine only defines a single scalar at an
unsynchronized step: ``filtered_probs = 0.5`` (its "probability of emitting a
1" default). It leaves the rest of the distribution undefined. To keep
``prediction_probs`` a well-formed dense array whose column ``1`` still
reproduces legacy ``filtered_probs``, every column of an unsynchronized row is
filled with the legacy ``0.5`` fallback. Synchronized rows are the model's
row-normalized emission distribution for the observed input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Hashable, Optional

import numpy as np

from .alphabet import _looks_like_ensemble

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .machine import EpsilonTransducer

__all__ = ["FilterResult", "filter_and_predict"]

# Sentinel output "index" used to mirror the legacy memoryless ``'n'`` output
# placeholder (which is deliberately *not* a real output symbol). It never
# matches a real, integer-indexed history key, exactly like the legacy 'n'.
_MEM_Y: int = -1


@dataclass
class FilterResult:
    """The output of :meth:`EpsilonTransducer.filter` / :func:`filter_and_predict`.

    Attributes
    ----------
    states : list of int
        The filtered causal-state sequence, one entry per time step. ``-1``
        marks steps where the filter is not (yet) synchronized to a causal
        state (legacy ``filtered_states``).
    predictions : list
        The MAP-predicted output *token* per step, or ``None`` where the state
        is ``-1`` (the legacy engine emitted the literal character ``'N'``
        there). Rendering ``None`` as ``'N'`` and string-joining single-token
        outputs reproduces the legacy ``stringY_pred``.
    prediction_probs : numpy.ndarray
        Array of shape ``(T, n_y)``. Row ``t`` is ``P(Y_t = y | X_t, past)``
        for the filtered state and observed input at step ``t``. At
        unsynchronized steps (``states[t] == -1``) every column holds the
        legacy ``0.5`` fallback; consequently, for a binary output alphabet,
        ``prediction_probs[:, 1]`` equals the legacy ``filtered_probs`` vector
        exactly.
    """

    states: list[int]
    predictions: list[Optional[Hashable]]
    prediction_probs: np.ndarray


def _transitions_memoryless(model: "EpsilonTransducer") -> dict[tuple[int, int], int]:
    """Build an output-memoryless transition map ``(state, x_idx) -> next``.

    Mirrors legacy ``get_transitions(..., memoryless=True)``
    (``docs/SPEC-core.md`` §4.4), which substitutes the placeholder output
    ``'n'`` for the real output symbol in every key and lookup. Because a model
    trained on real outputs has no ``'n'``-keyed histories, this map is empty
    for such models and the memoryless filter never synchronizes -- matching
    the legacy behavior when ``memoryless=True`` is applied to a memoryful
    model. It is provided for signature/behavior completeness; the shipped
    fixtures and tests use ``memoryless=False``.
    """
    from .machine import _iter_transition_sources

    inv = model.histories_by_state
    trans: dict[tuple[int, int], int] = {}
    for state in range(model.n_states):
        for base_x, base_y in _iter_transition_sources(
            inv.get(state, []), model.L_max
        ):
            for x_idx in range(model.n_x):
                key = (base_x + (x_idx,), base_y + (_MEM_Y,))
                nxt = model.epsilon.get(key, -1)
                if nxt != -1:
                    trans[(state, x_idx)] = nxt
    return trans


def filter_and_predict(
    model: "EpsilonTransducer",
    xs: Any,
    ys: Any,
    memoryless: bool = False,
) -> FilterResult:
    """Filter causal states and predict outputs for one input/output sequence.

    Reproduces the legacy :func:`transCSSR.filter_and_predict` synchronization,
    state-following, resynchronization, and MAP-prediction semantics
    (``docs/SPEC-core.md`` §5) exactly on the *index sequences* obtained by
    encoding ``xs`` / ``ys`` through the model's alphabets. Predicted output
    indices are decoded back to tokens.

    Parameters
    ----------
    model : EpsilonTransducer
        A fitted model (or :class:`~transcssr.machine.EpsilonMachine`).
    xs, ys : sequence
        A **single** input / output realization each (``str``, ``list``/
        ``tuple`` of tokens, or 1-d :class:`numpy.ndarray`). Filtering is
        defined for a single sequence only, matching the legacy engine: an
        ensemble (a list whose elements are themselves sequences) is rejected
        with :class:`ValueError`. Consequently a single sequence of tuple
        tokens or multi-character (word) string tokens -- which is
        indistinguishable from an ensemble by type -- cannot be filtered.
    memoryless : bool, optional
        If ``True``, use output-memoryless synchronization/transitions (legacy
        ``memoryless=True``). See :func:`_transitions_memoryless`; not exercised
        by the shipped fixtures. Default ``False``.

    Returns
    -------
    FilterResult
        The filtered states, per-step predicted tokens, and the ``(T, n_y)``
        prediction-probability matrix. ``T == len(xs)``.

    Raises
    ------
    ValueError
        If ``xs`` or ``ys`` is an ensemble (list of sequences), or if the two
        realizations differ in length.
    KeyError
        If any token is not in the model's alphabet.
    """
    if _looks_like_ensemble(xs) or _looks_like_ensemble(ys):
        raise ValueError(
            "filter_and_predict is defined for a single input/output "
            "realization only; got what looks like an ensemble (list of "
            "sequences). Filter each realization separately."
        )

    ix = model.input_alphabet.encode(xs)
    iy = model.output_alphabet.encode(ys)
    if len(ix) != len(iy):
        raise ValueError(
            f"Input and output sequences must have the same length; got "
            f"{len(ix)} and {len(iy)}."
        )

    L = model.L_max
    n_x = model.n_x
    n_y = model.n_y
    epsilon = model.epsilon
    probs = model.emission_probs()  # (n_states, n_x * n_y), row-normalized

    if memoryless:
        trans_mem = _transitions_memoryless(model)
    else:
        trans = model.transitions()  # (state, x_idx, y_idx) -> next

    T = len(ix)

    def _row(state: int, x_ord: int) -> np.ndarray:
        """Return the length-``n_y`` prediction-prob row for a step."""
        if state == -1:
            # Legacy defines only the scalar filtered_probs = 0.5 here; we
            # broadcast it so column 1 reproduces the legacy vector.
            return np.full(n_y, 0.5)
        start = x_ord * n_y
        return probs[state, start : start + n_y]

    def _eps_lookup(x_slice: tuple[int, ...], y_slice: tuple[int, ...]) -> int:
        """History -> state lookup, honoring the ``memoryless`` placeholder."""
        if memoryless:
            y_key = (_MEM_Y,) * len(x_slice)
        else:
            y_key = y_slice
        return epsilon.get((x_slice, y_key), -1)

    def _step(state: int, x_ord: int, e_x: int, e_y: int) -> int:
        """Follow the transition of ``state`` on emission ``(e_x, e_y)``."""
        if memoryless:
            return trans_mem.get((state, e_x), -1)
        return trans.get((state, e_x, e_y), -1)

    filtered_states: list[int] = []
    prob_rows: list[np.ndarray] = []

    # -- First L-1 steps cannot be synchronized (SPEC §5.2). ---------------
    for _ in range(L - 1):
        filtered_states.append(-1)
        prob_rows.append(np.full(n_y, 0.5))

    # -- Initial state at index L-1. ---------------------------------------
    s0 = _eps_lookup(ix[: L - 1], iy[: L - 1])
    if model.n_states == 1:
        # Single-state machine: we always know the state (SPEC §5.2).
        s0 = 0
    filtered_states.append(s0)
    cur_x_ord = ix[L - 1]
    prob_rows.append(_row(s0, cur_x_ord))

    # -- Slide forward until first synchronization (SPEC §5.2). ------------
    num_ahead = 0
    while s0 == -1 and len(filtered_states) < len(ix) - 1:
        s0 = _eps_lookup(ix[num_ahead : L + num_ahead], iy[num_ahead : L + num_ahead])
        filtered_states.append(s0)
        # LEGACY QUIRK (SPEC §5.2): the while-loop reads the input at
        # ``L + num_ahead + 1`` (an off-by-one vs. the appended index). It only
        # affects the probability row on the step that first synchronizes;
        # replicated for exact parity.
        cur_x_ord = ix[L + num_ahead + 1]
        prob_rows.append(_row(s0, cur_x_ord))
        num_ahead += 1

    # -- Main loop: follow transitions / resynchronize (SPEC §5.2). --------
    for ind in range(L - 1 + num_ahead, len(ix) - 1):
        e_x, e_y = ix[ind], iy[ind]
        if s0 == -1:
            # Try to resynchronize by a direct length-L window lookup.
            s1 = _eps_lookup(ix[ind - L + 1 : ind + 1], iy[ind - L + 1 : ind + 1])
        else:
            s1 = _step(s0, ix[ind], e_x, e_y)
        filtered_states.append(s1)
        cur_x_ord = ix[ind + 1]
        prob_rows.append(_row(s1, cur_x_ord))
        s0 = s1

    # -- MAP prediction per step (SPEC §5.3). ------------------------------
    predictions: list[Optional[Hashable]] = []
    for t_ind in range(len(filtered_states)):
        state = filtered_states[t_ind]
        if state == -1:
            predictions.append(None)  # legacy 'N' placeholder
            continue
        x_ord = ix[t_ind]
        start = x_ord * n_y
        row = probs[state, start : start + n_y]
        # numpy.argmax first-max tie-break (SPEC §5.3, §6.1(7)).
        y_idx = int(np.argmax(row))
        predictions.append(model.output_alphabet.token(y_idx))

    prediction_probs = np.asarray(prob_rows, dtype=float)
    return FilterResult(
        states=[int(s) for s in filtered_states],
        predictions=predictions,
        prediction_probs=prediction_probs,
    )
