"""History counting for the token-first transCSSR engine.

This module reproduces the counting semantics of the legacy
:func:`transCSSR.estimate_predictive_distributions` (``counting_method=0``)
but operates on *index sequences* -- tuples of non-negative integers produced
by an :class:`~transcssr.alphabet.Alphabet` -- rather than on character
strings. See ``docs/SPEC-core.md`` sections 0 and 1 for the behavioral
contract.

Two count tables are produced, mirroring the legacy structures exactly but
keyed by index-tuple histories instead of strings:

``marg`` (legacy ``word_lookup_marg``)
    Maps a history ``(x_hist, y_hist)`` where ``x_hist`` is **one token longer**
    than ``y_hist`` (``len(x_hist) == len(y_hist) + 1``) to its occurrence count.
    This is the count of ``(X_{t-l..t}, Y_{t-l..t-1})`` -- the output past plus
    the input including the *current* input ``X_t`` -- and serves as the
    denominator for ``P(Y_t | X_t, past)``.

``fut`` (legacy ``word_lookup_fut``)
    Maps a history ``(x_hist, y_hist)`` of **equal length** to its occurrence
    count. This is the count of ``(X_{t-l+1..t}, Y_{t-l+1..t})`` -- a past
    including the current output symbol -- and serves as the numerator.

Both tables reproduce the legacy right-boundary edge effect: a joint word of
length ``l <= L_max`` is counted only when it occurs as a prefix of a window
starting at positions ``0 .. T - L_max - 1``. Occurrences that begin in the
final ``L_max`` positions of a sequence are never counted. Length-``(L_max+1)``
words are counted once per occurrence.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence, Tuple

from .machine import History  # single-home History alias (docs/DESIGN.md)

__all__ = ["estimate_counts", "History"]

IndexSeq = Sequence[int]


def estimate_counts(
    xs: Sequence[IndexSeq],
    ys: Sequence[IndexSeq],
    L_max: int,
    counting_method: int = 0,
) -> Tuple[Counter, Counter]:
    """Count joint input/output histories from one or more index sequences.

    Reproduces legacy ``estimate_predictive_distributions`` with
    ``counting_method=0``. The caller is responsible for normalizing a single
    realization vs. an ensemble and for encoding tokens to indices: a single
    realization must be passed as a **one-element list**. This unifies the
    legacy ``is_multiline=False`` / ``is_multiline=True`` code paths, which for
    ``counting_method=0`` produce identical counts (they run the same per-line
    loop; a one-element ensemble is bit-for-bit identical to the single-sequence
    path).

    Parameters
    ----------
    xs : sequence of sequence of int
        A list of input index sequences. Each element is one realization,
        given as a tuple/list of non-negative integer token indices.
    ys : sequence of sequence of int
        A list of output index sequences, aligned one-to-one with ``xs``. Each
        ``ys[i]`` must have the same length as ``xs[i]``.
    L_max : int
        Maximum history length. Windows have length ``L_max + 1``; sub-words of
        length ``1 .. L_max + 1`` are counted as prefixes of each window.
    counting_method : int, optional
        Only ``0`` (the default) is supported. Method ``1`` is a
        marginalization strategy that produces identical counts on every
        length-``>= 1`` key (see ``docs/SPEC-core.md`` section 1.3); it is not
        reimplemented here.

    Returns
    -------
    marg : collections.Counter
        Maps ``(x_hist, y_hist)`` with ``len(x_hist) == len(y_hist) + 1`` to
        its count. Mirrors legacy ``word_lookup_marg``.
    fut : collections.Counter
        Maps ``(x_hist, y_hist)`` with equal lengths to its count. Mirrors
        legacy ``word_lookup_fut``.

    Raises
    ------
    NotImplementedError
        If ``counting_method`` is not ``0``.
    ValueError
        If ``xs`` and ``ys`` differ in the number of realizations, or if any
        aligned pair of realizations differ in length, or if ``L_max`` is
        negative.

    Notes
    -----
    Sequences shorter than ``L_max + 1`` contribute nothing: the window loop
    ``range(T - L_max)`` is empty (or, for ``T < L_max``, over a negative
    bound, hence also empty). The right-boundary edge effect is applied
    per realization, so no window ever spans a boundary between realizations.
    """
    if counting_method != 0:
        raise NotImplementedError(
            "Only counting_method=0 is implemented; methods 0 and 1 agree on "
            "all length->=1 keys (see docs/SPEC-core.md section 1.3 and the "
            "legacy transCSSR.estimate_predictive_distributions)."
        )
    if L_max < 0:
        raise ValueError(f"L_max must be non-negative, got {L_max}.")
    if len(xs) != len(ys):
        raise ValueError(
            f"xs and ys must contain the same number of realizations, "
            f"got {len(xs)} and {len(ys)}."
        )

    marg: Counter = Counter()
    fut: Counter = Counter()

    for line_ind, (seq_x, seq_y) in enumerate(zip(xs, ys)):
        x = tuple(seq_x)
        y = tuple(seq_y)
        if len(x) != len(y):
            raise ValueError(
                f"Realization {line_ind}: input and output must have the same "
                f"length, got {len(x)} and {len(y)}."
            )

        T = len(x)

        for t_ind in range(T - L_max):
            cur_x = x[t_ind : t_ind + L_max + 1]
            cur_y = y[t_ind : t_ind + L_max + 1]

            marg[(cur_x, cur_y[:-1])] += 1
            fut[(cur_x, cur_y)] += 1

            for remove_inds in range(1, L_max + 1):
                trunc_x = cur_x[:-remove_inds]
                trunc_y = cur_y[:-remove_inds]

                marg[(trunc_x, trunc_y[:-1])] += 1
                fut[(trunc_x, trunc_y)] += 1

    return marg, fut
