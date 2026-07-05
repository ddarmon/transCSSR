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
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .machine import History  # single-home History alias (docs/DESIGN.md)

__all__ = ["estimate_counts", "History"]

IndexSeq = Sequence[int]

# Largest value an int64 can hold; used by the integer-packing overflow guard.
_INT64_MAX = int(np.iinfo(np.int64).max)

# Below this many total windows the numpy set-up cost outweighs the vectorized
# gain, so the pure-Python reference is used directly. Purely a performance
# switch: both paths produce identical count tables.
_TINY_TOTAL_WINDOWS = 16

# If the fraction of distinct longest-windows exceeds this, the data is treated
# as near-incompressible and the reference path is used (see
# :func:`_worth_vectorizing`). Purely a performance switch: identical output.
_COMPRESS_RATIO = 0.5

# The compressibility pre-check subsamples at most this many longest-windows so
# its cost stays bounded regardless of total sequence length.
_PRECHECK_SAMPLE = 1_000_000


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

    This is a vectorized reimplementation of the reference triple loop
    (:func:`_estimate_counts_reference`, retained verbatim for tiny inputs and
    as the test oracle). It produces **bit-for-bit identical** ``Counter``
    tables -- same keys (tuples of :class:`int`, nested ``(x_hist, y_hist)``)
    and same counts.

    The fast path materializes each realization's length-``(L_max + 1)``
    sliding windows with :func:`numpy.lib.stride_tricks.sliding_window_view`
    (whose row set ``0 .. T - L_max - 1`` reproduces the edge effect exactly)
    and, for every sub-word length ``l`` in ``1 .. L_max + 1``, counts the
    distinct ``(x[:l], y[:l])`` / ``(x[:l], y[:l-1])`` rows with
    :func:`numpy.unique`. Fixed-length rows are packed to a single ``int64``
    in base ``B = max_index + 1`` when ``B ** ncols`` fits ``int64`` (``ncols``
    is at most ``2 * (L_max + 1)``); otherwise a structured, overflow-free
    ``numpy.unique(..., axis=0)`` is used. Identical input (``x``) history
    tuples are interned through a per-call cache so equal keys (notably the
    all-constant ``x`` windows of the epsilon-machine path) share one object.

    A cheap compressibility pre-check (:func:`_worth_vectorizing`) defers to the
    reference for near-incompressible data (e.g. i.i.d. tokens), where the
    reference triple loop is already near-optimal and the vectorized decode
    would only add overhead; the vectorized path is taken for the repetitive,
    highly compressible data typical of natural language.
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

    # Validate aligned lengths, convert to int64 arrays, and gather the window
    # count / alphabet bound. Realizations shorter than L_max + 1 yield no
    # windows and are skipped (but still length-validated).
    window = L_max + 1
    kept: List[Tuple[np.ndarray, np.ndarray]] = []
    total_windows = 0
    max_index = -1
    for line_ind, (seq_x, seq_y) in enumerate(zip(xs, ys)):
        x_arr = np.asarray(seq_x, dtype=np.int64).reshape(-1)
        y_arr = np.asarray(seq_y, dtype=np.int64).reshape(-1)
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Realization {line_ind}: input and output must have the same "
                f"length, got {x_arr.shape[0]} and {y_arr.shape[0]}."
            )
        n_windows = x_arr.shape[0] - L_max
        if n_windows <= 0:
            continue
        total_windows += n_windows
        max_index = max(max_index, int(x_arr.max()), int(y_arr.max()))
        kept.append((x_arr, y_arr))

    marg: Counter = Counter()
    fut: Counter = Counter()

    if total_windows == 0:
        return marg, fut
    if total_windows <= _TINY_TOTAL_WINDOWS:
        # numpy overhead not worth it; defer to the reference (identical output).
        return _estimate_counts_reference(xs, ys, L_max)

    base = max_index + 1  # every counted index lies in [0, base)

    # Adaptive dispatch: the vectorized path wins only when the data is
    # repetitive enough that np.unique compresses many windows into few keys
    # (the natural-language regime). On near-incompressible data (e.g. i.i.d.
    # tokens) almost every window is a distinct key, so the output is inherently
    # huge and dominated by per-key tuple materialization -- exactly what the
    # reference triple loop already does near-optimally. A cheap pre-check on
    # the longest (least compressible) window keeps us from regressing there.
    if not _worth_vectorizing(kept, window, base, total_windows):
        return _estimate_counts_reference(xs, ys, L_max)

    intern: Dict[tuple, tuple] = {}

    for x_arr, y_arr in kept:
        x_win = sliding_window_view(x_arr, window)  # (n_windows, window)
        y_win = sliding_window_view(y_arr, window)
        for l in range(1, window + 1):
            x_pref = x_win[:, :l]
            # fut key: (x[:l], y[:l]); marg key: (x[:l], y[:l-1]).
            fut_rows = np.concatenate((x_pref, y_win[:, :l]), axis=1)
            marg_rows = np.concatenate((x_pref, y_win[:, : l - 1]), axis=1)
            _accumulate(fut, fut_rows, l, l, base, intern)
            _accumulate(marg, marg_rows, l, l - 1, base, intern)

    return marg, fut


def _worth_vectorizing(
    kept: List[Tuple[np.ndarray, np.ndarray]],
    window: int,
    base: int,
    total_windows: int,
) -> bool:
    """Estimate whether the vectorized path will beat the reference.

    Counts the distinct longest (length-``window``) joint windows -- the least
    compressible sub-word length -- on a bounded subsample and compares that to
    the number of windows sampled. When the distinct fraction is high the data
    is near-incompressible: the output Counter is inherently large and the
    reference triple loop is already near-optimal, so the vectorized decode
    (per-key tuple materialization) would only add overhead. Returns ``True`` to
    vectorize, ``False`` to defer to the reference.

    The pre-check subsamples at most ``_PRECHECK_SAMPLE`` longest-windows (via a
    fixed stride) so its cost stays bounded even for very long sequences, and
    uses ``int64`` packing when it fits, otherwise an overflow-free structured
    unique.
    """
    ncols = 2 * window
    fits = _fits_int64(base, ncols)
    powers = (
        np.array([base**k for k in range(ncols)], dtype=np.int64) if fits else None
    )
    step = max(1, total_windows // _PRECHECK_SAMPLE)

    sampled = []
    for x_arr, y_arr in kept:
        x_win = sliding_window_view(x_arr, window)[::step]
        y_win = sliding_window_view(y_arr, window)[::step]
        rows = np.concatenate((x_win, y_win), axis=1)
        sampled.append(rows @ powers if fits else rows)

    data = sampled[0] if len(sampled) == 1 else np.concatenate(sampled)
    n_sampled = data.shape[0]
    if fits:
        n_unique = int(np.unique(data).size)
    else:
        n_unique = int(np.unique(data, axis=0).shape[0])
    return n_unique <= _COMPRESS_RATIO * n_sampled


def _fits_int64(base: int, ncols: int) -> bool:
    """Return whether base-``base`` packing of ``ncols`` symbols fits ``int64``.

    The packed value of a row is ``sum_k symbol_k * base**k`` with each symbol
    in ``[0, base)``, so its maximum is ``base**ncols - 1``. Packing is safe iff
    ``base**ncols - 1 <= _INT64_MAX``.
    """
    if base <= 1 or ncols == 0:
        return True
    return base**ncols - 1 <= _INT64_MAX


def _accumulate(
    counter: Counter,
    rows: np.ndarray,
    l_x: int,
    l_y: int,
    base: int,
    intern: Dict[tuple, tuple],
) -> None:
    """Count distinct ``rows`` and fold them into ``counter`` as history keys.

    ``rows`` has shape ``(n_windows, l_x + l_y)``: the first ``l_x`` columns are
    the input history and the remaining ``l_y`` the output history. Distinct
    rows are found with :func:`numpy.unique` -- via a single-``int64`` packing
    when it cannot overflow (:func:`_fits_int64`), otherwise via the
    overflow-free structured ``axis=0`` form. The ``x`` half of every key is
    interned so identical input histories (notably the all-constant windows of
    the epsilon-machine path) share one object.
    """
    ncols = l_x + l_y
    if _fits_int64(base, ncols):
        powers = np.array([base**k for k in range(ncols)], dtype=np.int64)
        packed = rows @ powers  # injective: base > every symbol, no overflow
        _, first_idx, counts = np.unique(
            packed, return_index=True, return_counts=True
        )
        unique_rows = rows[first_idx]
    else:
        unique_rows, counts = np.unique(rows, axis=0, return_counts=True)

    # Split columns once (C-level), then build keys. Intern the x half so the
    # constant-input machine path shares a single tuple per length.
    x_parts = unique_rows[:, :l_x].tolist()
    y_parts = unique_rows[:, l_x:].tolist()
    intern_get = intern.setdefault
    for x_part, y_part, count in zip(x_parts, y_parts, counts.tolist()):
        x_hist = tuple(x_part)
        x_hist = intern_get(x_hist, x_hist)
        counter[(x_hist, tuple(y_part))] += count


def _estimate_counts_reference(
    xs: Sequence[IndexSeq],
    ys: Sequence[IndexSeq],
    L_max: int,
    counting_method: int = 0,
) -> Tuple[Counter, Counter]:
    """Pure-Python reference counter (the original triple loop).

    Kept verbatim as the ground-truth semantics that
    :func:`estimate_counts` reproduces, and used directly by it for tiny inputs.
    Same signature and output contract as :func:`estimate_counts`.
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
