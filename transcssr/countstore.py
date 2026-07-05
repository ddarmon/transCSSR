"""Abstract count-access interface for the CSSR engine (token-first).

:func:`transcssr.cssr.run_cssr` reads history counts through a *narrow*
query surface: for any given history it needs only two answers, and it derives
both from the ``marg`` / ``fut`` :class:`collections.Counter` tables produced by
:func:`transcssr.counting.estimate_counts`. This module factors that surface out
into the :class:`CountStore` abstraction so alternative backends (e.g. a
memory-lean suffix-array / FM-index backend that answers per-history queries
on demand rather than materializing every history) can slot in behind the same
interface without touching the algorithm.

The two queries are:

``children(history)``
    The sorted ``(y_ind, x_ind)`` child prepends of ``history`` that occur --
    i.e. the emissions ``(y_ind, x_ind)`` for which some longer input context
    exists, so that ``(( x_ind,) + hx + (ax2,), (y_ind,) + hy)`` was counted in
    ``marg`` for some next input ``ax2``. Stage 1 of CSSR iterates exactly these
    children when growing a history one step into the past.

``extensions(history)``
    The observed emission extensions of ``history`` as ``(flat_pos, count)``
    pairs, where ``flat_pos = n_y * ex + ey`` is the flat morph index of the
    emission ``(ex, ey)`` (input outer, output inner). Scattering these into a
    zeroed length-``n_x * n_y`` vector reconstructs the dense morph of the
    history in O(observed support) time.

The interface is deliberately *per-history and on-demand*: neither method
requires the store to enumerate all histories up front, so a suffix-array
backend can answer each call by a bounded number of substring lookups.

The concrete :class:`CounterCountStore` builds the two inverted indexes from
``(marg, fut, n_y)`` exactly as the CSSR engine used to inline them; the key
inversion and ordering rationale lives with it below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np

from ._suffix import SuffixIndex
from .machine import History

__all__ = ["CountStore", "CounterCountStore", "SuffixArrayCountStore"]


class CountStore(ABC):
    """Per-history count-query interface consumed by :func:`run_cssr`.

    A ``CountStore`` answers, for any history ``(x_hist, y_hist)``, the two
    queries the CSSR state-splitting / determinization algorithm needs (see the
    module docstring). Implementations must be **deterministic** and must not
    require materializing every history in advance; each query is answered on
    demand for the single history passed in.

    The store must know ``n_y`` (the output-alphabet size) because emission
    extensions are reported as flat morph indices ``n_y * ex + ey``. It is
    exposed as the read-only :attr:`n_y` property; concrete implementations
    decide how it is supplied (a constructor argument for
    :class:`CounterCountStore`).
    """

    @property
    @abstractmethod
    def n_y(self) -> int:
        """Number of output symbols; sets the flat-index stride ``n_y``."""
        raise NotImplementedError

    @abstractmethod
    def children(self, history: History) -> Sequence[tuple[int, int]]:
        """Occurring ``(y_ind, x_ind)`` child prepends of ``history``.

        Parameters
        ----------
        history : History
            A ``(x_hist, y_hist)`` pair of equal-length index tuples.

        Returns
        -------
        sequence of tuple of int
            The ``(y_ind, x_ind)`` child pairs that occur, in ascending
            ``(y_ind, x_ind)`` order. Empty (``()`` is acceptable) when the
            history has no occurring children.
        """
        raise NotImplementedError

    @abstractmethod
    def extensions(self, history: History) -> Sequence[tuple[int, int]]:
        """Observed emission extensions of ``history`` as ``(flat_pos, count)``.

        Parameters
        ----------
        history : History
            A ``(x_hist, y_hist)`` pair of equal-length index tuples.

        Returns
        -------
        sequence of tuple of int
            ``(flat_pos, count)`` pairs with ``flat_pos = n_y * ex + ey`` for
            each observed emission ``(ex, ey)`` of ``history`` and its count.
            Each ``flat_pos`` occurs at most once. Empty (``()`` is acceptable)
            when the history has no observed extensions. Order is irrelevant to
            callers (the values are scattered into a zeroed morph vector).
        """
        raise NotImplementedError


class CounterCountStore(CountStore):
    """:class:`CountStore` backed by the legacy ``marg`` / ``fut`` Counters.

    Reproduces the exact inverted-index construction the CSSR engine used to
    inline: one pass over ``marg`` to build the child index and one pass over
    ``fut`` to build the extension index. This is the default backend used by
    :func:`run_cssr` when it is handed plain Counters.

    Parameters
    ----------
    marg : collections.Counter
        Legacy ``word_lookup_marg``: maps ``(x_hist, y_hist)`` with
        ``len(x_hist) == len(y_hist) + 1`` to its count.
    fut : collections.Counter
        Legacy ``word_lookup_fut``: maps equal-length ``(x_hist, y_hist)`` to
        its count.
    n_y : int
        Number of output symbols; the flat-index stride for
        :meth:`extensions`.
    """

    def __init__(self, marg: Counter, fut: Counter, n_y: int) -> None:
        self._n_y = int(n_y)

        # ``children[(hx, hy)]`` -> the ``(y_ind, x_ind)`` child prepends of
        # history ``(hx, hy)`` that occur, i.e. for which some next input
        # ``ax2`` has ``marg[((x_ind,) + hx + (ax2,), (y_ind,) + hy)] != 0``. A
        # ``marg`` key ``(mx, my)`` with ``len(mx) >= 2`` is exactly such an
        # occurrence of the child ``((mx[0],) + mx[1:-1], (my[0],) + my[1:])``
        # of history ``(mx[1:-1], my[1:])``, so we invert the marg keys
        # directly. The child generation order in Stage 1 is output-outer /
        # input-inner (legacy ``for ay: for ax:``), which fixes new-state
        # numbering (SPEC §6.1(1)); returning the occurring ``(y_ind, x_ind)``
        # pairs in ``(y_ind, x_ind)`` sorted order is equivalent because
        # non-occurring children only ``continue``.
        occ: dict[History, set[tuple[int, int]]] = defaultdict(set)
        for (mx, my), count in marg.items():
            if count == 0 or len(mx) < 2:
                continue
            history = (mx[1:-1], my[1:])
            occ[history].add((my[0], mx[0]))
        self._children: dict[History, list[tuple[int, int]]] = {
            hist: sorted(pairs) for hist, pairs in occ.items()
        }

        # ``extensions[(hx, hy)]`` -> list of ``(flat_pos, count)`` where a
        # length-``L+1`` ``fut`` key ``(fx, fy)`` extends history
        # ``(fx[:-1], fy[:-1])`` by emission ``(fx[-1], fy[-1])`` at flat morph
        # index ``n_y * fx[-1] + fy[-1]``. Each ``(ex, ey)`` occurs at most once
        # per history (unique ``fut`` keys), so scattering with ``=`` reproduces
        # the dense ``[fut[...] for ex, ey in e_symbols]`` vector exactly.
        extensions: dict[History, list[tuple[int, int]]] = defaultdict(list)
        for (fx, fy), count in fut.items():
            if count == 0 or len(fx) == 0:
                continue
            history = (fx[:-1], fy[:-1])
            pos = self._n_y * fx[-1] + fy[-1]
            extensions[history].append((pos, count))
        self._extensions: dict[History, list[tuple[int, int]]] = dict(extensions)

    @property
    def n_y(self) -> int:
        return self._n_y

    def children(self, history: History) -> Sequence[tuple[int, int]]:
        return self._children.get(history, ())

    def extensions(self, history: History) -> Sequence[tuple[int, int]]:
        return self._extensions.get(history, ())


class SuffixArrayCountStore(CountStore):
    """:class:`CountStore` backed by a suffix array, never materializing Counters.

    A memory-lean alternative to :class:`CounterCountStore` that answers the two
    per-history queries directly from a :class:`~transcssr._suffix.SuffixIndex`
    over the joint process, plus a small edge-effect correction table. It
    reproduces the counting semantics of :func:`transcssr.counting.estimate_counts`
    (``counting_method=0``) exactly -- the same partition, the same morph counts
    -- while storing only ``O(T)`` integers rather than one
    :class:`collections.Counter` entry per distinct observed window.

    z-coding
    --------
    The joint input/output process is encoded as a single product-alphabet
    stream ``z_t = x_t * n_y + y_t`` (so ``vocab_size = n_x * n_y``; for the
    epsilon-machine path ``n_x == 1`` this degenerates to ``z == y``). One
    :class:`~transcssr._suffix.SuffixIndex` is built over the ``z``-streams of
    all realizations. A joint word -- a ``fut`` key ``(fx, fy)`` of length ``l``
    -- is exactly the contiguous ``z``-word ``fx[i] * n_y + fy[i]``. The flat
    morph index of an emission ``(ex, ey)`` is ``n_y * ex + ey``, which is
    *identically* its ``z``-symbol: **the z-coding is the flat morph index**, so
    :meth:`extensions` maps a following ``z``-symbol straight to its ``flat_pos``.

    Edge-effect tail correction
    ---------------------------
    Legacy counting (see :mod:`transcssr.counting`) counts a length-``l`` word
    (``l <= L_max``) only when it starts at within-realization offsets
    ``0 .. T_r - L_max - 1``; the raw suffix-array count instead includes every
    in-realization start ``0 .. T_r - l``. The surplus starts -- the *tail*
    region ``T_r - L_max .. T_r - l`` -- are removed by a precomputed
    :class:`collections.Counter` ``tail``: for each realization ``r``, each start
    ``t`` in ``max(0, T_r - L_max) .. T_r - 1``, and each length ``l`` in
    ``1 .. min(L_max, T_r - t)``, ``tail[z[t:t+l]] += 1``. Then

        ``corrected_count(w) = suffix_index.count(w) - tail.get(w, 0)``

    for ``len(w) <= L_max``, and the raw suffix-array count for
    ``len(w) == L_max + 1`` (whose start range ``0 .. T_r - L_max - 1`` already
    coincides with the legacy window, so no correction is due). Realizations with
    ``T_r <= L_max`` contribute nothing in legacy counting; for them the tail
    loop enumerates *every* subword of the realization, so the correction
    subtracts the realization's entire suffix-array contribution -- matching the
    legacy empty-window loop exactly. The table holds
    ``O(n_realizations * L_max**2)`` keys.

    Queries
    -------
    ``extensions(history)``
        Let ``hz`` be the ``z``-word of ``history`` (length ``l``). Enumerate
        :meth:`~transcssr._suffix.SuffixIndex.right_extensions` of ``hz``; each
        extended word ``hz + (z,)`` has length ``l + 1``, so its count is
        corrected via ``tail`` when ``l + 1 <= L_max``. Entries whose corrected
        count is ``<= 0`` are dropped; the surviving following symbol ``z`` is
        ``flat_pos`` directly.
    ``children(history)``
        A child ``(y0, x0)`` occurs iff the ``z``-word ``(z0,) + hz`` (with
        ``z0 = x0 * n_y + y0``, length ``l + 1``) occurs at a within-realization
        offset ``<= T_r - L_max - 1`` -- exactly the corrected count of that
        word being positive (a following input symbol then exists automatically,
        because ``l + 1 <= L_max`` places the next position inside the
        realization). Enumerate
        :meth:`~transcssr._suffix.SuffixIndex.left_extensions` of ``hz``, keep
        the preceding symbols ``z0`` with positive corrected count, map
        ``z0 -> (z0 % n_y, z0 // n_y) == (y0, x0)``, and **re-sort** ascending by
        ``(y0, x0)`` (the ``z``-order is input-major and differs from
        ``(y_ind, x_ind)`` order when ``n_x > 1``).

    Both queries of the null history ``((), ())`` reduce to the corrected
    unigram distribution (right / left extensions of the empty ``z``-word).

    Results are memoized per history in a plain dict: :func:`run_cssr` re-queries
    the same history across Stages 1, 1b, and 3, and the memo makes those repeats
    free. Determinism is unaffected (the memo is keyed by the exact history and
    every query is a pure function of the immutable index).

    Parameters
    ----------
    xs_idx : sequence of sequence of int
        Input index realizations (as :func:`transcssr.counting.estimate_counts`
        takes them): a list of one realization each, tokens in ``[0, n_x)``.
    ys_idx : sequence of sequence of int
        Output index realizations aligned one-to-one with ``xs_idx``; each pair
        must have equal length, tokens in ``[0, n_y)``.
    L_max : int
        Maximum history length (windows have length ``L_max + 1``).
    n_x, n_y : int
        Input / output alphabet sizes. ``vocab_size = n_x * n_y``.

    Raises
    ------
    ValueError
        If ``xs_idx`` and ``ys_idx`` differ in the number of realizations or in
        any aligned realization's length, if ``L_max`` is negative, if
        ``n_x`` / ``n_y`` are not positive, or if any index is out of range.
    """

    def __init__(
        self,
        xs_idx: Sequence[Sequence[int]],
        ys_idx: Sequence[Sequence[int]],
        L_max: int,
        n_x: int,
        n_y: int,
    ) -> None:
        if L_max < 0:
            raise ValueError(f"L_max must be non-negative, got {L_max}.")
        if n_x < 1 or n_y < 1:
            raise ValueError(
                f"n_x and n_y must be positive, got n_x={n_x}, n_y={n_y}."
            )
        if len(xs_idx) != len(ys_idx):
            raise ValueError(
                f"xs_idx and ys_idx must contain the same number of "
                f"realizations, got {len(xs_idx)} and {len(ys_idx)}."
            )

        self._n_y = int(n_y)
        self._n_x = int(n_x)
        self._L_max = int(L_max)
        vocab_size = self._n_x * self._n_y

        # Encode each realization to its z-stream z_t = x_t * n_y + y_t and, in
        # the same pass, accumulate the edge-effect tail-correction table.
        z_streams: list[np.ndarray] = []
        tail: Counter = Counter()
        for line_ind, (seq_x, seq_y) in enumerate(zip(xs_idx, ys_idx)):
            x_arr = np.asarray(seq_x, dtype=np.int64).reshape(-1)
            y_arr = np.asarray(seq_y, dtype=np.int64).reshape(-1)
            if x_arr.shape[0] != y_arr.shape[0]:
                raise ValueError(
                    f"Realization {line_ind}: input and output must have the "
                    f"same length, got {x_arr.shape[0]} and {y_arr.shape[0]}."
                )
            if x_arr.size:
                if int(x_arr.min()) < 0 or int(x_arr.max()) >= self._n_x:
                    raise ValueError(
                        f"Realization {line_ind}: input index out of range "
                        f"[0, {self._n_x})."
                    )
                if int(y_arr.min()) < 0 or int(y_arr.max()) >= self._n_y:
                    raise ValueError(
                        f"Realization {line_ind}: output index out of range "
                        f"[0, {self._n_y})."
                    )
            z_arr = x_arr * self._n_y + y_arr
            z_streams.append(z_arr)

            # Tail region: starts t in [max(0, T-L_max), T-1], each length l in
            # [1, min(L_max, T-t)]. For T <= L_max this enumerates every subword
            # (subtracting the realization's whole contribution, as legacy's
            # empty window loop does).
            z_list = z_arr.tolist()
            T_r = len(z_list)
            for t in range(max(0, T_r - self._L_max), T_r):
                max_len = min(self._L_max, T_r - t)
                for ell in range(1, max_len + 1):
                    tail[tuple(z_list[t : t + ell])] += 1

        self._index = SuffixIndex(z_streams, vocab_size=vocab_size)
        self._tail = tail

        # Per-history memo (History -> materialized result). run_cssr re-queries
        # the same histories across Stages 1, 1b, and 3.
        self._children_memo: dict[History, tuple[tuple[int, int], ...]] = {}
        self._extensions_memo: dict[History, tuple[tuple[int, int], ...]] = {}

    @property
    def n_y(self) -> int:
        return self._n_y

    def _z_word(self, history: History) -> tuple[int, ...]:
        """The z-word ``x_i * n_y + y_i`` of ``history`` (a ``(hx, hy)`` pair)."""
        hx, hy = history
        n_y = self._n_y
        return tuple(hx[i] * n_y + hy[i] for i in range(len(hx)))

    def _corrected(self, word: tuple[int, ...], raw: int) -> int:
        """Legacy count of ``word``: raw SA count minus the tail correction.

        The correction applies for ``len(word) <= L_max``; a length-``(L_max+1)``
        word needs none (its SA start range already equals the legacy window).
        """
        if len(word) <= self._L_max:
            return raw - self._tail.get(word, 0)
        return raw

    def children(self, history: History) -> Sequence[tuple[int, int]]:
        cached = self._children_memo.get(history)
        if cached is not None:
            return cached
        hz = self._z_word(history)
        n_y = self._n_y
        # A child word has length len(hz) + 1 and must invert from a marg key,
        # whose history side caps at length L_max - 1 (marg x-length <= L_max+1).
        # Longer histories therefore have no children -- matching the Counter
        # store, and keeping the "a following input exists" reduction valid
        # (child length len(hz)+1 <= L_max).
        if len(hz) >= self._L_max:
            self._children_memo[history] = ()
            return ()
        pairs: list[tuple[int, int]] = []
        for z0, raw in self._index.left_extensions(hz):
            word = (z0,) + hz  # length len(hz) + 1
            if self._corrected(word, raw) > 0:
                # z0 = x0 * n_y + y0  ->  child (y_ind, x_ind).
                pairs.append((z0 % n_y, z0 // n_y))
        # left_extensions yields z0 in z-order (input-major); re-sort by
        # (y_ind, x_ind) as the CountStore contract requires.
        pairs.sort()
        result = tuple(pairs)
        self._children_memo[history] = result
        return result

    def extensions(self, history: History) -> Sequence[tuple[int, int]]:
        cached = self._extensions_memo.get(history)
        if cached is not None:
            return cached
        hz = self._z_word(history)
        # An extension word has length len(hz) + 1 and must be a fut key, whose
        # length caps at L_max + 1; histories longer than L_max have no observed
        # extensions -- matching the Counter store.
        if len(hz) > self._L_max:
            self._extensions_memo[history] = ()
            return ()
        out: list[tuple[int, int]] = []
        for z_ext, raw in self._index.right_extensions(hz):
            word = hz + (z_ext,)  # length len(hz) + 1
            count = self._corrected(word, raw)
            if count > 0:
                # flat_pos = n_y * ex + ey == z_ext exactly (the z-coding is the
                # flat morph index).
                out.append((z_ext, count))
        result = tuple(out)
        self._extensions_memo[history] = result
        return result
