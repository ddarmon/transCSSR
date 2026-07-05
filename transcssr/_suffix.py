"""Infini-gram-style suffix-array query engine over integer token streams.

This private module provides :class:`SuffixIndex`, a memory-lean substrate for
answering n-gram count / extension queries on demand over a corpus of one or
more integer token realizations, without ever materializing a
:class:`collections.Counter` of every observed window. It is the intended
backend for a future ``CountStore`` (see ``WHATS_NEXT.md`` §2 "scale walls";
infini-gram, Liu et al., COLM 2024, arXiv:2401.17377): a suffix array over the
raw stream costs ``O(T)`` integers and answers contiguous-subword count queries
in ``O(|word| log T)`` time.

Design notes
------------
Corpus layout
    The input realizations ``R_0, R_1, ...`` are concatenated with a single
    *sentinel* value placed between every pair of realizations *and* appended
    once at the very end::

        corpus = R_0 ++ [S] ++ R_1 ++ [S] ++ ... ++ R_{m-1} ++ [S]

    The sentinel ``S = vocab_size`` lies strictly outside the token range
    ``[0, vocab_size)``. Because a query word contains only real tokens (never
    ``S``), no reported occurrence can ever span a realization boundary: any
    window crossing a boundary would have to contain the sentinel. A single
    sentinel value is therefore sufficient for correctness -- the suffix array
    is still a strict total order because every suffix ends at the final
    sentinel and no two suffixes are equal.

Suffix array construction
    Pure numpy prefix-doubling (rank-doubling) via :func:`numpy.lexsort`, which
    is ``O(T log^2 T)`` and fully deterministic. If :mod:`pydivsufsort` is
    importable it is used as an ``O(T)`` fast path; it is strictly optional
    (the import is wrapped, any failure falls back to numpy, and the result is
    byte-identical because the suffix array is unique). ``pydivsufsort`` is
    *not* a declared dependency.

Reversed index
    Left-extension queries are answered through a second suffix array built
    over the *reversed* corpus (a left extension of ``word`` is a right
    extension of ``reversed(word)`` in the reversed corpus). The reversed index
    is built lazily on first use, so corpora that are never queried for left
    extensions pay nothing for it.

Realization mapping
    Construction records the corpus start offset of each realization. A corpus
    position ``p`` maps to realization ``searchsorted(offsets, p, "right") - 1``
    and within-realization offset ``p - offsets[r]``. This lets a caller apply
    CSSR's per-realization right-boundary edge effect (count a length-``L`` word
    only when it starts at within-realization offsets ``0 .. T_r - L_max - 1``)
    by filtering :meth:`positions` through :meth:`realization_of` /
    :meth:`offset_in_realization`.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np

try:  # optional fast path; never required, never a declared dependency
    import pydivsufsort as _pydivsufsort
except Exception:  # pragma: no cover - exercised only when the pkg is absent
    _pydivsufsort = None

__all__ = ["SuffixIndex"]

Word = Union[Sequence[int], np.ndarray]


def _suffix_array_numpy(corpus: np.ndarray) -> np.ndarray:
    """Build the suffix array of ``corpus`` with pure-numpy prefix doubling.

    The suffix array is the permutation ``sa`` of ``0 .. n-1`` that orders the
    suffixes ``corpus[sa[0]:], corpus[sa[1]:], ...`` lexicographically, with a
    shorter suffix ordered before a longer one it is a prefix of (the standard
    implicit-terminator convention, matching :func:`pydivsufsort.divsufsort`).

    Parameters
    ----------
    corpus : numpy.ndarray
        1-d array of non-negative integers.

    Returns
    -------
    numpy.ndarray
        The suffix array, ``int32`` when ``len(corpus) <= 2**31 - 1`` else
        ``int64``.

    Notes
    -----
    Complexity is ``O(n log^2 n)`` time (``log n`` doubling rounds, each an
    ``O(n log n)`` :func:`numpy.lexsort`) and ``O(n)`` extra space. Fully
    deterministic: ``lexsort`` is stable and the terminating ranks are a
    permutation, so ``sa`` is unique.
    """
    n = int(corpus.shape[0])
    sa_dtype = np.int32 if n <= np.iinfo(np.int32).max else np.int64
    if n == 0:
        return np.empty(0, dtype=sa_dtype)

    # Initial ranks: order by the single leading character.
    order = np.argsort(corpus, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    sorted_vals = corpus[order]
    rank[order[0]] = 0
    if n > 1:
        rank[order[1:]] = np.cumsum(sorted_vals[1:] != sorted_vals[:-1])

    k = 1
    while True:
        # Second sort key: the rank of the suffix k positions to the right,
        # with -1 (smaller than any real rank) for suffixes that end early.
        rank2 = np.full(n, -1, dtype=np.int64)
        if k < n:
            rank2[: n - k] = rank[k:]
        sa = np.lexsort((rank2, rank))  # primary key = rank, secondary = rank2

        r1 = rank[sa]
        r2 = rank2[sa]
        changed = np.empty(n, dtype=bool)
        changed[0] = False
        changed[1:] = (r1[1:] != r1[:-1]) | (r2[1:] != r2[:-1])
        new_rank = np.empty(n, dtype=np.int64)
        new_rank[sa] = np.cumsum(changed)

        if int(new_rank.max()) == n - 1:
            # All ranks distinct: the current order is the final suffix array.
            return sa.astype(sa_dtype)
        rank = new_rank
        k <<= 1


def _build_suffix_array(corpus: np.ndarray) -> np.ndarray:
    """Suffix array of ``corpus``: :mod:`pydivsufsort` fast path or numpy.

    Falls back to :func:`_suffix_array_numpy` when :mod:`pydivsufsort` is
    unavailable or raises. Both routes return the same (unique) suffix array.
    """
    n = int(corpus.shape[0])
    sa_dtype = np.int32 if n <= np.iinfo(np.int32).max else np.int64
    if n == 0:
        return np.empty(0, dtype=sa_dtype)
    if _pydivsufsort is not None:
        try:
            sa = _pydivsufsort.divsufsort(np.ascontiguousarray(corpus))
            return np.asarray(sa, dtype=sa_dtype)
        except Exception:  # pragma: no cover - defensive fallback
            pass
    return _suffix_array_numpy(corpus)


class SuffixIndex:
    """Suffix-array query engine over one or more integer token realizations.

    Parameters
    ----------
    realizations : sequence of sequence of int
        The realizations, each a 1-d sequence (list / tuple / 1-d
        :class:`numpy.ndarray`) of non-negative integer tokens strictly less
        than ``vocab_size``. An empty realization is allowed; an empty outer
        list yields an empty index.
    vocab_size : int, optional
        The token alphabet size; tokens must lie in ``[0, vocab_size)``. If
        omitted it is inferred as ``max(token) + 1`` (``0`` for an empty
        corpus). The sentinel value is ``vocab_size``.

    Attributes
    ----------
    vocab_size : int
        The token alphabet size.
    sentinel : int
        The reserved separator value (``vocab_size``).
    corpus_size : int
        Length of the internal corpus, including sentinels.
    n_realizations : int
        Number of realizations.

    Raises
    ------
    ValueError
        If any token is negative, if any token is ``>= vocab_size`` when
        ``vocab_size`` is given, or if a realization is not 1-d.

    Notes
    -----
    The corpus is stored as ``int32`` when ``vocab_size`` fits (i.e.
    ``vocab_size <= 2**31 - 2``) and ``int64`` otherwise; the suffix array is
    ``int32`` when the corpus length fits in ``int32``. Everything is
    deterministic (no randomness, no wall-clock, no unordered iteration).
    """

    __slots__ = (
        "_vocab_size",
        "_sentinel",
        "_corpus",
        "_sa",
        "_realization_lengths",
        "_realization_offsets",
        "_rev_corpus",
        "_rev_sa",
    )

    def __init__(
        self,
        realizations: Sequence[Word],
        vocab_size: int | None = None,
    ) -> None:
        seqs: List[np.ndarray] = []
        for r in realizations:
            arr = np.asarray(r, dtype=np.int64)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if arr.ndim != 1:
                raise ValueError(
                    f"Each realization must be 1-d; got ndim={arr.ndim}."
                )
            if arr.size and int(arr.min()) < 0:
                raise ValueError("Tokens must be non-negative integers.")
            seqs.append(arr)

        max_token = max((int(s.max()) for s in seqs if s.size), default=-1)
        if vocab_size is None:
            vocab_size = max_token + 1
        elif max_token >= vocab_size:
            raise ValueError(
                f"Token {max_token} is out of range for vocab_size={vocab_size}."
            )
        self._vocab_size = int(vocab_size)
        self._sentinel = int(vocab_size)

        dtype = (
            np.int32 if self._sentinel <= np.iinfo(np.int32).max else np.int64
        )
        lengths = np.array([s.size for s in seqs], dtype=np.int64)
        self._realization_lengths = lengths

        m = len(seqs)
        if m == 0:
            self._corpus = np.empty(0, dtype=dtype)
            self._realization_offsets = np.empty(0, dtype=np.int64)
        else:
            total = int(lengths.sum()) + m  # one sentinel after each realization
            corpus = np.empty(total, dtype=dtype)
            offsets = np.empty(m, dtype=np.int64)
            pos = 0
            for i, s in enumerate(seqs):
                offsets[i] = pos
                length = int(s.size)
                if length:
                    corpus[pos : pos + length] = s
                corpus[pos + length] = self._sentinel
                pos += length + 1
            self._corpus = corpus
            self._realization_offsets = offsets

        self._sa = _build_suffix_array(self._corpus)
        self._rev_corpus: np.ndarray | None = None
        self._rev_sa: np.ndarray | None = None

    # -- properties --------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """The token alphabet size (tokens lie in ``[0, vocab_size)``)."""
        return self._vocab_size

    @property
    def sentinel(self) -> int:
        """The reserved separator value (``== vocab_size``)."""
        return self._sentinel

    @property
    def corpus_size(self) -> int:
        """Length of the internal corpus, including sentinels."""
        return int(self._corpus.shape[0])

    @property
    def n_realizations(self) -> int:
        """Number of realizations indexed."""
        return int(self._realization_lengths.shape[0])

    @property
    def realization_lengths(self) -> np.ndarray:
        """Per-realization token counts (excluding sentinels), ``int64``."""
        return self._realization_lengths

    @property
    def realization_offsets(self) -> np.ndarray:
        """Corpus start position of each realization, ``int64``."""
        return self._realization_offsets

    @property
    def corpus(self) -> np.ndarray:
        """The internal corpus array (read-only; do not mutate)."""
        return self._corpus

    @property
    def suffix_array(self) -> np.ndarray:
        """The forward suffix array (read-only; do not mutate)."""
        return self._sa

    @property
    def nbytes(self) -> int:
        """Total bytes held by the index arrays (for benchmarking).

        Includes the corpus, forward suffix array, and the two per-realization
        bookkeeping arrays; the reversed corpus and its suffix array are added
        only once the reversed index has been built (lazily, on the first
        :meth:`left_extensions` call).
        """
        total = (
            self._corpus.nbytes
            + self._sa.nbytes
            + self._realization_offsets.nbytes
            + self._realization_lengths.nbytes
        )
        if self._rev_sa is not None:
            total += self._rev_corpus.nbytes + self._rev_sa.nbytes
        return int(total)

    # -- internal helpers --------------------------------------------------

    def _as_word_array(self, word: Word) -> np.ndarray:
        """Coerce ``word`` to a 1-d ``int64`` array."""
        w = np.asarray(word, dtype=np.int64)
        if w.ndim == 0:
            w = w.reshape(1)
        if w.ndim != 1:
            raise ValueError(f"word must be 1-d; got ndim={w.ndim}.")
        return w

    def _out_of_range(self, w: np.ndarray) -> bool:
        """True if ``w`` (non-empty) contains a token outside ``[0, vocab)``."""
        return bool(int(w.min()) < 0 or int(w.max()) >= self._vocab_size)

    @staticmethod
    def _range_in(
        corpus: np.ndarray, sa: np.ndarray, w: np.ndarray
    ) -> Tuple[int, int]:
        """SA interval ``[lo, hi)`` of suffixes of ``corpus`` starting with ``w``.

        Locates the interval by two suffix-compare binary searches over the
        whole suffix array (never materializing an interval). The suffixes are
        sorted, so the sign of ``cmp(sa[i])`` -- ``-1`` if the suffix at
        ``sa[i]`` is lexicographically below ``w`` over its first ``len(w)``
        characters, ``0`` if it starts with ``w``, ``+1`` if it is above -- is
        non-decreasing in ``i``. ``lo`` is the first index whose sign is ``>= 0``
        and ``hi`` the first whose sign is ``> 0``, so ``[lo, hi)`` is exactly
        the block of suffixes starting with ``w``.

        Each comparison reads at most ``len(w)`` scalar corpus columns (it stops
        at the first mismatch, and the corpus-terminating sentinel -- larger than
        every real token -- caps any read that runs off a short suffix), so the
        whole location costs ``O(|w| log T)`` time and ``O(1)`` extra space,
        never the ``O(interval size)`` column gathers of a set-refinement scheme.
        """
        n = int(sa.shape[0])
        length = int(w.shape[0])
        if length == 0:
            return 0, n
        w_list = w.tolist()

        def _cmp(pos: int) -> int:
            """Sign of suffix-at-``pos`` vs. ``w`` over ``w``'s first chars."""
            for c in range(length):
                sv = int(corpus[pos + c])
                wv = w_list[c]
                if sv < wv:
                    return -1
                if sv > wv:
                    return 1
            return 0

        # lo: first index whose suffix does NOT sort below w (sign >= 0).
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if _cmp(int(sa[mid])) < 0:
                lo = mid + 1
            else:
                hi = mid
        left = lo
        # hi: first index whose suffix sorts strictly above w (sign > 0).
        lo, hi = left, n
        while lo < hi:
            mid = (lo + hi) // 2
            if _cmp(int(sa[mid])) <= 0:
                lo = mid + 1
            else:
                hi = mid
        return left, lo

    def _extensions_in(
        self, corpus: np.ndarray, sa: np.ndarray, w: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Distinct symbols following ``w`` in ``corpus`` with counts, sorted.

        Refines to the SA interval of ``w`` and reads the next character column
        ``corpus[sa[lo:hi] + len(w)]``, which is sorted; :func:`numpy.unique`
        yields the distinct following symbols and their counts in symbol order.
        Sentinels are excluded. ``O(|w| log T + m)`` time where ``m`` is the
        number of occurrences of ``w`` (the interval width).
        """
        length = int(w.shape[0])
        if length == 0:
            lo, hi = 0, int(corpus.shape[0])
        else:
            if self._out_of_range(w):
                return []
            lo, hi = self._range_in(corpus, sa, w)
        if lo >= hi:
            return []
        column = corpus[sa[lo:hi] + length]
        symbols, counts = np.unique(column, return_counts=True)
        return [
            (int(s), int(c))
            for s, c in zip(symbols.tolist(), counts.tolist())
            if s != self._sentinel
        ]

    def _ensure_reverse(self) -> None:
        """Build the reversed-corpus suffix array on first use.

        The forward corpus already ends in a sentinel, so every non-sentinel
        match is followed by an in-bounds column. The plain reverse
        ``corpus[::-1]`` ends in a *real* token (the corpus starts with one), so
        a trailing sentinel is appended to restore that invariant; the extra
        terminator never matches a real query and does not affect boundaries.
        """
        if self._rev_sa is None:
            rev = self._corpus[::-1]
            terminator = np.array([self._sentinel], dtype=self._corpus.dtype)
            self._rev_corpus = np.concatenate([rev, terminator])
            self._rev_sa = _build_suffix_array(self._rev_corpus)

    # -- queries -----------------------------------------------------------

    def range(self, word: Word) -> Tuple[int, int]:
        """Suffix-array interval ``[lo, hi)`` of suffixes starting with ``word``.

        Parameters
        ----------
        word : sequence of int
            The contiguous query word. The empty word yields the full interval
            ``(0, corpus_size)``. A word containing an out-of-range token
            (``< 0`` or ``>= vocab_size``, which includes the sentinel) yields
            an empty interval.

        Returns
        -------
        (int, int)
            ``lo, hi`` with ``hi - lo`` the number of occurrences (for a
            non-empty in-range word). ``hi <= lo`` denotes an empty interval.

        Notes
        -----
        ``O(|word| log T)`` time. Because the empty-word interval spans
        sentinels, use :meth:`count` -- not ``hi - lo`` -- for the empty word.
        """
        w = self._as_word_array(word)
        n = int(self._corpus.shape[0])
        if w.shape[0] == 0:
            return 0, n
        if self._out_of_range(w):
            return 0, 0
        return self._range_in(self._corpus, self._sa, w)

    def count(self, word: Word) -> int:
        """Number of occurrences of the contiguous ``word`` in the corpus.

        Occurrences never span a realization boundary (the sentinel guards
        every boundary). ``O(|word| log T)`` time.

        Parameters
        ----------
        word : sequence of int
            The contiguous query word. The empty word returns the total number
            of non-sentinel tokens (``sum(realization_lengths)``). An
            out-of-range word returns ``0``.

        Returns
        -------
        int
            The occurrence count.
        """
        w = self._as_word_array(word)
        if w.shape[0] == 0:
            return int(self._realization_lengths.sum())
        if self._out_of_range(w):
            return 0
        lo, hi = self._range_in(self._corpus, self._sa, w)
        return hi - lo

    def right_extensions(self, word: Word) -> List[Tuple[int, int]]:
        """Distinct symbols ``s`` with ``word + (s,)`` occurring, and counts.

        Parameters
        ----------
        word : sequence of int
            The contiguous query word. For the empty word this returns the
            unigram distribution (every distinct first token and its count).

        Returns
        -------
        list of (int, int)
            ``(symbol, count)`` pairs sorted by ``symbol``, sentinels excluded.

        Notes
        -----
        ``O(|word| log T + m)`` time, where ``m`` is the number of occurrences
        of ``word``. The enumeration is vectorized (:func:`numpy.unique` over
        the sorted next-character column) rather than a Python scan.
        """
        w = self._as_word_array(word)
        return self._extensions_in(self._corpus, self._sa, w)

    def left_extensions(self, word: Word) -> List[Tuple[int, int]]:
        """Distinct symbols ``s`` with ``(s,) + word`` occurring, and counts.

        Answered via the reversed-corpus suffix array (a left extension of
        ``word`` is a right extension of ``reversed(word)`` in the reversed
        corpus). The reversed index is built lazily on the first call.

        Parameters
        ----------
        word : sequence of int
            The contiguous query word. For the empty word this returns the
            unigram distribution (every distinct token that occurs).

        Returns
        -------
        list of (int, int)
            ``(symbol, count)`` pairs sorted by ``symbol``, sentinels excluded.

        Notes
        -----
        ``O(|word| log T + m)`` query time after an ``O(T log^2 T)`` one-time
        reversed-index build (or ``O(T)`` with the :mod:`pydivsufsort` fast
        path).
        """
        self._ensure_reverse()
        w = self._as_word_array(word)
        assert self._rev_corpus is not None and self._rev_sa is not None
        return self._extensions_in(self._rev_corpus, self._rev_sa, w[::-1])

    def positions(self, word: Word) -> np.ndarray:
        """Sorted corpus positions of the occurrences of ``word``.

        Parameters
        ----------
        word : sequence of int
            The contiguous query word. The empty word returns every
            non-sentinel corpus position. An out-of-range word returns an empty
            array.

        Returns
        -------
        numpy.ndarray
            1-d ``int64`` array of corpus start positions, ascending.

        Notes
        -----
        ``O(|word| log T + m log m)`` time (the interval is sorted). Map results
        to (realization, offset) with :meth:`realization_of` /
        :meth:`offset_in_realization`.
        """
        w = self._as_word_array(word)
        if w.shape[0] == 0:
            return np.nonzero(self._corpus != self._sentinel)[0].astype(np.int64)
        if self._out_of_range(w):
            return np.empty(0, dtype=np.int64)
        lo, hi = self._range_in(self._corpus, self._sa, w)
        return np.sort(self._sa[lo:hi]).astype(np.int64)

    def count_in_starts(self, word: Word, max_start: int) -> int:
        """Occurrences of ``word`` whose corpus start position is ``<= max_start``.

        A convenience over :meth:`positions` for a single *global* corpus-position
        bound. CSSR's right-boundary edge effect is *per realization* (a word is
        counted only when it starts at within-realization offsets
        ``0 .. T_r - L_max - 1``); when realization lengths differ, apply that
        bound with :meth:`positions` + :meth:`offset_in_realization` +
        :attr:`realization_lengths` rather than a single ``max_start``.

        Parameters
        ----------
        word : sequence of int
            The contiguous query word.
        max_start : int
            Inclusive upper bound on the corpus start position.

        Returns
        -------
        int
            The number of qualifying occurrences. ``O(|word| log T + m)`` time.
        """
        pos = self.positions(word)
        return int(np.count_nonzero(pos <= max_start))

    # -- realization mapping ----------------------------------------------

    def realization_of(self, positions: Union[int, np.ndarray]) -> np.ndarray:
        """Realization index containing each corpus ``position``.

        ``r = searchsorted(realization_offsets, position, "right") - 1``. A
        sentinel position maps to the realization it terminates.

        Parameters
        ----------
        positions : int or numpy.ndarray
            Corpus position(s), each in ``[0, corpus_size)``.

        Returns
        -------
        numpy.ndarray
            ``int64`` array (0-d preserved as 1-element) of realization indices.
        """
        p = np.asarray(positions, dtype=np.int64)
        return np.searchsorted(self._realization_offsets, p, side="right") - 1

    def offset_in_realization(
        self, positions: Union[int, np.ndarray]
    ) -> np.ndarray:
        """Within-realization offset of each corpus ``position``.

        Parameters
        ----------
        positions : int or numpy.ndarray
            Corpus position(s), each in ``[0, corpus_size)``.

        Returns
        -------
        numpy.ndarray
            ``int64`` array of offsets ``position - realization_offsets[r]``.
        """
        p = np.asarray(positions, dtype=np.int64)
        r = self.realization_of(p)
        return p - self._realization_offsets[r]
