"""Token <-> index bijection for the token-first ``transcssr`` package.

The :class:`Alphabet` maps arbitrary hashable *tokens* (integers from NLP
tokenizers, single-character strings, tuples, ...) to contiguous integer
indices ``0 .. n-1`` and back. All algorithm code in the package operates on
index tuples; tokens are only ever mapped, never interpreted.
"""

from __future__ import annotations

from typing import Any, Hashable, Iterable, Sequence

import numpy as np

__all__ = ["Alphabet"]


def _normalize_token(token: Any) -> Hashable:
    """Coerce numpy scalar tokens to their native Python equivalents.

    ``numpy.int64(0)`` hashes and compares equal to ``0``, but normalizing on
    the way in keeps the stored ``tokens`` tuple free of numpy types so that
    serialization and ``repr`` are clean and deterministic.

    Parameters
    ----------
    token : object
        A candidate token.

    Returns
    -------
    object
        ``token.item()`` if ``token`` is a numpy scalar, otherwise ``token``
        unchanged.
    """
    if isinstance(token, np.generic):
        return token.item()
    return token


def _is_sequence(obj: Any) -> bool:
    """Return ``True`` if ``obj`` is an accepted sequence container."""
    return isinstance(obj, (str, list, tuple, np.ndarray))


def _iter_tokens(seq: Any) -> Iterable[Hashable]:
    """Yield normalized tokens from a single sequence.

    Accepts ``str`` (each character a token), ``list``/``tuple`` of tokens, or
    a 1-d :class:`numpy.ndarray`.
    """
    if isinstance(seq, str):
        yield from seq
        return
    if isinstance(seq, np.ndarray):
        if seq.ndim != 1:
            raise ValueError(
                f"Only 1-d numpy arrays are accepted as sequences; got "
                f"ndim={seq.ndim}."
            )
        for token in seq.tolist():
            yield _normalize_token(token)
        return
    if isinstance(seq, (list, tuple)):
        for token in seq:
            yield _normalize_token(token)
        return
    raise TypeError(
        f"Unsupported sequence type {type(seq).__name__!r}; expected str, "
        f"list, tuple, or 1-d numpy.ndarray."
    )


class Alphabet:
    """Immutable ordered bijection between tokens and integer indices.

    Parameters
    ----------
    tokens : sequence of hashable
        The tokens, in the desired index order. Must be unique.

    Attributes
    ----------
    tokens : tuple
        The tokens in index order.

    Raises
    ------
    ValueError
        If ``tokens`` contains duplicates.
    """

    __slots__ = ("_tokens", "_index")

    def __init__(self, tokens: Sequence[Hashable]) -> None:
        normalized = tuple(_normalize_token(t) for t in tokens)
        index: dict[Hashable, int] = {}
        for i, token in enumerate(normalized):
            if token in index:
                raise ValueError(
                    f"Duplicate token {token!r} at position {i}; alphabet "
                    f"tokens must be unique."
                )
            index[token] = i
        self._tokens: tuple[Hashable, ...] = normalized
        self._index: dict[Hashable, int] = index

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_data(cls, seqs: Any) -> "Alphabet":
        """Build an alphabet by scanning one or more sequences.

        Token order is the sorted order when all observed tokens are mutually
        orderable, otherwise first-appearance order. The result is fully
        deterministic in both cases.

        Parameters
        ----------
        seqs : sequence or ensemble
            Either a single sequence (``str``, ``list``/``tuple`` of scalar
            tokens, or 1-d :class:`numpy.ndarray`) or an *ensemble* -- a
            ``list``/``tuple`` whose elements are themselves sequences. A list
            whose elements are all scalar tokens is treated as a single
            sequence; a ``list[str]`` is treated as an ensemble of character
            sequences (matching legacy multiline usage).

        Returns
        -------
        Alphabet
            An alphabet containing every distinct token observed.
        """
        seen: dict[Hashable, None] = {}
        for token in _iter_all_tokens(seqs):
            if token not in seen:
                seen[token] = None
        tokens = list(seen)
        try:
            tokens_sorted = sorted(tokens)
        except TypeError:
            # Tokens are not mutually orderable: keep first-appearance order.
            return cls(tokens)
        return cls(tokens_sorted)

    # -- lookups -----------------------------------------------------------

    def index(self, token: Hashable) -> int:
        """Return the integer index of ``token``.

        Raises
        ------
        KeyError
            If ``token`` is not in the alphabet.
        """
        key = _normalize_token(token)
        try:
            return self._index[key]
        except KeyError:
            raise KeyError(
                f"Token {token!r} is not in the alphabet {list(self._tokens)!r}."
            ) from None

    def token(self, i: int) -> Hashable:
        """Return the token at index ``i``.

        Raises
        ------
        IndexError
            If ``i`` is out of range.
        """
        try:
            return self._tokens[i]
        except IndexError:
            raise IndexError(
                f"Index {i} is out of range for alphabet of size "
                f"{len(self._tokens)}."
            ) from None

    def encode(self, seq: Any) -> tuple[int, ...]:
        """Encode a token sequence to a tuple of indices.

        Parameters
        ----------
        seq : str, list, tuple, or 1-d numpy.ndarray
            The token sequence to encode.

        Returns
        -------
        tuple of int
            The indices of the tokens, in order.

        Raises
        ------
        KeyError
            If any token is not in the alphabet.
        """
        return tuple(self.index(token) for token in _iter_tokens(seq))

    def decode(self, idxs: Iterable[int]) -> tuple[Hashable, ...]:
        """Decode an iterable of indices back to a tuple of tokens.

        Raises
        ------
        IndexError
            If any index is out of range.
        """
        return tuple(self.token(int(i)) for i in idxs)

    # -- dunder ------------------------------------------------------------

    @property
    def tokens(self) -> tuple[Hashable, ...]:
        """The tokens in index order."""
        return self._tokens

    def __len__(self) -> int:
        return len(self._tokens)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Alphabet):
            return NotImplemented
        return self._tokens == other._tokens

    def __hash__(self) -> int:
        return hash(self._tokens)

    def __repr__(self) -> str:
        return f"Alphabet({list(self._tokens)!r})"


def _iter_all_tokens(seqs: Any) -> Iterable[Hashable]:
    """Yield every token from ``seqs``, treating it as a sequence or ensemble.

    An ensemble is a ``list``/``tuple`` whose elements are themselves
    sequences (``str``/``list``/``tuple``/``ndarray``). A ``list``/``tuple``
    whose elements are all scalar tokens is a single sequence.
    """
    if isinstance(seqs, (list, tuple)) and len(seqs) > 0 and all(
        _is_sequence(el) for el in seqs
    ):
        for seq in seqs:
            yield from _iter_tokens(seq)
    else:
        yield from _iter_tokens(seqs)
