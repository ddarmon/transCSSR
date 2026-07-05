"""Unit tests for the private :mod:`transcssr._suffix` suffix-array engine.

The engine is checked against a naive Python oracle over random small corpora
(multiple realizations, varying vocab sizes) for every query, plus determinism,
edge cases, an optional :mod:`pydivsufsort` parity gate, and a modest scale
sanity test.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np
import pytest

from transcssr._suffix import SuffixIndex, _suffix_array_numpy


# --------------------------------------------------------------------------- #
# Naive Python oracle (operates on the same [tokens ... sentinel] corpus)
# --------------------------------------------------------------------------- #

def _build_corpus(realizations: Sequence[Sequence[int]], vocab_size: int):
    """Reconstruct the corpus + offsets exactly as ``SuffixIndex`` does."""
    sentinel = vocab_size
    corpus: List[int] = []
    offsets: List[int] = []
    for r in realizations:
        offsets.append(len(corpus))
        corpus.extend(int(t) for t in r)
        corpus.append(sentinel)
    return corpus, offsets, sentinel


def _oracle_positions(corpus, sentinel, word) -> List[int]:
    """All corpus start positions of the contiguous ``word`` (empty -> reals)."""
    word = list(word)
    if len(word) == 0:
        return [i for i, v in enumerate(corpus) if v != sentinel]
    L = len(word)
    return [i for i in range(len(corpus) - L + 1) if corpus[i : i + L] == word]


def _oracle_count(corpus, sentinel, word, realization_lengths) -> int:
    if len(list(word)) == 0:
        return int(sum(realization_lengths))
    return len(_oracle_positions(corpus, sentinel, word))


def _oracle_unigram(corpus, sentinel) -> List[Tuple[int, int]]:
    """Distinct non-sentinel tokens with counts (== extensions of the empty word)."""
    counts: Counter = Counter(v for v in corpus if v != sentinel)
    return sorted((int(s), int(c)) for s, c in counts.items())


def _oracle_right(corpus, sentinel, word) -> List[Tuple[int, int]]:
    # Empty word: symbols s with (s,) occurring == unigram distribution.
    if len(list(word)) == 0:
        return _oracle_unigram(corpus, sentinel)
    counts: Counter = Counter()
    L = len(list(word))
    for i in _oracle_positions(corpus, sentinel, word):
        nxt = corpus[i + L]
        if nxt != sentinel:
            counts[nxt] += 1
    return sorted((int(s), int(c)) for s, c in counts.items())


def _oracle_left(corpus, sentinel, word) -> List[Tuple[int, int]]:
    # Empty word: symbols s with (s,) occurring == unigram distribution.
    if len(list(word)) == 0:
        return _oracle_unigram(corpus, sentinel)
    counts: Counter = Counter()
    for i in _oracle_positions(corpus, sentinel, word):
        if i > 0 and corpus[i - 1] != sentinel:
            counts[corpus[i - 1]] += 1
    return sorted((int(s), int(c)) for s, c in counts.items())


def _all_occurring_words(corpus, sentinel, max_len):
    """Every real (sentinel-free) contiguous word up to ``max_len`` that occurs."""
    words = {()}
    n = len(corpus)
    for L in range(1, max_len + 1):
        for i in range(n - L + 1):
            window = tuple(corpus[i : i + L])
            if sentinel not in window:
                words.add(window)
    return words


# --------------------------------------------------------------------------- #
# Exhaustive oracle comparison over random corpora
# --------------------------------------------------------------------------- #

_CORPUS_CASES = [
    # (seed, vocab_size, n_realizations, min_len, max_len)
    (0, 1, 3, 0, 6),      # vocab == 1 (all tokens identical), incl. empty line
    (1, 2, 4, 1, 8),      # binary
    (2, 3, 3, 1, 10),     # ternary
    (3, 5, 5, 0, 12),     # larger vocab, empty realizations mixed in
    (4, 4, 6, 1, 3),      # short realizations (near/under word length)
    (5, 2, 1, 20, 20),    # single realization
    (6, 7, 4, 1, 1),      # single-token realizations
]


@pytest.mark.parametrize("seed,vocab,m,lo,hi", _CORPUS_CASES)
def test_all_queries_match_oracle(seed, vocab, m, lo, hi):
    rng = np.random.default_rng(seed)
    realizations = [
        rng.integers(0, vocab, size=int(rng.integers(lo, hi + 1))).tolist()
        for _ in range(m)
    ]
    si = SuffixIndex(realizations, vocab_size=vocab)
    corpus, offsets, sentinel = _build_corpus(realizations, vocab)

    # Words to probe: all occurring words up to length 4, plus non-occurring
    # samples (out-of-range token, sentinel, a random unseen word).
    words = _all_occurring_words(corpus, sentinel, 4)
    non_occurring = [
        (vocab,),                 # the sentinel value (out of range)
        (vocab + 3,),             # out of range
        (0, vocab + 1),           # contains out-of-range token
        tuple(rng.integers(0, vocab, size=6).tolist()),  # likely absent len-6
    ]
    probe = list(words) + non_occurring

    for word in probe:
        out_of_range = len(word) > 0 and any(t < 0 or t >= vocab for t in word)
        lo_i, hi_i = si.range(word)
        if out_of_range:
            # Words with a token outside [0, vocab) (incl. the sentinel) never
            # occur, regardless of what the raw corpus scan would find.
            assert si.count(word) == 0, word
            assert hi_i <= lo_i, word
            assert si.right_extensions(word) == [], word
            assert si.left_extensions(word) == [], word
            assert si.positions(word).tolist() == [], word
            continue

        assert si.count(word) == _oracle_count(
            corpus, sentinel, word, si.realization_lengths.tolist()
        ), word

        if len(word) == 0:
            assert (lo_i, hi_i) == (0, len(corpus))
        else:
            assert hi_i - lo_i == len(_oracle_positions(corpus, sentinel, word))

        assert si.right_extensions(word) == _oracle_right(corpus, sentinel, word), word
        assert si.left_extensions(word) == _oracle_left(corpus, sentinel, word), word

        got = si.positions(word).tolist()
        assert got == _oracle_positions(corpus, sentinel, word), word

    # count_in_starts against oracle for a few bounds on an occurring word.
    if len(words) > 1:
        w = sorted(w for w in words if len(w) > 0)[0]
        all_pos = _oracle_positions(corpus, sentinel, w)
        for bound in (-1, 0, len(corpus) // 2, len(corpus)):
            assert si.count_in_starts(w, bound) == sum(p <= bound for p in all_pos)


@pytest.mark.parametrize("seed,vocab,m,lo,hi", _CORPUS_CASES)
def test_realization_mapping_matches_construction(seed, vocab, m, lo, hi):
    rng = np.random.default_rng(seed + 100)
    realizations = [
        rng.integers(0, vocab, size=int(rng.integers(lo, hi + 1))).tolist()
        for _ in range(m)
    ]
    si = SuffixIndex(realizations, vocab_size=vocab)
    _, offsets, _ = _build_corpus(realizations, vocab)

    # Every real position maps back to (realization, within-offset).
    for r, start in enumerate(offsets):
        for off in range(len(realizations[r])):
            p = start + off
            assert int(si.realization_of(p)) == r
            assert int(si.offset_in_realization(p)) == off

    # Vectorized form agrees with the scalar form.
    positions = np.arange(si.corpus_size, dtype=np.int64)
    r_vec = si.realization_of(positions)
    assert np.array_equal(
        r_vec, np.searchsorted(np.asarray(offsets), positions, side="right") - 1
    )


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def test_construction_is_deterministic():
    rng = np.random.default_rng(1234)
    realizations = [rng.integers(0, 5, size=50).tolist() for _ in range(4)]
    a = SuffixIndex(realizations, vocab_size=5)
    b = SuffixIndex(realizations, vocab_size=5)
    assert np.array_equal(a.suffix_array, b.suffix_array)
    assert np.array_equal(a.corpus, b.corpus)
    # Reversed index too.
    a.left_extensions((0,))
    b.left_extensions((0,))
    assert a.nbytes == b.nbytes


def test_suffix_array_is_a_valid_sorted_permutation():
    rng = np.random.default_rng(7)
    corpus = rng.integers(0, 4, size=200).astype(np.int32)
    corpus[-1] = 4  # a sentinel-like terminator
    sa = _suffix_array_numpy(corpus)
    assert sorted(sa.tolist()) == list(range(len(corpus)))
    suffixes = [corpus[i:].tolist() for i in sa]
    assert suffixes == sorted(suffixes)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

def test_empty_word_semantics():
    si = SuffixIndex([[0, 1, 0], [1, 1]], vocab_size=2)
    assert si.count(()) == 5  # total non-sentinel tokens
    assert si.range(()) == (0, si.corpus_size)  # spans sentinels
    assert si.positions(()).tolist() == [0, 1, 2, 4, 5]  # non-sentinel positions
    # Empty-word extensions == unigram distribution.
    assert si.right_extensions(()) == [(0, 2), (1, 3)]
    assert si.left_extensions(()) == [(0, 2), (1, 3)]


def test_word_longer_than_corpus():
    si = SuffixIndex([[0, 1]], vocab_size=2)
    long_word = (0, 1, 0, 1, 0)
    assert si.count(long_word) == 0
    lo, hi = si.range(long_word)
    assert hi <= lo
    assert si.positions(long_word).tolist() == []
    assert si.right_extensions(long_word) == []
    assert si.left_extensions(long_word) == []


def test_out_of_range_and_sentinel_symbols():
    si = SuffixIndex([[0, 1, 2, 1, 0]], vocab_size=3)
    for bad in [(3,), (99,), (0, 3), (-1,), (1, 5, 2)]:
        assert si.count(bad) == 0
        assert si.right_extensions(bad) == []
        assert si.left_extensions(bad) == []
        assert si.positions(bad).tolist() == []


def test_empty_realization_list():
    si = SuffixIndex([], vocab_size=4)
    assert si.corpus_size == 0
    assert si.n_realizations == 0
    assert si.count(()) == 0
    assert si.count((0,)) == 0
    assert si.positions((0,)).tolist() == []
    assert si.right_extensions((0,)) == []
    assert si.left_extensions((0,)) == []
    assert si.suffix_array.shape == (0,)


def test_realization_shorter_than_query():
    si = SuffixIndex([[0], [1, 0], []], vocab_size=2)
    assert si.count((0, 1, 0)) == 0
    # A length-2 word only occurs where a realization is long enough.
    assert si.count((1, 0)) == 1
    assert si.count((0,)) == 2


def test_vocab_inferred_when_omitted():
    si = SuffixIndex([[0, 3, 1], [2, 2]])
    assert si.vocab_size == 4
    assert si.sentinel == 4
    assert si.count((3,)) == 1
    assert si.count((2,)) == 2


def test_construction_rejects_bad_tokens():
    with pytest.raises(ValueError):
        SuffixIndex([[0, -1]])
    with pytest.raises(ValueError):
        SuffixIndex([[0, 5]], vocab_size=3)
    with pytest.raises(ValueError):
        SuffixIndex([np.zeros((2, 2), dtype=int)])


def test_nbytes_grows_when_reverse_built():
    rng = np.random.default_rng(3)
    si = SuffixIndex([rng.integers(0, 6, size=100).tolist()], vocab_size=6)
    before = si.nbytes
    si.left_extensions((0,))  # triggers reversed-index build
    after = si.nbytes
    assert after > before
    # int32 corpus when vocab fits.
    assert si.corpus.dtype == np.int32
    assert si.suffix_array.dtype == np.int32


# --------------------------------------------------------------------------- #
# Optional pydivsufsort parity
# --------------------------------------------------------------------------- #

def test_pydivsufsort_parity():
    pd = pytest.importorskip("pydivsufsort")
    rng = np.random.default_rng(11)
    for _ in range(5):
        corpus = rng.integers(0, 6, size=int(rng.integers(50, 500))).astype(np.int32)
        corpus[-1] = 6
        numpy_sa = _suffix_array_numpy(corpus)
        fast_sa = np.asarray(pd.divsufsort(np.ascontiguousarray(corpus)))
        assert np.array_equal(numpy_sa.astype(np.int64), fast_sa.astype(np.int64))


# --------------------------------------------------------------------------- #
# Modest scale sanity test (T ~ 2e5, vocab 256)
# --------------------------------------------------------------------------- #

def test_scale_sanity():
    rng = np.random.default_rng(2024)
    vocab = 256
    # A few realizations summing to ~2e5 tokens.
    realizations = [rng.integers(0, vocab, size=n).tolist() for n in (80_000, 70_000, 50_000)]
    t0 = time.perf_counter()
    si = SuffixIndex(realizations, vocab_size=vocab)
    build_s = time.perf_counter() - t0
    assert build_s < 15.0  # generous; typically well under a second

    corpus, _, sentinel = _build_corpus(realizations, vocab)

    # A handful of oracle-checked queries: single tokens + a guaranteed 2-gram.
    probe = [(0,), (255,), (128,)]
    # Take a real 2-gram from the first realization so it certainly occurs.
    probe.append(tuple(realizations[0][:2]))
    for word in probe:
        assert si.count(word) == _oracle_count(
            corpus, sentinel, word, si.realization_lengths.tolist()
        )
        assert si.right_extensions(word) == _oracle_right(corpus, sentinel, word)
        assert si.left_extensions(word) == _oracle_left(corpus, sentinel, word)
