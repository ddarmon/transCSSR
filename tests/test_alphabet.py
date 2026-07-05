"""Unit tests for :class:`transcssr.Alphabet`."""

from __future__ import annotations

import numpy as np
import pytest

from transcssr import Alphabet


def test_construction_and_bijection():
    alpha = Alphabet(["a", "b", "c"])
    assert len(alpha) == 3
    assert alpha.tokens == ("a", "b", "c")
    assert alpha.index("a") == 0
    assert alpha.index("c") == 2
    assert alpha.token(1) == "b"


def test_duplicate_tokens_raise():
    with pytest.raises(ValueError, match="Duplicate token"):
        Alphabet(["a", "b", "a"])


def test_integer_tokens():
    alpha = Alphabet([0, 1, 2])
    assert alpha.index(1) == 1
    assert alpha.token(2) == 2
    assert alpha.encode([2, 0, 1]) == (2, 0, 1)


def test_from_data_sorted_when_orderable():
    alpha = Alphabet.from_data("cabca")
    # Sorted order for mutually orderable tokens.
    assert alpha.tokens == ("a", "b", "c")


def test_from_data_integer_sorted():
    alpha = Alphabet.from_data([3, 1, 2, 1, 3])
    assert alpha.tokens == (1, 2, 3)


def test_from_data_first_appearance_when_not_orderable():
    # Mixing str and int is not mutually orderable in Python 3.
    alpha = Alphabet.from_data(["b", 1, "a", 1])
    assert alpha.tokens == ("b", 1, "a")


def test_from_data_ensemble():
    alpha = Alphabet.from_data(["abc", "cad"])
    assert alpha.tokens == ("a", "b", "c", "d")


def test_from_data_deterministic():
    seqs = ["zya", "axz"]
    a1 = Alphabet.from_data(seqs)
    a2 = Alphabet.from_data(seqs)
    assert a1 == a2
    assert a1.tokens == ("a", "x", "y", "z")


def test_encode_decode_roundtrip_str():
    alpha = Alphabet(["a", "b", "c"])
    idxs = alpha.encode("abcab")
    assert idxs == (0, 1, 2, 0, 1)
    assert alpha.decode(idxs) == ("a", "b", "c", "a", "b")


def test_encode_list_and_tuple():
    alpha = Alphabet(["x", "y"])
    assert alpha.encode(["x", "y", "x"]) == (0, 1, 0)
    assert alpha.encode(("y", "y")) == (1, 1)


def test_encode_ndarray():
    alpha = Alphabet([10, 20, 30])
    arr = np.array([30, 10, 20], dtype=np.int64)
    assert alpha.encode(arr) == (2, 0, 1)


def test_encode_ndarray_string_tokens():
    alpha = Alphabet(["a", "b"])
    arr = np.array(["b", "a", "b"])
    assert alpha.encode(arr) == (1, 0, 1)


def test_encode_rejects_2d_ndarray():
    alpha = Alphabet([0, 1])
    with pytest.raises(ValueError, match="1-d"):
        alpha.encode(np.zeros((2, 2)))


def test_unknown_token_error_message():
    alpha = Alphabet(["a", "b"])
    with pytest.raises(KeyError) as excinfo:
        alpha.index("z")
    assert "'z'" in str(excinfo.value)
    assert "not in the alphabet" in str(excinfo.value)


def test_token_out_of_range():
    alpha = Alphabet(["a", "b"])
    with pytest.raises(IndexError, match="out of range"):
        alpha.token(5)


def test_equality_and_repr():
    a1 = Alphabet(["a", "b"])
    a2 = Alphabet(["a", "b"])
    a3 = Alphabet(["b", "a"])
    assert a1 == a2
    assert a1 != a3
    assert repr(a1) == "Alphabet(['a', 'b'])"


def test_numpy_scalar_tokens_normalized():
    alpha = Alphabet(np.array([5, 6, 7]))
    assert alpha.tokens == (5, 6, 7)
    assert all(isinstance(t, int) for t in alpha.tokens)
    assert alpha.index(np.int64(6)) == 1
