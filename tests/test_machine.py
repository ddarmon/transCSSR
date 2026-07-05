"""Unit tests for :mod:`transcssr.machine`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transcssr import Alphabet, EpsilonMachine, EpsilonTransducer


def _even_process_machine() -> EpsilonMachine:
    """A hand-built 2-state even-process epsilon-machine (L_max=2).

    State 0 = "even parity" (A), state 1 = "odd trailing 1" (B). Output
    alphabet {0, 1}, single dummy input.
    """
    out = Alphabet([0, 1])
    epsilon = {
        # length-1 histories (input all-zeros dummy)
        ((0,), (0,)): 0,
        ((0,), (1,)): 1,
        # length-2 histories (transition targets)
        ((0, 0), (0, 0)): 0,
        ((0, 0), (0, 1)): 1,
        ((0, 0), (1, 0)): 0,
        ((0, 0), (1, 1)): 0,
    }
    morph_counts = np.array([[2, 1], [0, 3]])  # (n_states=2, n_y=2)
    return EpsilonMachine(out, n_states=2, epsilon=epsilon,
                          morph_counts=morph_counts, L_max=2)


def test_transducer_construction_and_sizes():
    model = _even_process_machine()
    assert model.n_states == 2
    assert model.n_x == 1
    assert model.n_y == 2
    assert model.L_max == 2
    assert isinstance(model, EpsilonTransducer)
    assert model.input_alphabet == Alphabet([0])


def test_bad_morph_shape_raises():
    out = Alphabet([0, 1])
    with pytest.raises(ValueError, match="shape"):
        EpsilonMachine(out, n_states=2, epsilon={},
                       morph_counts=np.zeros((2, 5)), L_max=1)


def test_histories_by_state():
    model = _even_process_machine()
    inv = model.histories_by_state
    assert ((0,), (0,)) in inv[0]
    assert ((0,), (1,)) in inv[1]
    assert ((0, 0), (1, 1)) in inv[0]


def test_emission_probs_normalized():
    model = _even_process_machine()
    probs = model.emission_probs()
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs[0], [2 / 3, 1 / 3])
    np.testing.assert_allclose(probs[1], [0.0, 1.0])


def test_transitions_even_process():
    model = _even_process_machine()
    trans = model.transitions()
    # need_Lmax is False for both states (each has a length-1 history), so
    # transitions are read off the length-1 histories only.
    assert trans == {
        (0, 0, 0): 0,
        (0, 0, 1): 1,
        (1, 0, 0): 0,
        (1, 0, 1): 0,
    }


def test_canonical_relabeling():
    out = Alphabet([0, 1])
    # State 0's smallest history is larger than state 1's -> should swap.
    epsilon = {
        ((1,), (1,)): 0,
        ((0,), (0,)): 1,
    }
    morph_counts = np.array([[7, 8], [1, 2]])
    model = EpsilonMachine(out, n_states=2, epsilon=epsilon,
                           morph_counts=morph_counts, L_max=1)
    canon = model.canonicalize()
    # After canonicalization, the state whose smallest history is ((0,),(0,))
    # becomes state 0.
    assert canon.epsilon[((0,), (0,))] == 0
    assert canon.epsilon[((1,), (1,))] == 1
    # Morph rows follow the relabeling.
    np.testing.assert_array_equal(canon.morph_counts[0], [1, 2])
    np.testing.assert_array_equal(canon.morph_counts[1], [7, 8])
    assert isinstance(canon, EpsilonMachine)


def test_json_roundtrip_string_tokens(tmp_path):
    in_alpha = Alphabet(["a", "b"])
    out_alpha = Alphabet(["0", "1"])
    epsilon = {
        ((0,), (0,)): 0,
        ((1,), (1,)): 1,
    }
    morph_counts = np.array([[3, 4, 1, 0], [0, 1, 2, 5]])  # (2, n_x*n_y=4)
    model = EpsilonTransducer(in_alpha, out_alpha, 2, epsilon, morph_counts, 1)

    path = tmp_path / "model.json"
    model.to_json(path)
    restored = EpsilonTransducer.from_json(path)

    assert restored.input_alphabet == in_alpha
    assert restored.output_alphabet == out_alpha
    assert restored.n_states == 2
    assert restored.L_max == 1
    assert restored.epsilon == epsilon
    np.testing.assert_array_equal(restored.morph_counts, morph_counts)


def test_json_roundtrip_integer_tokens(tmp_path):
    model = _even_process_machine()
    path = tmp_path / "machine.json"
    model.to_json(path)
    restored = EpsilonTransducer.from_json(path)
    assert restored.output_alphabet == Alphabet([0, 1])
    assert restored.input_alphabet == Alphabet([0])
    assert restored.epsilon == model.epsilon
    np.testing.assert_array_equal(restored.morph_counts, model.morph_counts)


def test_machine_json_roundtrip_preserves_class(tmp_path):
    """A machine survives JSON roundtrip as an EpsilonMachine (Fix 1).

    ``EpsilonMachine.from_json`` must return an :class:`EpsilonMachine` (not a
    demoted :class:`EpsilonTransducer`), so the output-only ``.filter(ys)``
    surface keeps working and produces the same result as before serialization.
    """
    from transcssr import infer_machine

    y = (
        (Path(__file__).resolve().parent.parent / "data" / "even.dat")
        .read_text()
        .strip()[:5000]
    )
    machine = infer_machine(y, L_max=3)
    before = machine.filter(y)  # output-only surface

    path = tmp_path / "machine.json"
    machine.to_json(path)
    restored = EpsilonMachine.from_json(path)

    assert isinstance(restored, EpsilonMachine)
    after = restored.filter(y)  # would TypeError on a demoted transducer
    assert after.states == before.states
    np.testing.assert_array_equal(after.prediction_probs, before.prediction_probs)


def test_machine_from_json_kind_wins_over_cls(tmp_path):
    """``kind`` in the file, not the calling class, decides the type (Fix 1)."""
    machine = _even_process_machine()
    path = tmp_path / "machine.json"
    machine.to_json(path)

    # Calling EpsilonTransducer.from_json on a machine file still yields a machine.
    restored = EpsilonTransducer.from_json(path)
    assert isinstance(restored, EpsilonMachine)


def test_transducer_json_kind_field():
    """``to_dict`` tags transducers vs. machines with a ``kind`` field (Fix 1)."""
    out = Alphabet([0, 1])
    machine = EpsilonMachine(out, n_states=1, epsilon={((0,), (0,)): 0},
                             morph_counts=np.array([[1, 1]]), L_max=1)
    assert machine.to_dict()["kind"] == "machine"

    in_alpha = Alphabet(["a"])
    trans = EpsilonTransducer(in_alpha, out, 1, {((0,), (0,)): 0},
                              np.array([[1, 1]]), 1)
    assert trans.to_dict()["kind"] == "transducer"


def test_json_non_representable_tokens_error():
    out = Alphabet([frozenset({1}), frozenset({2})])
    model = EpsilonMachine(out, n_states=1, epsilon={((0,), (0,)): 0},
                           morph_counts=np.array([[1, 1]]), L_max=1)
    with pytest.raises(TypeError, match="non-JSON-representable"):
        model.to_dict()


def test_to_dot_smoke_multistate(tmp_path):
    model = _even_process_machine()
    path = tmp_path / "model.dot"
    model.to_dot(path)
    text = path.read_text()
    assert "digraph" in text
    assert "->" in text
    # Edge labels follow the y|x:prob convention.
    assert "|" in text
    assert ":" in text
    assert "label" in text


def test_to_dot_smoke_single_state(tmp_path):
    out = Alphabet([0, 1])
    model = EpsilonMachine(out, n_states=1, epsilon={((0,), (0,)): 0},
                           morph_counts=np.array([[3, 2]]), L_max=1)
    path = tmp_path / "single.dot"
    model.to_dot(path)
    text = path.read_text()
    assert "A -> A" in text
    assert "|" in text
