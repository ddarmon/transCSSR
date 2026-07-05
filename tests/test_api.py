"""Tests for the public inference API (``transcssr.api``).

Covers: end-to-end ``infer_machine`` on a real dataset; integer-token parity
with the single-character-string pipeline; the documented ensemble-detection
rule; and error handling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transcssr import Alphabet, EpsilonMachine, infer_machine, infer_transducer
from transcssr.api import _normalize_sequences

REPO_ROOT = Path(__file__).resolve().parent.parent


def _even_head(n: int = 5000) -> str:
    return (REPO_ROOT / "data" / "even.dat").read_text().strip()[:n]


# ---------------------------------------------------------------------------
# infer_machine end to end
# ---------------------------------------------------------------------------


def test_infer_machine_even_process_two_states():
    y = _even_head(5000)
    machine = infer_machine(y, L_max=3)
    assert isinstance(machine, EpsilonMachine)
    assert machine.n_states == 2
    # Output-only: single dummy input token.
    assert machine.n_x == 1
    assert machine.input_alphabet.tokens == (0,)
    assert machine.output_alphabet.tokens == ("0", "1")

    # The machine can filter its own training series.
    result = machine.filter(y)
    assert len(result.states) == len(y)
    assert result.prediction_probs.shape == (len(y), 2)


def test_infer_machine_integer_token_end_to_end():
    y = _even_head(5000)
    y_int = [10 if c == "0" else 20 for c in y]

    machine_str = infer_machine(y, L_max=3)
    machine_int = infer_machine(y_int, L_max=3)  # list[int] = single sequence

    # Isomorphic partition: identical index-space histories.
    def index_partition(m):
        inv = m.histories_by_state
        return {frozenset(inv[s]) for s in range(m.n_states)}

    assert index_partition(machine_str) == index_partition(machine_int)
    assert machine_int.n_states == machine_str.n_states == 2

    # Identical predictions modulo token type: compare predicted *indices*.
    r_str = machine_str.filter(y)
    r_int = machine_int.filter(y_int)
    assert len(r_str.predictions) == len(r_int.predictions)
    for ps, pi in zip(r_str.predictions, r_int.predictions):
        if ps is None or pi is None:
            assert ps is None and pi is None
        else:
            assert machine_str.output_alphabet.index(ps) == (
                machine_int.output_alphabet.index(pi)
            )
    assert np.allclose(r_str.prediction_probs, r_int.prediction_probs,
                       equal_nan=True)


# ---------------------------------------------------------------------------
# Ensemble-detection rule (docs/DESIGN.md "Sequence input contract")
# ---------------------------------------------------------------------------


def test_ensemble_detection_list_of_str_is_ensemble():
    assert _normalize_sequences(["01", "10"], "ys") == ["01", "10"]


def test_ensemble_detection_list_of_int_is_single_sequence():
    assert _normalize_sequences([0, 1, 0, 1], "ys") == [[0, 1, 0, 1]]


def test_ensemble_detection_list_of_list_is_ensemble():
    assert _normalize_sequences([[0, 1], [1, 0]], "ys") == [[0, 1], [1, 0]]


def test_ensemble_detection_single_str_is_single_sequence():
    assert _normalize_sequences("0101", "ys") == ["0101"]


def test_infer_machine_ensemble_vs_single_disambiguation():
    # list[int] -> single realization; list[list[int]] -> ensemble of two.
    seq = [0, 1, 0, 0, 1, 0, 0, 1, 0] * 100
    single = infer_machine(seq, L_max=2)
    ensemble = infer_machine([seq, seq], L_max=2)
    # Both are valid fits; pooling two identical realizations doubles counts
    # but yields the same partition structure.
    assert single.n_states == ensemble.n_states


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        infer_transducer("000", "01", L_max=1)


def test_mismatched_realization_count_raises():
    with pytest.raises(ValueError):
        infer_transducer(["00", "11"], ["01"], L_max=1)


def test_unknown_token_raises():
    with pytest.raises(KeyError):
        infer_transducer(
            "00", "01", L_max=1, output_alphabet=["0"]  # '1' absent
        )


def test_bad_stats_mode_raises():
    with pytest.raises(ValueError):
        infer_machine("0101010101", L_max=2, stats_mode="bogus")


def test_bad_sequence_type_raises():
    with pytest.raises(TypeError):
        infer_machine(12345, L_max=2)  # not a sequence


# ---------------------------------------------------------------------------
# infer_transducer smoke: delay channel -> two states
# ---------------------------------------------------------------------------


def test_infer_transducer_delay_channel_two_states():
    rng = np.random.default_rng(777)
    n = 2000
    xb = rng.integers(0, 2, size=n)
    yb = np.empty(n, dtype=int)
    yb[0] = 0
    yb[1:] = xb[:-1]
    sx = "".join(map(str, xb))
    sy = "".join(map(str, yb))
    model = infer_transducer(sx, sy, L_max=2)
    assert model.n_states == 2
    assert model.n_x == 2 and model.n_y == 2
