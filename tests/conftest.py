import itertools
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pytest

from transCSSR import (
    estimate_predictive_distributions,
    filter_and_predict,
    run_transCSSR,
)


# ---------------------------------------------------------------------------
# Shared live-parity helpers (imported by test_cssr.py and test_predict.py)
# ---------------------------------------------------------------------------


def _random_binary(seed: int, n: int) -> str:
    """A length-``n`` random binary string from a seeded generator."""
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("01"), size=n))


def _delay_process(seed: int, n: int):
    """A delay channel ``Y_t = X_{t-1}`` (one bit of memory -> 2 states)."""
    rng = np.random.default_rng(seed)
    xb = rng.integers(0, 2, size=n)
    yb = np.empty(n, dtype=int)
    yb[0] = 0
    yb[1:] = xb[:-1]
    return "".join(map(str, xb)), "".join(map(str, yb))


# (seed, L_max, kind) -- two independent-random cases and two delay cases.
LIVE_CASES = [
    (101, 2, "indep"),
    (202, 3, "indep"),
    (303, 2, "delay"),
    (404, 3, "delay"),
]


@dataclass(frozen=True)
class SequencePair:
    stringX: str
    stringY: str
    axs: List[str]
    ays: List[str]
    L_max: int

    @property
    def e_symbols(self) -> List[Tuple[str, str]]:
        return list(itertools.product(self.axs, self.ays))


@pytest.fixture(scope="module")
def toy_sequence_pair() -> SequencePair:
    """Single short sequence used across unit tests."""
    return SequencePair(
        stringX="abacab",
        stringY="010001",
        axs=["a", "b", "c"],
        ays=["0", "1"],
        L_max=2,
    )


@pytest.fixture(scope="module")
def toy_expected_counts() -> Tuple[Counter, Counter]:
    """Baseline marginal and future counters for the toy sequence."""
    marginal = Counter({
        ("a", ""): 2,
        ("ab", "0"): 1,
        ("aba", "01"): 1,
        ("ac", "0"): 1,
        ("aca", "00"): 1,
        ("b", ""): 1,
        ("ba", "1"): 1,
        ("bac", "10"): 1,
        ("c", ""): 1,
        ("ca", "0"): 1,
        ("cab", "00"): 1,
    })
    futures = Counter({
        ("a", "0"): 2,
        ("ab", "01"): 1,
        ("aba", "010"): 1,
        ("ac", "00"): 1,
        ("aca", "000"): 1,
        ("b", "1"): 1,
        ("ba", "10"): 1,
        ("bac", "100"): 1,
        ("c", "0"): 1,
        ("ca", "00"): 1,
        ("cab", "001"): 1,
    })

    return marginal, futures


@pytest.fixture(scope="module")
def toy_baseline(toy_sequence_pair: SequencePair) -> Dict[str, object]:
    """Pre-computed pipeline outputs for the toy sequence."""
    seq = toy_sequence_pair
    word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(
        seq.stringX,
        seq.stringY,
        seq.L_max,
        axs=seq.axs,
        ays=seq.ays,
    )

    epsilon, invepsilon, morph_by_state = run_transCSSR(
        word_lookup_marg,
        word_lookup_fut,
        seq.L_max,
        seq.axs,
        seq.ays,
        seq.e_symbols,
        "toy_input",
        "toy_output",
    )

    filtered_states, filtered_probs, stringY_pred = filter_and_predict(
        seq.stringX,
        seq.stringY,
        epsilon,
        invepsilon,
        morph_by_state,
        seq.axs,
        seq.ays,
        seq.e_symbols,
        seq.L_max,
    )

    return {
        "word_lookup_marg": word_lookup_marg,
        "word_lookup_fut": word_lookup_fut,
        "epsilon": dict(epsilon),
        "invepsilon_keys": set(invepsilon.keys()),
        "morph_by_state": {state: list(counts) for state, counts in morph_by_state.items()},
        "filtered_states": list(filtered_states),
        "filtered_probs": [float(p) for p in filtered_probs],
        "stringY_pred": stringY_pred,
    }


@pytest.fixture(scope="module")
def multiline_sequences() -> Tuple[List[str], List[str], List[str], List[str], int]:
    stringX_list = [
        "abacabac",
        "babacaba",
        "cabacabc",
    ]
    stringY_list = [
        "01000100",
        "10100101",
        "00100010",
    ]
    axs = ["a", "b", "c"]
    ays = ["0", "1"]
    L_max = 2

    return stringX_list, stringY_list, axs, ays, L_max


@pytest.fixture(scope="module")
def multiline_expected_counts() -> Tuple[Counter, Counter]:
    marginal = Counter({
        ("a", ""): 9,
        ("ab", "0"): 5,
        ("ab", "1"): 1,
        ("aba", "01"): 4,
        ("aba", "10"): 1,
        ("abc", "01"): 1,
        ("ac", "0"): 3,
        ("aca", "00"): 3,
        ("b", ""): 5,
        ("ba", "1"): 5,
        ("bab", "10"): 1,
        ("bac", "10"): 4,
        ("c", ""): 4,
        ("ca", "0"): 4,
        ("cab", "00"): 3,
        ("cab", "01"): 1,
    })

    futures = Counter({
        ("a", "0"): 8,
        ("a", "1"): 1,
        ("ab", "01"): 5,
        ("ab", "10"): 1,
        ("aba", "010"): 4,
        ("aba", "101"): 1,
        ("abc", "010"): 1,
        ("ac", "00"): 3,
        ("aca", "000"): 2,
        ("aca", "001"): 1,
        ("b", "1"): 5,
        ("ba", "10"): 5,
        ("bab", "101"): 1,
        ("bac", "100"): 4,
        ("c", "0"): 4,
        ("ca", "00"): 3,
        ("ca", "01"): 1,
        ("cab", "001"): 3,
        ("cab", "010"): 1,
    })

    return marginal, futures


@pytest.fixture(scope="module")
def multiline_baseline(multiline_sequences):
    stringX_list, stringY_list, axs, ays, L_max = multiline_sequences
    e_symbols = list(itertools.product(axs, ays))

    word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(
        stringX_list,
        stringY_list,
        L_max,
        axs=axs,
        ays=ays,
        is_multiline=True,
    )

    epsilon, invepsilon, morph_by_state = run_transCSSR(
        word_lookup_marg,
        word_lookup_fut,
        L_max,
        axs,
        ays,
        e_symbols,
        "test_input_multiline",
        "test_output_multiline",
    )

    return {
        "word_lookup_marg": word_lookup_marg,
        "word_lookup_fut": word_lookup_fut,
        "epsilon": dict(epsilon),
        "invepsilon_keys": set(invepsilon.keys()),
        "morph_by_state": {state: list(counts) for state, counts in morph_by_state.items()},
    }


@pytest.fixture(scope="module")
def even_sequence_pair() -> SequencePair:
    with open("data/even.dat") as fh:
        stringY = fh.readline().strip()
    stringX = "0" * len(stringY)
    return SequencePair(
        stringX=stringX,
        stringY=stringY,
        axs=["0"],
        ays=["0", "1"],
        L_max=3,
    )


@pytest.fixture(scope="module")
def even_baseline(even_sequence_pair: SequencePair):
    seq = even_sequence_pair
    word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(
        seq.stringX,
        seq.stringY,
        seq.L_max,
        axs=seq.axs,
        ays=seq.ays,
    )

    epsilon, invepsilon, morph_by_state = run_transCSSR(
        word_lookup_marg,
        word_lookup_fut,
        seq.L_max,
        seq.axs,
        seq.ays,
        seq.e_symbols,
        "zeros",
        "even",
    )

    filtered_states, filtered_probs, stringY_pred = filter_and_predict(
        seq.stringX,
        seq.stringY,
        epsilon,
        invepsilon,
        morph_by_state,
        seq.axs,
        seq.ays,
        seq.e_symbols,
        seq.L_max,
    )

    return {
        "word_lookup_marg": word_lookup_marg,
        "word_lookup_fut": word_lookup_fut,
        "epsilon": dict(epsilon),
        "invepsilon_keys": set(invepsilon.keys()),
        "morph_by_state": {state: list(counts) for state, counts in morph_by_state.items()},
        "filtered_states": list(filtered_states),
        "filtered_probs": [float(p) for p in filtered_probs[:10]],
        "stringY_pred": stringY_pred,
    }


@pytest.fixture(scope="module")
def integer_encoded_toy() -> Dict[str, object]:
    """Integer token version of the toy sequence (for future parity tests)."""
    return {
        "X": [0, 1, 0, 2, 0, 1],
        "Y": [0, 1, 0, 0, 0, 1],
        "axs": [0, 1, 2],
        "ays": [0, 1],
        "L_max": 2,
    }


@pytest.fixture(scope="module")
def integer_encoded_even(even_sequence_pair: SequencePair) -> Dict[str, object]:
    """Integer token version of the even process data."""
    seq = even_sequence_pair
    return {
        "X": [0] * len(seq.stringX),
        "Y": [int(ch) for ch in seq.stringY],
        "axs": [0],
        "ays": [0, 1],
        "L_max": seq.L_max,
    }
