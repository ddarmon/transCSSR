import itertools

import pytest

from transCSSR import estimate_predictive_distributions, filter_and_predict, run_transCSSR


def test_toy_pipeline_outputs(toy_sequence_pair, toy_baseline):
    results = toy_baseline

    assert results["invepsilon_keys"] == {0}
    assert results["morph_by_state"][0] == [4, 0, 0, 2, 2, 0]
    assert results["filtered_states"] == [-1, 0, 0, 0, 0, 0]
    assert results["filtered_probs"] == [0.5, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert results["stringY_pred"] == "N10001"

    epsilon = results["epsilon"]
    for hist in [("a", "0"), ("b", "1"), ("c", "0")]:
        assert epsilon[hist] == 0


def test_multiline_pipeline(multiline_baseline):
    results = multiline_baseline

    assert results["invepsilon_keys"] == {0}
    assert results["morph_by_state"][0] == [14, 2, 2, 9, 8, 0]

    epsilon = results["epsilon"]
    for hist in [("a", "0"), ("a", "1"), ("b", "1"), ("c", "0")]:
        assert epsilon[hist] == 0


def test_even_process_regression(even_sequence_pair, even_baseline):
    results = even_baseline

    assert results["invepsilon_keys"] == {1, 3}
    assert results["stringY_pred"][:20] == "NNNNNNNNN01010000100"
    assert results["filtered_states"][:10] == [-1] * 9 + [1]

    expected_probs = [0.5] * 9 + [pytest.approx(0.4965026618, rel=1e-9)]
    for observed, expected in zip(results["filtered_probs"], expected_probs):
        assert observed == expected


def test_toy_pipeline_with_integer_tokens(integer_encoded_toy):
    """End-to-end integer-token inference via the new token-first engine.

    Same data as the string toy pipeline (a->0, b->1, c->2); expectations
    mirror the legacy string results ('N10001' becomes [None, 1, 0, 0, 0, 1]
    in token space).
    """
    from transcssr import infer_transducer

    payload = integer_encoded_toy
    model = infer_transducer(
        payload["X"],
        payload["Y"],
        L_max=payload["L_max"],
        input_alphabet=payload["axs"],
        output_alphabet=payload["ays"],
    )
    result = model.filter(payload["X"], payload["Y"])

    assert model.n_states == 1
    assert list(result.states) == [-1, 0, 0, 0, 0, 0]
    assert list(result.predictions) == [None, 1, 0, 0, 0, 1]
