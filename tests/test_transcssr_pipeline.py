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


@pytest.mark.xfail(reason="Integer token pipeline not yet supported")
def test_toy_pipeline_with_integer_tokens(integer_encoded_toy):
    payload = integer_encoded_toy
    e_symbols = list(itertools.product(payload["axs"], payload["ays"]))

    word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(
        payload["X"],
        payload["Y"],
        payload["L_max"],
        axs=payload["axs"],
        ays=payload["ays"],
    )

    epsilon, invepsilon, morph_by_state = run_transCSSR(
        word_lookup_marg,
        word_lookup_fut,
        payload["L_max"],
        payload["axs"],
        payload["ays"],
        e_symbols,
        "toy_input_int",
        "toy_output_int",
    )

    filtered_states, _, stringY_pred = filter_and_predict(
        payload["X"],
        payload["Y"],
        epsilon,
        invepsilon,
        morph_by_state,
        payload["axs"],
        payload["ays"],
        e_symbols,
        payload["L_max"],
    )

    assert list(filtered_states) == [-1, 0, 0, 0, 0, 0]
    assert stringY_pred == "N10001"
