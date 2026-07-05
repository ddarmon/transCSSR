import itertools

import pytest

from transCSSR import estimate_predictive_distributions, filter_and_predict, run_transCSSR


def test_demo_even_data_matches_regression():
    with open("data/even.dat") as fh:
        stringY = fh.readline().strip()
    stringX = "0" * len(stringY)

    axs = ["0"]
    ays = ["0", "1"]
    L_max = 3
    e_symbols = list(itertools.product(axs, ays))

    word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(
        stringX,
        stringY,
        L_max,
        axs=axs,
        ays=ays,
    )

    epsilon, invepsilon, morph_by_state = run_transCSSR(
        word_lookup_marg,
        word_lookup_fut,
        L_max,
        axs,
        ays,
        e_symbols,
        "zeros",
        "even",
    )

    filtered_states, filtered_probs, stringY_pred = filter_and_predict(
        stringX,
        stringY,
        epsilon,
        invepsilon,
        morph_by_state,
        axs,
        ays,
        e_symbols,
        L_max,
    )

    assert len(invepsilon) == 2
    assert stringY_pred[:20] == "NNNNNNNNN01010000100"
    assert filtered_states[:10] == [-1] * 9 + [1]
    assert filtered_probs[9] == pytest.approx(0.4965026618, rel=1e-9)
