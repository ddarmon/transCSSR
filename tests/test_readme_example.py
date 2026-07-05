import itertools

from transCSSR import estimate_predictive_distributions, filter_and_predict, run_transCSSR


def test_readme_example_pipeline_behaves_as_documented():
    stringX = "abacabacabac"
    stringY = "010001000100"
    axs = ["a", "b", "c"]
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
        "test_input",
        "test_output",
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

    assert len(invepsilon) >= 1
    assert len(filtered_states) == len(stringX)
    assert len(filtered_probs) == len(stringX)
    assert len(stringY_pred) == len(stringY)
