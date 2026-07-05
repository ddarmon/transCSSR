import itertools

from transCSSR import estimate_predictive_distributions, run_transCSSR


def test_multiline_pipeline_infers_single_state():
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

    assert len(invepsilon) == 1
    (state_id,) = invepsilon.keys()
    assert morph_by_state[state_id] == [14, 2, 2, 9, 8, 0]
