from collections import Counter

import pytest

from transCSSR import estimate_predictive_distributions


def _normalize(counter):
    """Return a plain Counter to avoid defaultdict comparison issues."""
    return Counter(counter)


def test_single_sequence_counts(toy_sequence_pair, toy_expected_counts):
    seq = toy_sequence_pair
    expected_marg, expected_fut = toy_expected_counts

    word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(
        seq.stringX,
        seq.stringY,
        seq.L_max,
        axs=seq.axs,
        ays=seq.ays,
    )

    assert _normalize(word_lookup_marg) == expected_marg
    assert _normalize(word_lookup_fut) == expected_fut


def test_single_sequence_counts_via_counting_method_one(toy_sequence_pair, toy_expected_counts):
    seq = toy_sequence_pair
    expected_marg, expected_fut = toy_expected_counts

    word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(
        seq.stringX,
        seq.stringY,
        seq.L_max,
        counting_method=1,
        axs=seq.axs,
        ays=seq.ays,
    )

    assert _normalize(word_lookup_marg) == expected_marg
    assert _normalize(word_lookup_fut) == expected_fut


def test_multiline_counts(multiline_sequences, multiline_expected_counts):
    stringX_list, stringY_list, axs, ays, L_max = multiline_sequences
    expected_marg, expected_fut = multiline_expected_counts

    word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(
        stringX_list,
        stringY_list,
        L_max,
        axs=axs,
        ays=ays,
        is_multiline=True,
    )

    assert _normalize(word_lookup_marg) == expected_marg
    assert _normalize(word_lookup_fut) == expected_fut


def test_single_sequence_counts_with_integers(integer_encoded_toy, toy_expected_counts):
    """Integer tokens through the new engine reproduce the string baseline.

    The integer toy is the char toy under a->0, b->1, c->2 / '0'->0, '1'->1,
    so the new engine's count tables must equal the legacy string-keyed
    baselines after translating keys through those maps.
    """
    from transcssr.counting import estimate_counts

    payload = integer_encoded_toy
    marg, fut = estimate_counts(
        [tuple(payload["X"])],
        [tuple(payload["Y"])],
        payload["L_max"],
    )

    x_map = {"a": 0, "b": 1, "c": 2}
    y_map = {"0": 0, "1": 1}

    def translate(counter):
        return Counter({
            (
                tuple(x_map[ch] for ch in x_hist),
                tuple(y_map[ch] for ch in y_hist),
            ): count
            for (x_hist, y_hist), count in counter.items()
        })

    expected_marg, expected_fut = toy_expected_counts
    assert _normalize(marg) == translate(expected_marg)
    assert _normalize(fut) == translate(expected_fut)
