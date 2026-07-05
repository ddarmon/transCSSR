"""Capture legacy transCSSR engine outputs as golden-parity JSON fixtures.

For each configured case this script runs the legacy pipeline
(estimate_predictive_distributions -> run_transCSSR -> filter_and_predict)
and dumps a canonicalized snapshot to tests/golden/<case>.json.

State labels produced by run_transCSSR are arbitrary, so we canonicalize:
each causal state is represented by the sorted list of its histories
(pairs of x_hist / y_hist strings); states are then ordered by their
first (smallest) history. All downstream references (morphs, filtered
state sequence) are remapped to this canonical, deterministic order.

Run with:
    mamba run -n transcssr-int python scripts/capture_golden.py

Deterministic: no randomness, fixed truncations.
"""

import itertools
import json
import os

from transCSSR import (
    estimate_predictive_distributions,
    filter_and_predict,
    run_transCSSR,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "golden")


def load_series(rel_path, truncate=None):
    """Read the first line of a .dat file, optionally truncating symbols."""
    path = os.path.join(REPO_ROOT, rel_path)
    with open(path) as fh:
        series = fh.readline().strip()
    if truncate is not None:
        series = series[:truncate]
    return series


def canonicalize(invepsilon):
    """Return (canonical_states, label_to_index).

    canonical_states: list of states in canonical order, where each state is
        the sorted list of [x_hist, y_hist] history pairs.
    label_to_index: maps the engine's original state label to the canonical
        integer index.
    """
    per_label = {}
    for label, histories in invepsilon.items():
        hist_pairs = sorted([x_hist, y_hist] for (x_hist, y_hist) in histories.keys())
        per_label[label] = hist_pairs

    # Order states by their first (lexicographically smallest) history.
    ordered_labels = sorted(per_label, key=lambda lbl: per_label[lbl][0])

    canonical_states = [per_label[lbl] for lbl in ordered_labels]
    label_to_index = {lbl: idx for idx, lbl in enumerate(ordered_labels)}
    return canonical_states, label_to_index


def to_int_counts(counts):
    """Convert a morph counts container (possibly numpy) to plain ints."""
    return [int(c) for c in counts]


def run_case(case):
    name = case["name"]
    axs = case["axs"]
    ays = case["ays"]
    L_max = case["L_max"]
    alpha = case.get("alpha", 0.001)
    test_type = case.get("test_type", "chi2")
    e_symbols = list(itertools.product(axs, ays))

    is_multiline = case.get("is_multiline", False)
    stringX = case["stringX"]
    stringY = case["stringY"]

    word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(
        stringX,
        stringY,
        L_max,
        axs=axs,
        ays=ays,
        is_multiline=is_multiline,
    )

    epsilon, invepsilon, morph_by_state = run_transCSSR(
        word_lookup_marg,
        word_lookup_fut,
        L_max,
        axs,
        ays,
        e_symbols,
        case["Xt_name"],
        case["Yt_name"],
        alpha=alpha,
        test_type=test_type,
    )

    canonical_states, label_to_index = canonicalize(invepsilon)

    # Morphs aligned to canonical state order.
    ordered_labels = sorted(label_to_index, key=lambda lbl: label_to_index[lbl])
    morphs = [to_int_counts(morph_by_state[lbl]) for lbl in ordered_labels]

    result = {
        "meta": {
            "name": name,
            "dataset": case.get("dataset"),
            "truncation": case.get("truncation"),
            "is_multiline": is_multiline,
            "L_max": L_max,
            "alpha": alpha,
            "test_type": test_type,
            "axs": list(axs),
            "ays": list(ays),
            "e_symbols": [list(pair) for pair in e_symbols],
        },
        "partition": canonical_states,
        "morphs": morphs,
    }

    # filter_and_predict operates on single strings only.
    if is_multiline:
        result["filter"] = None
        result["meta"]["filter_note"] = (
            "filter_and_predict is not defined for multiline/ensemble input; skipped."
        )
    else:
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
        # Map engine state labels to canonical indices; -1 (no state) stays -1.
        mapped_states = [
            -1 if s == -1 else label_to_index[s] for s in filtered_states
        ]
        result["filter"] = {
            "filtered_states": mapped_states,
            "filtered_probs": [round(float(p), 12) for p in filtered_probs],
            "stringY_pred": stringY_pred,
        }

    return result


def build_cases():
    cases = []

    # (c) Toy string fixture from conftest.py.
    cases.append({
        "name": "toy_string",
        "dataset": None,
        "truncation": None,
        "stringX": "abacab",
        "stringY": "010001",
        "axs": ["a", "b", "c"],
        "ays": ["0", "1"],
        "L_max": 2,
        "Xt_name": "golden_toy_input",
        "Yt_name": "golden_toy_output",
    })

    # (d) Multiline / ensemble fixture from conftest.py.
    cases.append({
        "name": "multiline_ensemble",
        "dataset": None,
        "truncation": None,
        "is_multiline": True,
        "stringX": ["abacabac", "babacaba", "cabacabc"],
        "stringY": ["01000100", "10100101", "00100010"],
        "axs": ["a", "b", "c"],
        "ays": ["0", "1"],
        "L_max": 2,
        "Xt_name": "golden_multiline_input",
        "Yt_name": "golden_multiline_output",
    })

    # (a) Even process (output-only). Truncate to first 20000 symbols.
    even_trunc = 20000
    even_Y = load_series("data/even.dat", truncate=even_trunc)
    cases.append({
        "name": "even_process",
        "dataset": "data/even.dat",
        "truncation": {"symbols": even_trunc},
        "stringX": "0" * len(even_Y),
        "stringY": even_Y,
        "axs": ["0"],
        "ays": ["0", "1"],
        "L_max": 3,
        "Xt_name": "golden_even_zeros",
        "Yt_name": "golden_even",
    })

    # (b) Genuine input-output transducer #1: the z-channel (Barnett-Crutchfield).
    z_trunc = 10000
    zX = load_series("data/Xt_z-channel.dat", truncate=z_trunc)
    zY = load_series("data/Yt_z-channel.dat", truncate=z_trunc)
    cases.append({
        "name": "z_channel",
        "dataset": ["data/Xt_z-channel.dat", "data/Yt_z-channel.dat"],
        "truncation": {"symbols": z_trunc},
        "stringX": zX,
        "stringY": zY,
        "axs": ["0", "1"],
        "ays": ["0", "1"],
        "L_max": 3,
        "Xt_name": "golden_Xt_z_channel",
        "Yt_name": "golden_Yt_z_channel",
    })

    # (b) Genuine input-output transducer #2: the delay-channel (Barnett-Crutchfield).
    d_trunc = 10000
    dX = load_series("data/Xt_delay-channel.dat", truncate=d_trunc)
    dY = load_series("data/Yt_delay-channel.dat", truncate=d_trunc)
    cases.append({
        "name": "delay_channel",
        "dataset": ["data/Xt_delay-channel.dat", "data/Yt_delay-channel.dat"],
        "truncation": {"symbols": d_trunc},
        "stringX": dX,
        "stringY": dY,
        "axs": ["0", "1"],
        "ays": ["0", "1"],
        "L_max": 3,
        "Xt_name": "golden_Xt_delay_channel",
        "Yt_name": "golden_Yt_delay_channel",
    })

    return cases


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    summary = []
    for case in build_cases():
        result = run_case(case)
        out_path = os.path.join(GOLDEN_DIR, "{}.json".format(case["name"]))
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, sort_keys=False)
            fh.write("\n")
        n_states = len(result["partition"])
        summary.append((case["name"], n_states, out_path))
        print("[{}] {} states -> {}".format(case["name"], n_states, out_path))

    print("\nCaptured {} golden cases.".format(len(summary)))


if __name__ == "__main__":
    main()
