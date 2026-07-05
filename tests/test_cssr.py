"""Parity tests for the token-first CSSR engine (``transcssr.cssr``).

Three layers of coverage:

1. **Golden parity** (the acceptance gate): reconstruct each frozen golden
   fixture's inputs, run the token-first pipeline, and assert an *exact* match
   of the canonical partition and per-state morph count vectors against the
   legacy-captured golden JSON.
2. **Live parity**: run the token-first engine and the legacy
   ``run_transCSSR`` side by side on fresh (non-golden) cases and assert the
   partitions (up to relabeling) and morphs agree.
3. **Integer-token smoke**: run the pipeline on an integer-token version of a
   process and assert the resulting partition is isomorphic to the
   single-character-string version's (index-space histories coincide).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from transcssr import Alphabet
from transcssr.counting import estimate_counts
from transcssr.cssr import build_transducer, run_cssr

# Legacy engine (parity oracle) -- tests may import both engines.
from transCSSR import estimate_predictive_distributions, run_transCSSR

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"

# Inline conftest fixtures (recorded in tests/golden/README.md), keyed by name.
_INLINE_INPUTS = {
    "toy_string": ("abacab", "010001"),
    "multiline_ensemble": (
        ["abacabac", "babacaba", "cabacabc"],
        ["01000100", "10100101", "00100010"],
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_golden(name: str) -> dict:
    with open(GOLDEN_DIR / f"{name}.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _reconstruct_inputs(meta: dict):
    """Reconstruct ``(stringX, stringY)`` (or ensembles) from a fixture's meta."""
    name = meta["name"]
    dataset = meta["dataset"]
    truncation = meta["truncation"]

    if dataset is None:
        return _INLINE_INPUTS[name]

    n = truncation["symbols"] if truncation else None
    if isinstance(dataset, str):
        # Output-only process: input is the dummy '0' stream.
        y = (REPO_ROOT / dataset).read_text().strip()
        if n is not None:
            y = y[:n]
        x = "0" * len(y)
        return x, y

    x = (REPO_ROOT / dataset[0]).read_text().strip()
    y = (REPO_ROOT / dataset[1]).read_text().strip()
    if n is not None:
        x, y = x[:n], y[:n]
    return x, y


def _encode(inputs, in_alpha: Alphabet, out_alpha: Alphabet, is_multiline: bool):
    """Encode string(s) to a list of index sequences (one-element for single)."""
    string_x, string_y = inputs
    if is_multiline:
        xs = [in_alpha.encode(s) for s in string_x]
        ys = [out_alpha.encode(s) for s in string_y]
    else:
        xs = [in_alpha.encode(string_x)]
        ys = [out_alpha.encode(string_y)]
    return xs, ys


def _canonicalize_model(model):
    """Canonical (partition, morphs): each state a sorted list of ``[x, y]``
    string histories; states sorted by their first history."""
    in_alpha = model.input_alphabet
    out_alpha = model.output_alphabet
    inv = model.histories_by_state
    entries = []
    for state in range(model.n_states):
        hists = sorted(
            [
                [
                    "".join(str(t) for t in in_alpha.decode(xh)),
                    "".join(str(t) for t in out_alpha.decode(yh)),
                ]
                for xh, yh in inv[state]
            ]
        )
        morph = [int(v) for v in model.morph_counts[state]]
        entries.append((hists[0], hists, morph))
    entries.sort(key=lambda t: t[0])
    return [e[1] for e in entries], [e[2] for e in entries]


def _canonicalize_legacy(invepsilon, morph_by_state):
    """Same canonical form, from legacy ``(invepsilon, morph_by_state)``."""
    entries = []
    for state, hists in invepsilon.items():
        if len(hists) == 0:
            continue
        sorted_hists = sorted([[xh, yh] for (xh, yh) in hists])
        morph = [int(v) for v in morph_by_state[state]]
        entries.append((sorted_hists[0], sorted_hists, morph))
    entries.sort(key=lambda t: t[0])
    return [e[1] for e in entries], [e[2] for e in entries]


def _run_new(inputs, axs, ays, L_max, alpha, test, is_multiline):
    in_alpha = Alphabet(axs)
    out_alpha = Alphabet(ays)
    xs, ys = _encode(inputs, in_alpha, out_alpha, is_multiline)
    marg, fut = estimate_counts(xs, ys, L_max)
    # Parity against the legacy engine requires its (buggy) chi2 stride;
    # stats_mode='correct' is the user-facing default (see cssr.py docstring).
    epsilon, _, morph_by_state = run_cssr(
        marg, fut, L_max, len(axs), len(ays), alpha=alpha, test=test,
        stats_mode="legacy",
    )
    return build_transducer(epsilon, morph_by_state, in_alpha, out_alpha, L_max)


# ---------------------------------------------------------------------------
# 1. Golden parity (acceptance gate)
# ---------------------------------------------------------------------------

GOLDEN_NAMES = [
    "toy_string",
    "multiline_ensemble",
    "even_process",
    "z_channel",
    "delay_channel",
]


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_golden_parity(name):
    golden = _load_golden(name)
    meta = golden["meta"]
    inputs = _reconstruct_inputs(meta)
    model = _run_new(
        inputs,
        meta["axs"],
        meta["ays"],
        meta["L_max"],
        meta["alpha"],
        meta["test_type"],
        meta["is_multiline"],
    )
    partition, morphs = _canonicalize_model(model)

    assert partition == golden["partition"], f"partition mismatch for {name}"
    assert morphs == golden["morphs"], f"morph mismatch for {name}"


def test_stats_modes_diverge_on_stride_bug_regime():
    """Guard the correct-vs-legacy stats_mode distinction.

    The ``multiline_ensemble`` fixture (n_x=3, n_y=2) triggers the legacy
    chi2 stride bug (SPEC §2.1). ``stats_mode='legacy'`` must reproduce the
    golden partition (tested above); ``stats_mode='correct'`` intentionally
    diverges here. If this test ever fails because the two modes agree,
    the mode plumbing has probably been broken or optimized away.
    """
    golden = _load_golden("multiline_ensemble")
    meta = golden["meta"]
    inputs = _reconstruct_inputs(meta)
    in_alpha = Alphabet(meta["axs"])
    out_alpha = Alphabet(meta["ays"])
    xs, ys = _encode(inputs, in_alpha, out_alpha, meta["is_multiline"])
    marg, fut = estimate_counts(xs, ys, meta["L_max"])
    epsilon, _, morph_by_state = run_cssr(
        marg, fut, meta["L_max"], len(meta["axs"]), len(meta["ays"]),
        alpha=meta["alpha"], test=meta["test_type"], stats_mode="correct",
    )
    model = build_transducer(epsilon, morph_by_state, in_alpha, out_alpha,
                             meta["L_max"])
    partition, _ = _canonicalize_model(model)
    assert partition != golden["partition"]


# ---------------------------------------------------------------------------
# 2. Live parity vs the legacy engine (non-golden cases)
# ---------------------------------------------------------------------------


def _random_binary(seed: int, n: int) -> str:
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("01"), size=n))


def _delay_process(seed: int, n: int):
    """A delay channel: ``Y_t = X_{t-1}`` (needs one bit of memory -> 2 states)."""
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


@pytest.mark.parametrize("seed,L_max,kind", LIVE_CASES)
def test_live_parity_vs_legacy(seed, L_max, kind, tmp_path):
    if kind == "indep":
        string_x = _random_binary(seed, 2000)
        string_y = _random_binary(seed + 1, 2000)
    else:
        string_x, string_y = _delay_process(seed, 2000)

    axs, ays = ["0", "1"], ["0", "1"]
    e_symbols = list(itertools.product(axs, ays))

    # Legacy engine (writes byproducts into transCSSR_results/; tolerable).
    marg_l, fut_l = estimate_predictive_distributions(
        string_x, string_y, L_max, axs=axs, ays=ays
    )
    _, invepsilon, morph_by_state = run_transCSSR(
        marg_l,
        fut_l,
        L_max,
        axs,
        ays,
        e_symbols,
        "live_x",
        "live_y",
        fname=f"live_{seed}_{L_max}_{kind}",
    )
    legacy_partition, legacy_morphs = _canonicalize_legacy(invepsilon, morph_by_state)

    # Token-first engine.
    model = _run_new((string_x, string_y), axs, ays, L_max, 0.001, "chi2", False)
    new_partition, new_morphs = _canonicalize_model(model)

    assert new_partition == legacy_partition
    assert new_morphs == legacy_morphs


# ---------------------------------------------------------------------------
# 3. Integer-token smoke test (isomorphism to the string version)
# ---------------------------------------------------------------------------


def _partition_index_space(invepsilon):
    """Partition as a set of frozensets of index-tuple histories."""
    return {
        frozenset(invepsilon[s].keys())
        for s in invepsilon
        if len(invepsilon[s]) > 0
    }


def test_integer_token_isomorphism():
    # A delay process -> two causal states, exercising real structure.
    string_x, string_y = _delay_process(seed=555, n=2000)

    # String-token pipeline.
    s_in, s_out = Alphabet(["0", "1"]), Alphabet(["0", "1"])
    marg_s, fut_s = estimate_counts(
        [s_in.encode(string_x)], [s_out.encode(string_y)], 2
    )
    eps_s, inv_s, morph_s = run_cssr(marg_s, fut_s, 2, 2, 2)

    # Integer-token pipeline: map characters to arbitrary integer tokens.
    char_to_int = {"0": 10, "1": 20}
    int_x = [char_to_int[c] for c in string_x]
    int_y = [char_to_int[c] for c in string_y]
    i_in, i_out = Alphabet([10, 20]), Alphabet([10, 20])
    marg_i, fut_i = estimate_counts([i_in.encode(int_x)], [i_out.encode(int_y)], 2)
    eps_i, inv_i, morph_i = run_cssr(marg_i, fut_i, 2, 2, 2)

    # The alphabets assign identical indices (10->0, 20->1 mirrors '0'->0,
    # '1'->1), so the index-space partitions must coincide exactly.
    assert _partition_index_space(inv_s) == _partition_index_space(inv_i)

    # And the morph multisets must match.
    morphs_s = sorted(tuple(v) for v in morph_s.values())
    morphs_i = sorted(tuple(v) for v in morph_i.values())
    assert morphs_s == morphs_i
    assert len(_partition_index_space(inv_s)) >= 2  # genuinely multi-state
