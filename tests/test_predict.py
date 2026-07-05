"""Parity tests for the token-first filtering/prediction path (``predict.py``).

Two layers:

1. **Golden filter parity** (acceptance gate): for every golden fixture that
   carries a ``filter`` section, rebuild the model through the token-first
   pipeline (``stats_mode='legacy'`` -- the golden fixtures encode the legacy
   chi2 stride, see ``transcssr.cssr``), run ``model.filter`` on the training
   series, and assert an EXACT match of the filtered state sequence, the
   binary ``prediction_probs[:, 1]`` vector (== legacy ``filtered_probs``),
   and the predicted output string.
2. **Live filter parity**: run the legacy ``filter_and_predict`` and the new
   ``model.filter`` side by side on fresh seeded cases and assert states match
   (after mapping legacy labels to canonical), probabilities to 1e-10, and the
   predicted strings are identical.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from transcssr import Alphabet, infer_transducer
from transcssr.counting import estimate_counts
from transcssr.cssr import build_transducer, run_cssr

# Legacy engine (parity oracle) -- tests may import both engines.
from transCSSR import (
    estimate_predictive_distributions,
    filter_and_predict as legacy_filter_and_predict,
    run_transCSSR,
)

# Shared live-parity helpers live in conftest.py (imported by both suites).
from conftest import LIVE_CASES, _delay_process, _random_binary

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"

_INLINE_INPUTS = {
    "toy_string": ("abacab", "010001"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_golden(name: str) -> dict:
    with open(GOLDEN_DIR / f"{name}.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _reconstruct_single(meta: dict):
    """Reconstruct ``(stringX, stringY)`` for a single-string fixture."""
    name = meta["name"]
    dataset = meta["dataset"]
    truncation = meta["truncation"]
    if dataset is None:
        return _INLINE_INPUTS[name]
    n = truncation["symbols"] if truncation else None
    if isinstance(dataset, str):
        y = (REPO_ROOT / dataset).read_text().strip()
        if n is not None:
            y = y[:n]
        return "0" * len(y), y
    x = (REPO_ROOT / dataset[0]).read_text().strip()
    y = (REPO_ROOT / dataset[1]).read_text().strip()
    if n is not None:
        x, y = x[:n], y[:n]
    return x, y


def _build_legacy_stride_model(x, y, axs, ays, L_max, alpha, test):
    """Build a canonical model via the token-first pipeline, legacy stride."""
    in_alpha = Alphabet(axs)
    out_alpha = Alphabet(ays)
    marg, fut = estimate_counts([in_alpha.encode(x)], [out_alpha.encode(y)], L_max)
    epsilon, _, morph_by_state = run_cssr(
        marg, fut, L_max, len(axs), len(ays), alpha=alpha, test=test,
        stats_mode="legacy",
    )
    return build_transducer(epsilon, morph_by_state, in_alpha, out_alpha, L_max)


def _render_pred(predictions) -> str:
    """Render predicted tokens to a legacy-style string (None -> 'N')."""
    return "".join("N" if p is None else str(p) for p in predictions)


def _legacy_canonical_map(invepsilon) -> dict[int, int]:
    """Map legacy engine state labels to canonical indices (string histories)."""
    per_label = {}
    for label, hists in invepsilon.items():
        if len(hists) == 0:
            continue
        per_label[label] = sorted([xh, yh] for (xh, yh) in hists.keys())
    ordered = sorted(per_label, key=lambda lbl: per_label[lbl][0])
    return {lbl: idx for idx, lbl in enumerate(ordered)}


# ---------------------------------------------------------------------------
# 1. Golden filter parity
# ---------------------------------------------------------------------------

GOLDEN_FILTER_NAMES = ["toy_string", "even_process", "z_channel", "delay_channel"]


@pytest.mark.parametrize("name", GOLDEN_FILTER_NAMES)
def test_golden_filter_parity(name):
    golden = _load_golden(name)
    meta = golden["meta"]
    filt = golden["filter"]
    assert filt is not None, f"{name} has no filter section"

    x, y = _reconstruct_single(meta)
    model = _build_legacy_stride_model(
        x, y, meta["axs"], meta["ays"], meta["L_max"], meta["alpha"],
        meta["test_type"],
    )
    result = model.filter(x, y)

    # States: canonical model labels should already align with golden canonical.
    assert result.states == filt["filtered_states"], f"state mismatch for {name}"

    # prediction_probs[:, 1] == legacy filtered_probs (binary output).
    col1 = [round(float(v), 12) for v in result.prediction_probs[:, 1]]
    assert col1 == filt["filtered_probs"], f"prob mismatch for {name}"

    # Predicted tokens, None -> 'N', joined == legacy stringY_pred.
    assert _render_pred(result.predictions) == filt["stringY_pred"], (
        f"prediction mismatch for {name}"
    )

    # Shape sanity: (T, n_y) with T == len(x).
    assert result.prediction_probs.shape == (len(x), model.n_y)
    assert len(result.predictions) == len(x)


# ---------------------------------------------------------------------------
# 2. Live filter parity vs the legacy engine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed,L_max,kind", LIVE_CASES)
def test_live_filter_parity_vs_legacy(seed, L_max, kind):
    if kind == "indep":
        string_x = _random_binary(seed, 2000)
        string_y = _random_binary(seed + 1, 2000)
    else:
        string_x, string_y = _delay_process(seed, 2000)

    axs, ays = ["0", "1"], ["0", "1"]
    e_symbols = list(itertools.product(axs, ays))

    marg_l, fut_l = estimate_predictive_distributions(
        string_x, string_y, L_max, axs=axs, ays=ays
    )
    epsilon, invepsilon, morph_by_state = run_transCSSR(
        marg_l, fut_l, L_max, axs, ays, e_symbols, "live_x", "live_y",
        fname=f"live_filter_{seed}_{L_max}_{kind}",
    )
    legacy_states, legacy_probs, legacy_pred = legacy_filter_and_predict(
        string_x, string_y, epsilon, invepsilon, morph_by_state, axs, ays,
        e_symbols, L_max,
    )
    canon = _legacy_canonical_map(invepsilon)
    mapped_legacy = [-1 if s == -1 else canon[s] for s in legacy_states]

    # Token-first engine (legacy stride for exact parity).
    model = infer_transducer(
        string_x, string_y, L_max=L_max, stats_mode="legacy"
    )
    result = model.filter(string_x, string_y)

    assert result.states == mapped_legacy
    assert np.allclose(
        np.asarray(legacy_probs, dtype=float),
        result.prediction_probs[:, 1],
        atol=1e-10,
    )
    assert "".join(legacy_pred) == _render_pred(result.predictions)
