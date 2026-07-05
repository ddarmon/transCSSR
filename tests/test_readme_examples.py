"""Executable-documentation tests: the two README quickstarts.

Each test mirrors a fenced code block in ``README.md`` verbatim (modulo the
output/scratch file paths, which are redirected into pytest's ``tmp_path`` to
avoid polluting the working tree). Keep these tests and the README in sync: if
you edit a quickstart in one place, edit it in the other. The asserted values
are exactly the inline ``# ->`` comments shown in the README.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_readme_quickstart_1_even_process(tmp_path):
    """Mirrors README "Quickstart 1: an epsilon-machine from the even process"."""
    from transcssr import infer_machine

    # Load the output-only sequence (each character is one token).
    ys = open(REPO_ROOT / "data" / "even.dat").readline().strip()

    # Infer the epsilon-machine.
    machine = infer_machine(ys, L_max=3, alpha=0.001)
    assert machine.n_states == 2

    # Filter the causal-state path and the per-step output distribution.
    result = machine.filter(ys)
    assert result.prediction_probs.shape == (len(ys), 2)
    # -1 while unsynchronized, then real (0/1) state labels.
    assert all(s in (-1, 0, 1) for s in result.states)
    assert set(result.states) == {-1, 0, 1}

    # Export a Graphviz .dot for visualization (README writes "even.dot").
    machine.to_dot(tmp_path / "even.dot")
    assert (tmp_path / "even.dot").exists()


def test_readme_quickstart_2_integer_tokens(tmp_path):
    """Mirrors README "Quickstart 2: an epsilon-transducer over integer tokens"."""
    from transcssr import EpsilonTransducer, infer_transducer

    # A toy tokenizer: words <-> integer ids, like an NLP vocabulary.
    vocab = {"the": 0, "cat": 1, "sat": 2, "dog": 3, "barked": 4}
    inv_vocab = {i: w for w, i in vocab.items()}

    # A tiny deterministic "grammar": two sentences, repeated.
    sentence = ["the", "cat", "sat", "the", "dog", "barked"]
    corpus = [vocab[w] for w in sentence] * 200      # a list[int] token stream

    # Next-token prediction: input is the current token, output is the next.
    xs = corpus[:-1]
    ys = corpus[1:]

    model = infer_transducer(
        xs, ys, L_max=2, alpha=0.001,
        input_alphabet=list(vocab.values()),
        output_alphabet=list(vocab.values()),
    )
    assert model.n_states == 3

    # Predict the next token at each step; None marks unsynchronized steps.
    result = model.filter(xs, ys)
    predicted_words = [inv_vocab[t] if t is not None else None
                       for t in result.predictions]
    assert predicted_words[1:7] == ["sat", "the", "dog", "barked", "the", "cat"]

    # JSON round-trip: serialize and restore the model (README writes CWD).
    path = tmp_path / "transducer.json"
    model.to_json(path)
    restored = EpsilonTransducer.from_json(path)
    assert restored.n_states == 3

    # The restored model reproduces the original filter output exactly.
    restored_result = restored.filter(xs, ys)
    assert restored_result.states == result.states
    assert np.allclose(
        restored_result.prediction_probs,
        result.prediction_probs,
        equal_nan=True,
    )
