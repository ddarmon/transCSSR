# Migrating to the token-first `transcssr` package

Version 1.1 introduces a new, token-first engine in the `transcssr`
package. It works with sequences of *arbitrary hashable tokens* (integer
ids from an NLP tokenizer, strings, tuples, ...), needs only `numpy` and
`scipy`, and exposes a small class-based API. The old flat `transCSSR.py`
module is **unchanged and still importable** --- existing code keeps
working --- and now serves as the parity oracle for the new engine.

This document maps the old API onto the new one, lists what has and has
not been ported, and documents the behavioral differences honestly.

## What has been ported

The core inference and prediction path is fully reimplemented in
`transcssr` and verified against the legacy engine (see
[Verifying parity yourself](#verifying-parity-yourself)):

-   **History counting** (legacy `estimate_predictive_distributions`,
    `counting_method=0`).
-   **The CSSR algorithm** (legacy `run_transCSSR`): Stage 1 homogenize,
    Stage 1b prune, transient removal, determinization, cleanup. Iteration
    order --- which is semantically load-bearing --- is replicated exactly.
-   **Filtering / prediction** (legacy `filter_and_predict`):
    synchronization, resynchronization, and MAP prediction.
-   **A model class** (`EpsilonTransducer` / `EpsilonMachine`) with derived
    emission probabilities, the inverse partition, and the transition map.
-   **JSON serialization** (`to_json` / `from_json`), a new versioned
    interchange format.
-   **DOT export** (`to_dot`), matching the legacy `draw_dot_singlearrows`
    visual conventions.

### Old function -> new API

| Legacy (`transCSSR.py`)                              | Token-first (`transcssr`)                                     |
| --------------------------------------------------- | ------------------------------------------------------------- |
| `estimate_predictive_distributions(...)`            | `transcssr.counting.estimate_counts(...)` (usually implicit)  |
| `run_transCSSR(...)`                                 | `infer_transducer(xs, ys, L_max, ...)`                        |
| `run_transCSSR(...)` with `stringX = '0'*len(...)`   | `infer_machine(ys, L_max, ...)`                               |
| `filter_and_predict(...)`                            | `model.filter(xs, ys)` (or `machine.filter(ys)`)              |
| `epsilon` (history -> state dict)                    | `model.epsilon` (keyed by index-tuple histories)             |
| `invepsilon` (state -> histories)                    | `model.histories_by_state`                                    |
| `morph_by_state` (predictive distributions)          | `model.morph_counts` (counts) + `model.emission_probs()`      |
| `get_transitions(...)`                               | `model.transitions()`                                         |
| `filtered_states`                                    | `FilterResult.states`                                         |
| `stringY_pred` (with `'N'` placeholder)              | `FilterResult.predictions` (with `None` placeholder)          |
| `filtered_probs` (binary-only scalar per step)       | `FilterResult.prediction_probs` (dense `(T, n_y)` array)      |
| `draw_dot_singlearrows(...)` / `.dot` side effects   | `model.to_dot(path)` (explicit export)                        |
| *(none)*                                             | `model.to_json(path)` / `EpsilonTransducer.from_json(path)`   |
| `axs` / `ays` symbol lists                           | `Alphabet` (or a token list, or inferred from data)           |
| `is_multiline=True`                                  | pass an ensemble (a list of sequences); detected by type      |

A minimal before/after:

``` python
# --- Legacy ---
from transCSSR import estimate_predictive_distributions, run_transCSSR, filter_and_predict
import itertools

axs, ays = ['0', '1'], ['0', '1']
e_symbols = list(itertools.product(axs, ays))
marg, fut = estimate_predictive_distributions(stringX, stringY, L_max, axs=axs, ays=ays)
epsilon, invepsilon, morph_by_state = run_transCSSR(
    marg, fut, L_max, axs, ays, e_symbols, 'Xt', 'Yt')
filtered_states, filtered_probs, stringY_pred = filter_and_predict(
    stringX, stringY, epsilon, invepsilon, morph_by_state, axs, ays, e_symbols, L_max)

# --- Token-first ---
from transcssr import infer_transducer

model = infer_transducer(stringX, stringY, L_max=L_max)
result = model.filter(stringX, stringY)
# result.states, result.predictions, result.prediction_probs
```

## What stays legacy-only (for now)

These remain available only through `transCSSR.py`; they will be ported
later, on demand, against the JSON model format:

-   **Mixed-state / presynchronization analysis**: `predict_presynch_*`,
    `filter_and_pred_probs*`.
-   **Simulation**: `simulate_eM*`, `simulate_eT`.
-   **Information measures**: `compute_ict_measures`,
    `compute_conditional_measures`, and the associated word maps.
-   **Bootstrap**: `computational_mechanics_bootstrap`.
-   **Model-order selection**: `choose_L_eM`.
-   **DOT *parsing*** (`load_transition_matrix_*`). The new package only
    *exports* DOT; JSON is the interchange format.

If you rely on any of these, keep using the legacy engine (install the
`[legacy]` extra) alongside the new package. Nothing in `transcssr`
imports from the legacy modules, so the two coexist freely.

## Behavioral differences (read before relying on parity)

The new engine reproduces the legacy engine's *partition* exactly on the
core path, but a few behaviors changed deliberately. Where a legacy bug
was faithful-reproducible, an opt-in flag restores it.

### (a) chi2 / G-test row stride --- new default is correct

The legacy `chisquared_test` slices each per-input-symbol morph block with
stride `len(axs)` (the number of input symbols) instead of the correct
`len(ays)` (`docs/SPEC-core.md` §2.1). The two strides agree whenever the
number of input symbols is `1` (an epsilon-machine) or equals the number
of output symbols --- so the bug is invisible for output-only processes
and for symmetric alphabets. It only bites when
`len(axs) not in {1, len(ays)}`, where it can change the inferred
partition, not merely the state numbering.

The new engine defaults to `stats_mode="correct"` (the mathematically
correct `len(ays)` stride), so users get correct statistics. To reproduce
the legacy engine exactly --- for instance to match previously published
results --- pass `stats_mode="legacy"` to `infer_transducer` /
`infer_machine`. The captured golden fixtures were produced by the legacy
engine and therefore encode the buggy stride; parity tests opt into
`stats_mode="legacy"` for that reason.

### (b) Legacy Stage-4 morph re-estimation omitted

The legacy `run_transCSSR` has an optional Stage 4 that re-estimates
morphs by re-filtering when `stringX` / `stringY` are passed. It is buggy
for more than one input symbol --- it drops the `len(ays)*x_ind` offset and
collapses every input into row 0 (`docs/SPEC-core.md` §6.3.2) --- and is
unused by the shipped demos (which never pass `stringX` / `stringY`). It
is **not ported**. The token-first engine computes morphs once, correctly.

### (c) `filtered_probs` (binary-only) -> full `prediction_probs`

The legacy `filtered_probs` is a single scalar per step, hardwired to
output index `1` (`prob_by_state[...][len(ays)*x + 1]`), which is only
meaningful for a **binary** output alphabet (`docs/SPEC-core.md` §6.3.6).
The new `FilterResult.prediction_probs` is a dense `(T, n_y)` array of the
full per-step output distribution, valid for any output alphabet size.

For a binary output, **column 1 corresponds exactly**:
`result.prediction_probs[:, 1]` reproduces the legacy `filtered_probs`
vector, including its `0.5` fallback at unsynchronized steps (every column
of an unsynchronized row is filled with that `0.5`).

### (d) `'N'` placeholder -> `None` tokens

Where the legacy engine emits the literal character `'N'` in
`stringY_pred` (an unsynchronized step with no prediction), the new engine
emits `None` in `FilterResult.predictions`. This avoids conflating a real
output token with the placeholder in token space. To recover the legacy
string for a single-character output alphabet, render `None` as `'N'` and
join.

### (e) Causal states are canonically renumbered

The legacy state labels depend on iteration order and are not stable. The
new engine relabels states `0 .. n_states-1` in a deterministic canonical
order (states sorted by their lexicographically smallest history). The
*partition* matches the legacy engine; the *numbering* generally does not.
Do not hard-code specific legacy state ids.

### (f) No `.dot` side effects during inference; igraph dropped

The legacy `run_transCSSR` writes `.dot` / results files as a side effect
of inference and depends on `python-igraph` for connected-component
analysis. The new engine does neither: inference is pure and produces a
model object, and DOT / JSON are **explicit exports** (`to_dot`,
`to_json`). `python-igraph` is no longer a dependency of the engine (SCC
analysis is hand-rolled). The token-first package depends only on `numpy`
and `scipy`.

## Verifying parity yourself

The parity gate lives in the CSSR and predict test modules. Run them in
the project environment:

``` bash
mamba run -n transcssr-int python -m pytest -q tests/test_cssr.py tests/test_predict.py
```

These reconstruct each golden fixture through the token-first pipeline
(with `stats_mode="legacy"`, since the fixtures encode the legacy stride)
and assert exact partition, morph, filtered-state, probability, and
prediction parity against the legacy engine --- including live
legacy-vs-new runs on freshly generated cases. The full suite
(`python -m pytest -q`) runs everything.
