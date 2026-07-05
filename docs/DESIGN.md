# DESIGN: token-first `transcssr` package

This document is the architectural contract for the token-first rewrite of
transCSSR. The companion document `docs/SPEC-core.md` is the *behavioral*
contract: it documents the exact semantics of the legacy engine
(`transCSSR.py` at the repo root), which the new engine must reproduce.
Where this document and SPEC-core.md appear to conflict on algorithm
behavior, SPEC-core.md wins.

## Goal

Replace the character-first legacy engine with a token-first package:
sequences of **arbitrary hashable tokens** (integers from NLP tokenizers,
strings, tuples, ...) are first-class inputs. Single-character strings
become just one token type among many. The legacy module stays importable
at the repo root and serves as the parity oracle until retirement.

## Package layout

```
transcssr/
    __init__.py     # public API re-exports (see below)
    alphabet.py     # Alphabet: token <-> index mapping
    counting.py     # history counting (estimate_counts)
    stats.py        # predictive-distribution comparison tests (chi2, G)
    cssr.py         # the CSSR algorithm, operating on index tuples
    machine.py      # EpsilonTransducer / EpsilonMachine + serialization + DOT export
    predict.py      # filtering / prediction with a fitted model
tests/
    test_alphabet.py, test_machine.py, ...   # unit tests for new modules
    test_parity.py                           # legacy-vs-new parity tests
    golden/*.json                            # captured legacy outputs
```

The new package lives beside the legacy flat modules (`transCSSR.py`,
`filter_data_methods.py`). Nothing in `transcssr/` may import from the
legacy modules; only tests may import both.

## Public API

```python
from transcssr import (
    Alphabet,
    EpsilonMachine,
    EpsilonTransducer,
    FilterResult,       # return type of model.filter(...)
    infer_machine,      # epsilon-machine from an output-only process
    infer_transducer,   # epsilon-transducer from an input/output process
)

model = infer_transducer(
    xs, ys,                  # sequence or list-of-sequences (ensemble)
    L_max=3,
    alpha=0.001,
    test="chi2",             # or "G"
    input_alphabet=None,     # Alphabet | list of tokens | None (infer from data)
    output_alphabet=None,
    ensemble=None,           # None = auto-detect single-vs-ensemble by type;
                             # True = force ensemble; False = force single sequence
                             # (escape hatch for tuple-token / word-token inputs)
)

machine = infer_machine(ys, L_max=3, alpha=0.001, test="chi2",
                        alphabet=None, ensemble=None)

result = model.filter(xs, ys)        # FilterResult
result.states                        # list[int], one state per time step (-1 = unsynchronized)
result.predictions                   # list of predicted output *tokens* (None where legacy emitted 'N')
result.prediction_probs              # ndarray (T, |Y|): P(Y_t | X_t, past) per step
                                     # (legacy filtered_probs == prediction_probs[:, 1] for binary Y)

model.to_json(path); EpsilonTransducer.from_json(path)
model.to_dot(path)                   # export-only visualization
```

### Sequence input contract

Accepted sequence types at every public boundary: `str` (each character a
token), `list`/`tuple` of tokens, 1-d `numpy.ndarray`. An *ensemble* is a
`list` of such sequences (the legacy `is_multiline=True`); ensembles are
detected by type, not by flag: a `list` whose elements are themselves
sequences (str/list/tuple/ndarray) is an ensemble. To avoid ambiguity for
`list[int]` (a single integer sequence, not an ensemble): a list whose
elements are all *scalar tokens* (not str/list/tuple/ndarray) is a single
sequence. A `list[str]` is an ensemble of character sequences (matches
legacy multiline usage). Document this rule in docstrings.

Two traps follow directly from this type-based rule: (1) a single sequence of
**tuple tokens** `[(0,0), (0,1), ...]` auto-detects as an ensemble of length-2
sequences; (2) a single sequence of multi-character **word (string) tokens**
`['the', 'cat', ...]` auto-detects as a character ensemble. The
`ensemble: bool | None` parameter on `infer_transducer`/`infer_machine` is the
escape hatch: `None` (default) keeps the auto-detection above; `True` forces
ensemble handling; `False` forces single-sequence handling, so the elements are
treated as tokens even when they look like sequences. (The filtering path,
`model.filter` / `filter_and_predict`, accepts a single realization only and
rejects anything that looks like an ensemble; it does not take `ensemble=`.)

## Core representations

- **Token**: any hashable object. Never interpreted, only mapped.
- **Alphabet** (`alphabet.py`): immutable ordered token<->index bijection.
  - `Alphabet(tokens)` — explicit order (recommended for users).
  - `Alphabet.from_data(seqs)` — sorted token order when tokens are
    mutually orderable, else first-appearance order. Deterministic.
  - `alpha.index(token) -> int`, `alpha.token(i)`, `len(alpha)`,
    `alpha.encode(seq) -> tuple[int, ...]`, `alpha.decode(idxs)`,
    `alpha.tokens -> tuple`. Raises `KeyError` with a helpful message on
    unknown tokens.
- **Index sequence**: `tuple[int, ...]` internally. All algorithm code
  (counting, cssr, predict) sees only index tuples — no tokens, no strings.
- **History**: `(x_hist, y_hist)` pair of equal-length index tuples.
- **Counts** (`counting.py`): mirror the legacy `word_lookup_marg` /
  `word_lookup_fut` structures exactly (see SPEC-core.md), but keyed by
  index-tuple histories instead of strings. Implement legacy
  `counting_method=0` semantics; if SPEC-core.md shows `counting_method=1`
  is used by any core path, implement it too, otherwise raise
  `NotImplementedError` with a message pointing at the legacy engine.

## Model classes (`machine.py`)

```python
@dataclass
class EpsilonTransducer:
    input_alphabet: Alphabet
    output_alphabet: Alphabet
    n_states: int
    epsilon: dict[History, int]          # history -> state (all retained lengths <= L_max)
    morph_counts: numpy.ndarray          # shape (n_states, |X|*|Y|), legacy flat
                                         # indexing: entry len(ays)*x_ind + y_ind
    L_max: int
```

- Derived, computed on demand: `emission_probs()` (row-normalized per input
  symbol), `histories_by_state` (inverse of epsilon), transition map
  `(state, x_idx, y_idx) -> state` derived per legacy `get_transitions`
  semantics.
- State labels are contiguous ints `0..n_states-1` in a deterministic
  canonical order (sort states by their lexicographically smallest
  history). Matching legacy state *numbering* is NOT a goal; matching the
  *partition* is.
- `EpsilonMachine`: an epsilon-transducer whose input alphabet is the
  single dummy token `0` (legacy convention: `stringX = '0'*len(stringY)`).
  Provide it as a subclass or thin wrapper exposing output-only signatures.
- **Serialization**: `to_json`/`from_json` — a versioned dict
  (`{"format": "transcssr-model-v1", ...}`) with alphabets stored as token
  lists (JSON-representable tokens only; raise a clear error otherwise).
  JSON is the interchange format. DOT is export-only and never re-parsed
  by the new package.
- `to_dot(path)`: same visual conventions as legacy `draw_dot_singlearrows`
  (edge labels `y|x:prob`), with multi-character tokens rendered via
  `str(token)`.

## Algorithm modules

- `stats.py`: reimplement `chisquared_test` semantics (chi2 and G) on
  morph count vectors, using the CORRECT per-input-symbol row stride
  `len(ays)` (SPEC-core.md §2.1 documents a legacy stride bug,
  `row_ind*len(axs)`, triggered whenever `len(axs) not in {1, len(ays)}`).
  Data-dependent `df`, skip rows unseen in either morph. Pure functions.
  **Stride policy** (decided after the `multiline_ensemble` golden fixture
  — `n_x=3, n_y=2` — turned out to TRIGGER the bug): `run_cssr` takes
  `stats_mode='correct'|'legacy'`, defaulting to `'correct'` so users get
  right statistics; parity tests pass `'legacy'` (a faithful buggy-stride
  comparator kept private in `cssr.py`) because the golden fixtures encode
  legacy behavior. The public `infer_*` entry points expose the same flag,
  default `'correct'`. Divergence regime documented in MIGRATION.md.
- `cssr.py`: `run_cssr(counts, L_max, n_x, n_y, alpha, test) -> partition`
  following SPEC-core.md's staged walkthrough of legacy `run_transCSSR`
  (Stage 1 homogenize → 1b prune → transient removal → Stage 3 determinize
  → cleanup → transient removal). **Order-dependence is semantics here**
  (SPEC §6.1): the resulting partition — not just numbering — depends on
  iteration order. The implementation must replicate legacy ordering
  exactly: insertion-ordered dicts for states/histories, child generation
  `for ay in ays: for ax in axs:`, first-wins strict-`>` argmax state
  matching, mid-pass parent-morph mutation, and the determinization
  pilot-selection order. Canonical relabeling of final states is applied
  only AFTER the algorithm completes (pure renaming — safe).
  Replace igraph with hand-rolled **strongly connected components**
  (Tarjan or Kosaraju — legacy `clusters()` default is STRONG mode);
  transient removal keeps the largest recurrent SCC, smallest SCC id on
  size ties (legacy `scipy.stats.mode` behavior).
  Legacy Stage 4 (optional morph re-estimation when `stringX`/`stringY`
  are passed) is OUT of phase-1 scope: it is unused by demos and buggy
  for `len(axs) > 1` (SPEC §6.3.2); raise `NotImplementedError`.
- `predict.py`: reimplement `filter_and_predict` semantics (synchronization
  per SPEC §5.2, the legacy `'N'` placeholder becomes `None` in token
  space, `memoryless` flag) against the model class. Return the FULL
  per-step distribution `prediction_probs` of shape `(T, |Y|)` — the
  legacy `filtered_probs` is hardwired to output index 1 (binary-only,
  SPEC §6.3.6); parity compares legacy `filtered_probs` against
  `prediction_probs[:, 1]` on binary cases only.

## Parity requirements (tested in `tests/test_parity.py`)

For each golden case and each live legacy-vs-new run:

1. **Partition parity**: the set of causal states, each represented as the
   set of its histories (index tuples decoded back to legacy strings for
   comparison), must be identical up to state relabeling.
2. **Morph parity**: per matched state, morph count vectors identical
   (integer-exact).
3. **Filter parity**: filtered state sequences identical after mapping
   both sides to canonical labels; probabilities equal to within 1e-10;
   predictions identical (including tie-breaking — replicate legacy
   argmax tie behavior, which is numpy first-max).
4. The two currently-xfailed integer-token tests in
   `tests/test_transcssr_pipeline.py` must pass against the new engine.

## Dependency and style policy

- Runtime deps of `transcssr/`: **numpy and scipy only.** No igraph,
  pandas, matplotlib, tqdm, or joblib in the package. (Legacy modules keep
  their deps; that's fine.)
- Python >= 3.10. Full type hints. 4-space indents, PEP8 naming.
  Docstrings in numpydoc style (matches the legacy house style).
- No prints in library code; use warnings/exceptions. A `verbose` flag may
  gate optional progress output in `infer_*` only.
- Determinism everywhere: no wall-clock, no randomness, no reliance on
  unordered iteration.

## Out of scope for phase 1 (stays legacy, listed in MIGRATION.md)

Mixed-state/presynchronization analysis (`predict_presynch_*`,
`filter_and_pred_probs*`), simulation (`simulate_eM*`, `simulate_eT`),
information measures (`compute_ict_measures`, `compute_conditional_measures`,
word maps), bootstrap (`computational_mechanics_bootstrap`), `choose_L_eM`,
and DOT *parsing* (`load_transition_matrix_*`). These port later, on
demand, against the JSON model format.
