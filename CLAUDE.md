# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

transCSSR implements the Causal State Splitting Reconstruction (CSSR) algorithm for inferring epsilon-machines and epsilon-transducers from discrete-valued, discrete-time input/output systems. As of version 1.1 the codebase has two engines:

- **`transcssr/` (primary, token-first).** The current engine. Sequences of arbitrary hashable tokens (integer tokenizer ids, strings, tuples, 1-d numpy arrays) are first-class inputs. Depends only on `numpy` and `scipy`. This is what new code should use.
- **`transCSSR.py` + `filter_data_methods.py` (legacy).** The original character-first flat modules, kept importable and **unchanged**. They serve as the **parity oracle** for the new engine and still back the demo scripts. Needs the extra deps (`python-igraph`, `pandas`, `matplotlib`, `joblib`, `tqdm`).

Nothing in `transcssr/` may import from the legacy modules; only tests import both.

## Contracts (read these before changing behavior)

- **`docs/DESIGN.md`** --- architectural contract for the token-first package: package layout, public API, model classes, scope policy (what is in / out of phase 1).
- **`docs/SPEC-core.md`** --- behavioral contract: the exact semantics of the legacy engine that the new engine must reproduce. Where DESIGN and SPEC conflict on algorithm behavior, SPEC wins. §6.3 is the legacy bug list; §2.1 is the chi2 stride bug.

## The token-first package (`transcssr/`)

Modules and responsibilities:

- `alphabet.py` --- `Alphabet`: immutable token <-> index bijection.
- `counting.py` --- `estimate_counts`: history counting (legacy `counting_method=0`).
- `countstore.py` --- `CountStore`: the two-query count-access abstraction `run_cssr` reads through (`children` / `extensions`, per-history, on-demand). Backends: `CounterCountStore` (materialized `marg`/`fut` Counters; default and parity oracle) and `SuffixArrayCountStore` (on-demand suffix-array counts over the product-coded joint stream with an edge-effect tail-correction table; **bit-exact same models**, O(T) memory instead of per-window Counter keys — for large-T runs). Selected via `count_backend="counter"|"suffix"` on `infer_*`.
- `_suffix.py` --- private `SuffixIndex`: pure-numpy suffix array (sentinel-separated realizations, lazy reversed-corpus twin) with count / range / left- and right-extension / position queries; `pydivsufsort` is an optional, never-declared construction fast path.
- `stats.py` --- `compare_morphs`: chi2 / G morph comparison, correct `len(ays)` stride.
- `cssr.py` --- `run_cssr` / `build_transducer`: the state-splitting + determinization algorithm on index tuples. Order-dependence is semantics; legacy iteration order is replicated exactly (see the module docstring and its `LEGACY QUIRK` comments).
- `machine.py` --- `EpsilonTransducer` / `EpsilonMachine`: model classes, JSON serialization, DOT export.
- `predict.py` --- `filter_and_predict` / `FilterResult`: filtering and MAP prediction (the legacy "hard" filter: exact-window synchronization, 0.5 placeholder when unsynchronized).
- `msp.py` --- mixed-state-presentation prediction: `stationary_distribution`, `mixed_transition_matrix`, `msp_filter` / `msp_filter_machine` (Bayesian belief filtering from the stationary prior; always-defined predictive probabilities; restarts on forbidden transitions), and `log_loss` (held-out scorer in nats and bits; zero-probability events are counted in `n_zero_prob`, not smoothed). Port of legacy `filter_and_pred_probs` / `compute_mixed_transition_matrix` / `compute_channel_states_distribution`; parity is float-tolerance (`np.allclose`), not bit-exact, and one documented deviation adds a termination guard where legacy would loop forever on a globally-impossible symbol.
- `api.py` --- `infer_transducer` / `infer_machine`: public inference entry points (normalize -> encode -> count -> CSSR -> build model).
- `cli.py` --- the `transcssr` command (typer app): subcommands `infer` / `filter` / `show` / `export-dot`, plus `--version`. Pure argument-parsing + I/O glue; all inference logic stays in `api`/`machine`. **`typer` is imported ONLY here** (the sole non-numpy/scipy runtime dep); engine modules must never import `cli` or `typer`.

Public API:

```python
from transcssr import Alphabet, EpsilonMachine, EpsilonTransducer, infer_machine, infer_transducer
model = infer_transducer(xs, ys, L_max=3, alpha=0.001, test="chi2")
machine = infer_machine(ys, L_max=3)
result = model.filter(xs, ys)   # .states, .predictions, .prediction_probs (T, n_y)
model.to_json(path); EpsilonTransducer.from_json(path); model.to_dot(path)

# MSP (belief-state) prediction and held-out scoring:
from transcssr import msp_filter_machine, log_loss
probs = msp_filter_machine(machine, ys).pred_probs   # (T, n_y), always defined
score = log_loss(machine, ys)   # .per_symbol_bits, .total_nats, .n_zero_prob
```

## stats_mode policy

Morph comparison has two modes, both exposed on `infer_*` and `run_cssr`:

- **`stats_mode="correct"` (default)** --- the mathematically correct per-input-symbol stride `len(ays)`. Users get correct statistics.
- **`stats_mode="legacy"`** --- reproduces the legacy `chisquared_test` stride bug (`row_ind*len(axs)`, SPEC §2.1) for exact partition parity with the legacy engine.

The two modes agree whenever `n_x` is `1` or equals `n_y`; they can differ (different partition, not just numbering) when `n_x` is neither. The golden fixtures were captured from the legacy engine and therefore encode the buggy stride, so **parity tests pass `stats_mode="legacy"`**.

## Testing

Canonical command (uv-native): run the full suite from the repo root.

```bash
uv sync --group dev            # first time / after dependency changes
uv run python -m pytest -q     # 351 passed, 1 skipped
```

The `dev` dependency group (PEP 735 `[dependency-groups]` in
`pyproject.toml`) installs `pytest` plus the `[legacy]` extra, so the
legacy-parity tests run. uv provisions the interpreter pinned in
`.python-version` (3.13) and resolves from the committed `uv.lock`.

Legacy fallback (the mamba env still works): `mamba run -n transcssr-int
python -m pytest -q`.

- Unit tests: `tests/test_alphabet.py`, `test_counting.py`, `test_stats.py`, `test_machine.py`, etc.
- Parity gate: `tests/test_cssr.py` and `tests/test_predict.py` compare the new engine against the legacy oracle (partition, morphs, filtered states, probabilities, predictions).
- Executable docs: `tests/test_readme_examples.py` mirrors the two README quickstarts verbatim --- keep them in sync when editing either.
- CLI tests: `tests/test_cli.py` drives the `transcssr` command in-process via `typer.testing.CliRunner`.
- Golden fixtures: `tests/golden/*.json` are captured legacy outputs; regenerate with `scripts/capture_golden.py` (do not hand-edit).

## Packaging

`pyproject.toml` (setuptools backend). Distribution name `transCSSR`,
version 1.1.x.
- `packages = ["transcssr"]` ONLY. The legacy flat modules are
  deliberately NOT shipped: an installed `transCSSR.py` and the
  `transcssr/` package collide on case-insensitive filesystems
  (macOS/Windows), making `import transCSSR` nondeterministic. The
  legacy engine is repo-checkout-only; anything importing it must run
  from the repo root (tests do) or put the root first on `sys.path`
  (the example scripts do, via a bootstrap block).
- Engine deps: `numpy`, `scipy`, plus `typer` for the CLI (imported only by `cli.py`). Legacy deps are the optional `[legacy]` extra (`python-igraph`, `pandas`, `matplotlib`, `joblib`, `tqdm`).
- Console script: `[project.scripts]` `transcssr = "transcssr.cli:app"`.
- Dev tooling: the PEP 735 `[dependency-groups]` `dev` group = `pytest` + a self-reference to the `transCSSR[legacy]` extra (so the full/parity suite runs without duplicating the legacy dep list).
- Python `>=3.11`; `.python-version` pins `3.13` for the uv workflow.
- uv-native: `uv sync` / `uv sync --group dev`, resolved from the committed `uv.lock` (tracked; `.venv/` is gitignored).
- Editable install (fallback): `pip install -e .` (or `pip install -e ".[legacy]"` for demos).

## Directory structure

- `transcssr/` --- token-first package (primary engine).
- `transCSSR.py`, `filter_data_methods.py` --- legacy flat modules (parity oracle + demo backend; repo-only, not installed).
- `docs/` --- `DESIGN.md`, `SPEC-core.md` (the contracts).
- `tests/` --- unit + parity tests; `tests/golden/` captured fixtures.
- `scripts/capture_golden.py` --- regenerates golden fixtures from the legacy engine.
- `data/` --- datasets for demos and quickstarts.
- `examples/` --- legacy-engine demos + notebooks (run from repo root); `examples/experimental/` research scripts; `examples/simulation*/` data-generation code for `data/`.
- `transCSSR_results/` --- generated `.dot` / analysis outputs, hardcoded output dir of the legacy engine (gitignored; some files tracked historically). Retires with the legacy engine.
- `legacy_code/` --- Python 2 compatible legacy implementation.

## Conventions

- Token-first package: numpy/scipy only, Python >= 3.11, full type hints, numpydoc docstrings, no prints in library code (warnings/exceptions instead), determinism everywhere (no wall-clock, no randomness, no unordered iteration).
- Legacy modules: do not modify to change behavior; they are the parity oracle.
