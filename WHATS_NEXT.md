# WHAT'S NEXT

Tracker for the remaining work after the token-first migration
(branch `transcssr-int`, 2026-07). See `MIGRATION.md` for what is already
ported and how the new engine differs from legacy; see `docs/DESIGN.md`
and `docs/SPEC-core.md` for the contracts that govern further porting.

## Where things stand

The core inference path (counting → CSSR → model → filter/predict, plus
JSON serialization and DOT export) is fully ported to the token-first
`transcssr/` package, with exact-parity tests against the legacy engine
(144 tests passing). Everything below is the legacy-only "analysis tail"
in `transCSSR.py`, plus infrastructure follow-ups.

The porting pattern that worked, for reuse on each item below:
spec the legacy semantics first (extend `docs/SPEC-core.md`) → capture
golden outputs from legacy → implement token-first against the
`EpsilonMachine`/`EpsilonTransducer` model classes → parity-test → review.

## 1. Porting backlog (analysis tail)

Ordered by suggested priority. All new code should consume the model
classes / JSON format — never `.dot` files — and should FIX the legacy
bugs listed in SPEC §6.3 rather than replicate them (the parity-quirk
policy applies only to the already-ported partition algorithm).

- [ ] **Simulation** — `simulate_eM`, `simulate_eM_fast`, `simulate_eT`.
      Needs a stationary-distribution utility on the model classes
      (currently computed ad hoc from DOT-parsed matrices). Natural first
      port: self-contained, easy to test (simulate → re-infer → compare
      partitions).
- [ ] **Model-file compatibility** — a one-time converter from legacy
      `.dot` machine/transducer files to `transcssr-model-v1` JSON, so
      existing saved results remain usable once DOT parsing
      (`load_transition_matrix_machine/transducer`) is retired.
- [ ] **Information / complexity measures** — `compute_ict_measures`
      (statistical complexity, entropy rate, E, etc.),
      `compute_conditional_measures`, `generate_word_probs_eM`,
      `generate_wordmap`, `map_words`. Depends on stationary
      distributions + mixed-state machinery.
- [ ] **Presynchronization & mixed-state prediction** —
      `predict_presynch_eM`, `predict_presynch_eT`,
      `filter_and_pred_probs` (+ `_nonsynch`, `_breakforbidden` variants),
      `compute_mixed_transition_matrix`,
      `compute_channel_states_distribution`,
      `compute_output_transition_matrix`. The largest and subtlest chunk;
      spec it before touching it. Known legacy landmines: binary-output
      assumptions and the `scipy.stats.mode` API drift (SPEC §6.3.10).
- [ ] **Model selection & uncertainty** — `choose_L_eM`,
      `computational_mechanics_bootstrap`. Port after simulation (the
      bootstrap needs it).
- [ ] **Demo scripts** — once the tail is ported, rewrite `demo_*.py`
      against the new API (or clearly label them legacy-only until then).

## 2. Engine follow-ups (not porting)

- [ ] **Performance profiling** on large tokenized corpora (the original
      motivation was NLP-tokenizer input; vocabulary sizes of 10k+ and
      long sequences are untested at scale). Candidate optimizations:
      vectorized counting, morph-vector caching in Stage 1.
- [ ] **`memoryless=True` filtering semantics** — currently a faithful
      structural port that never synchronizes on models trained with real
      outputs (documented in `transcssr/predict.py`). Decide: implement
      proper memoryless support end-to-end, or deprecate the flag.
- [ ] **Determinization off-by-one** (SPEC §6.3.3) — replicated for
      parity. Once legacy retires, decide whether to fix it behind
      `stats_mode`-style versioning or accept the legacy behavior as
      canonical.
- [ ] **`counting_method=1`** — currently `NotImplementedError` (provably
      equivalent to method 0 on all length-≥1 keys, SPEC §1.3). Decide:
      drop the parameter or implement for completeness.
- [ ] **DOT export niceties** — optional decoded-token node labels for
      large vocabularies (carried over from the old adapter-branch plan).

## 3. Infrastructure

- [ ] **CI** — GitHub Actions: `uv sync --group dev` + full pytest
      (the parity gate runs on every push). Consider a matrix over
      Python 3.10–3.13.
- [ ] **Open the PR** — `transcssr-int` → `master` (branch is
      merge-ready; PR intentionally not opened yet).
- [ ] **Release** — tag v1.1.0 after merge; decide whether to publish to
      PyPI (name `transCSSR` is currently unclaimed there — verify).
- [ ] **Legacy retirement** — after the analysis tail is ported and has
      survived a release cycle: move `transCSSR.py` +
      `filter_data_methods.py` to `legacy_code/`, drop the `[legacy]`
      extra, and keep `tests/golden/*.json` (they are frozen snapshots
      and outlive the legacy engine that produced them). The live
      legacy-vs-new parity tests retire with it; golden parity stays.
- [ ] **Delete the `tokens` branch** once its exporter ideas (decoded
      DOT labels) are re-implemented or rejected — the TokenAdapter
      approach itself is superseded by this migration.

## Decided and closed (do not reopen without new evidence)

- New engine defaults to the **correct** chi2/G stride;
  `stats_mode="legacy"` exists solely for parity (see `MIGRATION.md`).
- Legacy **Stage-4** morph re-estimation is not ported (unused + buggy,
  SPEC §6.3.2).
- The **TokenAdapter** design (branch `tokens`) is superseded.
- DOT is **export-only**; JSON is the interchange format.
