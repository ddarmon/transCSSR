# WHAT'S NEXT

Tracker for the remaining work after the token-first migration
(branch `feature/token-first-engine`, 2026-07; PR #4 → `main`). See
`MIGRATION.md` for what is already ported and how the new engine differs
from legacy; see `docs/DESIGN.md` and `docs/SPEC-core.md` for the
contracts that govern further porting.

## Where things stand

The core inference path (counting → CSSR → model → filter/predict, plus
JSON serialization and DOT export) is fully ported to the token-first
`transcssr/` package, with exact-parity tests against the legacy engine,
plus mixed-state-presentation prediction/scoring (`transcssr/msp.py`) and
large-alphabet performance work (282 tests passing). Everything below is
the remaining legacy-only "analysis tail" in `transCSSR.py`, plus
infrastructure follow-ups.

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
      The stationary-distribution utility it needs now exists
      (`transcssr.msp.stationary_distribution`). Natural first port:
      self-contained, easy to test (simulate → re-infer → compare
      partitions).
- [ ] **Model-file compatibility** — a one-time converter from legacy
      `.dot` machine/transducer files to `transcssr-model-v1` JSON, so
      existing saved results remain usable once DOT parsing
      (`load_transition_matrix_machine/transducer`) is retired.
- [ ] **Information / complexity measures** — `compute_ict_measures`
      (statistical complexity, entropy rate, E, etc.),
      `compute_conditional_measures`, `generate_word_probs_eM`,
      `generate_wordmap`, `map_words`. The stationary-distribution and
      mixed-state machinery it depends on is now in `transcssr/msp.py`.
- [x] **Presynchronization & mixed-state prediction** — DONE 2026-07-05
      as `transcssr/msp.py`: `stationary_distribution`,
      `mixed_transition_matrix`, `msp_filter` / `msp_filter_machine`
      (ports `filter_and_pred_probs` and subsumes `predict_presynch_eM` /
      `predict_presynch_eT`), plus a `log_loss` held-out scorer.
      Float-tolerance parity vs the legacy oracle (~1e-16); documented
      termination-guard deviation for globally-impossible symbols.
      NOT ported (decide whether they're needed): the `_nonsynch` and
      `_breakforbidden` variants, `compute_output_transition_matrix`.
- [ ] **Model selection & uncertainty** — `choose_L_eM`,
      `computational_mechanics_bootstrap`. Port after simulation (the
      bootstrap needs it).
- [ ] **Demo scripts** — once the tail is ported, rewrite `demo_*.py`
      against the new API (or clearly label them legacy-only until then).

## 2. Engine follow-ups (not porting)

- [x] **Vectorized counting + Stage-1 hot path** — DONE 2026-07-05:
      adaptive vectorized `estimate_counts` (exact Counter parity),
      occurrence/extension indexes + numpy morphs in `run_cssr`,
      vectorized `compare_morphs` (7.9–10.8× primitives at n_y=1024);
      partitions bit-identical in both stats modes.
- [x] **Scale wall (a): Counter materialization memory** — DONE
      2026-07-05 on `feature/optimizations`: `CountStore` abstraction
      (`transcssr/countstore.py`) + suffix-array backend
      (`SuffixArrayCountStore` over `transcssr/_suffix.py`,
      infini-gram-style, arXiv:2401.17377), selected via
      `count_backend="suffix"` on `infer_*`. Better than planned: a
      precomputed edge-effect tail-correction table makes it
      **bit-exact** with the Counter backend (not a documented
      deviation), verified store-level and end-to-end in
      `tests/test_suffix_countstore.py`. O(T) integers instead of
      per-window Counter keys (benchmark: peak RSS 232 vs 404 MB at
      T=1e6, n_y=8; count structure 0.8 vs 36.7 MB at n_y=64); trades a
      modest per-query cost, so `counter` stays the default. Also
      unlocks adaptive history depth later (no L_max in the memory
      equation).
- [ ] **Scale wall (b): the O(#states) existing-state matching loop in
      Stage 1** — order-dependent semantics, so any approximate matching
      must be an opt-in, clearly-non-parity mode. E1 pilot evidence
      (2026-07-05, now in the transcssr-experiments repo,
      github.com/ddarmon/transcssr-experiments: 1M chars of text8, L=3):
      real English yields **3,684 states** and inference is dominated by
      this loop (~212 s), so the wall binds on natural language too —
      this is now the top engineering target for scaling E1/E2,
      alongside a held-out backoff policy (1.47% zero-probability events
      on 200k held-out chars; `log_loss` reports, deliberately does not
      smooth).
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
      Python 3.11–3.13.
- [x] **Open the PR** — DONE 2026-07-05: branch renamed
      `transcssr-int` → `feature/token-first-engine`, default branch
      renamed `master` → `main`, PR #4 opened
      (https://github.com/ddarmon/transCSSR/pull/4).
- [ ] **Release** — tag v1.1.0 after PR #4 merges; decide whether to
      publish to PyPI (name `transCSSR` verified unclaimed as of
      2026-07-05: pypi.org returns 404 for the project).
- [ ] **Legacy retirement** — after the analysis tail is ported and has
      survived a release cycle: move `transCSSR.py` +
      `filter_data_methods.py` to `legacy_code/`, drop the `[legacy]`
      extra, and keep `tests/golden/*.json` (they are frozen snapshots
      and outlive the legacy engine that produced them). The live
      legacy-vs-new parity tests retire with it; golden parity stays.
- [ ] **Delete the `tokens` branch** once its exporter ideas (decoded
      DOT labels) are re-implemented or rejected — the TokenAdapter
      approach itself is superseded by this migration.

## 4. Research thread: CSSR on natural language (the E-experiments)

The research thread — E1 scale-up, the mini-E2 vocab ladder, and the
experiment log — now lives in the private repo
github.com/ddarmon/transcssr-experiments. This section keeps only the
items that are package (not experiment) work.

Decided 2026-07-05: **skip the standalone E0 feasibility study** and go
straight to scaling the E1 pilot (details in transcssr-experiments).

- [ ] **Simulation port** (`simulate_eM` etc., see §1) — no longer
      gated on an E0 study, but still wanted for recovery sanity checks
      and the bootstrap; cheap, self-contained.
- Deferred until the scale-up data says otherwise: approximate state
  matching (opt-in non-parity mode) and adaptive-depth CSSR (unlocked
  by the suffix backend; the statistically-honest replacement for
  fixed L_max).

## Decided and closed (do not reopen without new evidence)

- New engine defaults to the **correct** chi2/G stride;
  `stats_mode="legacy"` exists solely for parity (see `MIGRATION.md`).
- Legacy **Stage-4** morph re-estimation is not ported (unused + buggy,
  SPEC §6.3.2).
- The **TokenAdapter** design (branch `tokens`) is superseded.
- DOT is **export-only**; JSON is the interchange format.
