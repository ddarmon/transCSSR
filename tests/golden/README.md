# Golden-output parity fixtures

These JSON files are frozen snapshots of the **legacy** `transCSSR.py` engine's
output on a fixed set of datasets. They exist so a token-first rewrite can be
tested for **exact behavioral parity** against the reference implementation.

Regenerate with:

```bash
mamba run -n transcssr-int python scripts/capture_golden.py
```

The generator is deterministic (no randomness, fixed truncations), so a clean
re-run must reproduce these files byte-for-byte.

## Cases

| File | Source | Kind | States |
|------|--------|------|--------|
| `toy_string.json` | conftest toy fixture (`abacab` / `010001`) | single string I/O | 1 |
| `multiline_ensemble.json` | conftest multiline fixture (3 seqs) | ensemble / multiline | 1 |
| `even_process.json` | `data/even.dat` (first 20000 symbols) | output-only | 2 |
| `z_channel.json` | `data/Xt_z-channel.dat` + `data/Yt_z-channel.dat` (10000) | input/output | 1 (memoryless) |
| `delay_channel.json` | `data/Xt_delay-channel.dat` + `data/Yt_delay-channel.dat` (10000) | input/output | 2 |

The z-channel collapses to a single memoryless state (output is 0 when the
input is 0, and roughly 50/50 when the input is 1); the delay-channel needs one
bit of memory, hence two states. Both are the expected, correct behavior.

## JSON schema

Each file has three top-level keys.

### `meta`
Everything needed to reproduce the run:
- `name` — case identifier.
- `dataset` — source `.dat` path(s), or `null` for inline conftest fixtures.
- `truncation` — `{"symbols": N}` when the series was truncated to its first
  `N` symbols, else `null`.
- `is_multiline` — whether the input was an ensemble (list of sequences).
- `L_max`, `alpha`, `test_type` — engine parameters.
- `axs`, `ays` — input / output alphabets.
- `e_symbols` — the `(x, y)` emission-symbol pairs (as `[x, y]` lists).

### `partition`
The causal-state partition in **canonical** form. Engine state labels are
arbitrary, so each state is represented by the **sorted** list of its histories,
where each history is a `[x_hist, y_hist]` pair of strings. States are ordered by
their first (lexicographically smallest) history. This ordering is the canonical
state index used everywhere else in the file.

### `morphs`
One entry per canonical state, in canonical order. Each entry is the
`morph_by_state` count list for that state (plain integers). Counts are laid out
row-major over `(x, y)`: entry `len(ays) * x_index + y_index`.

### `filter`
Result of `filter_and_predict` on the (possibly truncated) training series:
- `filtered_states` — filtered causal-state sequence, remapped to **canonical
  state indices** (`-1` means "no state yet", preserved as-is).
- `filtered_probs` — filtered probabilities as floats, rounded to 12 decimals.
- `stringY_pred` — the predicted output string (`N` = no prediction yet).

`filter` is `null` for the multiline/ensemble case, since `filter_and_predict`
is only defined for single-string input (see `meta.filter_note`).
