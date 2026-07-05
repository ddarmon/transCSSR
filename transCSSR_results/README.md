# transCSSR_results/

The **hardcoded output directory of the legacy engine**: legacy
`run_transCSSR` writes `.dot` and `.dat_results` files here (relative to
the working directory), and several legacy analysis functions read
machine/transducer `.dot` files back from here. That is why this
directory must sit at the repository root for as long as the legacy
engine lives.

- Newly generated files are gitignored.
- The tracked files are historical results kept because legacy analysis
  paths and experimental scripts consume some of them as reference
  machines.
- The token-first `transcssr` package does not use this directory at
  all: models are saved/loaded as JSON wherever you choose, and DOT is
  an explicit, export-only format (`model.to_dot(path)` /
  `transcssr export-dot`).

This directory retires together with the legacy engine (see
`WHATS_NEXT.md`).
