# Examples

Demos and research scripts. Everything here drives the **legacy** engine
(`transCSSR.py` at the repo root) and therefore needs the `[legacy]`
extras (`uv sync --group dev` provides them). For the current token-first
API, see the quickstarts in the top-level README and the `transcssr` CLI.

**Run the demos from the repository root** — the legacy engine writes its
outputs into `transCSSR_results/` relative to the working directory:

```bash
uv run python examples/demo_transCSSR.py
```

Each script starts with a small `sys.path` bootstrap so that
`import transCSSR` resolves to the legacy module at the repo root rather
than the installed `transcssr` package (the two names collide on
case-insensitive filesystems).

## Contents

- `demo_transCSSR.py` — end-to-end epsilon-transducer inference and
  prediction (with plots).
- `demo_CSSR.py` — epsilon-machine inference and prediction.
- `demo_computational_mechanics_bootstrap.py` — bootstrap assessment of
  inferred models.
- `demo_CSSR.ipynb`, `demo_computational_mechanics_bootstrap.ipynb` —
  notebook versions. The notebooks assume the repo root as the working
  directory and do not include the `sys.path` bootstrap; start Jupyter
  from the repo root.
- `experimental/` — research-era scripts (mixed-state analysis,
  presynchronization prediction, simulation from fitted models,
  `dit` integration). Archival: kept runnable where cheap to do so, but
  not maintained or tested.
- `simulation/`, `simulation-paper/` — the data-generation scripts that
  produced the datasets in `data/` (the `-paper` variants generated the
  datasets used in the transCSSR paper). Run from inside their own
  directory; they write into `../../data/`.
