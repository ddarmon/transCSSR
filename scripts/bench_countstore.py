#!/usr/bin/env python
"""Benchmark the Counter vs. suffix-array count backends for the CSSR engine.

Measures, per backend, on Markov-structured synthetic output data (the
epsilon-machine path, ``n_x = 1``): the count-store **build time**, the
``run_cssr`` **wall time**, and **peak memory**. Memory is measured honestly --
each ``(backend, config)`` cell runs in a *fresh subprocess* and reports
``resource.getrusage(RUSAGE_SELF).ru_maxrss`` (peak RSS), alongside the suffix
store's exact ``nbytes`` and a ``sys.getsizeof``-based estimate of the two
Counters. A small aligned table is printed.

The two backends produce identical partitions by construction; this script is
purely about build time / query time / memory, not correctness (see
``tests/test_suffix_countstore.py`` for the parity gate).

Usage
-----
    uv run python scripts/bench_countstore.py                 # default grid
    uv run python scripts/bench_countstore.py --T 100000 1000000 --n-y 8 64 --L 3

This file is outside the ``transcssr`` package, so ``print`` is fine.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

# Run from the repo root so the in-tree package imports cleanly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcssr.counting import estimate_counts  # noqa: E402
from transcssr.countstore import (  # noqa: E402
    CounterCountStore,
    SuffixArrayCountStore,
)
from transcssr.cssr import run_cssr  # noqa: E402


def _maxrss_mb() -> float:
    """Peak resident set size in MiB (``ru_maxrss`` is bytes on macOS, KiB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return rss * scale / (1024.0 * 1024.0)


def _counter_bytes(marg, fut) -> int:
    """A ``sys.getsizeof``-based estimate of the two Counters' footprint.

    Counts the two dict containers plus, per key, the outer ``(x_hist, y_hist)``
    tuple and its two inner index tuples. Interned small ints are shared and
    deliberately not re-counted, so this is a lower-ish estimate of the true
    resident cost -- honest about being an estimate, not a measurement.
    """
    total = sys.getsizeof(marg) + sys.getsizeof(fut)
    for counter in (marg, fut):
        for key in counter:
            total += sys.getsizeof(key) + sys.getsizeof(key[0]) + sys.getsizeof(key[1])
    return total


def _markov_chain(T: int, n_y: int, seed: int, order: int = 1) -> np.ndarray:
    """A length-``T`` order-``order`` Markov chain over ``n_y`` symbols.

    The per-context next-symbol distributions are drawn from a low-concentration
    Dirichlet so each context is *peaked* (structured, compressible data -- the
    regime the suffix backend targets), NOT i.i.d. noise. Deterministic in
    ``seed`` so both backend workers see the identical stream.
    """
    rng = np.random.default_rng(seed)
    n_contexts = n_y ** order
    # Peaked rows: concentration < 1 pushes mass onto a few successors.
    trans = rng.dirichlet(np.full(n_y, 0.3), size=n_contexts)
    cdf = np.cumsum(trans, axis=1)

    out = np.empty(T, dtype=np.int64)
    out[:order] = rng.integers(0, n_y, size=order)
    ctx = 0
    for i in range(order):
        ctx = ctx * n_y + int(out[i])
    draws = rng.random(T)
    for t in range(order, T):
        row = cdf[ctx]
        sym = int(np.searchsorted(row, draws[t]))
        if sym >= n_y:
            sym = n_y - 1
        out[t] = sym
        ctx = (ctx * n_y + sym) % n_contexts
    return out


def _run_worker(backend: str, T: int, n_y: int, L: int, seed: int) -> dict:
    """Build the store and run CSSR for one cell; return timings + memory."""
    y = _markov_chain(T, n_y, seed).tolist()
    xs = [[0] * T]
    ys = [y]
    n_x = 1

    result: dict = {"backend": backend, "T": T, "n_y": n_y, "L": L}

    if backend == "counter":
        t0 = time.perf_counter()
        marg, fut = estimate_counts(xs, ys, L)
        store = CounterCountStore(marg, fut, n_y)
        result["build_s"] = time.perf_counter() - t0
        result["struct_mb"] = _counter_bytes(marg, fut) / (1024.0 * 1024.0)
        result["n_keys"] = len(marg) + len(fut)
    elif backend == "suffix":
        t0 = time.perf_counter()
        store = SuffixArrayCountStore(xs, ys, L, n_x, n_y)
        result["build_s"] = time.perf_counter() - t0
        result["struct_mb"] = store._index.nbytes / (1024.0 * 1024.0)
        result["n_keys"] = len(store._tail)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    t0 = time.perf_counter()
    eps, _, morph = run_cssr(store, None, L, n_x, n_y)
    result["cssr_s"] = time.perf_counter() - t0
    result["n_states"] = len(morph)
    result["peak_mb"] = _maxrss_mb()
    return result


def _spawn(backend: str, T: int, n_y: int, L: int, seed: int) -> dict:
    """Run one cell in a fresh subprocess (isolated peak-RSS measurement)."""
    import subprocess

    cmd = [
        sys.executable, os.path.abspath(__file__), "--worker",
        "--backend", backend, "--T", str(T), "--n-y", str(n_y),
        "--L", str(L), "--seed", str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"worker failed for {backend} T={T} n_y={n_y}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _print_table(rows: list[dict]) -> None:
    header = [
        "T", "n_y", "L", "backend", "build_s", "cssr_s",
        "peak_mb", "struct_mb", "n_keys", "n_states",
    ]
    widths = {h: len(h) for h in header}
    fmt_rows = []
    for r in rows:
        cells = {
            "T": f"{r['T']:.0e}",
            "n_y": str(r["n_y"]),
            "L": str(r["L"]),
            "backend": r["backend"],
            "build_s": f"{r['build_s']:.3f}",
            "cssr_s": f"{r['cssr_s']:.3f}",
            "peak_mb": f"{r['peak_mb']:.1f}",
            "struct_mb": f"{r['struct_mb']:.1f}",
            "n_keys": f"{r['n_keys']:,}",
            "n_states": str(r["n_states"]),
        }
        fmt_rows.append(cells)
        for h in header:
            widths[h] = max(widths[h], len(cells[h]))

    line = "  ".join(h.rjust(widths[h]) for h in header)
    print(line)
    print("  ".join("-" * widths[h] for h in header))
    for cells in fmt_rows:
        print("  ".join(cells[h].rjust(widths[h]) for h in header))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=["counter", "suffix"])
    parser.add_argument("--T", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--n-y", type=int, nargs="+", default=[8, 64])
    parser.add_argument("--L", type=int, nargs="+", default=[3])
    parser.add_argument("--seed", type=int, default=20240705)
    args = parser.parse_args()

    if args.worker:
        # Single-cell worker: emit one JSON line on stdout.
        res = _run_worker(args.backend, args.T[0], args.n_y[0], args.L[0], args.seed)
        print(json.dumps(res))
        return

    rows: list[dict] = []
    for T in args.T:
        for n_y in args.n_y:
            for L in args.L:
                for backend in ("counter", "suffix"):
                    print(
                        f"# running {backend:8s} T={T:.0e} n_y={n_y} L={L} ...",
                        file=sys.stderr,
                        flush=True,
                    )
                    rows.append(_spawn(backend, T, n_y, L, args.seed))
    print()
    _print_table(rows)


if __name__ == "__main__":
    main()
