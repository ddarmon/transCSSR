"""E1 pilot: char-level CSSR on English text, with generation.

Infers an epsilon-machine from a prefix of text8 (or any text file),
scores held-out bits/char with the MSP scorer, reports complexity
measures, and samples text from the fitted machine by following its
transitions from the stationary distribution.

Usage (from the repo root):

    python examples/experimental/e1_pilot.py                    # text8, 1M chars, L=3
    python examples/experimental/e1_pilot.py --L 4 --train-chars 5000000
    python examples/experimental/e1_pilot.py --data my.txt --gen-chars 1000

text8 (100M chars of Wikipedia, alphabet = 'a'-'z' + space) is downloaded
to data/text8 on first use (~31 MB zip).
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

# Run from anywhere: put the repo root (two levels up) first on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from transcssr import infer_machine  # noqa: E402
from transcssr.msp import log_loss, stationary_distribution  # noqa: E402

TEXT8_URLS = [
    "https://mattmahoney.net/dc/text8.zip",
    "http://mattmahoney.net/dc/text8.zip",
]


def load_text(path: Path, n_chars: int) -> str:
    """Load and normalize text to the text8 alphabet ('a'-'z' + space)."""
    if path.name == "text8" and not path.exists():
        download_text8(path)
    raw = path.read_text(errors="ignore")[: 4 * n_chars]
    norm = re.sub(r"[^a-z ]", " ", raw.lower())
    norm = re.sub(r" +", " ", norm)
    if len(norm) < n_chars:
        raise SystemExit(
            f"Only {len(norm)} normalized chars available in {path}; "
            f"need {n_chars}. Reduce --train-chars/--test-chars."
        )
    return norm


def download_text8(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in TEXT8_URLS:
        try:
            print(f"Downloading text8 from {url} ...", flush=True)
            with urllib.request.urlopen(url, timeout=120) as resp:
                payload = resp.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                dest.write_bytes(zf.read("text8"))
            print(f"Saved {dest} ({dest.stat().st_size/1e6:.0f} MB).")
            return
        except Exception as err:  # noqa: BLE001 - report and try next mirror
            last_err = err
            print(f"  failed: {err}")
    raise SystemExit(f"Could not download text8: {last_err}")


def entropy_rate_bits(pi: np.ndarray, probs: np.ndarray) -> float:
    """h_mu = sum_s pi_s * H(emission_s), in bits/char."""
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(probs > 0, probs * np.log2(probs), 0.0)
    return float(-(pi @ np.nansum(plogp, axis=1)))


def generate(
    machine, n_chars: int, seed: int
) -> tuple[str, int]:
    """Sample text from the machine; returns (text, n_restarts).

    Starts in a stationary-distribution state; emits from the state's
    morph distribution; follows the (unifilar) transition. If an emitted
    symbol has no defined transition (possible: emissions are estimated
    from counts, transitions from retained histories), restarts from the
    stationary distribution and counts the event.
    """
    rng = np.random.default_rng(seed)
    pi = stationary_distribution(machine)
    probs = machine.emission_probs()  # (n_states, n_y); n_x == 1
    trans = machine.transitions()  # (state, 0, y_idx) -> next state
    alpha = machine.alphabet
    n_y = machine.n_y

    out: list[str] = []
    restarts = 0
    state = int(rng.choice(len(pi), p=pi))
    while len(out) < n_chars:
        p = probs[state]
        if not np.all(np.isfinite(p)):
            state = int(rng.choice(len(pi), p=pi))
            restarts += 1
            continue
        y = int(rng.choice(n_y, p=p))
        out.append(str(alpha.token(y)))
        nxt = trans.get((state, 0, y), -1)
        if nxt == -1:
            state = int(rng.choice(len(pi), p=pi))
            restarts += 1
        else:
            state = nxt
    return "".join(out), restarts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=_REPO_ROOT / "data" / "text8")
    ap.add_argument("--train-chars", type=int, default=1_000_000)
    ap.add_argument("--test-chars", type=int, default=200_000)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--test", choices=["chi2", "G"], default="chi2")
    ap.add_argument("--gen-chars", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    text = load_text(args.data, args.train_chars + args.test_chars)
    train = text[: args.train_chars]
    test = text[args.train_chars : args.train_chars + args.test_chars]
    n_y = len(set(train))
    print(
        f"corpus={args.data.name} train={len(train):,} test={len(test):,} "
        f"alphabet={n_y} L={args.L} alpha={args.alpha} test={args.test}"
    )

    t0 = time.perf_counter()
    machine = infer_machine(train, L_max=args.L, alpha=args.alpha, test=args.test)
    t_fit = time.perf_counter() - t0
    print(f"\ninferred: {machine.n_states} states in {t_fit:.1f}s")

    pi = stationary_distribution(machine)
    c_mu = float(-(pi @ np.log2(np.where(pi > 0, pi, 1.0))))
    h_mu = entropy_rate_bits(pi, machine.emission_probs())
    print(f"C_mu (statistical complexity) = {c_mu:.3f} bits")
    print(f"h_mu (entropy rate)           = {h_mu:.3f} bits/char")

    t0 = time.perf_counter()
    score = log_loss(machine, test)
    t_score = time.perf_counter() - t0
    print(
        f"held-out log-loss             = {score.per_symbol_bits:.3f} bits/char "
        f"({score.n_symbols:,} symbols, {score.n_zero_prob} zero-prob, "
        f"{t_score:.1f}s)"
    )
    finite = np.isfinite(score.per_step_nats)
    if finite.any():
        finite_bits = float(score.per_step_nats[finite].mean() / np.log(2.0))
        print(
            f"  finite steps only           = {finite_bits:.3f} bits/char "
            f"over {int(finite.sum()):,}/{score.n_symbols:,} steps "
            f"({100 * (1 - finite.mean()):.2f}% zero-prob, excluded; a real "
            f"score needs an explicit backoff policy)"
        )
    print("reference: PPM/CTW ~1.7-2.0, char n-grams ~2.0-2.5, "
          "small transformers ~1.0-1.2 bpc on text8")

    sample, restarts = generate(machine, args.gen_chars, args.seed)
    print(f"\n--- sample ({args.gen_chars} chars, {restarts} restarts) ---")
    for i in range(0, len(sample), 78):
        print(sample[i : i + 78])


if __name__ == "__main__":
    main()
