"""Command-line interface for the token-first ``transcssr`` package.

This module is the *only* place in the package that imports :mod:`typer`; the
engine modules (``api``, ``machine``, ``predict``, ``cssr``, ``counting``,
``stats``, ``alphabet``) stay pure numpy/scipy. The CLI is a thin
argument-parsing and I/O glue layer: all inference and model logic lives in
:mod:`transcssr.api` and :mod:`transcssr.machine`.

Subcommands
-----------
``infer``
    Fit an epsilon-machine (output only) or epsilon-transducer (with ``-x``)
    from data files and write the model as JSON.
``filter``
    Load a model and filter/predict over a data file, emitting a CSV.
``show``
    Print a readable summary of a model, including a per-state emission table.
``export-dot``
    Export a model to a Graphviz ``.dot`` file.

User errors (missing file, bad format, unknown token, machine/transducer
mismatch) exit with code ``2`` and a one-line message on stderr; no traceback.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

import typer

from .api import infer_machine, infer_transducer
from .machine import EpsilonMachine, EpsilonTransducer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Token-first CSSR: infer epsilon-machines/transducers and filter data.",
)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class _UserError(Exception):
    """A user-facing error: reported on stderr with exit code 2, no traceback."""


def _fail(message: str) -> "typer.Exit":
    """Print ``message`` to stderr and return a ``typer.Exit(2)`` to raise."""
    typer.echo(message, err=True)
    return typer.Exit(code=2)


# ---------------------------------------------------------------------------
# Data-file reading (CLI-only glue; never imported by engine modules)
# ---------------------------------------------------------------------------


def _is_int_like(token: str) -> bool:
    """Return ``True`` if ``token`` parses as a Python ``int``."""
    try:
        int(token)
    except ValueError:
        return False
    return True


def _decide_int(tokens: list[str], token_type: str) -> bool:
    """Decide whether whitespace tokens should be parsed as ints.

    ``token_type`` is one of ``'auto'``, ``'int'``, ``'str'``. ``'auto'``
    chooses int only when EVERY token is integer-like (and there is at least
    one token).
    """
    if token_type == "int":
        return True
    if token_type == "str":
        return False
    return len(tokens) > 0 and all(_is_int_like(t) for t in tokens)


def _convert(tokens: list[str], as_int: bool, source: Path) -> list:
    """Convert raw string tokens to ints or leave them as strings."""
    if not as_int:
        return list(tokens)
    out = []
    for t in tokens:
        try:
            out.append(int(t))
        except ValueError:
            raise _fail(
                f"Error: token {t!r} in {source} is not an integer "
                f"(required by --token-type int)."
            )
    return out


def read_data(
    path: Path,
    fmt: str,
    multiline: bool,
    token_type: str,
):
    """Read a data file into a sequence (single realization) or an ensemble.

    Parameters
    ----------
    path : Path
        Data file to read.
    fmt : {'chars', 'tokens'}
        ``'chars'`` treats each character as a token (legacy ``.dat`` style);
        ``'tokens'`` splits on whitespace.
    multiline : bool
        If ``True``, each nonempty LINE is one realization (an ensemble); if
        ``False``, the file is a single realization (the first line for
        ``chars``, matching the legacy ``.dat`` convention; the whole file,
        whitespace-split, for ``tokens``).
    token_type : {'auto', 'int', 'str'}
        Token parsing for the ``tokens`` format only (ignored for ``chars``).

    Returns
    -------
    (data, is_ensemble) : tuple
        ``data`` is a single sequence or a list of realizations; ``is_ensemble``
        says which, so callers can pass ``ensemble=`` to the inference API.

    Raises
    ------
    typer.Exit
        (code 2) if the file is missing/unreadable, or a token fails to parse.
    """
    if not path.exists():
        raise _fail(f"Error: data file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - unusual I/O failure
        raise _fail(f"Error: could not read {path}: {exc}")

    if fmt == "chars":
        if multiline:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return lines, True
        lines = text.splitlines()
        first = lines[0].strip() if lines else ""
        return first, False

    # fmt == "tokens"
    if multiline:
        raw = [ln.split() for ln in text.splitlines() if ln.split()]
        flat = [t for r in raw for t in r]
        as_int = _decide_int(flat, token_type)
        return [_convert(r, as_int, path) for r in raw], True
    raw = text.split()
    as_int = _decide_int(raw, token_type)
    return _convert(raw, as_int, path), False


def _validate_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    """Raise a code-2 error if ``value`` is not one of ``choices``."""
    if value not in choices:
        raise _fail(
            f"Error: invalid --{name} {value!r}; choose from "
            f"{', '.join(choices)}."
        )


def _load_model(path: Path) -> EpsilonTransducer:
    """Load a model from JSON, mapping user errors to exit code 2."""
    if not path.exists():
        raise _fail(f"Error: model file not found: {path}")
    try:
        return EpsilonTransducer.from_json(path)
    except (ValueError, KeyError) as exc:
        raise _fail(f"Error: could not load model {path}: {exc}")


# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import PackageNotFoundError, version

        try:
            ver = version("transCSSR")
        except PackageNotFoundError:  # pragma: no cover - unusual install state
            ver = "unknown"
        typer.echo(f"transcssr {ver}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the package version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Token-first CSSR command-line interface."""


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------


@app.command()
def infer(
    outputs: Path = typer.Option(
        ..., "-y", "--outputs", help="Output-symbol data file (required)."
    ),
    l_max: int = typer.Option(
        ..., "-L", "--l-max", help="Maximum history length (required)."
    ),
    output: Path = typer.Option(
        ..., "-o", "--output", help="Destination model JSON path (required)."
    ),
    inputs: Optional[Path] = typer.Option(
        None,
        "-x",
        "--inputs",
        help="Input-symbol data file. Given -> transducer; omitted -> machine.",
    ),
    alpha: float = typer.Option(
        0.001, "--alpha", help="Significance level for the split test."
    ),
    test: str = typer.Option("chi2", "--test", help="Split statistic: chi2 or G."),
    stats_mode: str = typer.Option(
        "correct", "--stats-mode", help="Morph-comparison stride: correct or legacy."
    ),
    dot: Optional[Path] = typer.Option(
        None, "--dot", help="Also export a Graphviz .dot to this path."
    ),
    fmt: str = typer.Option(
        "chars", "--format", help="Data format: chars or tokens."
    ),
    multiline: bool = typer.Option(
        False, "--multiline", help="Each nonempty line is one realization (ensemble)."
    ),
    token_type: str = typer.Option(
        "auto", "--token-type", help="Token parsing for --format tokens: auto|int|str."
    ),
) -> None:
    """Infer an epsilon-machine or epsilon-transducer and write it as JSON."""
    _validate_choice("format", fmt, ("chars", "tokens"))
    _validate_choice("test", test, ("chi2", "G"))
    _validate_choice("stats-mode", stats_mode, ("correct", "legacy"))
    _validate_choice("token-type", token_type, ("auto", "int", "str"))

    ys, ys_ensemble = read_data(outputs, fmt, multiline, token_type)

    try:
        if inputs is not None:
            xs, _ = read_data(inputs, fmt, multiline, token_type)
            model: EpsilonTransducer = infer_transducer(
                xs,
                ys,
                l_max,
                alpha=alpha,
                test=test,
                stats_mode=stats_mode,
                ensemble=ys_ensemble,
            )
            kind = "transducer"
        else:
            model = infer_machine(
                ys,
                l_max,
                alpha=alpha,
                test=test,
                stats_mode=stats_mode,
                ensemble=ys_ensemble,
            )
            kind = "machine"
    except (ValueError, TypeError) as exc:
        raise _fail(f"Error: {exc}")

    try:
        model.to_json(output)
    except (OSError, TypeError) as exc:
        raise _fail(f"Error: could not write model to {output}: {exc}")

    if dot is not None:
        try:
            model.to_dot(dot)
        except OSError as exc:
            raise _fail(f"Error: could not write DOT to {dot}: {exc}")

    if kind == "transducer":
        summary = (
            f"infer: transducer, {model.n_states} state(s), "
            f"n_x={model.n_x}, n_y={model.n_y}, L_max={l_max}."
        )
    else:
        summary = (
            f"infer: machine, {model.n_states} state(s), "
            f"n_y={model.n_y}, L_max={l_max}."
        )
    typer.echo(summary, err=True)


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


@app.command()
def filter(
    model: Path = typer.Option(..., "-m", "--model", help="Model JSON file."),
    outputs: Path = typer.Option(
        ..., "-y", "--outputs", help="Output-symbol data file."
    ),
    inputs: Optional[Path] = typer.Option(
        None, "-x", "--inputs", help="Input-symbol data file (required for transducers)."
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="CSV output path (default: stdout)."
    ),
    fmt: str = typer.Option("chars", "--format", help="Data format: chars or tokens."),
    token_type: str = typer.Option(
        "auto", "--token-type", help="Token parsing for --format tokens: auto|int|str."
    ),
) -> None:
    """Filter causal states and predict outputs, writing a CSV table."""
    _validate_choice("format", fmt, ("chars", "tokens"))
    _validate_choice("token-type", token_type, ("auto", "int", "str"))

    m = _load_model(model)
    ys, _ = read_data(outputs, fmt, False, token_type)

    is_machine = isinstance(m, EpsilonMachine)
    try:
        if is_machine:
            if inputs is not None:
                raise _fail(
                    "Error: model is an epsilon-machine (output-only); do not "
                    "pass -x/--inputs."
                )
            result = m.filter(ys)
        else:
            if inputs is None:
                raise _fail(
                    "Error: model is an epsilon-transducer; -x/--inputs is "
                    "required."
                )
            xs, _ = read_data(inputs, fmt, False, token_type)
            result = m.filter(xs, ys)
    except (ValueError, KeyError) as exc:
        raise _fail(f"Error: {exc}")

    out_tokens = list(m.output_alphabet.tokens)
    header = ["t", "state", "prediction"] + [f"p_{tok}" for tok in out_tokens]

    def _write(handle) -> None:
        writer = csv.writer(handle)
        writer.writerow(header)
        probs = result.prediction_probs
        for t, state in enumerate(result.states):
            pred = result.predictions[t]
            pred_cell = "" if pred is None else str(pred)
            prob_cells = [_fmt(probs[t, j]) for j in range(len(out_tokens))]
            writer.writerow([t, state, pred_cell] + prob_cells)

    if output is None:
        _write(sys.stdout)
    else:
        try:
            with open(output, "w", encoding="utf-8", newline="") as handle:
                _write(handle)
        except OSError as exc:
            raise _fail(f"Error: could not write CSV to {output}: {exc}")


def _fmt(value: float) -> str:
    """Format a probability for CSV output."""
    return f"{float(value):.6g}"


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@app.command()
def show(
    model: Path = typer.Option(..., "-m", "--model", help="Model JSON file."),
) -> None:
    """Print a readable summary of a model and its emission-probability table."""
    m = _load_model(model)
    is_machine = isinstance(m, EpsilonMachine)
    kind = "machine" if is_machine else "transducer"

    in_tokens = list(m.input_alphabet.tokens)
    out_tokens = list(m.output_alphabet.tokens)

    lines = [
        f"kind: {kind}",
        f"n_states: {m.n_states}",
        f"L_max: {m.L_max}",
    ]
    if not is_machine:
        lines.append(
            f"input_alphabet ({m.n_x}): "
            + ", ".join(repr(t) for t in in_tokens)
        )
    lines.append(
        f"output_alphabet ({m.n_y}): "
        + ", ".join(repr(t) for t in out_tokens)
    )
    typer.echo("\n".join(lines))
    typer.echo("")

    # Emission-probability table: rows = states, columns = input|output pairs.
    probs = m.emission_probs()
    if is_machine:
        col_labels = [f"{ot}" for ot in out_tokens]
    else:
        col_labels = [
            f"{it}|{ot}" for it in in_tokens for ot in out_tokens
        ]
    typer.echo(_render_table(probs, col_labels))


def _render_table(probs, col_labels: list[str]) -> str:
    """Render a plain-text emission-probability table."""
    import numpy as np

    n_states = probs.shape[0]
    row_header = "state"
    cells = []
    for label in col_labels:
        cells.append(label)
    # Column widths.
    body = []
    for s in range(n_states):
        row = [str(s)]
        for j in range(len(col_labels)):
            val = probs[s, j]
            row.append("nan" if np.isnan(val) else f"{val:.3f}")
        body.append(row)
    widths = [max(len(row_header), max((len(r[0]) for r in body), default=0))]
    for j, label in enumerate(col_labels):
        w = len(label)
        for r in body:
            w = max(w, len(r[j + 1]))
        widths.append(w)

    def fmt_row(values: list[str]) -> str:
        return "  ".join(v.rjust(widths[i]) for i, v in enumerate(values))

    header = fmt_row([row_header] + col_labels)
    sep = "  ".join("-" * w for w in widths)
    return "\n".join([header, sep] + [fmt_row(r) for r in body])


# ---------------------------------------------------------------------------
# export-dot
# ---------------------------------------------------------------------------


@app.command(name="export-dot")
def export_dot(
    model: Path = typer.Option(..., "-m", "--model", help="Model JSON file."),
    output: Path = typer.Option(
        ..., "-o", "--output", help="Destination .dot path."
    ),
) -> None:
    """Export a model to a Graphviz .dot file."""
    m = _load_model(model)
    try:
        m.to_dot(output)
    except OSError as exc:
        raise _fail(f"Error: could not write DOT to {output}: {exc}")
    typer.echo(f"Wrote DOT to {output}", err=True)


if __name__ == "__main__":  # pragma: no cover
    app()
