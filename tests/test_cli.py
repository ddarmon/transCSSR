"""In-process CLI tests for :mod:`transcssr.cli` using ``typer.testing``.

Data files are truncated into ``tmp_path`` copies so runtimes stay small.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from transcssr.cli import app

runner = CliRunner()

_HEAD = 5000  # truncate data-file copies to keep inference fast


@pytest.fixture(scope="module")
def even_file(tmp_path_factory) -> Path:
    """A truncated copy of the even-process output data."""
    src = Path("data/even.dat").read_text().strip()[:_HEAD]
    dst = tmp_path_factory.mktemp("data") / "even.dat"
    dst.write_text(src)
    return dst


@pytest.fixture(scope="module")
def delay_files(tmp_path_factory) -> tuple[Path, Path]:
    """Truncated copies of the delay-channel input/output data."""
    xt = Path("data/Xt_delay-channel.dat").read_text().strip()[:_HEAD]
    yt = Path("data/Yt_delay-channel.dat").read_text().strip()[:_HEAD]
    d = tmp_path_factory.mktemp("delay")
    xp, yp = d / "Xt.dat", d / "Yt.dat"
    xp.write_text(xt)
    yp.write_text(yt)
    return xp, yp


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------


def test_infer_machine_even(even_file, tmp_path):
    out = tmp_path / "even.json"
    result = runner.invoke(
        app, ["infer", "-y", str(even_file), "-L", "3", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "machine" in result.output
    data = json.loads(out.read_text())
    assert data["kind"] == "machine"
    assert data["n_states"] == 2

    # And `show` reports 2 states.
    show = runner.invoke(app, ["show", "-m", str(out)])
    assert show.exit_code == 0
    assert "n_states: 2" in show.output


def test_infer_transducer_delay(delay_files, tmp_path):
    xp, yp = delay_files
    out = tmp_path / "delay.json"
    result = runner.invoke(
        app,
        ["infer", "-y", str(yp), "-x", str(xp), "-L", "2", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "transducer" in result.output
    data = json.loads(out.read_text())
    assert data["kind"] == "transducer"
    assert data["n_states"] == 2


def test_infer_dot_side_export(even_file, tmp_path):
    out = tmp_path / "even.json"
    dot = tmp_path / "even.dot"
    result = runner.invoke(
        app,
        ["infer", "-y", str(even_file), "-L", "3", "-o", str(out), "--dot", str(dot)],
    )
    assert result.exit_code == 0, result.output
    assert dot.exists()
    assert "digraph" in dot.read_text()


def test_infer_tokens_format(tmp_path):
    data = tmp_path / "toks.txt"
    data.write_text(" ".join(["0", "1"] * 100))
    out = tmp_path / "m.json"
    result = runner.invoke(
        app,
        ["infer", "-y", str(data), "-L", "2", "--format", "tokens", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    model = json.loads(out.read_text())
    # integer tokens preserved as ints in the alphabet
    assert model["output_alphabet"] == [0, 1]


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def test_filter_csv_shape(even_file, tmp_path):
    model = tmp_path / "even.json"
    runner.invoke(app, ["infer", "-y", str(even_file), "-L", "3", "-o", str(model)])
    result = runner.invoke(app, ["filter", "-m", str(model), "-y", str(even_file)])
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln]
    assert lines[0] == "t,state,prediction,p_0,p_1"
    # one data row per output symbol.
    n_symbols = len(even_file.read_text().strip())
    assert len(lines) - 1 == n_symbols
    # Unsynchronized rows: state -1, empty prediction cell.
    first = lines[1].split(",")
    assert first[1] == "-1"
    assert first[2] == ""
    # Once synchronized, a real state and prediction appear somewhere.
    assert any(row.split(",")[1] in {"0", "1"} for row in lines[1:])


def test_filter_transducer_csv(delay_files, tmp_path):
    xp, yp = delay_files
    model = tmp_path / "delay.json"
    runner.invoke(
        app, ["infer", "-y", str(yp), "-x", str(xp), "-L", "2", "-o", str(model)]
    )
    out = tmp_path / "out.csv"
    result = runner.invoke(
        app,
        ["filter", "-m", str(model), "-y", str(yp), "-x", str(xp), "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    lines = [ln for ln in out.read_text().splitlines() if ln]
    assert lines[0] == "t,state,prediction,p_0,p_1"
    assert len(lines) - 1 == len(yp.read_text().strip())


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def test_show_fields(even_file, tmp_path):
    model = tmp_path / "even.json"
    runner.invoke(app, ["infer", "-y", str(even_file), "-L", "3", "-o", str(model)])
    result = runner.invoke(app, ["show", "-m", str(model)])
    assert result.exit_code == 0, result.output
    for field in ("kind: machine", "n_states:", "L_max:", "output_alphabet"):
        assert field in result.output
    # emission table header row (state + output token labels).
    assert "state" in result.output


# ---------------------------------------------------------------------------
# export-dot
# ---------------------------------------------------------------------------


def test_export_dot(delay_files, tmp_path):
    xp, yp = delay_files
    model = tmp_path / "delay.json"
    runner.invoke(
        app, ["infer", "-y", str(yp), "-x", str(xp), "-L", "2", "-o", str(model)]
    )
    dot = tmp_path / "delay.dot"
    result = runner.invoke(app, ["export-dot", "-m", str(model), "-o", str(dot)])
    assert result.exit_code == 0, result.output
    text = dot.read_text()
    assert "digraph" in text
    # edge labels use the y|x:prob convention.
    assert "->" in text
    assert "label" in text


# ---------------------------------------------------------------------------
# --version and error cases
# ---------------------------------------------------------------------------


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "transcssr" in result.output


def test_error_missing_file(tmp_path):
    result = runner.invoke(
        app, ["infer", "-y", str(tmp_path / "nope.dat"), "-L", "3", "-o", str(tmp_path / "o.json")]
    )
    assert result.exit_code == 2
    assert "not found" in result.output


def test_error_x_with_machine(even_file, tmp_path):
    model = tmp_path / "even.json"
    runner.invoke(app, ["infer", "-y", str(even_file), "-L", "3", "-o", str(model)])
    result = runner.invoke(
        app, ["filter", "-m", str(model), "-y", str(even_file), "-x", str(even_file)]
    )
    assert result.exit_code == 2
    assert "epsilon-machine" in result.output


def test_error_transducer_without_x(delay_files, tmp_path):
    xp, yp = delay_files
    model = tmp_path / "delay.json"
    runner.invoke(
        app, ["infer", "-y", str(yp), "-x", str(xp), "-L", "2", "-o", str(model)]
    )
    result = runner.invoke(app, ["filter", "-m", str(model), "-y", str(yp)])
    assert result.exit_code == 2
    assert "required" in result.output


def test_error_bad_format(even_file, tmp_path):
    result = runner.invoke(
        app,
        ["infer", "-y", str(even_file), "-L", "3", "--format", "bogus", "-o", str(tmp_path / "o.json")],
    )
    assert result.exit_code == 2
    assert "invalid --format" in result.output
