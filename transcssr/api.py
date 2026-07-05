"""Public inference entry points for the token-first ``transcssr`` package.

Two functions assemble the full inference pipeline
(normalize -> encode -> count -> CSSR -> build model):

* :func:`infer_transducer` -- learn an :class:`~transcssr.machine.EpsilonTransducer`
  for an input/output process ``P(Y_t | X_t, X_past, Y_past)``.
* :func:`infer_machine` -- learn an :class:`~transcssr.machine.EpsilonMachine`
  for an output-only process, wrapping the legacy dummy-constant-input
  convention internally so callers never see the dummy input.

Both accept token sequences of any hashable token type (integers, strings,
tuples, ...); see the "Sequence input contract" below.
"""

from __future__ import annotations

from typing import Any

from .alphabet import Alphabet, _is_sequence, _looks_like_ensemble
from .counting import estimate_counts
from .countstore import SuffixArrayCountStore
from .cssr import build_transducer, run_cssr
from .machine import EpsilonMachine, EpsilonTransducer

__all__ = ["infer_transducer", "infer_machine"]


# ---------------------------------------------------------------------------
# Sequence input contract (docs/DESIGN.md)
# ---------------------------------------------------------------------------
#
# Accepted sequence types: ``str`` (each character a token), ``list``/``tuple``
# of tokens, and 1-d ``numpy.ndarray``. An *ensemble* (the legacy
# ``is_multiline=True``) is detected by TYPE, not a flag:
#
#   * a ``list``/``tuple`` whose elements are ALL sequences
#     (str/list/tuple/ndarray) is an ensemble of realizations;
#   * a ``list``/``tuple`` whose elements are scalar tokens is a SINGLE
#     realization (this disambiguates ``list[int]`` -> one integer sequence);
#   * a ``list[str]`` is therefore an ensemble of character sequences (matches
#     legacy multiline usage); ``list[list[int]]`` is an ensemble.
#
# Two documented TRAPS follow directly from this type-based rule, and the
# ``ensemble=`` parameter of ``infer_transducer`` / ``infer_machine`` is the
# escape hatch for both:
#
#   * TUPLE TOKENS: ``[(0, 0), (0, 1), ...]`` -- a single sequence of tuple
#     tokens -- auto-detects as an ENSEMBLE of length-2 sequences, because each
#     tuple element is itself a sequence. Pass ``ensemble=False`` to treat the
#     whole list as one sequence whose tokens are the tuples.
#   * WORD TOKENS: ``['the', 'cat', ...]`` -- a single sequence of multi-char
#     string tokens -- auto-detects as an ENSEMBLE of character sequences (each
#     ``str`` is a sequence of characters). Pass ``ensemble=False`` to treat the
#     whole list as one sequence whose tokens are the whole strings.
#
# ``ensemble=None`` (default) auto-detects as above; ``ensemble=True`` forces
# ensemble handling (``seq`` must be a list/tuple of realizations);
# ``ensemble=False`` forces single-realization handling (``seq`` itself is one
# realization whose elements are tokens, even if they look like sequences).


def _normalize_sequences(
    seq: Any, name: str, ensemble: bool | None = None
) -> list:
    """Normalize a public sequence argument to a list of realizations.

    A single realization becomes a one-element list; an ensemble is returned
    as a list of its realizations. See the module-level "Sequence input
    contract" for the auto-detection rule and the ``ensemble`` escape hatch.

    Parameters
    ----------
    seq : sequence or ensemble
        The public sequence argument.
    name : str
        Argument name, used in error messages.
    ensemble : bool or None, optional
        ``None`` (default) auto-detects single-vs-ensemble by type. ``True``
        forces ensemble handling (``seq`` must be a list/tuple of realizations).
        ``False`` forces single-realization handling: ``seq`` is treated as one
        realization whose elements are tokens, even if those elements look like
        sequences (the escape hatch for tuple-token and word-token sequences).

    Raises
    ------
    ValueError
        If the normalized data contains no symbols at all (e.g. ``''``, ``[]``,
        or ``['', '']``).
    TypeError
        If ``seq`` is not a supported sequence type, or ``ensemble=True`` is
        given for a non-list/tuple ``seq``.
    """
    if ensemble is None:
        if _looks_like_ensemble(seq):
            realizations = list(seq)
        elif _is_sequence(seq):
            realizations = [seq]
        else:
            raise TypeError(
                f"{name} must be a sequence (str, list, tuple, or 1-d "
                f"numpy.ndarray) or an ensemble (list of such sequences); got "
                f"{type(seq).__name__!r}."
            )
    elif ensemble:
        if not isinstance(seq, (list, tuple)):
            raise TypeError(
                f"{name}: ensemble=True requires a list/tuple of realizations; "
                f"got {type(seq).__name__!r}."
            )
        realizations = list(seq)
    else:  # ensemble is False: seq itself is a single realization.
        if not _is_sequence(seq):
            raise TypeError(
                f"{name} must be a sequence (str, list, tuple, or 1-d "
                f"numpy.ndarray); got {type(seq).__name__!r}."
            )
        realizations = [seq]

    if sum(len(r) for r in realizations) == 0:
        raise ValueError(
            f"{name} contains no symbols; cannot infer a model from empty "
            f"input. Pass a non-empty sequence or ensemble."
        )
    return realizations


def _resolve_alphabet(
    spec: Alphabet | list | tuple | None,
    realizations: list,
) -> Alphabet:
    """Resolve an alphabet argument to an :class:`Alphabet`.

    Parameters
    ----------
    spec : Alphabet, list, tuple, or None
        An explicit :class:`Alphabet`, an explicit token list/tuple, or
        ``None`` to infer from data via :meth:`Alphabet.from_data`.
    realizations : list
        The normalized list of realizations, used only when ``spec is None``.
    """
    if isinstance(spec, Alphabet):
        return spec
    if spec is None:
        return Alphabet.from_data(realizations)
    if isinstance(spec, (list, tuple)):
        return Alphabet(spec)
    raise TypeError(
        f"alphabet must be an Alphabet, a list/tuple of tokens, or None; got "
        f"{type(spec).__name__!r}."
    )


def _encode_realizations(realizations: list, alphabet: Alphabet) -> list[tuple[int, ...]]:
    """Encode each realization to an index tuple through ``alphabet``."""
    return [alphabet.encode(seq) for seq in realizations]


def _run_cssr_backend(
    xs_idx: list,
    ys_idx: list,
    L_max: int,
    n_x: int,
    n_y: int,
    alpha: float,
    test: str,
    stats_mode: str,
    count_backend: str,
):
    """Run CSSR with the chosen count backend (``"counter"`` or ``"suffix"``).

    Both backends read counts through the identical
    :class:`~transcssr.countstore.CountStore` surface and therefore produce the
    same partition; they differ only in how the counts are stored (materialized
    Counters vs. an on-demand suffix array). Returns the ``run_cssr`` triple.
    """
    if count_backend == "counter":
        marg, fut = estimate_counts(xs_idx, ys_idx, L_max)
        return run_cssr(
            marg, fut, L_max, n_x, n_y, alpha=alpha, test=test, stats_mode=stats_mode
        )
    if count_backend == "suffix":
        store = SuffixArrayCountStore(xs_idx, ys_idx, L_max, n_x, n_y)
        return run_cssr(
            store, None, L_max, n_x, n_y, alpha=alpha, test=test, stats_mode=stats_mode
        )
    raise ValueError(
        f"Unknown count_backend {count_backend!r}; expected 'counter' or 'suffix'."
    )


def infer_transducer(
    xs: Any,
    ys: Any,
    L_max: int,
    alpha: float = 0.001,
    test: str = "chi2",
    input_alphabet: Alphabet | list | tuple | None = None,
    output_alphabet: Alphabet | list | tuple | None = None,
    stats_mode: str = "correct",
    ensemble: bool | None = None,
    count_backend: str = "counter",
    verbose: bool = False,
) -> EpsilonTransducer:
    """Infer an epsilon-transducer from an input/output process.

    Pipeline: normalize the input/output sequences (single realization vs.
    ensemble, detected by type -- see the module "Sequence input contract"),
    resolve/encode the alphabets, count joint histories, run the CSSR
    state-splitting algorithm, and assemble a canonicalized model.

    Parameters
    ----------
    xs, ys : sequence or ensemble
        The input and output realizations. Each is either a single sequence
        (``str``, ``list``/``tuple`` of tokens, or 1-d :class:`numpy.ndarray`)
        or an ensemble (a ``list``/``tuple`` of such sequences). ``xs`` and
        ``ys`` must have the same structure: the same number of realizations,
        pairwise equal lengths.

        Ensembles are auto-detected by TYPE (``docs/DESIGN.md``): a list whose
        elements are all sequences is an ensemble. Two traps follow from this
        rule -- a list of TUPLE tokens ``[(0, 0), (0, 1), ...]`` is otherwise
        read as an ensemble of length-2 sequences, and a list of multi-char
        STRING (word) tokens ``['the', 'cat', ...]`` as a character ensemble.
        Use ``ensemble=False`` to force single-sequence handling in both cases.
    L_max : int
        Maximum history length.
    alpha : float, optional
        Significance level for the split test. Default ``0.001``.
    test : {'chi2', 'G'}, optional
        Statistic used for morph comparison. Default ``'chi2'``.
    input_alphabet, output_alphabet : Alphabet, list, tuple, or None, optional
        Token alphabets. An :class:`Alphabet` is used as-is; a list/tuple of
        tokens builds one in that order; ``None`` (default) infers a
        deterministic alphabet from the data via :meth:`Alphabet.from_data`.
    stats_mode : {'correct', 'legacy'}, optional
        Morph-comparison stride policy (``docs/DESIGN.md``). ``'correct'``
        (default) uses the mathematically correct per-input-symbol stride;
        ``'legacy'`` reproduces the legacy stride bug for exact parity with the
        legacy engine (matters only when ``n_x`` is neither ``1`` nor ``n_y``).
    ensemble : bool or None, optional
        Override the single-vs-ensemble auto-detection for both ``xs`` and
        ``ys``. ``None`` (default) auto-detects by type; ``True`` forces
        ensemble handling; ``False`` forces single-sequence handling (each
        element of ``xs`` / ``ys`` is a token, even tuple or word tokens that
        look like sequences). See the "Sequence input contract" traps above.
    count_backend : {'counter', 'suffix'}, optional
        Which history-count backend to use. ``'counter'`` (default) materializes
        the legacy ``marg`` / ``fut`` :class:`collections.Counter` tables and is
        the parity oracle. ``'suffix'`` answers the same count queries on demand
        from a :class:`~transcssr.countstore.SuffixArrayCountStore` (a suffix
        array over the ``z``-coded joint stream) without materializing any
        Counter, trading a modest per-query cost for much lower peak memory at
        large ``T``. The two backends yield **identical** models by
        construction (same partition, morph counts, and state count).
    verbose : bool, optional
        If ``True``, print a one-line summary of the fitted model. Default
        ``False``.

    Returns
    -------
    EpsilonTransducer
        The canonicalized model.

    Raises
    ------
    ValueError
        If ``xs`` and ``ys`` differ in the number of realizations or in any
        aligned realization's length, or if ``count_backend`` is not
        ``'counter'`` or ``'suffix'``.
    KeyError
        If a data token is absent from a provided alphabet.
    """
    xs_list = _normalize_sequences(xs, "xs", ensemble=ensemble)
    ys_list = _normalize_sequences(ys, "ys", ensemble=ensemble)
    if len(xs_list) != len(ys_list):
        raise ValueError(
            f"xs and ys must have the same number of realizations; got "
            f"{len(xs_list)} and {len(ys_list)}."
        )

    in_alpha = _resolve_alphabet(input_alphabet, xs_list)
    out_alpha = _resolve_alphabet(output_alphabet, ys_list)

    xs_idx = _encode_realizations(xs_list, in_alpha)
    ys_idx = _encode_realizations(ys_list, out_alpha)
    for i, (xi, yi) in enumerate(zip(xs_idx, ys_idx)):
        if len(xi) != len(yi):
            raise ValueError(
                f"Realization {i}: xs and ys must have the same length; got "
                f"{len(xi)} and {len(yi)}."
            )

    epsilon, _, morph_by_state = _run_cssr_backend(
        xs_idx,
        ys_idx,
        L_max,
        len(in_alpha),
        len(out_alpha),
        alpha,
        test,
        stats_mode,
        count_backend,
    )
    model = build_transducer(epsilon, morph_by_state, in_alpha, out_alpha, L_max)

    if verbose:
        print(
            f"infer_transducer: {model.n_states} state(s), "
            f"n_x={model.n_x}, n_y={model.n_y}, L_max={L_max}, "
            f"test={test!r}, stats_mode={stats_mode!r}."
        )
    return model


def infer_machine(
    ys: Any,
    L_max: int,
    alpha: float = 0.001,
    test: str = "chi2",
    alphabet: Alphabet | list | tuple | None = None,
    stats_mode: str = "correct",
    ensemble: bool | None = None,
    count_backend: str = "counter",
    verbose: bool = False,
) -> EpsilonMachine:
    """Infer an epsilon-machine from an output-only process.

    An epsilon-machine is the degenerate epsilon-transducer whose input is a
    single dummy token (legacy convention ``stringX = '0' * len(stringY)``).
    This function wraps that convention internally: it manufactures a constant
    dummy input stream, runs the transducer pipeline with a single input
    symbol, and returns an :class:`EpsilonMachine` whose output-only surface
    hides the dummy input.

    Parameters
    ----------
    ys : sequence or ensemble
        The output realization(s). A single sequence (``str``, ``list``/
        ``tuple`` of tokens, or 1-d :class:`numpy.ndarray`) or an ensemble (a
        list/tuple of such sequences). See the module "Sequence input contract".

        Ensembles are auto-detected by TYPE. Two traps follow: a list of TUPLE
        tokens ``[(0, 0), (0, 1), ...]`` reads as an ensemble of length-2
        sequences, and a list of multi-char STRING (word) tokens
        ``['the', 'cat', ...]`` as a character ensemble. Pass ``ensemble=False``
        to force single-sequence handling in both cases.
    L_max : int
        Maximum history length.
    alpha : float, optional
        Significance level for the split test. Default ``0.001``.
    test : {'chi2', 'G'}, optional
        Statistic used for morph comparison. Default ``'chi2'``.
    alphabet : Alphabet, list, tuple, or None, optional
        Output token alphabet. An :class:`Alphabet` is used as-is; a list/tuple
        builds one in that order; ``None`` (default) infers it from the data.
    stats_mode : {'correct', 'legacy'}, optional
        Morph-comparison stride policy. For an epsilon-machine ``n_x == 1``, so
        the two modes always agree; the flag is exposed for API symmetry.
        Default ``'correct'``.
    ensemble : bool or None, optional
        Override the single-vs-ensemble auto-detection for ``ys``. ``None``
        (default) auto-detects by type; ``True`` forces ensemble handling;
        ``False`` forces single-sequence handling (each element of ``ys`` is a
        token, even tuple or word tokens that look like sequences). See the
        "Sequence input contract" traps above.
    count_backend : {'counter', 'suffix'}, optional
        Which history-count backend to use. ``'counter'`` (default) materializes
        the legacy count Counters and is the parity oracle; ``'suffix'`` answers
        the same queries on demand from a
        :class:`~transcssr.countstore.SuffixArrayCountStore` (here ``n_x == 1``,
        so the ``z``-coding degenerates to the output stream), with much lower
        peak memory at large ``T``. Both yield **identical** machines by
        construction.
    verbose : bool, optional
        If ``True``, print a one-line summary of the fitted machine. Default
        ``False``.

    Returns
    -------
    EpsilonMachine
        The canonicalized output-only model. Its input alphabet is the single
        dummy token ``0``.
    """
    ys_list = _normalize_sequences(ys, "ys", ensemble=ensemble)
    out_alpha = _resolve_alphabet(alphabet, ys_list)
    ys_idx = _encode_realizations(ys_list, out_alpha)

    # Manufacture a constant dummy input stream (single symbol, index 0) of the
    # same shape as the output, matching the legacy `stringX = '0'*len(stringY)`.
    dummy_in = Alphabet([0])
    xs_idx = [tuple([0] * len(yi)) for yi in ys_idx]

    epsilon, _, morph_by_state = _run_cssr_backend(
        xs_idx,
        ys_idx,
        L_max,
        1,
        len(out_alpha),
        alpha,
        test,
        stats_mode,
        count_backend,
    )
    # Reuse the transducer assembly (compaction + canonicalization), then wrap
    # the canonical result in an EpsilonMachine that hides the dummy input.
    et = build_transducer(epsilon, morph_by_state, dummy_in, out_alpha, L_max)
    machine = EpsilonMachine(
        output_alphabet=out_alpha,
        n_states=et.n_states,
        epsilon=et.epsilon,
        morph_counts=et.morph_counts,
        L_max=L_max,
        input_alphabet=dummy_in,
    )

    if verbose:
        print(
            f"infer_machine: {machine.n_states} state(s), "
            f"n_y={machine.n_y}, L_max={L_max}, test={test!r}."
        )
    return machine
