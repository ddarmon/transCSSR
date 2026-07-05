"""The CSSR state-splitting / determinization algorithm (token-first).

This module reimplements the core inference path of the legacy
:func:`transCSSR.run_transCSSR` (see ``docs/SPEC-core.md`` §3-§4), operating
purely on **index tuples** produced by an :class:`~transcssr.alphabet.Alphabet`
rather than on character strings. A history is a pair ``(x_hist, y_hist)`` of
equal-length ``tuple[int, ...]``; a morph is a flat count vector of length
``n_x * n_y`` indexed ``n_y * x_ind + y_ind``.

The output partition is **order-dependent** (``docs/SPEC-core.md`` §6.1): the
concrete set of causal states, and even which histories land in which state,
depend on iteration order. This implementation therefore replicates the legacy
iteration order *exactly* -- insertion-ordered dicts for states and histories,
child generation ``for ay: for ax:`` (output-outer, input-inner), first-wins
strict-``>`` argmax existing-state matching, mid-pass parent-morph mutation, and
the determinization pilot-selection order (including a legacy off-by-one quirk,
flagged below). Several fragile legacy behaviors are replicated deliberately;
each is called out with a ``LEGACY QUIRK`` comment and a SPEC reference.

Scope (per ``docs/DESIGN.md``): Stage 4 (optional ``stringX``/``stringY`` morph
re-estimation) is omitted, as are the ``.dot``/``.dat_results`` side effects of
Stage 5. Canonical relabeling is applied only by :func:`build_transducer`,
never inside :func:`run_cssr`.

SPEC/legacy discrepancy (see :func:`_legacy_compare_morphs` below): the legacy
``chisquared_test`` uses row *stride* ``n_x`` (``docs/SPEC-core.md`` §2.1 flags
it as a bug); the corrected stride is ``n_y`` (``transcssr.stats``). The two
agree whenever ``n_x == 1`` or ``n_x == n_y`` but can produce DIFFERENT
partitions otherwise (e.g. the ``multiline_ensemble`` golden fixture,
``n_x=3, n_y=2``, captured from the buggy legacy engine). Policy
(``docs/DESIGN.md``): :func:`run_cssr` defaults to ``stats_mode='correct'``
so users get correct statistics; parity tests pass ``stats_mode='legacy'``
to reproduce the legacy engine exactly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping

import numpy as np
import scipy.stats

from .alphabet import Alphabet
from .machine import EpsilonTransducer
from .stats import compare_morphs as _correct_compare_morphs

__all__ = ["run_cssr", "build_transducer"]

History = tuple[tuple[int, ...], tuple[int, ...]]
Epsilon = dict[History, int]
InvEpsilon = dict[int, dict[History, bool]]
MorphByState = dict[int, list[int]]


def _emission_symbols(n_x: int, n_y: int) -> list[tuple[int, int]]:
    """Return the ``(x_ind, y_ind)`` emission symbols in legacy order.

    The legacy ``e_symbols = list(itertools.product(axs, ays))`` iterates the
    input index in the outer loop and the output index in the inner loop, so
    ``e_symbols[k] == (x_ind, y_ind)`` with ``k = n_y * x_ind + y_ind``.
    """
    return [(x_ind, y_ind) for x_ind in range(n_x) for y_ind in range(n_y)]


def _legacy_compare_morphs(
    morph1: list[int] | np.ndarray,
    morph2: list[int] | np.ndarray,
    n_x: int,
    n_y: int,
    alpha: float = 0.001,
    test: str = "chi2",
) -> tuple[bool, float]:
    """Compare two morphs, replicating the LEGACY ``chisquared_test`` exactly.

    LEGACY QUIRK / SPEC DISCREPANCY (``docs/SPEC-core.md`` §2.1): the legacy
    per-input-symbol row slice uses stride ``n_x`` (``row_ind * len(axs)``)
    rather than the mathematically correct stride ``n_y``. The corrected
    ``transcssr.stats.compare_morphs`` uses ``n_y``; the two agree whenever
    ``n_x == 1`` or ``n_x == n_y``. They *disagree* when ``n_x > 1`` and
    ``n_x != n_y`` -- e.g. the ``multiline_ensemble`` golden fixture
    (``n_x=3, n_y=2``), which was captured from the legacy engine and therefore
    encodes the buggy stride.

    This comparator exists for ``run_cssr(..., stats_mode='legacy')``, whose
    contract is exact partition parity with the legacy ``run_transCSSR`` (the
    parity oracle / golden fixtures). The DEFAULT ``stats_mode='correct'``
    uses ``transcssr.stats.compare_morphs`` instead, so users get correct
    statistics; parity tests opt into the legacy stride explicitly.

    Parameters and returns mirror :func:`transcssr.stats.compare_morphs`:
    returns ``(reject, pvalue)`` where ``reject`` is ``True`` when the null
    (same distribution) is rejected at level ``alpha``.
    """
    if test not in ("chi2", "G"):
        raise ValueError(f"Unknown test {test!r}; expected 'chi2' or 'G'.")
    a1 = np.asarray(morph1, dtype=float)
    a2 = np.asarray(morph2, dtype=float)

    statistic = 0.0
    df_x = 0
    for row_ind in range(n_x):
        start = row_ind * n_x  # LEGACY stride len(axs); see SPEC §2.1
        tmp1 = a1[start : start + n_y]
        tmp2 = a2[start : start + n_y]
        n1 = tmp1.sum()
        n2 = tmp2.sum()
        if n1 == 0 or n2 == 0:
            continue
        df_x += 1
        theta0 = (tmp1 + tmp2) / (n1 + n2)
        ns = (n1, n2)
        table = (tmp1, tmp2)
        for array_ind in range(len(tmp1)):
            for sample_ind in (0, 1):
                denom = ns[sample_ind] * theta0[array_ind]
                if denom == 0:
                    continue
                obs = table[sample_ind][array_ind]
                if test == "chi2":
                    statistic += (obs - denom) ** 2 / denom
                else:  # G-test
                    if obs != 0:
                        statistic += obs * np.log(obs / denom)
    if test == "G":
        statistic *= 2

    df = df_x * (n_y - 1)
    reject = bool(statistic > scipy.stats.chi2.ppf(1 - alpha, df))
    pvalue = float(1 - scipy.stats.chi2.cdf(statistic, df))
    return reject, pvalue


def _remove_state(
    state: int,
    epsilon: Epsilon,
    invepsilon: InvEpsilon,
    morph_by_state: MorphByState,
) -> None:
    """Delete ``state`` and all its histories from every structure in place.

    Mirrors legacy ``remove_state`` (``docs/SPEC-core.md`` §4.3). Remaining
    state labels are **not** renumbered (labels stay sparse).
    """
    for hist in list(invepsilon[state].keys()):
        epsilon.pop(hist, None)
    invepsilon.pop(state, None)
    morph_by_state.pop(state, None)


# ---------------------------------------------------------------------------
# Strongly connected components (igraph replacement)
# ---------------------------------------------------------------------------


def _strongly_connected_components(
    n_nodes: int, adj: Mapping[int, set[int]]
) -> list[int]:
    """Iterative Tarjan SCC over a directed graph on nodes ``0 .. n_nodes-1``.

    Replaces the legacy ``igraph.Graph.clusters()`` call, whose default mode is
    **strong** (``docs/SPEC-core.md`` §4.1). Component ids are assigned so that
    components are numbered by ascending minimum node index; this is a
    deterministic labeling whose "smallest id" tie-break matches the intent of
    the legacy ``scipy.stats.mode`` largest-component selection
    (``docs/SPEC-core.md`` §4.2, §6.1). Written iteratively (no recursion) to
    be safe against deep graphs.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    adj : mapping of int to set of int
        Directed adjacency: ``adj[i]`` is the set of heads of edges out of ``i``.

    Returns
    -------
    list of int
        ``membership[i]`` is the component id of node ``i``.
    """
    index_of: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    on_stack: dict[int, bool] = {}
    stack: list[int] = []
    counter = 0
    raw_components: list[list[int]] = []

    for root in range(n_nodes):
        if root in index_of:
            continue
        # Explicit DFS stack of (node, iterator-position) frames.
        work: list[tuple[int, int]] = [(root, 0)]
        while work:
            node, child_ind = work[-1]
            if child_ind == 0:
                index_of[node] = counter
                lowlink[node] = counter
                counter += 1
                stack.append(node)
                on_stack[node] = True
            neighbors = sorted(adj.get(node, ()))
            recursed = False
            for pos in range(child_ind, len(neighbors)):
                w = neighbors[pos]
                if w not in index_of:
                    work[-1] = (node, pos + 1)
                    work.append((w, 0))
                    recursed = True
                    break
                elif on_stack.get(w, False):
                    lowlink[node] = min(lowlink[node], index_of[w])
            if recursed:
                continue
            # Done with this node's neighbors.
            if lowlink[node] == index_of[node]:
                component: list[int] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == node:
                        break
                raw_components.append(component)
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])

    # Number components by ascending minimum node index (deterministic).
    raw_components.sort(key=min)
    membership = [0] * n_nodes
    for cid, component in enumerate(raw_components):
        for node in component:
            membership[node] = cid
    return membership


def _connected_component(
    epsilon: Epsilon,
    invepsilon: InvEpsilon,
    e_symbols: list[tuple[int, int]],
    L_max: int,
) -> tuple[list[int], dict[int, dict[int, bool]], dict[int, int]]:
    """Build the directed state graph and its SCC membership.

    Reproduces legacy ``get_connected_component`` (``docs/SPEC-core.md`` §4.1):
    node indices follow ``list(invepsilon.keys())`` enumeration order, and
    transition source histories follow the ``need_Lmax`` rule -- transitions are
    read off length-``(L_max - 1)`` histories (grow), falling back to
    length-``L_max`` histories (left-shift + append) only for states that have
    no length-``(L_max - 1)`` history.

    Returns
    -------
    membership : list of int
        SCC id per node index.
    trans_dict : dict
        ``{from_index: {to_index: True}}``.
    index_to_states : dict
        Node index -> original (sparse) state label.
    """
    states = list(invepsilon.keys())
    states_to_index = {s: i for i, s in enumerate(states)}
    index_to_states = {i: s for i, s in enumerate(states)}

    adj: dict[int, set[int]] = {i: set() for i in range(len(states))}
    trans_dict: dict[int, dict[int, bool]] = defaultdict(dict)

    for state in states:
        i = states_to_index[state]
        hists = list(invepsilon[state].keys())
        need_lmax = not any(len(h[0]) == L_max - 1 for h in hists)
        for hist in hists:
            if len(hist[0]) == L_max:
                if not need_lmax:
                    continue
                base_x, base_y = hist[0][1:], hist[1][1:]
            else:
                base_x, base_y = hist[0], hist[1]
            for ex, ey in e_symbols:
                s = epsilon.get((base_x + (ex,), base_y + (ey,)), -1)
                if s != -1:
                    j = states_to_index[s]
                    adj[i].add(j)
                    trans_dict[i][j] = True

    membership = _strongly_connected_components(len(states), adj)
    return membership, trans_dict, index_to_states


def _remove_transients(
    epsilon: Epsilon,
    invepsilon: InvEpsilon,
    morph_by_state: MorphByState,
    e_symbols: list[tuple[int, int]],
    L_max: int,
) -> None:
    """Remove transient states in place (``docs/SPEC-core.md`` §4.2).

    Pass 1 removes any state that is the source of an edge leaving its own SCC
    (a transient / non-recurrent state). Pass 2 recomputes SCCs and keeps only
    the largest recurrent component, breaking size ties toward the smallest
    component id (legacy ``scipy.stats.mode`` behavior; see
    :func:`_strongly_connected_components` for the id convention).
    """
    if not invepsilon:
        return

    membership, trans_dict, index_to_states = _connected_component(
        epsilon, invepsilon, e_symbols, L_max
    )
    remove: dict[int, bool] = {}
    for from_idx in trans_dict:
        for to_idx in trans_dict[from_idx]:
            if membership[from_idx] != membership[to_idx]:
                remove[index_to_states[from_idx]] = True
    for s in remove:
        _remove_state(s, epsilon, invepsilon, morph_by_state)

    if not invepsilon:
        return

    membership, _, index_to_states = _connected_component(
        epsilon, invepsilon, e_symbols, L_max
    )
    sizes = Counter(membership)
    # Largest component; smallest id on ties (mirrors scipy.stats.mode).
    keep_component = min(sizes, key=lambda cid: (-sizes[cid], cid))
    for state_idx, cid in enumerate(membership):
        if cid != keep_component:
            _remove_state(index_to_states[state_idx], epsilon, invepsilon, morph_by_state)


# ---------------------------------------------------------------------------
# The CSSR algorithm
# ---------------------------------------------------------------------------


def run_cssr(
    marg: Counter,
    fut: Counter,
    L_max: int,
    n_x: int,
    n_y: int,
    alpha: float = 0.001,
    test: str = "chi2",
    stats_mode: str = "correct",
) -> tuple[Epsilon, InvEpsilon, MorphByState]:
    """Infer an epsilon-transducer partition from history counts.

    Reimplements the legacy ``run_transCSSR`` Stages 0, 1, 1b, 2, 3, 3b, and 5
    (Stages 4 and the ``.dot`` side effects are out of scope; see
    ``docs/DESIGN.md`` and ``docs/SPEC-core.md`` §3), operating on index-tuple
    histories. State labels are non-negative integers assigned from a
    monotonically increasing counter; they are **not** guaranteed contiguous
    (splits and removals leave gaps) and are **not** canonicalized here.

    Parameters
    ----------
    marg : collections.Counter
        Legacy ``word_lookup_marg``: maps ``(x_hist, y_hist)`` with
        ``len(x_hist) == len(y_hist) + 1`` to its count.
    fut : collections.Counter
        Legacy ``word_lookup_fut``: maps equal-length ``(x_hist, y_hist)`` to
        its count.
    L_max : int
        Maximum history length.
    n_x, n_y : int
        Number of input / output symbols.
    alpha : float, optional
        Significance level for the split test. Default ``0.001``.
    test : {'chi2', 'G'}, optional
        Statistic used for morph comparison. Default ``'chi2'``.
    stats_mode : {'correct', 'legacy'}, optional
        Which morph comparator to use. ``'correct'`` (default) uses
        ``transcssr.stats.compare_morphs`` (mathematically correct
        per-input-symbol stride ``n_y``). ``'legacy'`` replicates the legacy
        ``chisquared_test`` stride bug (SPEC §2.1) for exact parity with the
        legacy engine; the two modes agree whenever ``n_x == 1`` or
        ``n_x == n_y`` and can produce different partitions otherwise.

    Returns
    -------
    epsilon : dict
        ``History -> int state`` for every retained history (lengths
        ``L_max - 1`` and ``L_max`` after pruning / determinization).
    invepsilon : dict
        ``int state -> {History: True}`` inverse partition (insertion-ordered).
    morph_by_state : dict
        ``int state -> list[int]`` morph count vector, length ``n_x * n_y``,
        indexed ``n_y * x_ind + y_ind``.
    """
    if stats_mode == "correct":
        compare_morphs = _correct_compare_morphs
    elif stats_mode == "legacy":
        compare_morphs = _legacy_compare_morphs
    else:
        raise ValueError(
            f"Unknown stats_mode {stats_mode!r}; expected 'correct' or 'legacy'."
        )

    e_symbols = _emission_symbols(n_x, n_y)
    n_e = len(e_symbols)

    epsilon: Epsilon = {}
    invepsilon: InvEpsilon = defaultdict(dict)
    morph_by_state: MorphByState = {}

    # -- Stage 0: initialize the null history in state 0. ------------------
    num_states = 0
    null_history: History = ((), ())
    epsilon[null_history] = 0
    invepsilon[0][null_history] = True
    morph_by_state[0] = [fut[((ex,), (ey,))] for ex, ey in e_symbols]
    num_states = 1

    # -- Stage 1: homogenize / split. -------------------------------------
    # Grow histories one step INTO THE PAST (prepend); target lengths L_cur+1.
    for L_cur in range(0, L_max):
        states = list(invepsilon.keys())  # snapshot in insertion order
        for state in states:
            histories = list(invepsilon[state].keys())  # snapshot
            for history in histories:
                if len(history[0]) != L_cur:
                    continue
                all_distinct = True
                # Child generation order: output-outer, input-inner
                # (legacy `for ay in ays: for ax in axs:`); this fixes the
                # order in which new states are numbered (SPEC §6.1(1)).
                for y_ind in range(n_y):
                    for x_ind in range(n_x):
                        new_history: History = (
                            (x_ind,) + history[0],
                            (y_ind,) + history[1],
                        )
                        # Occurrence test: does this child ever appear on any
                        # next input? (LEGACY QUIRK, SPEC §6.3.5: the inner
                        # `for ax` shadows the outer input loop var; harmless
                        # because new_history is already fixed.)
                        does_not_occur = True
                        for ax2 in range(n_x):
                            if marg[(new_history[0] + (ax2,), new_history[1])] != 0:
                                does_not_occur = False
                        if does_not_occur:
                            continue

                        morph_by_history = [
                            fut[(new_history[0] + (ex,), new_history[1] + (ey,))]
                            for ex, ey in e_symbols
                        ]
                        reject, _ = compare_morphs(
                            morph_by_history,
                            morph_by_state[state],
                            n_x,
                            n_y,
                            alpha=alpha,
                            test=test,
                        )
                        if reject:
                            is_matching_state = False
                            if num_states != 1:
                                cur_best_pvalue = -np.inf
                                cur_best_state = -1
                                for altstate in list(invepsilon.keys()):
                                    if altstate == state:
                                        continue
                                    _, pvalue = compare_morphs(
                                        morph_by_history,
                                        morph_by_state[altstate],
                                        n_x,
                                        n_y,
                                        alpha=alpha,
                                        test=test,
                                    )
                                    # Strict `>`: first state wins ties, in
                                    # insertion order (SPEC §6.1(2)).
                                    if pvalue > alpha and pvalue > cur_best_pvalue:
                                        cur_best_pvalue = pvalue
                                        cur_best_state = altstate
                                if cur_best_state != -1:
                                    is_matching_state = True
                                    epsilon[new_history] = cur_best_state
                                    invepsilon[cur_best_state][new_history] = True
                                    dst = morph_by_state[cur_best_state]
                                    for k in range(n_e):
                                        dst[k] += morph_by_history[k]
                            if not is_matching_state:
                                epsilon[new_history] = num_states
                                invepsilon[num_states][new_history] = True
                                morph_by_state[num_states] = list(morph_by_history)
                                num_states += 1
                        else:
                            # Do not reject: child belongs to its parent state.
                            # Parent morph mutates mid-pass, so later children
                            # are tested against the updated parent (SPEC
                            # §6.1(3)).
                            all_distinct = False
                            epsilon[new_history] = state
                            invepsilon[state][new_history] = True
                            dst = morph_by_state[state]
                            for k in range(n_e):
                                dst[k] += morph_by_history[k]
                if all_distinct:
                    # Every child of `history` split off elsewhere; drop it.
                    invepsilon[state].pop(history)
                    epsilon.pop(history)
                    if len(invepsilon[state]) == 0:
                        _remove_state(state, epsilon, invepsilon, morph_by_state)
                    else:
                        # LEGACY QUIRK (SPEC §3 Stage 1, §6.3.9): subtract the
                        # parent's own child counts back out of the state
                        # morph. The author flagged this bookkeeping as a
                        # suspected over/under-count source; replicate as-is.
                        old_morph = [
                            fut[(history[0] + (ex,), history[1] + (ey,))]
                            for ex, ey in e_symbols
                        ]
                        dst = morph_by_state[state]
                        for k in range(n_e):
                            dst[k] -= old_morph[k]

    # -- Stage 1b: prune histories shorter than L_max - 1. -----------------
    for hist in list(epsilon.keys()):
        if len(hist[0]) < L_max - 1:
            state = epsilon[hist]
            # LEGACY QUIRK (SPEC §3 Stage 1b, §6.3.9): subtract the pruned
            # history's child morph out of the state; replicate as-is.
            hist_morph = [
                fut[(hist[0] + (ex,), hist[1] + (ey,))] for ex, ey in e_symbols
            ]
            dst = morph_by_state[state]
            for k in range(n_e):
                dst[k] -= hist_morph[k]
            epsilon.pop(hist)
            invepsilon[state].pop(hist)
    for state in list(invepsilon.keys()):
        if len(invepsilon[state]) == 0:
            _remove_state(state, epsilon, invepsilon, morph_by_state)

    # -- Stage 2: first transient removal. --------------------------------
    _remove_transients(epsilon, invepsilon, morph_by_state, e_symbols, L_max)
    # (Legacy makes a dead get_connected_component call here whose result is
    # unused; SPEC §3 Stage 2, §6.3.8. Omitted -- no observable effect.)

    # -- Stage 3: determinize. --------------------------------------------
    def _trans(hist: History, e: tuple[int, int]) -> int:
        """Transition of ``hist`` on emission ``e``; -1 if none (SPEC §3)."""
        if len(hist[0]) == L_max - 1:
            return epsilon.get((hist[0] + (e[0],), hist[1] + (e[1],)), -1)
        # len == L_max: left-shift + append. (Other lengths are unreachable
        # after Stage 1b pruning, matching the legacy NameError-on-else.)
        return epsilon.get((hist[0][1:] + (e[0],), hist[1][1:] + (e[1],)), -1)

    recursive = False
    while not recursive:
        recursive = True
        states = list(invepsilon.keys())
        for state in states:
            has_split = False
            histories = list(invepsilon[state].keys())

            for e_symbol in e_symbols:
                # 1) pilot x0: first history with a non-null transition.
                x0_to_state = -1
                hist_ind = 0
                for hist_ind, x0 in enumerate(histories):
                    x0_to_state = _trans(x0, e_symbol)
                    if x0_to_state != -1:
                        break
                # LEGACY QUIRK (SPEC §6.3.3): `hist_ind == len(histories) - 1`
                # conflates "a transition was found only at the last history"
                # with "no history transitions" -- if ONLY the last history
                # transitions on this symbol it is silently skipped as a pilot.
                # REPLICATED intentionally (fixing it changes partitions).
                if hist_ind == len(histories) - 1:
                    continue

                # 2) first later history that transitions to a DIFFERENT
                #    non-null state.
                to_split: list[History] = []
                pilot_ind = hist_ind
                for pilot_ind in range(hist_ind + 1, len(histories)):
                    xa = histories[pilot_ind]
                    xa_to_state = _trans(xa, e_symbol)
                    if xa_to_state != x0_to_state and xa_to_state != -1:
                        to_split.append(xa)
                        break
                # LEGACY QUIRK (SPEC §6.3.3): the analogous off-by-one on
                # pilot_ind. Its guarded branch only skips an already-empty
                # `range(pilot_ind+1, len)` scan, so it is effectively inert;
                # replicated for faithfulness.
                if pilot_ind == len(histories) - 1:
                    pass
                else:
                    # 3) collect every later history transitioning like xa.
                    for new_ind in range(pilot_ind + 1, len(histories)):
                        xn = histories[new_ind]
                        if _trans(xn, e_symbol) == xa_to_state:
                            to_split.append(xn)

                if len(to_split) > 0:
                    recursive = False
                    has_split = True
                    morph_by_state[num_states] = [0] * n_e
                    candidates = dict.fromkeys(to_split, True)  # dedup, ordered
                    for candidate in candidates:
                        epsilon[candidate] = num_states
                        invepsilon[num_states][candidate] = True
                        invepsilon[state].pop(candidate)
                        m = [
                            fut[(candidate[0] + (ex,), candidate[1] + (ey,))]
                            for ex, ey in e_symbols
                        ]
                        dst_new = morph_by_state[num_states]
                        dst_old = morph_by_state[state]
                        for k in range(n_e):
                            dst_new[k] += m[k]
                            dst_old[k] -= m[k]
                    num_states += 1
                    break  # restart e_symbol scan
            if has_split:
                break  # restart state scan

    # -- Stage 3b: cleanup. -----------------------------------------------
    for state in list(invepsilon.keys()):
        if len(invepsilon[state]) == 0:
            _remove_state(state, epsilon, invepsilon, morph_by_state)
    _remove_transients(epsilon, invepsilon, morph_by_state, e_symbols, L_max)

    return dict(epsilon), dict(invepsilon), morph_by_state


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------


def build_transducer(
    epsilon: Epsilon,
    morph_by_state: MorphByState,
    input_alphabet: Alphabet,
    output_alphabet: Alphabet,
    L_max: int,
) -> EpsilonTransducer:
    """Assemble an :class:`EpsilonTransducer` from a :func:`run_cssr` partition.

    The sparse integer state labels from :func:`run_cssr` are compacted to
    contiguous ``0 .. n_states - 1`` and then **canonically relabeled** (states
    sorted by their lexicographically smallest history). Canonicalization
    happens only here -- never inside :func:`run_cssr` -- because it is a pure
    renaming that is safe once the order-dependent inference has completed.

    Parameters
    ----------
    epsilon : dict
        ``History -> int state`` from :func:`run_cssr`.
    morph_by_state : dict
        ``int state -> list[int]`` morph counts from :func:`run_cssr`.
    input_alphabet, output_alphabet : Alphabet
        Token alphabets. ``len(input_alphabet) == n_x``,
        ``len(output_alphabet) == n_y``.
    L_max : int
        Maximum history length used during inference.

    Returns
    -------
    EpsilonTransducer
        The canonicalized model.
    """
    old_states = list(morph_by_state.keys())
    old_to_new = {old: new for new, old in enumerate(old_states)}
    n_states = len(old_states)
    n_x = len(input_alphabet)
    n_y = len(output_alphabet)

    new_epsilon: Epsilon = {
        history: old_to_new[state] for history, state in epsilon.items()
    }
    morph_counts = np.array(
        [morph_by_state[old] for old in old_states], dtype=np.int64
    ).reshape(n_states, n_x * n_y)

    model = EpsilonTransducer(
        input_alphabet=input_alphabet,
        output_alphabet=output_alphabet,
        n_states=n_states,
        epsilon=new_epsilon,
        morph_counts=morph_counts,
        L_max=L_max,
    )
    return model.canonicalize()
