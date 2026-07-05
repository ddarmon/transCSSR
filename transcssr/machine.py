"""Model classes for the token-first ``transcssr`` package.

An :class:`EpsilonTransducer` is a learned causal-state model of an
input/output process ``P(Y_t | X_t, X_past, Y_past)``. An
:class:`EpsilonMachine` is the degenerate output-only case whose input
alphabet is a single dummy token.

Both store their morphs as integer *count* vectors, row-stacked by input
symbol (entry ``n_y * x_ind + y_ind``; see :mod:`transcssr.stats`). Derived
quantities -- emission probabilities, the inverse partition, and the
transition map -- are computed on demand. Serialization uses a versioned JSON
format; DOT export is visualization-only.
"""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable

import numpy as np

from .alphabet import Alphabet

__all__ = ["EpsilonTransducer", "EpsilonMachine", "History"]

# A history is a pair of equal-length index tuples: (input_past, output_past).
History = tuple[tuple[int, ...], tuple[int, ...]]

_MODEL_FORMAT = "transcssr-model-v1"


def _state_label(value: int) -> str:
    """Map a non-negative integer to a Roman-alphabet DOT node name.

    ``0 -> 'A'``, ``25 -> 'Z'``, ``26 -> 'AA'``, ... matching the legacy
    ``numeric_to_alpha`` convention.
    """
    remainder = value % 26
    integer_part = value // 26
    return (integer_part + 1) * string.ascii_uppercase[remainder]


@dataclass
class EpsilonTransducer:
    """A learned epsilon-transducer over token alphabets.

    Parameters
    ----------
    input_alphabet : Alphabet
        Alphabet of input tokens ``X``.
    output_alphabet : Alphabet
        Alphabet of output tokens ``Y``.
    n_states : int
        Number of causal states.
    epsilon : dict
        Mapping ``History -> int state`` for every retained history (lengths
        ``<= L_max``). ``History`` is a pair of equal-length index tuples.
    morph_counts : numpy.ndarray
        Integer array of shape ``(n_states, n_x * n_y)``. Row ``s`` is the
        morph of state ``s``, flat-indexed ``n_y * x_ind + y_ind``.
    L_max : int
        Maximum history length used during inference.
    """

    input_alphabet: Alphabet
    output_alphabet: Alphabet
    n_states: int
    epsilon: dict[History, int]
    morph_counts: np.ndarray
    L_max: int

    def __post_init__(self) -> None:
        self.morph_counts = np.asarray(self.morph_counts)
        expected = (self.n_states, self.n_x * self.n_y)
        if self.morph_counts.shape != expected:
            raise ValueError(
                f"morph_counts must have shape {expected}; got "
                f"{self.morph_counts.shape}."
            )

    # -- basic derived sizes ----------------------------------------------

    @property
    def n_x(self) -> int:
        """Number of input symbols."""
        return len(self.input_alphabet)

    @property
    def n_y(self) -> int:
        """Number of output symbols."""
        return len(self.output_alphabet)

    # -- derived structures -----------------------------------------------

    @property
    def histories_by_state(self) -> dict[int, list[History]]:
        """Inverse of ``epsilon``: state -> list of its histories.

        Histories are listed in ``epsilon`` insertion order.
        """
        inv: dict[int, list[History]] = {s: [] for s in range(self.n_states)}
        for history, state in self.epsilon.items():
            inv.setdefault(state, []).append(history)
        return inv

    def emission_probs(self) -> np.ndarray:
        """Row-normalized emission probabilities.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_states, n_x * n_y)``. Each per-input-symbol
            slice ``[x*n_y : x*n_y + n_y]`` sums to 1 (or is ``nan`` if that
            input row was never observed for the state).
        """
        counts = self.morph_counts.astype(float)
        probs = np.empty_like(counts)
        with np.errstate(divide="ignore", invalid="ignore"):
            for x_ind in range(self.n_x):
                sl = slice(x_ind * self.n_y, x_ind * self.n_y + self.n_y)
                row = counts[:, sl]
                totals = row.sum(axis=1, keepdims=True)
                probs[:, sl] = row / totals
        return probs

    def transitions(self) -> dict[tuple[int, int, int], int]:
        """Transition map ``(state, x_idx, y_idx) -> next_state``.

        Derived per the legacy ``get_transitions`` semantics (SPEC §4.4),
        including the ``need_Lmax`` source-history rule: transitions are read
        preferentially off the length-``(L_max - 1)`` histories (grow),
        falling back to length-``L_max`` histories (shift + append) only for
        states that have no length-``(L_max - 1)`` history.
        """
        inv = self.histories_by_state
        trans: dict[tuple[int, int, int], int] = {}
        for state in range(self.n_states):
            hists = inv.get(state, [])
            need_lmax = not any(len(xh) == self.L_max - 1 for xh, _ in hists)
            for xh, yh in hists:
                if len(xh) == self.L_max:
                    if not need_lmax:
                        continue
                    base_x, base_y = xh[1:], yh[1:]
                else:
                    base_x, base_y = xh, yh
                for x_idx in range(self.n_x):
                    for y_idx in range(self.n_y):
                        key = (base_x + (x_idx,), base_y + (y_idx,))
                        nxt = self.epsilon.get(key, -1)
                        if nxt != -1:
                            trans[(state, x_idx, y_idx)] = nxt
        return trans

    # -- canonicalization --------------------------------------------------

    def canonicalize(self) -> "EpsilonTransducer":
        """Return a copy with states relabeled to a canonical order.

        States are sorted by their lexicographically smallest history and
        relabeled ``0 .. n_states - 1``. This is a pure renaming: the
        partition and morphs are unchanged.
        """
        inv = self.histories_by_state
        order = sorted(
            range(self.n_states),
            key=lambda s: min(inv.get(s, [((), ())])),
        )
        old_to_new = {old: new for new, old in enumerate(order)}
        new_epsilon = {
            history: old_to_new[state] for history, state in self.epsilon.items()
        }
        new_morphs = self.morph_counts[order, :].copy()
        return type(self)._construct(
            input_alphabet=self.input_alphabet,
            output_alphabet=self.output_alphabet,
            n_states=self.n_states,
            epsilon=new_epsilon,
            morph_counts=new_morphs,
            L_max=self.L_max,
        )

    @classmethod
    def _construct(cls, **kwargs) -> "EpsilonTransducer":
        """Build an instance, tolerant of subclass ``__init__`` signatures."""
        if cls is EpsilonTransducer:
            return cls(**kwargs)
        # Subclasses (e.g. EpsilonMachine) drop input_alphabet from __init__.
        obj = EpsilonTransducer.__new__(cls)
        EpsilonTransducer.__init__(obj, **kwargs)
        return obj

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict in the ``transcssr-model-v1`` format.

        Raises
        ------
        TypeError
            If any alphabet token is not JSON-representable.
        """
        in_tokens = list(self.input_alphabet.tokens)
        out_tokens = list(self.output_alphabet.tokens)
        for name, tokens in (("input", in_tokens), ("output", out_tokens)):
            try:
                json.dumps(tokens)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{name} alphabet contains non-JSON-representable tokens "
                    f"{tokens!r}; JSON serialization requires str/int/float/"
                    f"bool/None tokens."
                ) from exc
        epsilon_list = [
            [list(xh), list(yh), int(state)]
            for (xh, yh), state in self.epsilon.items()
        ]
        return {
            "format": _MODEL_FORMAT,
            "input_alphabet": in_tokens,
            "output_alphabet": out_tokens,
            "n_states": int(self.n_states),
            "L_max": int(self.L_max),
            "epsilon": epsilon_list,
            "morph_counts": self.morph_counts.tolist(),
        }

    def to_json(self, path: str | Path) -> None:
        """Write the model to ``path`` as JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "EpsilonTransducer":
        """Reconstruct a model from a ``transcssr-model-v1`` dict."""
        fmt = data.get("format")
        if fmt != _MODEL_FORMAT:
            raise ValueError(
                f"Unsupported model format {fmt!r}; expected {_MODEL_FORMAT!r}."
            )
        input_alphabet = Alphabet(data["input_alphabet"])
        output_alphabet = Alphabet(data["output_alphabet"])
        epsilon: dict[History, int] = {
            (tuple(xh), tuple(yh)): int(state)
            for xh, yh, state in data["epsilon"]
        }
        morph_counts = np.asarray(data["morph_counts"])
        return EpsilonTransducer(
            input_alphabet=input_alphabet,
            output_alphabet=output_alphabet,
            n_states=int(data["n_states"]),
            epsilon=epsilon,
            morph_counts=morph_counts,
            L_max=int(data["L_max"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "EpsilonTransducer":
        """Read a model from a JSON file at ``path``."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    # -- DOT export --------------------------------------------------------

    def to_dot(self, path: str | Path, all_digits: bool = False) -> None:
        """Write a Graphviz ``.dot`` visualization to ``path``.

        Edge labels follow the legacy ``draw_dot_singlearrows`` convention
        ``y|x:prob``, where ``x`` and ``y`` are rendered with ``str(token)``.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path (written verbatim; no extension added).
        all_digits : bool, optional
            If ``True``, print full-precision probabilities; otherwise use
            ``{:.3}``. Default ``False``.
        """
        probs = self.emission_probs()
        header = (
            'digraph  {\nsize = "6,8.5";\nratio = "fill";\nnode\n'
            "[shape = circle];\nnode [fontsize = 24];\nnode [penwidth = 5];\n"
            "edge [fontsize = 24];\nnode [fontname = \"CMU Serif Roman\"];\n"
            "graph [fontname = \"CMU Serif Roman\"];\n"
            "edge [fontname = \"CMU Serif Roman\"];\n"
        )
        lines: list[str] = []

        def emit_label(x_idx: int, y_idx: int) -> str:
            ax = str(self.input_alphabet.token(x_idx))
            ay = str(self.output_alphabet.token(y_idx))
            ptrans = probs[state, self.n_y * x_idx + y_idx]
            if all_digits:
                return f"{ay}|{ax}:{ptrans}\\l"
            return f"{ay}|{ax}:{ptrans:.3}\\l"

        if self.n_states == 1:
            state = 0
            for y_idx in range(self.n_y):
                for x_idx in range(self.n_x):
                    ptrans = probs[state, self.n_y * x_idx + y_idx]
                    if not np.isnan(ptrans) and ptrans > 0:
                        lines.append(f'A -> A [label = "{emit_label(x_idx, y_idx)}"];\n')
        else:
            trans = self.transitions()
            # Accumulate edge labels grouped by (from_state, to_state).
            edge_labels: dict[tuple[int, int], str] = {}
            for state in range(self.n_states):
                for y_idx in range(self.n_y):
                    for x_idx in range(self.n_x):
                        to_state = trans.get((state, x_idx, y_idx), -1)
                        if to_state == -1:
                            continue
                        ptrans = probs[state, self.n_y * x_idx + y_idx]
                        if np.isnan(ptrans):
                            continue
                        edge_labels[(state, to_state)] = (
                            edge_labels.get((state, to_state), "")
                            + emit_label(x_idx, y_idx)
                        )
            for from_state in range(self.n_states):
                for to_state in range(self.n_states):
                    label = edge_labels.get((from_state, to_state))
                    if label:
                        lines.append(
                            f'{_state_label(from_state)} -> '
                            f'{_state_label(to_state)} [label = "{label}"];\n'
                        )

        dot_text = header + "".join(lines) + "}"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(dot_text)


class EpsilonMachine(EpsilonTransducer):
    """An epsilon-machine: an output-only causal-state model.

    The input alphabet is a single dummy token ``0`` (legacy convention
    ``stringX = '0' * len(stringY)``), so all input conditioning collapses.

    Parameters
    ----------
    output_alphabet : Alphabet
        Alphabet of output tokens ``Y``.
    n_states : int
        Number of causal states.
    epsilon : dict
        Mapping ``History -> int state``. Input histories are tuples of the
        single dummy index ``0``.
    morph_counts : numpy.ndarray
        Integer array of shape ``(n_states, n_y)`` (since ``n_x == 1``).
    L_max : int
        Maximum history length used during inference.
    input_alphabet : Alphabet, optional
        Overrides the default single-dummy-token alphabet ``Alphabet([0])``.
    """

    def __init__(
        self,
        output_alphabet: Alphabet,
        n_states: int,
        epsilon: dict[History, int],
        morph_counts: np.ndarray,
        L_max: int,
        input_alphabet: Alphabet | None = None,
    ) -> None:
        if input_alphabet is None:
            input_alphabet = Alphabet([0])
        super().__init__(
            input_alphabet=input_alphabet,
            output_alphabet=output_alphabet,
            n_states=n_states,
            epsilon=epsilon,
            morph_counts=morph_counts,
            L_max=L_max,
        )

    @property
    def alphabet(self) -> Alphabet:
        """The output alphabet (output-only convenience accessor)."""
        return self.output_alphabet
