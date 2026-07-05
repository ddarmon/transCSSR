"""Token-first reimplementation of the transCSSR library.

Public API (foundation modules). Inference entry points
(``infer_transducer`` / ``infer_machine``) are re-exported once implemented;
they are guarded here so the package imports cleanly before they exist.
"""

from __future__ import annotations

from .alphabet import Alphabet
from .machine import EpsilonMachine, EpsilonTransducer

__all__ = ["Alphabet", "EpsilonTransducer", "EpsilonMachine"]

try:  # pragma: no cover - optional until cssr.py lands
    from .cssr import infer_machine, infer_transducer  # type: ignore
except ImportError:
    pass
else:
    __all__ += ["infer_machine", "infer_transducer"]
