"""Token-first reimplementation of the transCSSR library.

Public API: alphabet + model classes, the two inference entry points
(``infer_transducer`` / ``infer_machine``), and filtering/prediction
(``filter_and_predict`` / ``FilterResult``).
"""

from __future__ import annotations

from .alphabet import Alphabet
from .api import infer_machine, infer_transducer
from .machine import EpsilonMachine, EpsilonTransducer
from .predict import FilterResult, filter_and_predict

__all__ = [
    "Alphabet",
    "EpsilonTransducer",
    "EpsilonMachine",
    "infer_transducer",
    "infer_machine",
    "filter_and_predict",
    "FilterResult",
]
