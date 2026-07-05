"""Token-first reimplementation of the transCSSR library.

Public API: alphabet + model classes, the two inference entry points
(``infer_transducer`` / ``infer_machine``), and filtering/prediction
(``filter_and_predict`` / ``FilterResult``).
"""

from __future__ import annotations

from .alphabet import Alphabet
from .api import infer_machine, infer_transducer
from .countstore import CountStore, CounterCountStore, SuffixArrayCountStore
from .machine import EpsilonMachine, EpsilonTransducer
from .msp import (
    LogLoss,
    MSPFilterResult,
    MixedModel,
    log_loss,
    machine_transition_matrix,
    mixed_transition_matrix,
    msp_filter,
    msp_filter_machine,
    stationary_distribution,
)
from .predict import FilterResult, filter_and_predict

__all__ = [
    "Alphabet",
    "CountStore",
    "CounterCountStore",
    "SuffixArrayCountStore",
    "EpsilonTransducer",
    "EpsilonMachine",
    "infer_transducer",
    "infer_machine",
    "filter_and_predict",
    "FilterResult",
    "machine_transition_matrix",
    "stationary_distribution",
    "mixed_transition_matrix",
    "MixedModel",
    "MSPFilterResult",
    "msp_filter",
    "msp_filter_machine",
    "LogLoss",
    "log_loss",
]
