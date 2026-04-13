"""
PDG-MOO: competitive Top-K expansion + bounded elite pool for stage-level DOP allocation.
"""

from . import types
from .types import DopConfig, Action, Solution, RoundLog, AlgoParams
from .run import run_pdg_moo
from .context import PDGContext
from .incremental_eval import IncrementalEvaluator
from .actions import build_A_balance, build_A_profit
from .pool import ElitePool

__all__ = [
    "run_pdg_moo",
    "DopConfig",
    "Action",
    "Solution",
    "RoundLog",
    "AlgoParams",
    "PDGContext",
    "IncrementalEvaluator",
    "build_A_balance",
    "build_A_profit",
    "ElitePool",
    "types",
]
