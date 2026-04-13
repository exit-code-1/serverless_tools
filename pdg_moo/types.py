"""
Type definitions for PDG-MOO: DopConfig, Action, Solution, RoundLog, AlgoParams.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple


# DOP configuration: stage_id -> dop
DopConfig = Dict[int, int]


@dataclass(frozen=True)
class Action:
    """A single action: change one stage's DOP by one step (delta_dop = +1 or -1)."""
    stage_id: int
    delta_dop: int  # +1 or -1 (adjacent slot in candidate list)
    # Optional tag for logging: "balance" or "profit"
    tag: str = ""


@dataclass
class Solution:
    """A solution with DOP config and evaluated objectives."""
    d: DopConfig
    L: float = 0.0   # total query latency
    C: float = 0.0  # total cost (sum over stages: dop * exec_time)
    # Optional: interference I = sum max(0, Delta t_seg) for last action
    I: float = 0.0
    # Unique id for logging (e.g. generation index or parent id)
    solution_id: Optional[int] = None


@dataclass
class RoundLog:
    """Per-round log: parents, actions with scores, Evict summary."""
    round_index: int
    parents: List[Tuple[Solution, int]]  # (solution, parent_rank) or (solution, id)
    # For each parent: list of (action, score, L, C) for evaluated actions
    action_results: List[List[Tuple[Action, float, float, float]]]
    # After evict
    pool_size: int
    L_min: float
    C_min: float
    n_new_in_pool: int = 0  # number of new solutions that entered P this round


@dataclass
class AlgoParams:
    """Algorithm hyper-parameters."""
    WL: float = 0.5      # weight for latency (WL + WC = 1)
    WC: float = 0.5      # weight for cost
    lambda_I: float = 0.1  # penalty weight for interference I/L
    B: int = 20          # elite pool capacity
    T: int = 20         # max rounds (or until no new solution enters P)
    K: int = 5          # top-K actions per parent to generate children
    b: int = 4          # top-b slowest segs for A_balance (3~5)
    p: int = 15         # top-p stages for A_profit decrease (10~30)
    gap_min: float = 0.05  # min gap (ratio or abs) for A_profit "non-bottleneck" filter
    n_parents_J: int = 2   # number of parents by smallest J
    n_parents_div: int = 2  # number of parents by diversity (nn_dist)
    rel_improve_threshold: float = 0.03  # for marginal-gain anchor d0
