"""
Entry point: run_pdg_moo(thread_blocks, all_nodes, WL, WC, B, T, K, ...).
Creates context, evaluator, pool, runs algorithm, returns (P, rounds_log).
"""

from typing import Dict, List, Any, Tuple, Optional

from . import types
from .context import PDGContext
from .incremental_eval import IncrementalEvaluator
from .pool import ElitePool
from .algorithm import run_algorithm


def run_pdg_moo(
    thread_blocks: Dict[int, Any],
    all_nodes: List[Any],
    WL: float = 0.5,
    WC: float = 0.5,
    B: int = 20,
    T: int = 20,
    K: int = 5,
    b: int = 4,
    p: int = 15,
    lambda_I: float = 0.1,
    gap_min: float = 0.05,
    n_parents_J: int = 2,
    n_parents_div: int = 2,
    rel_improve_threshold: float = 0.03,
    **kwargs: Any,
) -> Tuple[List[types.Solution], List[types.RoundLog]]:
    """
    Run PDG-MOO: competitive Top-K expansion + bounded elite pool.

    Args:
        thread_blocks: stage_id -> ThreadBlock (has .nodes, .pred_dop_exec_time, etc.)
        all_nodes: list of PlanNode for the query
        WL, WC: weights for L and C (WL + WC = 1)
        B: elite pool capacity
        T: max rounds (or until no new solution enters P)
        K: top-K actions per parent to expand
        b: top-b slowest segs for A_balance
        p: top-p stages for A_profit decrease
        lambda_I: penalty weight for I/L in score
        gap_min: min gap for A_profit non-bottleneck filter
        n_parents_J, n_parents_div: number of parents by J and by diversity
        rel_improve_threshold: for marginal-gain anchor d0

    Returns:
        (pool solutions, rounds_log)
    """
    params = types.AlgoParams(
        WL=WL,
        WC=WC,
        lambda_I=lambda_I,
        B=B,
        T=T,
        K=K,
        b=b,
        p=p,
        gap_min=gap_min,
        n_parents_J=n_parents_J,
        n_parents_div=n_parents_div,
        rel_improve_threshold=rel_improve_threshold,
    )

    context = PDGContext(thread_blocks, all_nodes)
    evaluator = IncrementalEvaluator(context)
    pool = ElitePool(context, evaluator, B=B, WL=WL, WC=WC)

    d0 = context.anchor_d0(rel_improve_threshold=params.rel_improve_threshold)
    pool.init_with_anchor(d0)

    return run_algorithm(context, evaluator, pool, params)
