"""
Main loop: Parents -> A(d) -> Top-K children -> Evict -> Round log.
Iterate T rounds or until no new solution enters P.
"""

from typing import List, Tuple, Any, Optional

from . import types
from .context import PDGContext
from .incremental_eval import IncrementalEvaluator
from .actions import build_A_balance, build_A_profit
from .actions import _adjacent_dop
from .pool import ElitePool


def run_algorithm(
    context: PDGContext,
    evaluator: IncrementalEvaluator,
    pool: ElitePool,
    params: types.AlgoParams,
) -> Tuple[List[types.Solution], List[types.RoundLog]]:
    """
    Run competitive Top-K expansion + bounded elite pool for T rounds (or until no new in P).
    Returns (final pool solutions, rounds_log).
    """
    rounds_log: List[types.RoundLog] = []
    B = params.B
    T = params.T
    K = params.K
    b = params.b
    p = params.p
    WL = params.WL
    WC = params.WC
    lambda_I = params.lambda_I
    gap_min = params.gap_min
    n_J = params.n_parents_J
    n_div = params.n_parents_div

    for r in range(T):
        parents = pool.select_parents(n_J=n_J, n_div=n_div)
        action_results_per_parent: List[List[Tuple[types.Action, float, float, float]]] = []

        Q: List[types.Solution] = []
        n_new_this_round = 0

        for parent in parents:
            d = parent.d
            evaluator.set_config(d)
            L_base = evaluator.get_L()
            C_base = evaluator.get_C()
            if L_base <= 0:
                L_base = 1e-12
            if C_base <= 0:
                C_base = 1e-12

            A_bal = build_A_balance(d, context, evaluator, b=b)
            A_prof = build_A_profit(d, context, evaluator, p=p, gap_min=gap_min)
            seen_new: set = set()
            cand: List[Tuple[types.Action, float, float, float, float, types.DopConfig]] = []

            for a in A_bal + A_prof:
                new_dop = _adjacent_dop(context, a.stage_id, d.get(a.stage_id), a.delta_dop)
                if new_dop is None:
                    continue
                key = (a.stage_id, new_dop)
                if key in seen_new:
                    continue
                seen_new.add(key)
                L_new, C_new, I, _ = evaluator.apply_action(a.stage_id, new_dop)
                d_new = dict(d)
                d_new[a.stage_id] = new_dop
                delta_L = L_new - L_base
                delta_C = C_new - C_base
                score = (
                    WL * (delta_L / L_base)
                    + WC * (delta_C / C_base)
                    + lambda_I * (I / L_base)
                )
                cand.append((a, score, L_new, C_new, I, d_new))

            cand.sort(key=lambda x: x[1])
            this_parent_results: List[Tuple[types.Action, float, float, float]] = []
            for (a, score, L_new, C_new, I, d_new) in cand[:K]:
                sol = pool.add(d_new, L_new, C_new, I=I)
                Q.append(sol)
                n_new_this_round += 1
                this_parent_results.append((a, score, L_new, C_new))
            action_results_per_parent.append(this_parent_results)

        # Evict: R = P ∪ Q, then evict to B
        R = list(pool.solutions)
        pool.evict(R)
        P_sols = pool.solutions
        L_min = min(s.L for s in P_sols) if P_sols else 0.0
        C_min = min(s.C for s in P_sols) if P_sols else 0.0

        log = types.RoundLog(
            round_index=r,
            parents=[(p, getattr(p, "solution_id", i)) for i, p in enumerate(parents)],
            action_results=action_results_per_parent,
            pool_size=len(P_sols),
            L_min=L_min,
            C_min=C_min,
            n_new_in_pool=n_new_this_round,
        )
        rounds_log.append(log)

        # Stop if no new solution entered P (evict might drop some; "no new" = none of Q survived evict)
        # Simplified: stop when n_new_this_round == 0 and we didn't add any that survived
        if r > 0 and n_new_this_round == 0 and len(Q) == 0:
            break

    return (pool.solutions, rounds_log)
