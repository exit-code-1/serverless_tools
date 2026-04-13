"""
Elite pool P: capacity B, init with anchor d0, select_parents (J-min + diversity), evict (J + nn_dist).
"""

from typing import Dict, List, Optional, Tuple, Any
import math

from . import types
from .context import PDGContext
from .incremental_eval import IncrementalEvaluator


def _dop_vector(d: types.DopConfig, stage_ids: List[int]) -> List[int]:
    """Ordered vector of DOPs for stable distance and dedup."""
    return [d.get(s, 0) for s in stage_ids]


def _norm_dist(a: List[int], b: List[int]) -> float:
    """Euclidean distance between two dop vectors."""
    if len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nn_dist(d: types.DopConfig, others: List[types.DopConfig], stage_ids: List[int]) -> float:
    """Min distance from d to any other in others."""
    v = _dop_vector(d, stage_ids)
    best = float("inf")
    for o in others:
        if o == d:
            continue
        best = min(best, _norm_dist(v, _dop_vector(o, stage_ids)))
    return best if best != float("inf") else 0.0


def compute_J(
    d: types.DopConfig,
    L: float,
    C: float,
    all_L: List[float],
    all_C: List[float],
    WL: float,
    WC: float,
) -> float:
    """J = WL * L_norm + WC * C_norm, normalized over (all_L, all_C)."""
    if not all_L or not all_C:
        return WL * L + WC * C
    min_L, max_L = min(all_L), max(all_L)
    min_C, max_C = min(all_C), max(all_C)
    L_norm = (L - min_L) / (max_L - min_L + 1e-12)
    C_norm = (C - min_C) / (max_C - min_C + 1e-12)
    return WL * L_norm + WC * C_norm


def config_key(d: types.DopConfig, stage_ids: List[int]) -> tuple:
    """Stable key for dedup."""
    return tuple((s, d.get(s, 0)) for s in stage_ids)


class ElitePool:
    """Bounded elite pool: solutions with (d, L, C), capacity B."""

    def __init__(
        self,
        context: PDGContext,
        evaluator: IncrementalEvaluator,
        B: int,
        WL: float = 0.5,
        WC: float = 0.5,
    ):
        self._ctx = context
        self._eval = evaluator
        self._B = B
        self._WL = WL
        self._WC = WC
        self._stage_ids = sorted(context.stage_ids)
        self._solutions: List[types.Solution] = []
        self._solution_id: int = 0

    def _next_id(self) -> int:
        self._solution_id += 1
        return self._solution_id

    def init_with_anchor(self, d0: types.DopConfig) -> None:
        """Set pool to single anchor solution d0."""
        L, C = self._eval.set_config(d0)
        sol = types.Solution(d=dict(d0), L=L, C=C, solution_id=self._next_id())
        self._solutions = [sol]

    def add(self, d: types.DopConfig, L: float, C: float, I: float = 0.0) -> types.Solution:
        """Add solution (caller should have already evaluated d). Returns the Solution."""
        sol = types.Solution(d=dict(d), L=L, C=C, I=I, solution_id=self._next_id())
        self._solutions.append(sol)
        return sol

    def select_parents(
        self,
        n_J: int = 2,
        n_div: int = 2,
    ) -> List[types.Solution]:
        """Select n_J by smallest J and n_div by largest nn_dist (diversity)."""
        if not self._solutions:
            return []
        all_L = [s.L for s in self._solutions]
        all_C = [s.C for s in self._solutions]
        J_list = [
            compute_J(s.d, s.L, s.C, all_L, all_C, self._WL, self._WC)
            for s in self._solutions
        ]
        # J smallest
        order_J = sorted(range(len(self._solutions)), key=lambda i: J_list[i])
        parents: List[types.Solution] = [
            self._solutions[i] for i in order_J[:n_J]
        ]
        # diversity: nn_dist relative to current pool
        others = [s.d for s in self._solutions]
        div_list = [
            nn_dist(s.d, [o for j, o in enumerate(others) if o != s.d], self._stage_ids)
            for s in self._solutions
        ]
        order_div = sorted(range(len(self._solutions)), key=lambda i: div_list[i], reverse=True)
        added = set(id(s) for s in parents)
        for i in order_div:
            if len(parents) >= n_J + n_div:
                break
            if id(self._solutions[i]) not in added:
                parents.append(self._solutions[i])
                added.add(id(self._solutions[i]))
        return parents

    def evict(self, R: List[types.Solution]) -> None:
        """
        Replace pool with eviction from R: dedup by config, then B0=floor(0.7*B) by smallest J,
        remaining by nn_dist descending.
        """
        # Dedup by config key
        seen: Dict[tuple, types.Solution] = {}
        for s in R:
            k = config_key(s.d, self._stage_ids)
            if k not in seen or s.L + s.C < seen[k].L + seen[k].C:
                seen[k] = s
        uniq = list(seen.values())
        if len(uniq) <= self._B:
            self._solutions = uniq
            return

        all_L = [s.L for s in uniq]
        all_C = [s.C for s in uniq]
        B0 = max(0, int(math.floor(0.7 * self._B)))
        J_list = [
            compute_J(s.d, s.L, s.C, all_L, all_C, self._WL, self._WC)
            for s in uniq
        ]
        order_J = sorted(range(len(uniq)), key=lambda i: J_list[i])
        kept_idx: set = set()
        for i in order_J[:B0]:
            kept_idx.add(i)
        # Fill rest by nn_dist desc (diversity)
        others_d = [uniq[i].d for i in kept_idx]
        remaining = [i for i in range(len(uniq)) if i not in kept_idx]
        while len(kept_idx) < self._B and remaining:
            best_i = None
            best_dist = -1.0
            for i in remaining:
                d = nn_dist(uniq[i].d, [uniq[j].d for j in kept_idx], self._stage_ids)
                if d > best_dist:
                    best_dist = d
                    best_i = i
            if best_i is None:
                break
            kept_idx.add(best_i)
            remaining.remove(best_i)
        self._solutions = [uniq[i] for i in sorted(kept_idx)]

    @property
    def solutions(self) -> List[types.Solution]:
        return self._solutions

    @property
    def capacity(self) -> int:
        return self._B
