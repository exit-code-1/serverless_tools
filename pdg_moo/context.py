"""
PDG context: build top_segment, all_segments, stage_to_segs, stage_exec_time, candidate_dops.
Uses core.pdg_builder.convert_stage_dag_to_pdg and optimization.moo_dop_optimizer for exec-time curve.
"""

import warnings
from typing import Dict, List, Set, Any, Optional
from . import types


def _collect_all_segments(top_segment) -> List[Any]:
    """DFS from top_segment to collect all segments (including virtual root)."""
    seen: Set[int] = set()
    out: List[Any] = []

    def dfs(seg):
        if seg is None or id(seg) in seen:
            return
        seen.add(id(seg))
        out.append(seg)
        for u in getattr(seg, "upstream_segments", []):
            dfs(u)

    dfs(top_segment)
    return out


def _compute_exec_time_from_params(pred_params: Any, dop: int, is_parallel: bool, base_time: float) -> float:
    """Compute execution time at given DOP. Formula: (b/dop^a) + (d*dop^c) + e. Guards against overflow/NaN."""
    if not is_parallel or pred_params is None:
        return base_time
    dop = max(1, min(int(dop), 2000))  # avoid 0 (division/0**neg) and huge dop (overflow in dop**c)
    try:
        a, b, c, d, e = pred_params[0], pred_params[1], pred_params[2], pred_params[3], pred_params[4]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t = (b / (dop ** a)) + (d * (dop ** c)) + e
        if not (t == t and abs(t) != float("inf")):  # NaN or inf
            return base_time if base_time > 0 else 100.0
        return max(float(t), 0.1)
    except (OverflowError, ZeroDivisionError, ValueError):
        return base_time if base_time > 0 else 100.0


def _compute_marginal_gain_dop(
    pred_params: Any,
    is_parallel: bool,
    base_time: float,
    candidate_dops: List[int],
    rel_improve_threshold: float = 0.03,
) -> int:
    """Return DOP where relative improvement becomes small (marginal-effect point)."""
    if not candidate_dops:
        return 8
    cds = sorted(candidate_dops)
    min_dop, max_dop = min(cds), max(cds)
    grid = [d for d in range(min_dop, max_dop + 1, 4) if d % 2 == 0]
    if not grid:
        grid = [d for d in range(min_dop, max_dop + 1) if d % 2 == 0] or cds
    times = [
        max(_compute_exec_time_from_params(pred_params, d, is_parallel, base_time), 1e-6)
        for d in grid
    ]
    best_dop = grid[-1]
    for i in range(1, len(grid)):
        prev_t, curr_t = times[i - 1], times[i]
        rel_improve = (prev_t - curr_t) / prev_t if prev_t > 0 else 0.0
        if rel_improve <= rel_improve_threshold:
            best_dop = grid[i]
            break
    # clamp to even
    v = max(min_dop, min(best_dop, max_dop))
    if v % 2 == 1:
        v = v - 1 if v > min_dop else v + 1
    return v


class PDGContext:
    """
    PDG context built from (thread_blocks, all_nodes).
    Provides: top_segment, all_segments, stage_to_segs, stage_exec_time, candidate_dops, seg_ratio(s, seg).
    """

    def __init__(self, thread_blocks: Dict[int, Any], all_nodes: List[Any]):
        """
        thread_blocks: dict stage_id -> ThreadBlock (has .nodes, .pred_dop_exec_time, etc.)
        all_nodes: list of PlanNode for the query.
        """
        from core.pdg_builder import convert_stage_dag_to_pdg

        self._thread_blocks = dict(thread_blocks)
        self._all_nodes = list(all_nodes)
        self._top_segment = convert_stage_dag_to_pdg(self._thread_blocks, self._all_nodes)
        self._all_segments = _collect_all_segments(self._top_segment)

        # stage_id -> list of segments that contain this stage
        self._stage_to_segs: Dict[int, List[Any]] = {}
        for seg in self._all_segments:
            for sid in getattr(seg, "get_stage_ids", lambda: [])():
                self._stage_to_segs.setdefault(sid, []).append(seg)

        # per-stage exec-time curve params (pred_params, is_parallel, base_time)
        self._stage_params: Dict[int, tuple] = {}
        for tid, tb in self._thread_blocks.items():
            pred_params = None
            is_parallel = True
            base_time = getattr(tb, "thread_execution_time", 0.0) or 0.0
            for node in getattr(tb, "nodes", []):
                if getattr(node, "is_parallel", True) and getattr(node, "pred_params", None) is not None:
                    pred_params = node.pred_params
                    break
            if all(not getattr(n, "is_parallel", True) for n in getattr(tb, "nodes", [])):
                is_parallel = False
            self._stage_params[tid] = (pred_params, is_parallel, base_time)

        # per-stage candidate DOPs (ordered)
        self._candidate_dops: Dict[int, List[int]] = {}
        for tid, tb in self._thread_blocks.items():
            cand = getattr(tb, "candidate_optimal_dops", None) or getattr(tb, "candidate_dops", None)
            if cand is not None:
                self._candidate_dops[tid] = sorted(cand) if isinstance(cand, (set, list)) else sorted(list(cand))
            else:
                keys = list(getattr(tb, "pred_dop_exec_time", {}).keys())
                self._candidate_dops[tid] = sorted(keys) if keys else [8, 16, 32, 64]

    @property
    def top_segment(self):
        return self._top_segment

    @property
    def all_segments(self) -> List[Any]:
        return self._all_segments

    @property
    def stage_to_segs(self) -> Dict[int, List[Any]]:
        return self._stage_to_segs

    @property
    def stage_ids(self) -> List[int]:
        return sorted(self._thread_blocks.keys())

    def stage_exec_time(self, stage_id: int, dop: int) -> float:
        """Execution time of stage at given DOP."""
        if stage_id not in self._stage_params:
            return 0.0
        pred_params, is_parallel, base_time = self._stage_params[stage_id]
        return _compute_exec_time_from_params(pred_params, dop, is_parallel, base_time)

    def candidate_dops(self, stage_id: int) -> List[int]:
        """Ordered list of candidate DOPs for this stage."""
        return self._candidate_dops.get(stage_id, [8, 16, 32, 64])

    def seg_ratio(self, stage_id: int, seg: Any) -> float:
        """Fraction of stage's nodes in this segment. t_{s,seg} = stage_exec_time(s,d) * seg_ratio(s,seg)."""
        dop_info = getattr(seg, "dop_info", {}) or {}
        if stage_id not in dop_info:
            return 0.0
        nodes_in_seg = dop_info[stage_id].get("nodes", [])
        tb = self._thread_blocks.get(stage_id)
        total_nodes = len(getattr(tb, "nodes", [])) if tb else 0
        if total_nodes == 0:
            return 1.0 if nodes_in_seg else 0.0
        return len(nodes_in_seg) / total_nodes

    def anchor_d0(self, rel_improve_threshold: float = 0.03) -> types.DopConfig:
        """Anchor solution: marginal-effect DOP per stage."""
        d: types.DopConfig = {}
        for tid in self.stage_ids:
            pred_params, is_parallel, base_time = self._stage_params[tid]
            cand = self.candidate_dops(tid)
            dop = _compute_marginal_gain_dop(
                pred_params, is_parallel, base_time, cand, rel_improve_threshold
            )
            d[tid] = dop
        return d
