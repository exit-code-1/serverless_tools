"""
Action generation: A_balance (top-b slow segs, bottleneck ±1), A_profit (low-yield reclaim).
"""

from typing import Dict, List, Any, Optional, Set

from . import types
from .context import PDGContext
from .incremental_eval import IncrementalEvaluator


def _adjacent_dop(
    context: PDGContext, stage_id: int, current_dop: int, delta: int
) -> Optional[int]:
    """Next/prev DOP in candidate list. delta in {-1, +1}. Returns None if no adjacent."""
    cand = context.candidate_dops(stage_id)
    if not cand:
        return None
    try:
        i = cand.index(current_dop)
    except ValueError:
        return cand[0] if delta > 0 and cand else (cand[-1] if delta < 0 and cand else None)
    ni = i + delta
    if 0 <= ni < len(cand):
        return cand[ni]
    return None


def build_A_balance(
    d: types.DopConfig,
    context: PDGContext,
    evaluator: IncrementalEvaluator,
    b: int = 4,
) -> List[types.Action]:
    """
    A_balance: top-b slowest segs; for each seg's bottleneck stage(s) (max1, ties), enumerate ±1 DOP.
    """
    t_seg_map = evaluator.get_t_seg_map()
    # Map seg_id -> seg for ordering; we need seg objects for get_max1_stages
    seg_to_id = {id(seg): seg for seg in context.all_segments}
    # Exclude virtual root (no stage_ids)
    segs_with_stages = [
        seg for seg in context.all_segments
        if getattr(seg, "get_stage_ids", lambda: [])()
    ]
    if not segs_with_stages:
        return []

    # Sort by t_seg descending (slowest first), take top b
    sorted_segs = sorted(
        segs_with_stages,
        key=lambda s: t_seg_map.get(id(s), 0.0),
        reverse=True,
    )
    top_b = sorted_segs[: min(b, len(sorted_segs))]
    seen: Set[tuple] = set()
    out: List[types.Action] = []

    for seg in top_b:
        bottleneck_stages = evaluator.get_max1_stages(seg)
        for stage_id in bottleneck_stages:
            if stage_id not in d:
                continue
            cur = d[stage_id]
            for delta in (-1, 1):
                adj = _adjacent_dop(context, stage_id, cur, delta)
                if adj is not None and adj != cur:
                    key = (stage_id, adj)
                    if key not in seen:
                        seen.add(key)
                        out.append(
                            types.Action(stage_id=stage_id, delta_dop=delta, tag="balance")
                        )
    return out


def build_A_profit(
    d: types.DopConfig,
    context: PDGContext,
    evaluator: IncrementalEvaluator,
    p: int = 15,
    gap_min: float = 0.05,
    gap_abs_min: float = 1.0,
) -> List[types.Action]:
    """
    A_profit: decrease(s) for stages that are non-bottleneck in all their segs with gap >= gap_min.
    Sort by reclaimable cost (dop * exec_time), take top p. Optional few increase(s) not added here.
    """
    stage_to_segs = context.stage_to_segs
    out: List[types.Action] = []
    # Candidate stages: those for which in every seg containing s, s is not max1 and gap >= gap_min
    candidates: List[tuple] = []  # (stage_id, reclaimable_cost)

    for stage_id in context.stage_ids:
        segs = stage_to_segs.get(stage_id, [])
        if not segs:
            continue
        # Check: in all segs involving s, s is not bottleneck and gap >= gap_min
        is_ok = True
        min_gap = float("inf")
        for seg in segs:
            bottleneck_stages = evaluator.get_max1_stages(seg)
            if stage_id in bottleneck_stages:
                is_ok = False
                break
            contrib = evaluator.get_contrib_for_seg(seg)
            t_s = contrib.get(stage_id, 0.0)
            max1 = max(contrib.values()) if contrib else 0.0
            gap = max1 - t_s
            if gap < gap_abs_min and (max1 <= 0 or (gap / max1) < gap_min):
                is_ok = False
                break
            min_gap = min(min_gap, gap)

        if not is_ok:
            continue
        # Reclaimable cost ~ dop * exec_time (decrease saves cost)
        dop = d.get(stage_id)
        if dop is None:
            continue
        exec_t = context.stage_exec_time(stage_id, dop)
        reclaim = dop * exec_t
        candidates.append((stage_id, reclaim))

    # Sort by reclaim desc, take top p; each yields one decrease action
    candidates.sort(key=lambda x: x[1], reverse=True)
    for stage_id, _ in candidates[:p]:
        cur = d.get(stage_id)
        if cur is None:
            continue
        adj = _adjacent_dop(context, stage_id, cur, -1)
        if adj is not None and adj < cur:
            out.append(types.Action(stage_id=stage_id, delta_dop=-1, tag="profit"))
    return out
