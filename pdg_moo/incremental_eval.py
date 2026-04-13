"""
Incremental evaluator: set_config(d), apply_action(stage_id, new_dop).
Only traverses segs affected by the changed stage; uses max1/max2 for fast t_seg update.
"""

from typing import Dict, List, Any, Optional, Tuple

from . import types


def eval_segment_with_t_seg(top_segment: Any, t_seg_map: Dict[int, float]) -> float:
    """
    Recursive completion time using t_seg_map[id(seg)] instead of segment_latency(seg).
    Same semantics as eval_segment: no upstream -> t_seg; else lup + t_seg.
    """
    def eval_rec(seg: Any) -> float:
        sid = id(seg)
        t_seg = t_seg_map.get(sid, 0.0)
        upstream = getattr(seg, "upstream_segments", []) or []
        if not upstream:
            return t_seg
        lup = max(eval_rec(u) for u in upstream)
        return lup + t_seg

    return eval_rec(top_segment)


class IncrementalEvaluator:
    """
    Maintains current d, t_seg map, per-seg contributions and max1/max2.
    set_config(d): full recompute of L, C, t_seg.
    apply_action(stage_id, new_dop): only update affected segs, return (L, C, I, delta_per_seg).
    """

    def __init__(self, context: Any):
        self._ctx = context
        self._d: types.DopConfig = {}
        # t_seg_map: id(seg) -> t_seg
        self._t_seg_map: Dict[int, float] = {}
        # per-seg: id(seg) -> {stage_id: t_{s,seg}} for cross-stage; for inner-stage single key
        self._contrib: Dict[int, Dict[int, float]] = {}
        self._L = 0.0
        self._C = 0.0

    def _seg_t_from_contrib(self, seg: Any, contrib: Dict[int, float]) -> float:
        is_inner = getattr(seg, "is_inner_stage", True)
        if is_inner and contrib:
            return next(iter(contrib.values()))
        return max(contrib.values()) if contrib else 0.0

    def set_config(self, d: types.DopConfig) -> Tuple[float, float]:
        """Full recompute of L, C and internal t_seg/contrib. Returns (L, C)."""
        self._d = dict(d)
        ctx = self._ctx
        self._t_seg_map.clear()
        self._contrib.clear()

        for seg in ctx.all_segments:
            sid = id(seg)
            stage_ids = getattr(seg, "get_stage_ids", lambda: [])()
            contrib: Dict[int, float] = {}
            for s in stage_ids:
                if s in d:
                    t_s = ctx.stage_exec_time(s, d[s]) * ctx.seg_ratio(s, seg)
                    contrib[s] = t_s
            self._contrib[sid] = contrib
            self._t_seg_map[sid] = self._seg_t_from_contrib(seg, contrib)

        self._L = eval_segment_with_t_seg(ctx.top_segment, self._t_seg_map)
        self._C = sum(
            ctx.stage_exec_time(s, dop) * dop
            for s, dop in self._d.items()
        )
        return (self._L, self._C)

    def get_L(self) -> float:
        return self._L

    def get_C(self) -> float:
        return self._C

    def get_t_seg_map(self) -> Dict[int, float]:
        """Snapshot of current t_seg (id(seg) -> t_seg)."""
        return dict(self._t_seg_map)

    def get_t_seg_for_seg(self, seg: Any) -> float:
        return self._t_seg_map.get(id(seg), 0.0)

    def get_max1_stages(self, seg: Any) -> List[int]:
        """Bottleneck stages (argmax of t_{s,seg}); may be ties."""
        sid = id(seg)
        contrib = self._contrib.get(sid, {})
        if not contrib:
            return []
        mx = max(contrib.values())
        return [s for s, t in contrib.items() if t >= mx - 1e-9]

    def get_contrib_for_seg(self, seg: Any) -> Dict[int, float]:
        """Per-stage contribution t_{s,seg} for this segment."""
        return dict(self._contrib.get(id(seg), {}))

    def apply_action(
        self, stage_id: int, new_dop: int
    ) -> Tuple[float, float, float, Dict[Any, float]]:
        """
        Apply changing stage_id to new_dop. Only updates segs in stage_to_segs[stage_id].
        Returns (L_new, C_new, I, delta_per_seg) where delta_per_seg maps seg -> Delta t_seg.
        """
        ctx = self._ctx
        old_dop = self._d.get(stage_id)
        if old_dop is None:
            self._d[stage_id] = new_dop
            return self.set_config(self._d)[0], self.set_config(self._d)[1], 0.0, {}

        affected = ctx.stage_to_segs.get(stage_id, [])
        delta_per_seg: Dict[Any, float] = {}
        I = 0.0

        for seg in affected:
            sid = id(seg)
            old_t = self._t_seg_map.get(sid, 0.0)
            contrib = dict(self._contrib.get(sid, {}))
            contrib[stage_id] = ctx.stage_exec_time(stage_id, new_dop) * ctx.seg_ratio(stage_id, seg)
            self._contrib[sid] = contrib
            new_t = self._seg_t_from_contrib(seg, contrib)
            self._t_seg_map[sid] = new_t
            delta = new_t - old_t
            delta_per_seg[seg] = delta
            I += max(0.0, delta)

        self._d[stage_id] = new_dop
        L_new = eval_segment_with_t_seg(ctx.top_segment, self._t_seg_map)
        old_exec = ctx.stage_exec_time(stage_id, old_dop)
        new_exec = ctx.stage_exec_time(stage_id, new_dop)
        C_new = self._C - (old_dop * old_exec) + (new_dop * new_exec)
        self._L = L_new
        self._C = C_new
        return (L_new, C_new, I, delta_per_seg)
