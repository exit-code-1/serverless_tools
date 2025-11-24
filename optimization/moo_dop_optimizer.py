"""
Multi-Objective Optimization for Pipeline DOP Selection with Flow Rate Matching
Using NSGA-II to optimize thread block DOP configurations considering:
1. Minimize total query execution time (latency)
2. Minimize total resource cost (DOP * latency)
3. Maximize pipeline flow rate matching quality
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
import math

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.core.sampling import Sampling
    from pymoo.core.repair import Repair
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("Warning: pymoo not available. Install with: pip install pymoo")


@dataclass
class ThreadBlockInfo:
    """Thread block information for MOO optimization"""
    thread_id: int
    candidate_dops: List[int]  # All candidate DOPs in the interval
    pred_dop_exec_time: Dict[int, float]  # DOP -> predicted execution time
    blocking_interval: Dict[int, float]  # DOP -> blocking time
    child_thread_ids: List[int]  # IDs of child thread blocks
    parent_thread_id: Optional[int]  # ID of parent thread block (None if root)
    pred_params: Optional[Any] = None  # Curve parameters for computing execution time at any DOP
    is_parallel: bool = True  # Whether the thread block supports parallel execution
    base_exec_time: float = 0.0  # Base execution time for non-parallel blocks


def compute_exec_time_from_params(pred_params: Any, dop: int, is_parallel: bool, base_time: float) -> float:
    """
    Compute execution time at given DOP using curve parameters
    
    Formula: exec_time = (b / dop^a) + (d * dop^c) + e
    where pred_params = [a, b, c, d, e]
    """
    if not is_parallel or pred_params is None:
        # Non-parallel nodes have constant execution time
        return base_time
    
    try:
        # Curve formula: exec_time = (b / dop^a) + (d * dop^c) + e
        a, b, c, d, e = pred_params[0], pred_params[1], pred_params[2], pred_params[3], pred_params[4]
        exec_time = (b / (dop ** a)) + (d * (dop ** c)) + e
        return max(exec_time, 0.1)  # Ensure positive value
    except Exception:
        # Fallback
        return base_time if base_time > 0 else 100.0


def calculate_flow_rate_mismatch(thread_blocks: List[ThreadBlockInfo], 
                                  dop_config: Dict[int, int]) -> float:
    """
    Calculate total pipeline flow rate mismatch penalty
    
    For each parent-child pair, compute:
    mismatch = |parent_exec_time - child_upload_time|
    
    Lower mismatch = better pipeline synchronization
    """
    total_mismatch = 0.0
    
    # Build parent->children mapping
    parent_children_map = {}
    for tb in thread_blocks:
        if tb.child_thread_ids:
            parent_children_map[tb.thread_id] = tb.child_thread_ids
    
    # Calculate mismatch for each parent-child pair
    for parent_tb in thread_blocks:
        parent_id = parent_tb.thread_id
        parent_dop = dop_config[parent_id]
        
        # Use curve parameters to compute parent execution time
        parent_exec_time = compute_exec_time_from_params(
            pred_params=parent_tb.pred_params,
            dop=parent_dop,
            is_parallel=parent_tb.is_parallel,
            base_time=parent_tb.base_exec_time
        )
        
        if parent_id not in parent_children_map:
            continue
            
        for child_id in parent_children_map[parent_id]:
            # Find child thread block
            child_tb = next((tb for tb in thread_blocks if tb.thread_id == child_id), None)
            if child_tb is None:
                continue
            
            child_dop = dop_config[child_id]
            
            # Use curve parameters to compute child execution time
            child_exec_time = compute_exec_time_from_params(
                pred_params=child_tb.pred_params,
                dop=child_dop,
                is_parallel=child_tb.is_parallel,
                base_time=child_tb.base_exec_time
            )
            child_blocking = child_tb.blocking_interval.get(child_dop, 0)
            child_upload_time = child_exec_time - child_blocking
            
            # Calculate mismatch (absolute difference)
            mismatch = abs(parent_exec_time - child_upload_time)
            total_mismatch += mismatch
    
    return total_mismatch


def calculate_query_execution_time_simple(thread_blocks: List[ThreadBlockInfo],
                                         dop_config: Dict[int, int]) -> float:
    """
    Calculate total query execution time given DOP configuration
    This is a simplified version that sums up execution times
    """
    # Build thread_id -> ThreadBlockInfo mapping
    tb_map = {tb.thread_id: tb for tb in thread_blocks}
    
    # Build parent->children mapping
    children_map = {}
    for tb in thread_blocks:
        if tb.child_thread_ids:
            children_map[tb.thread_id] = tb.child_thread_ids
    
    # Find root thread blocks (those with no parent)
    thread_ids = set(tb.thread_id for tb in thread_blocks)
    child_ids = set()
    for children in children_map.values():
        child_ids.update(children)
    root_ids = thread_ids - child_ids
    
    # Calculate execution time recursively from roots
    memo = {}
    
    def calc_time(thread_id: int) -> float:
        if thread_id in memo:
            return memo[thread_id]
        
        tb = tb_map[thread_id]
        dop = dop_config[thread_id]
        
        # Compute execution time using curve parameters (supports any DOP)
        exec_time = compute_exec_time_from_params(
            pred_params=tb.pred_params,
            dop=dop,
            is_parallel=tb.is_parallel,
            base_time=tb.base_exec_time
        )
        
        # If has children, consider their max completion time
        if thread_id in children_map:
            child_times = [calc_time(cid) for cid in children_map[thread_id]]
            max_child_time = max(child_times) if child_times else 0
            total_time = exec_time + max_child_time
        else:
            total_time = exec_time
        
        memo[thread_id] = total_time
        return total_time
    
    # Return max time among all roots
    if root_ids:
        return max(calc_time(rid) for rid in root_ids)
    else:
        return 0.0


def _ensure_even_dop(value: int, min_dop: int, max_dop: int) -> int:
    """Clamp value to [min_dop, max_dop] and make it even if possible."""
    v = max(min_dop, min(int(value), max_dop))
    if v % 2 == 1:
        if v > min_dop:
            v = v - 1
        elif v < max_dop:
            v = v + 1
    return v


def compute_marginal_gain_dop(tb: ThreadBlockInfo,
                              rel_improve_threshold: float = 0.03,
                              exec_time_cache: Optional[Dict[int, Dict[int, float]]] = None) -> int:
    """
    Find the marginal-gain DOP where the relative improvement becomes small.
    Uses the exec-time curve on even DOPs within the candidate range.
    Optimized: can use exec_time_cache to avoid repeated computation.
    """
    candidate_dops = sorted(tb.candidate_dops) if tb.candidate_dops else [8, 16, 32, 64]
    min_dop = min(candidate_dops)
    max_dop = max(candidate_dops)
    # Use sparse grid for speed: sample every 4 DOPs instead of every 2
    grid = [d for d in range(min_dop, max_dop + 1, 4) if d % 2 == 0]
    if not grid:
        grid = [d for d in range(min_dop, max_dop + 1) if d % 2 == 0] or candidate_dops
    # Evaluate times - use cache if available
    times = []
    for d in grid:
        if exec_time_cache and tb.thread_id in exec_time_cache and d in exec_time_cache[tb.thread_id]:
            t = exec_time_cache[tb.thread_id][d]
        else:
            t = compute_exec_time_from_params(tb.pred_params, d, tb.is_parallel, tb.base_exec_time)
        times.append(max(t, 1e-6))
    # Relative improvement compared to previous point
    best_dop = grid[-1]
    for i in range(1, len(grid)):
        prev_t, curr_t = times[i - 1], times[i]
        rel_improve = (prev_t - curr_t) / prev_t if prev_t > 0 else 0.0
        if rel_improve <= rel_improve_threshold:
            best_dop = grid[i]
            break
    return _ensure_even_dop(best_dop, min_dop, max_dop)


def compute_throughput_match_dop(parent_tb: ThreadBlockInfo,
                                 children: List[ThreadBlockInfo]) -> int:
    """
    Find the parent DOP that best matches downstream consumption rates.
    Heuristic: for each candidate parent DOP, compute sum of absolute mismatch
    to each child's upload time evaluated at the child's mid-range DOP.
    """
    if not children:
        # No downstream dependency; default to mid-range or current best
        cds = parent_tb.candidate_dops if parent_tb.candidate_dops else [8, 16, 32, 64]
        return _ensure_even_dop(int(np.median(cds)), min(cds), max(cds))
    
    parent_candidates = sorted(parent_tb.candidate_dops) if parent_tb.candidate_dops else [8, 16, 32, 64]
    pmin, pmax = min(parent_candidates), max(parent_candidates)
    pgrid = [d for d in range(pmin, pmax + 1) if d % 2 == 0] or parent_candidates
    
    # Precompute child's reference upload time at child's mid-range DOP
    child_refs = []
    for ct in children:
        cds = sorted(ct.candidate_dops) if ct.candidate_dops else [8, 16, 32, 64]
        cmin, cmax = min(cds), max(cds)
        cref = _ensure_even_dop(int((cmin + cmax) / 2), cmin, cmax)
        c_exec = compute_exec_time_from_params(ct.pred_params, cref, ct.is_parallel, ct.base_exec_time)
        c_block = ct.blocking_interval.get(cref, 0.0)
        child_refs.append(max(c_exec - c_block, 0.0))
    
    best_parent_dop = pgrid[0]
    best_mismatch = float("inf")
    for pd in pgrid:
        p_exec = compute_exec_time_from_params(parent_tb.pred_params, pd, parent_tb.is_parallel, parent_tb.base_exec_time)
        total_mismatch = 0.0
        for upload_time in child_refs:
            total_mismatch += abs(p_exec - upload_time)
        if total_mismatch < best_mismatch:
            best_mismatch = total_mismatch
            best_parent_dop = pd
    return _ensure_even_dop(best_parent_dop, pmin, pmax)


def compute_stage_dop_range(tb: ThreadBlockInfo,
                            tb_map: Dict[int, ThreadBlockInfo],
                            rel_improve_threshold: float = 0.03) -> Tuple[int, int]:
    """
    Compute [low, high] range using marginal-gain and throughput-matching endpoints.
    """
    # Marginal endpoint
    d_marg = compute_marginal_gain_dop(tb, rel_improve_threshold=rel_improve_threshold)
    # Matching endpoint against children
    children = [tb_map[cid] for cid in tb.child_thread_ids if cid in tb_map] if tb.child_thread_ids else []
    d_match = compute_throughput_match_dop(tb, children)
    
    low = min(d_marg, d_match)
    high = max(d_marg, d_match)
    # Ensure within candidate min/max
    cds = sorted(tb.candidate_dops) if tb.candidate_dops else [8, 16, 32, 64]
    cmin, cmax = min(cds), max(cds)
    low = max(low, cmin)
    high = min(high, cmax)
    # Make endpoints even
    low = _ensure_even_dop(low, cmin, cmax)
    high = _ensure_even_dop(high, cmin, cmax)
    if low > high:
        low, high = high, low
    return (low, high)


def update_candidate_dops_by_endpoints(thread_blocks: List[ThreadBlockInfo],
                                       rel_improve_threshold: float = 0.03,
                                       samples_per_stage: int = 5) -> Dict[int, Tuple[int, int]]:
    """
    Recompute candidate_dops for each stage using the two endpoints and return the ranges.
    Candidate DOPs are sampled evenly within [low, high] on even values.
    """
    tb_map = {tb.thread_id: tb for tb in thread_blocks}
    ranges: Dict[int, Tuple[int, int]] = {}
    for tb in thread_blocks:
        low, high = compute_stage_dop_range(tb, tb_map, rel_improve_threshold=rel_improve_threshold)
        ranges[tb.thread_id] = (low, high)
        if low == high:
            tb.candidate_dops = [low]
        else:
            # Build even grid within [low, high]
            grid = [d for d in range(low, high + 1) if d % 2 == 0]
            if len(grid) <= samples_per_stage:
                tb.candidate_dops = grid
            else:
                # Evenly pick samples_per_stage points
                idxs = np.linspace(0, len(grid) - 1, samples_per_stage).astype(int).tolist()
                tb.candidate_dops = [grid[i] for i in idxs]
    return ranges


def compute_child_match_dop(child_tb: ThreadBlockInfo, target_parent_time: float,
                            exec_time_cache: Optional[Dict[int, Dict[int, float]]] = None) -> int:
    """
    Given a parent's execution time target, find child's DOP that best matches its upload time to target.
    Optimized: can use exec_time_cache to avoid repeated computation.
    """
    cds = sorted(child_tb.candidate_dops) if child_tb.candidate_dops else [8, 16, 32, 64]
    cmin, cmax = min(cds), max(cds)
    # Use sparse grid for speed: sample every 4 DOPs instead of every 2
    grid = [d for d in range(cmin, cmax + 1, 4) if d % 2 == 0]
    if not grid:
        grid = [d for d in range(cmin, cmax + 1) if d % 2 == 0] or cds
    best_dop = grid[0]
    best_err = float("inf")
    for d in grid:
        # Use cache if available
        if exec_time_cache and child_tb.thread_id in exec_time_cache and d in exec_time_cache[child_tb.thread_id]:
            c_exec = exec_time_cache[child_tb.thread_id][d]
        else:
            c_exec = compute_exec_time_from_params(child_tb.pred_params, d, child_tb.is_parallel, child_tb.base_exec_time)
        c_block = child_tb.blocking_interval.get(d, 0.0)
        upload_time = max(c_exec - c_block, 0.0)
        err = abs(upload_time - target_parent_time)
        if err < best_err:
            best_err = err
            best_dop = d
    return _ensure_even_dop(best_dop, cmin, cmax)


class DependentSampling(Sampling):
    """
    Topology-aware sampling: generate an individual by assigning DOPs in topological order.
    For each downstream stage, its feasible interval is computed from the already-sampled parent DOP.
    Optimized: uses exec_time_cache and caches interval computations.
    """
    def __init__(self, rel_improve_threshold: float = 0.03):
        super().__init__()
        self.rel_improve_threshold = rel_improve_threshold
        self._interval_cache = {}  # Cache: (child_tid, parent_dop) -> (low, high)
    
    def _do(self, problem, n_samples, **kwargs):
        import numpy as _np
        n_var = problem.n_var
        X = _np.zeros((n_samples, n_var), dtype=int)
        
        # Build topo order using problem.root_ids and children_map (compute once)
        thread_blocks = list(problem.thread_blocks)
        id_to_index = {tb.thread_id: idx for idx, tb in enumerate(thread_blocks)}
        
        def topo_order():
            order = []
            visited = set()
            stack = list(problem.root_ids) if hasattr(problem, "root_ids") else [tb.thread_id for tb in thread_blocks]
            while stack:
                tid = stack.pop(0)
                if tid in visited:
                    continue
                visited.add(tid)
                order.append(tid)
                for cid in problem.children_map.get(tid, []):
                    if cid not in visited:
                        stack.append(cid)
            return order
        
        order = topo_order()
        exec_time_cache = getattr(problem, 'exec_time_cache', None)
        
        for i in range(n_samples):
            dop_config = {}
            for tid in order:
                tb = problem.tb_map[tid]
                # Determine feasible interval for this stage
                if tb.parent_thread_id is None or tb.parent_thread_id not in dop_config:
                    cds = sorted(tb.candidate_dops) if tb.candidate_dops else [8, 16, 32, 64]
                    low, high = min(cds), max(cds)
                else:
                    parent_tb = problem.tb_map[tb.parent_thread_id]
                    parent_dop = dop_config[tb.parent_thread_id]
                    # Check cache first
                    cache_key = (tid, parent_dop)
                    if cache_key in self._interval_cache:
                        low, high = self._interval_cache[cache_key]
                    else:
                        # Compute parent exec time using cache if available
                        if exec_time_cache and parent_tb.thread_id in exec_time_cache and parent_dop in exec_time_cache[parent_tb.thread_id]:
                            parent_exec = exec_time_cache[parent_tb.thread_id][parent_dop]
                        else:
                            parent_exec = compute_exec_time_from_params(parent_tb.pred_params, parent_dop, parent_tb.is_parallel, parent_tb.base_exec_time)
                        d_marg = compute_marginal_gain_dop(tb, rel_improve_threshold=self.rel_improve_threshold, exec_time_cache=exec_time_cache)
                        d_match_child = compute_child_match_dop(tb, parent_exec, exec_time_cache=exec_time_cache)
                        cds = sorted(tb.candidate_dops) if tb.candidate_dops else [8, 16, 32, 64]
                        low = max(min(d_marg, d_match_child), min(cds))
                        high = min(max(d_marg, d_match_child), max(cds))
                        low = _ensure_even_dop(low, min(cds), max(cds))
                        high = _ensure_even_dop(high, min(cds), max(cds))
                        if low > high:
                            low, high = high, low
                        # Cache the interval (limit cache size)
                        if len(self._interval_cache) < 1000:
                            self._interval_cache[cache_key] = (low, high)
                # Sample uniformly within [low, high] on even values
                grid = [d for d in range(low, high + 1) if d % 2 == 0] or [low]
                choice = grid[_np.random.randint(0, len(grid))]
                dop_config[tid] = choice
                X[i, id_to_index[tid]] = choice
        return X


class DependentRepair(Repair):
    """
    After crossover/mutation, project each individual's genes to respect upstream->downstream interval dependency.
    Optimized: uses exec_time_cache and caches interval computations.
    """
    def __init__(self, rel_improve_threshold: float = 0.03):
        super().__init__()
        self.rel_improve_threshold = rel_improve_threshold
        self._interval_cache = {}  # Cache: (child_tid, parent_dop) -> (low, high)
    
    def _do(self, problem, X, **kwargs):
        import numpy as _np
        X = _np.asarray(X, dtype=int)
        thread_blocks = list(problem.thread_blocks)
        id_to_index = {tb.thread_id: idx for idx, tb in enumerate(thread_blocks)}
        
        # Build topo order (compute once)
        def topo_order():
            order = []
            visited = set()
            stack = list(problem.root_ids) if hasattr(problem, "root_ids") else [tb.thread_id for tb in thread_blocks]
            while stack:
                tid = stack.pop(0)
                if tid in visited:
                    continue
                visited.add(tid)
                order.append(tid)
                for cid in problem.children_map.get(tid, []):
                    if cid not in visited:
                        stack.append(cid)
            return order
        
        order = topo_order()
        exec_time_cache = getattr(problem, 'exec_time_cache', None)
        
        for i in range(X.shape[0]):
            # Build/repair dop_config in topo order
            dop_config = {}
            for tid in order:
                tb = problem.tb_map[tid]
                j = id_to_index[tid]
                val = int(X[i, j])
                if tb.parent_thread_id is None or tb.parent_thread_id not in dop_config:
                    cds = sorted(tb.candidate_dops) if tb.candidate_dops else [8, 16, 32, 64]
                    fixed = _ensure_even_dop(val, min(cds), max(cds))
                else:
                    parent_tb = problem.tb_map[tb.parent_thread_id]
                    parent_dop = dop_config[tb.parent_thread_id]
                    # Check cache first
                    cache_key = (tid, parent_dop)
                    if cache_key in self._interval_cache:
                        low, high = self._interval_cache[cache_key]
                    else:
                        # Compute parent exec time using cache if available
                        if exec_time_cache and parent_tb.thread_id in exec_time_cache and parent_dop in exec_time_cache[parent_tb.thread_id]:
                            parent_exec = exec_time_cache[parent_tb.thread_id][parent_dop]
                        else:
                            parent_exec = compute_exec_time_from_params(parent_tb.pred_params, parent_dop, parent_tb.is_parallel, parent_tb.base_exec_time)
                        d_marg = compute_marginal_gain_dop(tb, rel_improve_threshold=self.rel_improve_threshold, exec_time_cache=exec_time_cache)
                        d_match_child = compute_child_match_dop(tb, parent_exec, exec_time_cache=exec_time_cache)
                        cds = sorted(tb.candidate_dops) if tb.candidate_dops else [8, 16, 32, 64]
                        low = max(min(d_marg, d_match_child), min(cds))
                        high = min(max(d_marg, d_match_child), max(cds))
                        low = _ensure_even_dop(low, min(cds), max(cds))
                        high = _ensure_even_dop(high, min(cds), max(cds))
                        if low > high:
                            low, high = high, low
                        # Cache the interval (limit cache size)
                        if len(self._interval_cache) < 1000:
                            self._interval_cache[cache_key] = (low, high)
                    fixed = max(low, min(val, high))
                    fixed = _ensure_even_dop(fixed, low, high)
                dop_config[tid] = fixed
                X[i, j] = fixed
        return X

class ThreadBlockDOPProblem(Problem):
    """Thread Block DOP optimization problem for NSGA-II"""
    
    def __init__(self, thread_blocks: List[ThreadBlockInfo], use_continuous_dop: bool = True, enable_cache: bool = True, mismatch_penalty: float = 0.0):
        """
        Initialize the optimization problem.
        
        Args:
            thread_blocks: List of thread block information
            use_continuous_dop: If True, search in continuous DOP interval [min, max].
                               If False, search only in discrete candidate list.
            mismatch_penalty: Weight for throughput mismatch penalty added to latency objective.
        """
        self.thread_blocks = thread_blocks
        self.thread_ids = [tb.thread_id for tb in thread_blocks]
        self.tb_map = {tb.thread_id: tb for tb in thread_blocks}
        self.use_continuous_dop = use_continuous_dop
        self.enable_cache = enable_cache
        self.mismatch_penalty = mismatch_penalty
        self._cache = {}  # Cache for objective function evaluations
        
        # Define variable boundaries
        n_vars = len(thread_blocks)
        
        if use_continuous_dop:
            # Continuous mode: DOP values in interval [min_dop, max_dop]
            xl = []  # Lower bound (min DOP value)
            xu = []  # Upper bound (max DOP value)
            
            for tb in thread_blocks:
                if tb.candidate_dops:
                    min_dop = min(tb.candidate_dops)
                    max_dop = max(tb.candidate_dops)
                else:
                    min_dop = 8
                    max_dop = 64
                xl.append(min_dop)
                xu.append(max_dop)
            
            xl = np.array(xl)
            xu = np.array(xu)
        else:
            # Discrete mode: indices in candidate list
            self.candidate_lists = {}
            xl = []  # Lower bound (index 0)
            xu = []  # Upper bound (last index)
            
            for tb in thread_blocks:
                self.candidate_lists[tb.thread_id] = sorted(tb.candidate_dops)
                xl.append(0)
                xu.append(len(tb.candidate_dops) - 1)
            
        xl = np.array(xl)
        xu = np.array(xu)
        
        # 2 objectives: latency and cost
        # Flow matching is incorporated via a penalty term (added to latency) when mismatch_penalty > 0
        # Use integer type to ensure DOP values are integers
        super().__init__(n_var=n_vars, n_obj=2, xl=xl, xu=xu, vtype=int)
        
        # Pre-compute data structures for performance (after super().__init__)
        self._precompute_mappings()
        self._precompute_exec_times()  # Pre-compute exec times for common DOPs
        
        # Pre-compute min/max DOPs for continuous mode optimization
        if use_continuous_dop:
            self.min_dops = {}
            self.max_dops = {}
            for tb in self.thread_blocks:
                if tb.candidate_dops:
                    self.min_dops[tb.thread_id] = min(tb.candidate_dops)
                    self.max_dops[tb.thread_id] = max(tb.candidate_dops)
                else:
                    self.min_dops[tb.thread_id] = 8
                    self.max_dops[tb.thread_id] = 64
        else:
            self.min_dops = {}
            self.max_dops = {}
    
    def _precompute_mappings(self):
        """Pre-compute parent-child mappings for faster evaluation"""
        # Build parent->children mapping
        self.children_map = {}
        for tb in self.thread_blocks:
            if tb.child_thread_ids:
                self.children_map[tb.thread_id] = tb.child_thread_ids
        
        # Find root thread blocks (those with no parent)
        thread_ids = set(tb.thread_id for tb in self.thread_blocks)
        child_ids = set()
        for children in self.children_map.values():
            child_ids.update(children)
        self.root_ids = thread_ids - child_ids
    
    def _precompute_exec_times(self):
        """Pre-compute execution times for all thread blocks at all candidate DOPs"""
        # Pre-compute exec times for faster lookup during evaluation
        # This significantly speeds up evaluation as we avoid repeated computation
        self.exec_time_cache = {}  # {thread_id: {dop: exec_time}}
        for tb in self.thread_blocks:
            self.exec_time_cache[tb.thread_id] = {}
            candidate_dops = tb.candidate_dops if tb.candidate_dops else [8, 16, 32, 64]
            
            # Pre-compute for all candidate DOPs
            for dop in candidate_dops:
                exec_time = compute_exec_time_from_params(
                    pred_params=tb.pred_params,
                    dop=dop,
                    is_parallel=tb.is_parallel,
                    base_time=tb.base_exec_time
                )
                self.exec_time_cache[tb.thread_id][dop] = exec_time
            
            # Also pre-compute for common DOP values that might be generated in continuous mode
            # This covers most values in the [min_dop, max_dop] range
            if self.use_continuous_dop and candidate_dops:
                min_dop = min(candidate_dops)
                max_dop = max(candidate_dops)
                # Pre-compute for even DOPs in range (most common case)
                for dop in range(min_dop, max_dop + 1, 2):  # Step by 2 for even numbers
                    if dop not in self.exec_time_cache[tb.thread_id]:
                        exec_time = compute_exec_time_from_params(
                            pred_params=tb.pred_params,
                            dop=dop,
                            is_parallel=tb.is_parallel,
                            base_time=tb.base_exec_time
                        )
                        self.exec_time_cache[tb.thread_id][dop] = exec_time
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objective functions for given solutions - optimized version"""
        n_solutions = x.shape[0]
        f = np.zeros((n_solutions, 2), dtype=np.float32)  # Use float32 for speed
        
        # Batch process configurations for better cache locality
        # Pre-extract common values to reduce repeated lookups
        n_threads = len(self.thread_blocks)
        thread_blocks_list = list(self.thread_blocks)  # Cache list conversion
        
        # Vectorized processing where possible
        for i in range(n_solutions):
            dop_config = {}
            
            # Fast DOP extraction - use cached list
            if self.use_continuous_dop:
                # Continuous mode: x[i, j] is directly the DOP value
                for j, tb in enumerate(thread_blocks_list):
                    dop = int(x[i, j])
                    
                    # Ensure DOP is even (if needed) and within bounds
                    # Use precomputed min/max DOPs for faster lookup
                    min_dop = self.min_dops[tb.thread_id]
                    max_dop = self.max_dops[tb.thread_id]
                    dop = max(min_dop, min(dop, max_dop))
                    
                    # Round to nearest even number (optimized)
                    if dop % 2 == 1:
                        if dop > min_dop:
                            dop = dop - 1
                        elif dop < max_dop:
                            dop = dop + 1
                    
                    dop_config[tb.thread_id] = dop
            else:
                # Discrete mode: x[i, j] is index in candidate list - faster lookup
                # Cache candidate_lists lookups
                for j, tb in enumerate(thread_blocks_list):
                    idx = int(x[i, j])
                    candidate_list = self.candidate_lists[tb.thread_id]
                    # Clamp index to valid range
                    idx = max(0, min(idx, len(candidate_list) - 1))
                    dop = candidate_list[idx]
                    dop_config[tb.thread_id] = dop
            
            # Check cache first to avoid redundant computation
            cache_key = tuple(sorted(dop_config.items()))
            if self.enable_cache and cache_key in self._cache:
                total_latency, total_cost = self._cache[cache_key]
            else:
                # Optimized objective computation
                # Objective 1: Total query execution time - use optimized version
                total_latency = self._compute_latency_optimized(dop_config)
                
                # Objective 2: Total resource cost - use precomputed exec times when possible
                total_cost = self._compute_cost_optimized(dop_config)
                
                # Cache the result (limit cache size to avoid memory issues)
                if self.enable_cache:
                    if len(self._cache) < 20000:  # Increased cache size for better hit rate
                        self._cache[cache_key] = (total_latency, total_cost)
            
            # Add throughput mismatch penalty to latency if enabled
            if self.mismatch_penalty and self.mismatch_penalty > 0.0:
                mismatch = calculate_flow_rate_mismatch(self.thread_blocks, dop_config)
                total_latency = total_latency + (self.mismatch_penalty * mismatch)
            
            # Set objective values
            f[i, 0] = total_latency  # Objective 1: minimize latency (with penalty if enabled)
            f[i, 1] = total_cost     # Objective 2: minimize cost
        
        out["F"] = f
    
    def _compute_latency_optimized(self, dop_config: Dict[int, int]) -> float:
        """Optimized version of latency calculation with precomputed mappings"""
        # Use precomputed mappings for faster lookup
        memo = {}
        
        def calc_time(thread_id: int) -> float:
            if thread_id in memo:
                return memo[thread_id]
            
            tb = self.tb_map[thread_id]
            dop = dop_config[thread_id]
            
            # Use precomputed exec time if available, otherwise compute on-the-fly
            if thread_id in self.exec_time_cache and dop in self.exec_time_cache[thread_id]:
                exec_time = self.exec_time_cache[thread_id][dop]
            else:
                # Fallback: compute on-the-fly
                exec_time = compute_exec_time_from_params(
                    pred_params=tb.pred_params,
                    dop=dop,
                    is_parallel=tb.is_parallel,
                    base_time=tb.base_exec_time
                )
            
            # If has children, consider their max completion time
            if thread_id in self.children_map:
                child_times = [calc_time(cid) for cid in self.children_map[thread_id]]
                max_child_time = max(child_times) if child_times else 0
                total_time = exec_time + max_child_time
            else:
                total_time = exec_time
            
            memo[thread_id] = total_time
            return total_time
        
        # Return max time among all roots (using precomputed root_ids)
        if self.root_ids:
            return max(calc_time(rid) for rid in self.root_ids)
        else:
            return 0.0
    
    def _compute_cost_optimized(self, dop_config: Dict[int, int]) -> float:
        """Optimized version of cost calculation using precomputed exec times"""
        total_cost = 0.0
        for tb in self.thread_blocks:
            dop = dop_config[tb.thread_id]
            
            # Use precomputed exec time if available
            if tb.thread_id in self.exec_time_cache and dop in self.exec_time_cache[tb.thread_id]:
                exec_time = self.exec_time_cache[tb.thread_id][dop]
            else:
                # Fallback: compute on-the-fly
                exec_time = compute_exec_time_from_params(
                    pred_params=tb.pred_params,
                    dop=dop,
                    is_parallel=tb.is_parallel,
                    base_time=tb.base_exec_time
                )
            total_cost += dop * exec_time
        
        return total_cost


def optimize_thread_block_dops_with_moo(
    thread_blocks: Dict[int, Any],  # thread_id -> ThreadBlock object
    population_size: int = 30,
    generations: int = 20,
    weight_latency: float = 0.9,
    weight_cost: float = 0.1,
    use_continuous_dop: bool = True,
    mismatch_penalty: float = 0.0,
    enable_endpoint_ranges: bool = True,
    rel_improve_threshold: float = 0.03,
    samples_per_stage: int = 5
) -> Tuple[Dict[int, int], float]:
    """
    Optimize thread block DOP configuration using MOO (NSGA-II)
    
    Args:
        thread_blocks: Dictionary of ThreadBlock objects
        population_size: NSGA-II population size
        generations: Number of generations
        weight_latency: Weight for latency objective
        weight_cost: Weight for cost objective
        use_continuous_dop: If True, search in continuous DOP interval [min, max].
                           If False, search only in discrete candidate list.
        mismatch_penalty: Weight for throughput mismatch penalty added to latency (0 disables).
        enable_endpoint_ranges: If True, recompute candidate_dops via endpoints before MOO.
        rel_improve_threshold: Relative improvement threshold for marginal-gain endpoint.
        samples_per_stage: Number of candidate samples within [low, high] for each stage.
        
    Returns:
        tuple: (Optimal DOP configuration: {thread_id: optimal_dop}, execution_time: float)
        
    Note:
        Flow matching is encouraged via a penalty term added to the latency objective.
    """
    if not PYMOO_AVAILABLE:
        print("Warning: pymoo not available, using greedy fallback")
        # Fallback: use existing candidate_optimal_dops
        return {tid: tb.optimal_dop for tid, tb in thread_blocks.items()}, 0.0
    
    # Convert ThreadBlock objects to ThreadBlockInfo
    thread_block_infos = []
    for tid in sorted(thread_blocks.keys()):
        tb = thread_blocks[tid]
        
        # Build parent relationship
        parent_id = None
        for other_tid, other_tb in thread_blocks.items():
            if tid in other_tb.child_thread_ids:
                parent_id = other_tid
                break
        
        # Extract prediction parameters from thread block nodes
        # Use the first parallel node's pred_params as representative
        pred_params = None
        is_parallel = True
        base_exec_time = tb.thread_execution_time if hasattr(tb, 'thread_execution_time') else 0.0
        
        if hasattr(tb, 'nodes') and tb.nodes:
            for node in tb.nodes:
                if hasattr(node, 'is_parallel') and node.is_parallel and hasattr(node, 'pred_params'):
                    if node.pred_params is not None:
                        pred_params = node.pred_params
                        break
            
            # Check if all nodes are non-parallel
            if all(not getattr(node, 'is_parallel', True) for node in tb.nodes):
                is_parallel = False
        
        info = ThreadBlockInfo(
            thread_id=tid,
            candidate_dops=tb.candidate_optimal_dops,
            pred_dop_exec_time=tb.pred_dop_exec_time,
            blocking_interval=tb.blocking_interval,
            child_thread_ids=list(tb.child_thread_ids),
            parent_thread_id=parent_id,
            pred_params=pred_params,
            is_parallel=is_parallel,
            base_exec_time=base_exec_time
        )
        thread_block_infos.append(info)
    
    # Optionally recompute candidate ranges by endpoints
    if enable_endpoint_ranges:
        ranges = update_candidate_dops_by_endpoints(
            thread_block_infos,
            rel_improve_threshold=rel_improve_threshold,
            samples_per_stage=samples_per_stage
        )
        # Optional: print brief diagnostic
        try:
            for tid in sorted(ranges.keys()):
                low, high = ranges[tid]
                print(f"    Stage {tid} endpoint range: [{low}, {high}] -> candidates={thread_block_infos[tid - 1].candidate_dops if 0 <= tid - 1 < len(thread_block_infos) else 'N/A'}")
        except Exception:
            pass
    
    print(f"  MOO optimization: {len(thread_block_infos)} thread blocks")
    if use_continuous_dop:
        print(f"  Mode: Continuous DOP search in interval [min, max]")
    else:
        print(f"  Mode: Discrete DOP search from candidate list")
    
    # Create optimization problem with caching enabled
    problem = ThreadBlockDOPProblem(
        thread_block_infos,
        use_continuous_dop=use_continuous_dop,
        enable_cache=True,
        mismatch_penalty=mismatch_penalty
    )
    
    # Configure NSGA-II algorithm with optimized parameters for speed
    # Reduce crossover/mutation probability slightly to reduce diversity but increase speed
    # Use dependent sampling & repair to enforce upstream->downstream interval dependency per individual
    algorithm = NSGA2(
        pop_size=population_size,
        n_offsprings=population_size,
        sampling=DependentSampling(rel_improve_threshold=rel_improve_threshold) if PYMOO_AVAILABLE else IntegerRandomSampling(),
        crossover=SBX(prob=0.8, eta=15),  # Slightly lower prob for faster convergence
        mutation=PM(eta=20),
        eliminate_duplicates=True,
        repair=DependentRepair(rel_improve_threshold=rel_improve_threshold) if PYMOO_AVAILABLE else None
    )
    
    # Set termination criterion: use early stopping if no improvement for 3 generations
    # This can significantly speed up when convergence is reached early
    try:
        from pymoo.termination import get_termination
        # Use multiple termination criteria: generation limit OR early stopping
        termination = get_termination("n_gen", generations)
        # Alternative: early stopping if no improvement (comment out for now to keep current behavior)
        # termination = get_termination("n_gen", generations) | get_termination("n_eval", max_eval=None)
    except:
        # Fallback if termination API changed
        termination = get_termination("n_gen", generations)
    
    # Run optimization - only time this actual algorithm execution
    import time
    moo_start_time = time.time()
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=False
    )
    moo_end_time = time.time()
    moo_execution_time = moo_end_time - moo_start_time
    print(f"  NSGA-II optimization execution time: {moo_execution_time:.4f} seconds")
    
    # Select best solution from Pareto front
    if res.F is None or len(res.F) == 0:
        print("  Warning: MOO optimization failed, using greedy fallback")
        return {tid: tb.optimal_dop for tid, tb in thread_blocks.items()}, moo_execution_time
    
    # Normalize objectives
    latencies = res.F[:, 0]
    costs = res.F[:, 1]
    
    min_lat, max_lat = np.min(latencies), np.max(latencies)
    min_cost, max_cost = np.min(costs), np.max(costs)
    
    # Calculate weighted scores
    best_idx = 0
    best_score = float('inf')
    
    for i in range(len(res.F)):
        norm_lat = (latencies[i] - min_lat) / (max_lat - min_lat + 1e-8)
        norm_cost = (costs[i] - min_cost) / (max_cost - min_cost + 1e-8)
        
        score = weight_latency * norm_lat + weight_cost * norm_cost
        
        if score < best_score:
            best_score = score
            best_idx = i
    
    # Convert best solution to DOP configuration
    best_solution = res.X[best_idx]
    dop_config = {}
    
    if use_continuous_dop:
        # Continuous mode: res.X contains DOP values directly
        for j, tb_info in enumerate(thread_block_infos):
            dop = int(best_solution[j])
            
            # Ensure DOP is within bounds and is even
            min_dop = min(tb_info.candidate_dops) if tb_info.candidate_dops else 8
            max_dop = max(tb_info.candidate_dops) if tb_info.candidate_dops else 64
            dop = max(min_dop, min(dop, max_dop))
            
            # Round to nearest even number
            if dop % 2 == 1 and dop > 1:
                dop = dop - 1 if dop > min_dop else dop + 1
            
            dop_config[tb_info.thread_id] = dop
    else:
        # Discrete mode: res.X contains indices
        for j, tb_info in enumerate(thread_block_infos):
            idx = int(best_solution[j])
            idx = max(0, min(idx, len(tb_info.candidate_dops) - 1))
            dop = tb_info.candidate_dops[idx]
            dop_config[tb_info.thread_id] = dop
    
    print(f"  MOO found Pareto front with {len(res.F)} solutions")
    print(f"  Best solution: latency={latencies[best_idx]:.2f}, cost={costs[best_idx]:.2f}")
    
    return dop_config, moo_execution_time

