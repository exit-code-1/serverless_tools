import os
import time
import numpy as np
import math
import onnxruntime
import torch
# import utils
# from structure import no_dop_operator_features, no_dop_operators_exec, no_dop_operators_mem, dop_operators_exec, dop_operators_mem, parallel_op
from .plan_node import PlanNode # 如果需要取消注释

class ThreadBlock:
    def __init__(self, thread_id, nodes):
        """
        :param thread_id: 线程块标识
        :param nodes: 属于该线程块的所有 PlanNode
        """
        self.thread_id = thread_id
        self.visit = False
        self.nodes = nodes
        self.candidate_dops = set()  # 可后续设置
        self.real_dop_exec_time = {}     
        self.pred_dop_exec_time = {}  
        self.thread_execution_time = 0
        self.thread_complete_time = 0
        self.local_data_transfer_start_time = 0
        self.up_data_transfer_start_time = 0
        self.child_thread_ids = set()  # 用于记录子线程块的 thread_id
        self.child_max_execution_time = 0  # 用于记录子线程内的最大执行时间
        self.optimal_dop = None        # 用于记录最终最优 dop
        self.blocking_interval = {}
        self.pred_time = 0
        self.candidate_optimal_dops = []
        self.total_exec_time = {} #加上发送时间
        self.dop_mismatch_penalties = {}  # {parent_dop: {child_dop: penalty}}

    def aggregate_metrics(self):
        """
        聚合线程块内所有节点的指标:
          - thread_execution_time:累加所有节点的执行时间
          - thread_complete_time:所有节点中最大的完成时间
          - local/up_data_transfer_start_time:取所有节点中的最大值(仅作参考)
          - candidate_dops:如果未设置,则取所有阻塞节点(materialized==True)的 matched_dops 的交集
          - blocking_interval:在 candidate_dops 下,
              从所有阻塞节点中选取 thread_execution_time 最大的那个作为“最上层阻塞节点”,
              然后直接使用该节点的 pred_dop_exec_map 作为阻塞时间
          - 同时,从该节点的 dop_exec_time_map 取得真实的累加执行时间
        """
        if not self.nodes:
            return

        # 聚合基本指标
        self.thread_execution_time = sum(node.execution_time for node in self.nodes)
        self.thread_complete_time = max(node.thread_complete_time for node in self.nodes)
        self.local_data_transfer_start_time = max(node.local_data_transfer_start_time for node in self.nodes)
        self.up_data_transfer_start_time = max(node.up_data_transfer_start_time for node in self.nodes)
        
        # 如果没有 candidate_dops,则取所有节点 matched_dops 的交集
        if not self.candidate_dops:
            candidate_dops = None
            for node in self.nodes:
                if not node.is_parallel or node.parent_node is None:
                    candidate_dops = {1}
                    break
                if candidate_dops is None:
                    candidate_dops = set(node.matched_dops)
                else:
                    candidate_dops = candidate_dops.intersection(node.matched_dops)
            self.candidate_dops = candidate_dops if candidate_dops is not None else set()
            
        # 在线程块内选出“最上层阻塞节点”:定义为 materialized==True 且 thread_execution_time 最大的那个节点
        top_blocking = None
        top_node = self.nodes[0]
        max_block_exec = -1
        max_exec = -1
        for node in self.nodes:
            if node.materialized and node.up_data_transfer_start_time > max_block_exec:
                max_block_exec = node.up_data_transfer_start_time
                top_blocking = node
            if node.thread_execution_time > max_exec:
                max_exec = node.thread_execution_time
                top_node = node

        self.real_dop_exec_time = top_node.true_dop_exec_map.copy()
        self.pred_dop_exec_time = top_node.pred_dop_exec_map.copy()
        # 如果存在阻塞节点,就直接从该节点获取累加的真实和预测执行时间
        if top_blocking:
            self.blocking_interval = {
            dop: top_blocking.pred_dop_exec_map.get(dop, float('inf')) - top_blocking.build_dop_exec_map.get(dop, 0)
            for dop in top_blocking.pred_dop_exec_map
        }
        else:
            self.blocking_interval = {dop: 0 for dop in self.candidate_dops}
            
    def choose_optimal_dop(self,
                           child_upload_times: dict,
                           min_improvement_ratio=0.15,
                           min_reduction_threshold=100,
                           interval_tolerance=0.3,  # Tolerance for interval matching (30%)
                           ):
        """
        Choose optimal DOP using pipeline flow rate matching with interval-based approach:
          1. Determine right boundary d_right based on performance curve
          2. For each child thread upload time, find matching DOP intervals (not just single points)
          3. Aggregate all matching intervals and fill candidates within the range
          4. Return all candidates between left boundary and right boundary

        :param child_upload_times: dict, key=child_thread_id, value=upload time (ms)
        :param interval_tolerance: tolerance ratio for interval matching (e.g., 0.3 = ±30%)
        :return: list of candidate DOPs
        """
        if not self.candidate_dops:
            return None
        if len(self.candidate_dops) == 1:
            only = next(iter(self.candidate_dops))
            self.optimal_dop = only
            self.pred_time = self.pred_dop_exec_time.get(only, float('inf'))
            self.candidate_optimal_dops = [only]
            return [only]

        # 1. Sort DOPs in ascending order
        sorted_dops = sorted(self.candidate_dops)

        # 2. Determine right boundary d_right (performance-based)
        d_right = sorted_dops[0]
        for d in sorted_dops[1:]:
            t_prev = self.pred_dop_exec_time.get(d_right, float('inf'))
            t_cur = self.pred_dop_exec_time.get(d, float('inf'))
            # Absolute reduction
            delta = t_prev - t_cur
            if delta < min_reduction_threshold:
                continue
            # Normalized gain
            factor = 1 + math.log2(d - d_right) if d > d_right else 1
            adj_gain = (delta / t_prev) / factor
            dynamic_min_improvement = min_improvement_ratio / math.log(delta)
            if adj_gain >= dynamic_min_improvement:
                d_right = d
            else:
                continue
        if d_right < 8:
            d_right = 8

        # 3. Generate matching intervals for ALL child upload times
        # Now considering DOP mismatch penalties
        matching_intervals = []  # List of (left_bound, right_bound) tuples
        
        for child_id, up_time in child_upload_times.items():
            # Define acceptable time range for this child
            time_lower = up_time * (1 - interval_tolerance)
            time_upper = up_time * (1 + interval_tolerance)
            
            # Find all DOPs whose execution time falls within this range
            # Also consider DOP mismatch penalty if available
            matching_dops = []
            for d in sorted_dops:
                if d == 1:
                    continue
                if d > d_right:  # Don't exceed right boundary
                    break
                    
                # Get base execution time
                t = self.pred_dop_exec_time.get(d, float('inf'))
                
                # Add DOP mismatch penalty if this thread block has children
                # The penalty was calculated in threading_utils.py
                if hasattr(self, 'dop_mismatch_penalties') and d in self.dop_mismatch_penalties:
                    # Find the maximum penalty across all parent-child DOP combinations
                    # We want to penalize configurations where parent DOP differs significantly from child
                    max_penalty = 0
                    for parent_dop, child_penalties in self.dop_mismatch_penalties.items():
                        if d in child_penalties:
                            max_penalty = max(max_penalty, child_penalties[d])
                    t += max_penalty
                
                if time_lower <= t <= time_upper:
                    matching_dops.append(d)
            
            # If we found matching DOPs, record the interval
            if matching_dops:
                interval_left = min(matching_dops)
                interval_right = max(matching_dops)
                matching_intervals.append((interval_left, interval_right))
        
        # 4. Determine the overall left boundary
        # Use the minimum of all interval left bounds (most conservative)
        # This ensures we don't miss good configurations
        if matching_intervals:
            d_left = min(interval[0] for interval in matching_intervals)
        else:
            # If no child constraints, use a default left boundary
            d_left = sorted_dops[0] if sorted_dops[0] >= 8 else 8

        # Ensure left boundary is reasonable
        if d_left < 8:
            d_left = 8
        
        # 5. Fill all candidates in the interval [d_left, d_right]
        candidates = set()
        for d in sorted_dops:
            if d_left <= d <= d_right:
                candidates.add(d)
        
        # 6. Also add specific matching points from each child for better coverage
        for interval_left, interval_right in matching_intervals:
            candidates.add(interval_left)
            candidates.add(interval_right)
            # Add middle point of each interval
            mid_candidates = [d for d in sorted_dops if interval_left <= d <= interval_right]
            if len(mid_candidates) > 2:
                candidates.add(mid_candidates[len(mid_candidates)//2])
        
        # Ensure boundaries are included
        candidates.add(d_left)
        candidates.add(d_right)
        
        # Convert to sorted list
        candidates = sorted(candidates)
        
        self.candidate_optimal_dops = candidates
        # Default to the largest (most aggressive)
        self.optimal_dop = candidates[-1]
        self.pred_time = self.pred_dop_exec_time.get(self.optimal_dop)
        
        return candidates
    def compute_consumption_rate(self, dop: int) -> float:
        """
        Compute consumption rate at given DOP.
        
        Consumption rate = execution_time - production_time
        For non-materialized operators: consumption_rate ≈ execution_time
        For materialized operators: consumption_rate = execution_time - blocking_interval
        
        :param dop: DOP value
        :return: consumption rate (time)
        """
        exec_time = self.pred_dop_exec_time.get(dop, float('inf'))
        blocking = self.blocking_interval.get(dop, 0)
        consumption_rate = exec_time - blocking
        return consumption_rate
    
    def compute_production_rate(self, dop: int) -> float:
        """
        Compute production rate (upload_time) at given DOP.
        
        Production rate = execution_time - blocking_time = upload_time
        This is the rate at which this pipeline produces data for downstream pipelines.
        
        :param dop: DOP value
        :return: production rate (upload_time)
        """
        exec_time = self.pred_dop_exec_time.get(dop, float('inf'))
        blocking = self.blocking_interval.get(dop, 0)
        production_rate = exec_time - blocking  # Same as upload_time
        return production_rate
    
    def choose_optimal_dop_top_down(self,
                                   all_thread_blocks: dict,
                                   parent_production_rate: float = None,
                                   min_improvement_ratio=0.2,
                                   min_reduction_threshold=200,
                                   interval_tolerance=0.3):
        """
        Choose optimal DOP top-down with interval-based pipeline flow matching.
        Starting from topmost pipeline (DOP=1), propagate production rate downward.
        
        According to paper: 
        - Topmost pipeline's consumption rate → determine downstream pipeline's feasible range
        - Once a DOP is selected, compute production rate → propagate to next downstream pipeline
        - For throughput matching: downstream consumption_rate ≈ upstream production_rate
        
        :param all_thread_blocks: dict of all thread blocks in the query
        :param parent_production_rate: production rate (upload_time) from upstream (parent) pipeline
        :param min_improvement_ratio: minimum performance improvement ratio
        :param min_reduction_threshold: minimum time reduction threshold (ms)
        :param interval_tolerance: tolerance for interval matching (e.g., 0.3 = ±30%)
        :return: consumption rate of this pipeline (for downstream propagation)
        """
        if getattr(self, 'visit', False):
            # Already processed, return consumption rate at optimal DOP
            return self.compute_consumption_rate(self.optimal_dop) if self.optimal_dop else self.pred_time
        self.visit = True
        
        # Check if this thread block contains any root node (parent_node is None)
        # Root pipeline thread blocks should always have DOP=1
        has_root_node = False
        for node in self.nodes:
            if node.parent_node is None:
                has_root_node = True
                break
        
        if has_root_node:
            # Root pipeline (topmost): force DOP=1 as baseline
            self.optimal_dop = 1
            self.pred_time = self.pred_dop_exec_time.get(1, float('inf'))
            self.candidate_optimal_dops = [1]
            
            # Compute consumption rate for downstream propagation
            consumption_rate = self.compute_consumption_rate(1)
            # Also compute production rate (upload_time) for downstream matching
            production_rate = self.compute_production_rate(1)
            
            # Process downstream pipelines (child thread blocks)
            # Note: child_id in execution plan refers to downstream pipeline
            # For throughput matching: downstream consumption_rate should match upstream production_rate
            for child_id in self.child_thread_ids:
                child_tb = all_thread_blocks.get(child_id)
                if child_tb is None:
                    continue
                # Propagate production rate downward for downstream to match
                # Downstream will use this as reference for matching its consumption rate
                child_tb.choose_optimal_dop_top_down(
                    all_thread_blocks,
                    parent_production_rate=production_rate,  # Pass production rate (upload_time)
                    min_improvement_ratio=min_improvement_ratio,
                    min_reduction_threshold=min_reduction_threshold,
                    interval_tolerance=interval_tolerance
                )
            
            return consumption_rate
        
        # Non-root pipeline: use parent's production rate to determine feasible range
        # According to paper: downstream consumption rate should match upstream production rate
        # In code context:
        # - parent_production_rate is upload_time from upstream pipeline (parent in execution plan)
        # - Current pipeline needs to match upstream's production rate with its consumption rate
        # - For throughput matching: upstream production_rate (upload_time) ≈ downstream consumption_rate
        # - choose_optimal_dop matches: downstream exec_time ≈ upstream upload_time
        # - For non-materialized operators: consumption_rate ≈ exec_time, production_rate ≈ upload_time
        # - So we can use parent_production_rate directly as upload_time reference for matching
        
        if parent_production_rate is not None:
            # Use upstream parent's production rate as reference for matching
            # choose_optimal_dop will match current pipeline's exec_time with parent_production_rate
            # This implements: downstream consumption_rate ≈ upstream production_rate
            parent_upload_times = {}
            # In choose_optimal_dop, child_upload_times represents upstream children's upload times
            # In top-down context, parent_production_rate is from upstream parent
            # We use it as reference for matching current pipeline's exec_time
            for child_id in self.child_thread_ids:
                # child_id here refers to downstream pipeline in execution plan
                # But for throughput matching, we use parent_production_rate as reference
                # which represents upstream parent's production rate we need to match
                parent_upload_times[child_id] = parent_production_rate
            
            self.choose_optimal_dop(
                child_upload_times=parent_upload_times,
                min_improvement_ratio=min_improvement_ratio,
                min_reduction_threshold=min_reduction_threshold,
                interval_tolerance=interval_tolerance
            )
        else:
            # No parent constraint, use default method
            self.choose_optimal_dop(
                child_upload_times={},
                min_improvement_ratio=min_improvement_ratio,
                min_reduction_threshold=min_reduction_threshold,
                interval_tolerance=interval_tolerance
            )
        
        # Compute production rate for downstream propagation
        # According to paper: once DOP is selected, compute production rate → propagate to next downstream
        production_rate = self.compute_production_rate(self.optimal_dop) if self.optimal_dop else self.pred_time
        
        # Process downstream pipelines (child thread blocks)
        # Propagate production rate downward so downstream can match it
        for child_id in self.child_thread_ids:
            child_tb = all_thread_blocks.get(child_id)
            if child_tb is None:
                continue
            # Propagate production rate downward for downstream to match
            child_tb.choose_optimal_dop_top_down(
                all_thread_blocks,
                parent_production_rate=production_rate,
                min_improvement_ratio=min_improvement_ratio,
                min_reduction_threshold=min_reduction_threshold,
                interval_tolerance=interval_tolerance
            )
        
        # Return consumption rate (for potential future use)
        consumption_rate = self.compute_consumption_rate(self.optimal_dop) if self.optimal_dop else self.pred_time
        return consumption_rate
    
    def recursive_choose_optimal_dop(self,
                                  all_thread_blocks: dict,
                                  min_improvement_ratio=0.2,
                                  min_reduction_threshold=200,
                                  interval_tolerance=0.3):
        """
        DEPRECATED: Use choose_optimal_dop_top_down instead.
        This method is kept for backward compatibility but now delegates to top-down approach.
        
        Recursively choose optimal DOP top-down with interval-based pipeline flow matching.
        Starting from topmost pipeline, propagate consumption rate downward.
        """
        return self.choose_optimal_dop_top_down(
            all_thread_blocks,
            parent_production_rate=None,
            min_improvement_ratio=min_improvement_ratio,
            min_reduction_threshold=min_reduction_threshold,
            interval_tolerance=interval_tolerance
        )
    
    def _get_redist_penalty(self, child_tb, parent_dop, child_dop):
        """
        Get redistribution penalty with interpolation support for continuous DOPs
        
        :param child_tb: Child thread block
        :param parent_dop: Parent's DOP (might be continuous value)
        :param child_dop: Child's DOP (might be continuous value)
        :return: Redistribution penalty cost
        """
        if not hasattr(child_tb, 'dop_mismatch_penalties') or len(child_tb.dop_mismatch_penalties) == 0:
            return 0
        
        penalties = child_tb.dop_mismatch_penalties
        
        # Find closest parent DOP
        available_parent_dops = sorted(penalties.keys())
        
        if parent_dop in penalties:
            parent_penalties = penalties[parent_dop]
        else:
            # Use closest parent DOP
            closest_parent = min(available_parent_dops, key=lambda x: abs(x - parent_dop))
            parent_penalties = penalties[closest_parent]
        
        # Get penalty for child DOP (with interpolation if needed)
        if child_dop in parent_penalties:
            return parent_penalties[child_dop]
        
        # Interpolate child DOP
        available_child_dops = sorted(parent_penalties.keys())
        if not available_child_dops:
            return 0
        
        if child_dop < available_child_dops[0]:
            return parent_penalties[available_child_dops[0]]
        elif child_dop > available_child_dops[-1]:
            return parent_penalties[available_child_dops[-1]]
        else:
            # Linear interpolation
            lower = max(d for d in available_child_dops if d <= child_dop)
            upper = min(d for d in available_child_dops if d >= child_dop)
            if lower == upper:
                return parent_penalties[lower]
            else:
                p_lower = parent_penalties[lower]
                p_upper = parent_penalties[upper]
                ratio = (child_dop - lower) / (upper - lower)
                return p_lower + ratio * (p_upper - p_lower)
    
    def _adjust_child_dops_for_redistribution(self, all_thread_blocks, 
                                              min_improvement_ratio=0.2,
                                              min_reduction_threshold=200):
        """
        Adjust child thread block DOPs to reduce redistribution overhead.
        
        Key insight: Increasing child DOP can reduce parent's waiting time for data transfer,
        which might be more beneficial than reducing child's execution time.
        
        :param all_thread_blocks: dict of all thread blocks
        :param min_improvement_ratio: minimum improvement ratio to consider adjustment
        :param min_reduction_threshold: minimum time reduction threshold (ms)
        """
        if not self.child_thread_ids:
            return
        
        my_optimal_dop = self.optimal_dop
        if my_optimal_dop is None:
            return
        
        for child_id in self.child_thread_ids:
            child_tb = all_thread_blocks.get(child_id)
            if child_tb is None:
                continue
            
            current_child_dop = child_tb.optimal_dop
            if current_child_dop is None:
                continue
            
            # Get current redistribution cost with interpolation support
            # NOTE: penalties are stored on the CHILD thread block, not parent
            current_redist_cost = self._get_redist_penalty(child_tb, my_optimal_dop, current_child_dop)
            
            # Current child execution time
            # If exact DOP not found, interpolate from nearby DOPs
            if current_child_dop in child_tb.pred_dop_exec_time:
                current_child_time = child_tb.pred_dop_exec_time[current_child_dop]
            elif len(child_tb.pred_dop_exec_time) > 0:
                # Interpolate: find closest DOPs and average
                available_dops = sorted(child_tb.pred_dop_exec_time.keys())
                if current_child_dop < available_dops[0]:
                    current_child_time = child_tb.pred_dop_exec_time[available_dops[0]]
                elif current_child_dop > available_dops[-1]:
                    current_child_time = child_tb.pred_dop_exec_time[available_dops[-1]]
                else:
                    # Linear interpolation
                    lower_dop = max(d for d in available_dops if d <= current_child_dop)
                    upper_dop = min(d for d in available_dops if d >= current_child_dop)
                    if lower_dop == upper_dop:
                        current_child_time = child_tb.pred_dop_exec_time[lower_dop]
                    else:
                        t_lower = child_tb.pred_dop_exec_time[lower_dop]
                        t_upper = child_tb.pred_dop_exec_time[upper_dop]
                        ratio = (current_child_dop - lower_dop) / (upper_dop - lower_dop)
                        current_child_time = t_lower + ratio * (t_upper - t_lower)
            else:
                current_child_time = float('inf')
            
            # Try higher DOPs for the child
            # Use all available DOPs (candidate_dops), not just the originally selected ones
            available_dops = sorted(child_tb.candidate_dops) if child_tb.candidate_dops else []
            
            if not available_dops:
                continue
            
            # Calculate data volume being transferred
            # Find the redistribution node that connects to this specific child
            redistribution_node = None
            for node in self.nodes:
                if 'streaming' in node.operator_type.lower() or 'redistribute' in node.operator_type.lower():
                    # Check if this streaming node's child is in the child thread
                    for child_plan in node.child_plans:
                        if getattr(child_plan, 'thread_id', None) == child_id:
                            redistribution_node = node
                            break
                    if redistribution_node:
                        break
            
            data_volume = 0
            if redistribution_node:
                data_volume = redistribution_node.actual_rows * redistribution_node.width
            else:
                # Fallback: use any streaming node in this thread
                for node in reversed(self.nodes):
                    if 'streaming' in node.operator_type.lower() or 'redistribute' in node.operator_type.lower():
                        redistribution_node = node
                        data_volume = redistribution_node.actual_rows * redistribution_node.width
                        break
            
            # For large data volumes (> 1GB), be more aggressive in adjusting child DOP
            # even if current redistribution cost seems moderate
            if data_volume > 1e9:  # > 1GB
                min_redist_threshold = 200  # Lower threshold for large data
                max_dop_factor = 8  # Allow more aggressive increase
                min_benefit_ratio = 2.0  # Less strict benefit requirement
            elif current_redist_cost > 1000:
                # High redistribution cost, be moderately aggressive
                min_redist_threshold = 500
                max_dop_factor = 4
                min_benefit_ratio = 3.0
            else:
                # Skip if redistribution cost is small and data volume is small
                if current_redist_cost < 500 and data_volume < 1e8:
                    continue
                min_redist_threshold = 500
                max_dop_factor = 4
                min_benefit_ratio = 3.0
            
            best_alternative_dop = current_child_dop
            best_total_benefit = 0
            best_benefit_ratio = 0  # Track benefit/cost ratio
            
            # Apply dynamic constraints based on data volume
            MAX_DOP_INCREASE_FACTOR = max_dop_factor
            MIN_BENEFIT_RATIO = min_benefit_ratio
            
            for candidate_dop in available_dops:
                if candidate_dop <= current_child_dop:
                    continue  # Only consider increasing DOP
                
                # Limit DOP increase to avoid overly aggressive changes
                if candidate_dop > current_child_dop * MAX_DOP_INCREASE_FACTOR:
                    continue
                
                # Get or estimate child execution time at this DOP
                if candidate_dop in child_tb.pred_dop_exec_time:
                    new_child_time = child_tb.pred_dop_exec_time[candidate_dop]
                    is_estimated = False
                else:
                    # Try interpolation first if we have pred data
                    if len(child_tb.pred_dop_exec_time) > 1:
                        available_dops_list = sorted(child_tb.pred_dop_exec_time.keys())
                        if candidate_dop < available_dops_list[0]:
                            # Extrapolate using scaling from first DOP
                            new_child_time = current_child_time * current_child_dop / (candidate_dop * 0.6)
                        elif candidate_dop > available_dops_list[-1]:
                            # Extrapolate using last DOP
                            new_child_time = current_child_time * current_child_dop / (candidate_dop * 0.6)
                        else:
                            # Interpolate
                            lower = max(d for d in available_dops_list if d <= candidate_dop)
                            upper = min(d for d in available_dops_list if d >= candidate_dop)
                            if lower == upper:
                                new_child_time = child_tb.pred_dop_exec_time[lower]
                            else:
                                t_lower = child_tb.pred_dop_exec_time[lower]
                                t_upper = child_tb.pred_dop_exec_time[upper]
                                ratio = (candidate_dop - lower) / (upper - lower)
                                new_child_time = t_lower + ratio * (t_upper - t_lower)
                    else:
                        # Fallback: estimate using DOP scaling
                        new_child_time = current_child_time * current_child_dop / (candidate_dop * 0.6)
                    is_estimated = True
                
                child_time_increase = new_child_time - current_child_time
                
                # New redistribution cost (lower because child sends faster)
                new_redist_cost = self._get_redist_penalty(child_tb, my_optimal_dop, candidate_dop)
                
                redist_time_saved = current_redist_cost - new_redist_cost
                
                # Net benefit: redistribution time saved - child execution time increase
                net_benefit = redist_time_saved - child_time_increase
                
                # Calculate benefit/cost ratio for conservative selection
                # If child time increases, the benefit must be significantly larger
                if child_time_increase > 0:
                    benefit_ratio = redist_time_saved / child_time_increase if child_time_increase > 0 else float('inf')
                else:
                    # If child time decreases, it's a win-win
                    benefit_ratio = float('inf')
                
                # Check if this is beneficial
                if net_benefit > best_total_benefit:
                    # Additional checks for conservative selection:
                    # 1. Redistribution saving should be significant
                    threshold_met = redist_time_saved > min_reduction_threshold
                    ratio_met = redist_time_saved / max(current_redist_cost, 1) > min_improvement_ratio
                    
                    # 2. If child time increases, benefit must be at least MIN_BENEFIT_RATIO times the cost
                    benefit_ratio_met = (child_time_increase <= 0 or benefit_ratio >= MIN_BENEFIT_RATIO)
                    
                    # 3. Net benefit should be positive and significant (at least 200ms)
                    net_benefit_significant = net_benefit > 200
                    
                    if (threshold_met or ratio_met) and benefit_ratio_met and net_benefit_significant:
                        best_total_benefit = net_benefit
                        best_alternative_dop = candidate_dop
                        best_benefit_ratio = benefit_ratio
            
            # If we found a better DOP for the child, update it
            if best_alternative_dop != current_child_dop and best_total_benefit > 0:
                # Get new redistribution cost with interpolation
                new_redist_cost = self._get_redist_penalty(child_tb, my_optimal_dop, best_alternative_dop)
                
                # Get child time at best DOP (with interpolation if needed)
                if best_alternative_dop in child_tb.pred_dop_exec_time:
                    new_child_time_final = child_tb.pred_dop_exec_time[best_alternative_dop]
                else:
                    # Estimate using the same logic as in the loop
                    new_child_time_final = current_child_time * current_child_dop / (best_alternative_dop * 0.6)
                
                child_increase_final = new_child_time_final - current_child_time
                
                print(f"  ✓ Adjusting child thread {child_id} DOP: {current_child_dop} -> {best_alternative_dop}")
                print(f"    Redistribution: {current_redist_cost:.1f} -> {new_redist_cost:.1f} ms (saved {current_redist_cost - new_redist_cost:.1f})")
                print(f"    Child exec: {current_child_time:.1f} -> {new_child_time_final:.1f} ms (Δ {child_increase_final:+.1f})")
                print(f"    Net benefit: {best_total_benefit:.1f} ms, Benefit ratio: {best_benefit_ratio:.1f}x")
                
                child_tb.optimal_dop = best_alternative_dop
                child_tb.pred_time = new_child_time_final

