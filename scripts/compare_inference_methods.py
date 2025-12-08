# -*- coding: utf-8 -*-
"""
Compare inference method accuracy differences
Compare accuracy differences between two methods:
1. Using Pipeline Dependency Evaluation Algorithm (PDEA) - based on paper formula, considers throughput matching and wait time
2. Using Thread-wise aggregation (sum within same thread, max across different threads)
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from collections import defaultdict
import torch

# Add parent directory to Python path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_environment, create_dataset_loader
from inference.predict_queries import (
    run_inference, 
    calculate_query_sum_time,
    calculate_query_memory,
    calculate_sum_memory,
    get_root_nodes
)
from core.onnx_manager import ONNXModelManager


def assign_thread_ids(root_node, thread_id=0):
    """
    Recursively assign thread_id to all nodes in the plan tree
    
    Args:
        root_node: Root PlanNode
        thread_id: Starting thread_id
    """
    root_node.thread_id = thread_id
    root_node.visit = False  # Reset visit flag
    
    is_streaming = 'streaming' in root_node.operator_type.lower()
    for child in root_node.child_plans:
        child_thread_id = thread_id + 1 if is_streaming else thread_id
        assign_thread_ids(child, child_thread_id)


class Pipeline:
    """Represents a pipeline with its operators and metrics"""
    def __init__(self, root_operator):
        self.root_operator = root_operator
        self.operators = []  # All operators in this pipeline
        self.child_pipelines = []  # Child pipelines (upstream pipelines)
        self.ts = 0  # Start timestamp
        self.tf = 0  # Finish timestamp
        self.tau_proc = 0  # Total processing time
        self.tau_prep = 0  # Preparation time (time until can output)
        self.tau_prod = 0  # Production time (tau_proc - tau_prep)
        self.tau_in = 0   # Input time (calculated based on throughput)
        self.tau_out = 0  # Output time (calculated based on throughput)
        
    def calculate_metrics(self):
        """Calculate pipeline metrics from operators
        
        Note: In comparison mode, pred_execution_time contains real execution times
        (set before calling calculate_pipeline_dependency_time).
        In inference mode, pred_execution_time would contain predicted values.
        """
        # tau_proc: sum of all operator execution times + transmission time (send_time)
        # According to the paper: M_P(D) = sum(M_i(D)) + M_s(D)
        # where M_i(D) is operator processing time and M_s(D) is transmission time
        # Note: pred_execution_time is set to real values in comparison mode (see compare_inference_methods)
        self.tau_proc = sum(op.pred_execution_time for op in self.operators if hasattr(op, 'pred_execution_time') and op.pred_execution_time > 0)
        
        # Add transmission time (send_time) for the root operator
        # According to the paper: "The data forwarding to downstream consumers is performed 
        # by the top operator of the producer pipeline, and we abstract this sending process 
        # as an additional streaming operation to model the transmission performance."
        if hasattr(self.root_operator, 'send_time') and self.root_operator.send_time > 0:
            self.tau_proc += self.root_operator.send_time
        
        # tau_prep: time from pipeline start until can send data to upper pipeline
        # Start from leaf operators (those receiving data from child_pipelines)
        # Accumulate execution time upward until can output (after build for join/aggregate)
        # Do not include the top operator's execution time (that's production time)
        
        operator_set = set(self.operators)
        
        # Find leaf operators: operators whose children are in child_pipelines (not in current pipeline)
        leaf_ops = []
        for op in self.operators:
            has_child_in_pipeline = False
            for child in op.child_plans:
                if child in operator_set:
                    has_child_in_pipeline = True
                    break
            if not has_child_in_pipeline and len(op.child_plans) > 0:
                # All children are in child_pipelines, this is a leaf in current pipeline
                leaf_ops.append(op)
        
        # If no such leaf operators, find true leaves (no children at all)
        if not leaf_ops:
            for op in self.operators:
                if len(op.child_plans) == 0:
                    leaf_ops.append(op)
        
        # Calculate tau_prep: max path from leaf to output point
        # From leaf operators, accumulate upward along data flow (child -> parent)
        
        def calc_prep_time_from_leaf(op, visited=None):
            """Calculate prep time from this operator upward until can output"""
            if visited is None:
                visited = set()
            if op in visited or op not in operator_set:
                return 0
            
            visited.add(op)
            
            # If this is the root operator, only count build_time (if needed), not execution time
            # Because execution time of root is production time (tau_prod)
            # Root operator is the end point, stop here
            if op == self.root_operator:
                build_only = 0
                op_type_lower = op.operator_type.lower()
                if hasattr(op, 'build_time') and op.build_time > 0:
                    if 'join' in op_type_lower or 'aggregate' in op_type_lower or 'hash' in op_type_lower:
                        build_only = op.build_time
                return build_only
            
            # For non-root operators, add execution time and build_time
            prep_time = 0
            
            # Add execution time
            if hasattr(op, 'pred_execution_time') and op.pred_execution_time > 0:
                prep_time += op.pred_execution_time
            
            # Add build_time for operators that need build (join, aggregate, etc.)
            op_type_lower = op.operator_type.lower()
            if hasattr(op, 'build_time') and op.build_time > 0:
                if 'join' in op_type_lower or 'aggregate' in op_type_lower or 'hash' in op_type_lower:
                    prep_time += op.build_time
            
            # Find parent in current pipeline and accumulate upward
            max_parent_prep = 0
            if hasattr(op, 'parent_node') and op.parent_node in operator_set:
                parent = op.parent_node
                # Only continue if parent is not a streaming operator (boundary)
                if 'streaming' not in parent.operator_type.lower():
                    parent_prep = calc_prep_time_from_leaf(parent, visited.copy())
                    max_parent_prep = max(max_parent_prep, parent_prep)
            
            # Total = current operator prep + max parent prep
            return prep_time + max_parent_prep
        
        # Calculate tau_prep as max over all leaf paths
        max_prep_time = 0
        for leaf_op in leaf_ops:
            prep_time = calc_prep_time_from_leaf(leaf_op)
            max_prep_time = max(max_prep_time, prep_time)
        
        self.tau_prep = max_prep_time
        
        # tau_prod: production time = total processing time - preparation time
        self.tau_prod = self.tau_proc - self.tau_prep if self.tau_proc > self.tau_prep else 0
        
        # tau_in and tau_out will be calculated in evaluate_pipeline_timestamp
        # based on throughput relationships with upstream and downstream pipelines
        self.tau_in = 0
        self.tau_out = 0


def build_pipeline_tree(root_node):
    """
    Build a pipeline tree from operator nodes
    Returns a Pipeline object representing the root pipeline
    
    Pipeline partitioning follows the same logic as assign_thread_ids:
    - If a node is streaming, its children belong to separate (child) pipelines
    - If a node is not streaming, its children belong to the current pipeline
    - Streaming operators themselves belong to the current pipeline
    
    Args:
        root_node: Root PlanNode
        
    Returns:
        Pipeline object
    """
    # Create a new pipeline starting from this operator
    pipeline = Pipeline(root_node)
    
    # Collect all operators in this pipeline
    visited = set()
    streaming_ops_with_children = []
    
    def collect_operators(node):
        if node in visited:
            return
        visited.add(node)
        
        # Add this node to current pipeline
        pipeline.operators.append(node)
        
        # Check if current node is streaming
        is_streaming = 'streaming' in node.operator_type.lower()
        
        # Recursively collect children
        for child in node.child_plans:
            if is_streaming:
                # If current node is streaming, its children start new pipelines
                # Mark this streaming node as having children that need separate pipelines
                if node not in streaming_ops_with_children:
                    streaming_ops_with_children.append(node)
            else:
                # If current node is not streaming, children continue in current pipeline
                collect_operators(child)
    
    collect_operators(root_node)
    
    # Calculate pipeline metrics
    pipeline.calculate_metrics()
    
    # Build child pipelines from streaming operators found
    # Children of streaming operators start new pipelines
    for streaming_op in streaming_ops_with_children:
        for child in streaming_op.child_plans:
            child_pipeline = build_pipeline_tree(child)
            pipeline.child_pipelines.append(child_pipeline)
    
    return pipeline


def evaluate_pipeline_timestamp(pipeline):
    """
    Pipeline Dependency Evaluation Algorithm (PDEA)
    Based on Equation in the paper
    Operates at PIPELINE level
    
    Args:
        pipeline: Pipeline object
        
    Returns:
        Tuple of (start_time, finish_time) for the pipeline
    """
    # If no upstream pipelines (child_pipelines), this is a leaf pipeline
    if not pipeline.child_pipelines:
        # For leaf pipeline: t_s = 0, t_f = tau_in + tau_out
        # tau_in = tau_prep (no upstream, so consumption is just preparation)
        # tau_out = tau_prod (production time)
        pipeline.tau_in = pipeline.tau_prep
        pipeline.tau_out = pipeline.tau_prod
        pipeline.ts = 0
        pipeline.tf = pipeline.tau_in + pipeline.tau_out
        return pipeline.ts, pipeline.tf
    
    # Recursively evaluate all upstream pipelines first
    for upstream_pipeline in pipeline.child_pipelines:
        evaluate_pipeline_timestamp(upstream_pipeline)
    
    # Calculate tau_in based on throughput relationships with upstream pipelines
    # tau_in(P) = tau_prep(P) if T_out(P_u) >= T_in(P), else tau_prod(P_u)
    # Throughput comparison: T_out(P_u) >= T_in(P) means tau_prod(P_u) <= tau_prep(P)
    # Since throughput = 1/time, higher throughput means smaller time
    
    # Calculate tau_in: check if any upstream has lower production throughput
    # (i.e., larger production time) than our consumption throughput
    # For each upstream, if T_out(P_u) < T_in(P), use tau_prod(P_u)
    # Otherwise, use tau_prep(P)
    tau_in = pipeline.tau_prep  # Default: preparation time
    
    for upstream_pipeline in pipeline.child_pipelines:
        # Check if upstream production throughput < downstream consumption throughput
        # T_out(P_u) < T_in(P) means 1/tau_prod(P_u) < 1/tau_prep(P)
        # This means tau_prod(P_u) > tau_prep(P)
        if upstream_pipeline.tau_prod > 0 and pipeline.tau_prep > 0:
            if upstream_pipeline.tau_prod > pipeline.tau_prep:
                # Upstream production throughput < downstream consumption throughput
                # Use upstream production time (max over all upstreams)
                tau_in = max(tau_in, upstream_pipeline.tau_prod)
        elif upstream_pipeline.tau_prod > 0:
            # If downstream tau_prep is 0, use upstream production time
            tau_in = max(tau_in, upstream_pipeline.tau_prod)
    
    pipeline.tau_in = tau_in
    
    # Calculate tau_out: for now, use tau_prod
    # tau_out will be adjusted based on downstream if we traverse from downstream
    # But since we're doing bottom-up, we'll use tau_prod for now
    # Actually, according to the formula, tau_out depends on downstream,
    # but we're calculating bottom-up, so we'll handle it later or use tau_prod
    pipeline.tau_out = pipeline.tau_prod
    
    # Calculate start timestamp
    # t_s(P) = max_{P_u in U(P)} (t_s(P_u) + tau_in(P_u))
    pipeline.ts = 0
    for upstream_pipeline in pipeline.child_pipelines:
        upstream_start_plus_input = upstream_pipeline.ts
        pipeline.ts = max(pipeline.ts, upstream_start_plus_input)
    
    # Calculate finish timestamp
    # Check if all upstream pipelines have T_out(P_u) >= T_in(P)
    # This means: tau_out(P_u) <= tau_in(P) for all upstream
    all_upstream_balanced = True
    for upstream_pipeline in pipeline.child_pipelines:
        # Check if upstream production throughput >= downstream consumption throughput
        # T_out(P_u) >= T_in(P) means tau_out(P_u) <= tau_in(P)
        if upstream_pipeline.tau_prod > pipeline.tau_in:
            all_upstream_balanced = False
            break
    
    # if all_upstream_balanced:
    #     # Case: all upstream pipelines have T_out(P_u) >= T_in(P)
    #     # t_f(P) = t_s(P) + tau_in(P) + tau_out(P)
    #     # When balanced, tau_in(P) = tau_prep(P)
    #     pipeline.tf = pipeline.ts + pipeline.tau_in + pipeline.tau_out
    # else:
    #     # Case: exists upstream pipeline with T_out(P_u) < T_in(P)
    #     # t_f(P) = max_{P_u in U(P)} t_f(P_u) + tau_out(P)
    #     max_upstream_finish = max((upstream_pipeline.tf for upstream_pipeline in pipeline.child_pipelines), default=0)
    #     pipeline.tf = max(pipeline.ts + pipeline.t, max_upstream_finish + pipeline.tau_out)
    max_upstream_finish = max((upstream_pipeline.tf for upstream_pipeline in pipeline.child_pipelines), default=0)
    pipeline.tf = max(pipeline.ts + + pipeline.tau_in + pipeline.tau_out, max_upstream_finish + pipeline.tau_out)
    return pipeline.ts, pipeline.tf


def calculate_pipeline_dependency_time(plan_tree):
    """
    Main entry point for Pipeline Dependency Evaluation Algorithm
    
    Args:
        plan_tree: List of PlanNode objects
        
    Returns:
        Total execution time for the query
    """
    if not plan_tree:
        return 0
    
    # Reset visit flags
    for node in plan_tree:
        node.visit = False
    
    # Find root nodes
    root_nodes = [node for node in plan_tree if node.parent_node is None]
    if not root_nodes and plan_tree:
        root_nodes = [plan_tree[0]]
    
    # Build pipeline tree and evaluate
    max_finish_time = 0
    for root in root_nodes:
        # For all root nodes, build pipeline tree
        # If root is not streaming, it will be the only operator in its pipeline
        root_pipeline = build_pipeline_tree(root)
        _, tf = evaluate_pipeline_timestamp(root_pipeline)
        max_finish_time = max(max_finish_time, tf)
    
    return max_finish_time if max_finish_time > 0 else 1e-6


def calculate_thread_wise_time(plan_tree, query_plan_data):
    """
    Thread-wise aggregation
    - Parent-child nodes in same thread_id: sum time (serial execution)
    - Nodes in different thread_id: take max (parallel execution)
    
    Args:
        plan_tree: List of PlanNode objects
        query_plan_data: DataFrame containing plan execution data
        
    Returns:
        Total execution time using thread-wise aggregation
    """
    # First, assign real execution times
    for node in plan_tree:
        node.visit = False
        # Assign real execution time
        node_row = query_plan_data[query_plan_data['plan_id'] == node.plan_id]
        if not node_row.empty:
            node.pred_execution_time = node_row['execution_time'].values[0]
    
    # Assign thread_ids to all nodes
    root_nodes = [node for node in plan_tree if node.parent_node is None]
    if not root_nodes and plan_tree:
        root_nodes = [plan_tree[0]]
    
    for root in root_nodes:
        assign_thread_ids(root, thread_id=0)
    
    # Step 1: Calculate time for each thread_id (sum within same thread_id)
    thread_time_map = defaultdict(float)
    for node in plan_tree:
        thread_time_map[node.thread_id] += node.pred_execution_time
    
    # Step 2: Build dependency graph between thread_ids
    # If a node in thread_id t2 has a parent in thread_id t1, then t2 depends on t1
    thread_dependencies = defaultdict(set)  # thread_id -> set of thread_ids it depends on
    reverse_deps = defaultdict(set)  # thread_id -> set of thread_ids that depend on it
    
    for node in plan_tree:
        if node.parent_node is not None:
            parent_thread = node.parent_node.thread_id
            child_thread = node.thread_id
            if parent_thread != child_thread:
                # child_thread depends on parent_thread
                thread_dependencies[child_thread].add(parent_thread)
                reverse_deps[parent_thread].add(child_thread)
    
    # Step 3: Build dependency tree and calculate time level by level
    # Start from leaf nodes (thread_ids that no one depends on)
    # Calculate time bottom-up: same level parallel (max), different level serial (sum)
    
    all_thread_ids = set(thread_time_map.keys())
    completed = set()  # Track completed thread_ids
    thread_finish_time = {}  # Track finish time for each thread_id
    
    # Find leaf nodes (thread_ids that no one depends on)
    def get_leaf_nodes():
        """Get thread_ids that have no dependencies or all dependencies are completed"""
        leaves = []
        for tid in all_thread_ids:
            if tid not in completed:
                # Check if all dependencies are completed
                deps = thread_dependencies.get(tid, set())
                if all(dep in completed for dep in deps):
                    leaves.append(tid)
        return leaves
    
    # Calculate time level by level
    total_time = 0
    while len(completed) < len(all_thread_ids):
        # Get current level leaf nodes (can start now)
        current_level = get_leaf_nodes()
        
        if not current_level:
            # Should not happen, but handle circular dependencies
            break
        
        # Calculate finish time for current level
        # Same level nodes run in parallel: level start = max(deps finish), level finish = start + max(execution)
        
        # Find level start time: max of all dependencies' finish time
        level_start_time = 0
        for tid in current_level:
            for dep_tid in thread_dependencies.get(tid, set()):
                level_start_time = max(level_start_time, thread_finish_time.get(dep_tid, 0))
        
        # Calculate each thread_id's finish time in this level
        level_max_execution = 0
        for tid in current_level:
            # Each thread_id starts at level_start_time and runs for its execution time
            thread_finish_time[tid] = level_start_time + thread_time_map[tid]
            level_max_execution = max(level_max_execution, thread_time_map[tid])
        
        # Update total time (max of all thread finish times)
        for tid in current_level:
            total_time = max(total_time, thread_finish_time[tid])
        
        # Mark current level as completed
        for tid in current_level:
            completed.add(tid)
    
    return total_time if total_time > 0 else 0


def calculate_thread_wise_memory(plan_tree, query_plan_data):
    """
    Thread-wise aggregation for memory
    Group nodes by thread_id, sum within same thread, then take max across different threads
    
    Args:
        plan_tree: List of PlanNode objects
        query_plan_data: DataFrame containing plan execution data
        
    Returns:
        Total memory using thread-wise aggregation
    """
    # First, assign real memory
    for node in plan_tree:
        node.visit = False
        # Assign real memory
        node_row = query_plan_data[query_plan_data['plan_id'] == node.plan_id]
        if not node_row.empty:
            node.pred_mem = node_row['peak_mem'].values[0]
    
    # Assign thread_ids to all nodes
    root_nodes = [node for node in plan_tree if node.parent_node is None]
    if not root_nodes and plan_tree:
        root_nodes = [plan_tree[0]]
    
    for root in root_nodes:
        assign_thread_ids(root, thread_id=0)
    
    # Group nodes by thread_id and sum within each thread
    thread_memory_map = defaultdict(float)
    
    for node in plan_tree:
        thread_memory_map[node.thread_id] += node.pred_mem
    
    # Take max across all threads
    if thread_memory_map:
        total_memory = max(thread_memory_map.values())
    else:
        total_memory = 0
    
    return total_memory


def estimate_startup_time(operator_num, streaming_count=0, query_plan_data=None):
    """
    Estimate executor startup time based on operator number and streaming operator count.
    
    Based on empirical analysis of TPCDS and TPCH data, startup time is related to:
    1. Number of operators (operator_num) - primary factor
    2. Number of streaming operators (streaming_count) - additional overhead for pipeline initialization
    
    Empirical observations:
    - Simple queries (9 operators): ~3ms
    - Medium queries (20-30 operators): ~50-200ms
    - Complex queries (60-100 operators): ~800-1000ms
    
    Args:
        operator_num: Number of operators in the query plan
        streaming_count: Number of streaming operators (optional, for better accuracy)
        query_plan_data: DataFrame containing plan execution data (optional, for actual streaming count)
        
    Returns:
        Estimated startup time in ms
    """
    # Linear model based on empirical data analysis
    # Base startup time + per-operator overhead + per-streaming-operator overhead
    base_startup = 0.0  # Base startup time (ms) - minimal base overhead
    per_operator = 10.0  # Per-operator overhead (ms) - based on observed data
    per_streaming = 5.0  # Per-streaming-operator overhead (ms) - additional pipeline setup cost
    
    if streaming_count > 0:
        # Use streaming count if provided for better accuracy
        estimated_time = base_startup + per_operator * operator_num + per_streaming * streaming_count
    else:
        # Fallback to simple linear model based on operator number only
        # This is a reasonable approximation when streaming count is not available
        estimated_time = base_startup + per_operator * operator_num
    
    return max(estimated_time, 0.0)  # Ensure non-negative


def count_streaming_operators(query_plan_data):
    """
    Count the number of streaming operators in a query plan.
    
    Args:
        query_plan_data: DataFrame containing plan execution data
        
    Returns:
        Number of streaming operators
    """
    if query_plan_data is None or query_plan_data.empty:
        return 0
    
    if 'operator_type' in query_plan_data.columns:
        streaming_ops = query_plan_data[
            query_plan_data['operator_type'].str.contains('Streaming', case=False, na=False)
        ]
        return len(streaming_ops)
    
    return 0


def compare_inference_methods(dataset: str, train_mode: str, use_estimates: bool = False, model_dataset: str = None):
    """
    Compare accuracy differences between two inference methods:
    1. Using Pipeline Dependency Evaluation Algorithm (PDEA) - based on paper formula, considers throughput matching and wait time
    2. Using Thread-wise aggregation (sum within same thread, max across different threads) to calculate query latency
    
    Args:
        dataset: Dataset name ('tpch' or 'tpcds')
        train_mode: Training mode ('exact_train' or 'estimated_train')
        use_estimates: Whether to use estimates
        model_dataset: Model dataset name ('tpch' or 'tpcds'), if None use dataset
    """
    print("=" * 80)
    print("Compare inference method accuracy differences")
    print("=" * 80)
    print(f"Dataset: {dataset}")
    if model_dataset is None:
        model_dataset = dataset
    print(f"Model dataset: {model_dataset}")
    print(f"Training mode: {train_mode}")
    print(f"Use estimates: {use_estimates}")
    print("=" * 80)
    
    # Setup environment
    setup_environment()
    
    # Create dataset loader
    loader = create_dataset_loader(dataset)
    file_paths = loader.get_file_paths('test')
    
    plan_csv_path = file_paths['plan_info']
    query_csv_path = file_paths['query_info']
    
    print(f"Plan info file: {plan_csv_path}")
    print(f"Query info file: {query_csv_path}")
    
    # Check if files exist
    if not os.path.exists(plan_csv_path):
        print(f"❌ Plan info file does not exist: {plan_csv_path}")
        return False
    if not os.path.exists(query_csv_path):
        print(f"❌ Query info file does not exist: {query_csv_path}")
        return False
    
    # Load query info
    df_query_info = pd.read_csv(query_csv_path, sep=';')
    print(f"Loaded {len(df_query_info)} query records")
    
    # Set model paths
    if model_dataset is None:
        model_dataset = dataset  # If not specified, use models trained on same dataset
    no_dop_model_dir = f"output/{model_dataset}/models/{train_mode}/operator_non_dop_aware"
    dop_model_dir = f"output/{model_dataset}/models/{train_mode}/operator_dop_aware"
    
    print(f"Non-DOP model directory: {no_dop_model_dir}")
    print(f"DOP model directory: {dop_model_dir}")
    
    # Check if model directories exist
    if not os.path.exists(no_dop_model_dir):
        print(f"❌ Non-DOP model directory does not exist: {no_dop_model_dir}")
        return False
    if not os.path.exists(dop_model_dir):
        print(f"❌ DOP model directory does not exist: {dop_model_dir}")
        return False
    
    # Create ONNX manager
    onnx_manager = ONNXModelManager(no_dop_model_dir, dop_model_dir)
    
    # Parse plans - use logic from run_inference
    print("\n📋 Parsing query plans...")
    
    # Read execution plan data
    df_plans = pd.read_csv(plan_csv_path, delimiter=';', encoding='utf-8')
    query_groups = df_plans.groupby(['query_id', 'query_dop'])
    
    # Create PlanNode objects and process tree structure for each query
    query_trees = {}
    
    # Only process queries from test data
    target_dop = 64
    for (query_id, query_dop), group_df in query_groups:
        if query_id > 0 and query_dop == target_dop:
            # Create PlanNode objects
            nodes = {}
            for _, row in group_df.iterrows():
                from core.plan_node import PlanNode
                node = PlanNode(row, onnx_manager, use_estimates)
                nodes[row['plan_id']] = node
            
            # Build parent-child relationships
            for _, row in group_df.iterrows():
                node = nodes[row['plan_id']]
                if pd.notna(row['child_plan']) and row['child_plan'] != '':
                    child_ids = [int(pid.strip()) for pid in str(row['child_plan']).split(',')]
                    for child_id in child_ids:
                        if child_id in nodes:
                            node.child_plans.append(nodes[child_id])
                            nodes[child_id].parent_node = node
            
            # Store query tree
            query_trees[(query_id, query_dop)] = list(nodes.values())
    
    print(f"Parsed {len(query_trees)} query plans (only DOP={target_dop})")
    
    # Store comparison results
    comparison_results = []
    
    print("\n🔍 Starting inference method comparison...")
    
    for i, ((query_id, query_dop), plan_tree) in enumerate(query_trees.items()):
        if i % 50 == 0:
            print(f"Processing progress: {i}/{len(query_trees)} ({i/len(query_trees)*100:.1f}%)")
        
        # Get actual execution time and memory usage
        actual_time_row = df_query_info[
            (df_query_info['query_id'] == query_id) & 
            (df_query_info['dop'] == query_dop)
        ]
        
        if actual_time_row.empty:
            continue
            
        actual_time = actual_time_row['execution_time'].values[0] - actual_time_row['executor_start_time'].values[0]
        actual_memory = actual_time_row['query_used_mem'].values[0]
        
        # Get plan data for this query
        query_plan_data = df_plans[
            (df_plans['query_id'] == query_id) & 
            (df_plans['query_dop'] == query_dop)
        ]
        
        # Method 1: Pipeline Dependency Evaluation Algorithm (PDEA) based on paper formula
        start_time = time.time()
        # First assign real execution time to each node
        for node in plan_tree:
            node.visit = False
            # Find corresponding real execution time
            node_row = query_plan_data[query_plan_data['plan_id'] == node.plan_id]
            if not node_row.empty:
                node.pred_execution_time = node_row['execution_time'].values[0]  # Use original execution_time
                # Also load real send_time and build_time for throughput calculation
                if 'stream_data_send_time' in query_plan_data.columns and 'stream_quota_time' in query_plan_data.columns:
                    node.send_time = node_row['stream_data_send_time'].values[0] - node_row['stream_quota_time'].values[0]
                if 'build_time' in query_plan_data.columns:
                    node.build_time = node_row['build_time'].values[0]
        if query_id == 34:
            print()
        predicted_time_mapping = calculate_pipeline_dependency_time(plan_tree)
        pred_exec_time_mapping = sum(node.pred_exec_time for node in plan_tree)
        
        # Add startup time to predicted time
        operator_num = actual_time_row['operator_num'].values[0]
        streaming_count = count_streaming_operators(query_plan_data)
        # startup_time_estimated = estimate_startup_time(operator_num, streaming_count, query_plan_data)
        startup_time_estimated = 0
        predicted_time_mapping = predicted_time_mapping + startup_time_estimated
        
        mapping_time = time.time() - start_time + pred_exec_time_mapping
        
        # Method 2: Thread-wise aggregation (sum within same thread, max across different threads)
        start_time = time.time()
        thread_wise_time = calculate_thread_wise_time(plan_tree, query_plan_data)
        thread_wise_time_elapsed = time.time() - start_time
        
        # Calculate memory prediction (based on real operator memory usage)
        start_time = time.time()
        # First assign real memory usage to each node
        for node in plan_tree:
            node_row = query_plan_data[query_plan_data['plan_id'] == node.plan_id]
            if not node_row.empty:
                node.pred_mem = node_row['peak_mem'].values[0]  # Use real value
        predicted_memory_mapping, _ = calculate_query_memory(plan_tree)
        pred_mem_time_mapping = sum(node.pred_mem_time for node in plan_tree)
        memory_mapping_time = time.time() - start_time + pred_mem_time_mapping
        
        # Method 2: Thread-wise aggregation for memory (sum within same thread, max across different threads)
        start_time = time.time()
        thread_wise_memory = calculate_thread_wise_memory(plan_tree, query_plan_data)
        thread_wise_memory_time = time.time() - start_time
        
        # Ensure predicted values are scalars
        if isinstance(predicted_time_mapping, (np.ndarray,)):
            predicted_time_mapping = predicted_time_mapping.item()
        if isinstance(predicted_memory_mapping, (np.ndarray,)):
            predicted_memory_mapping = predicted_memory_mapping.item()
        if isinstance(thread_wise_time, (np.ndarray,)):
            thread_wise_time = thread_wise_time.item()
        if isinstance(thread_wise_memory, (np.ndarray,)):
            thread_wise_memory = thread_wise_memory.item()
        
        # Calculate Q-error
        def calculate_q_error(actual, predicted):
            if actual == 0 or predicted == 0:
                return float('inf')
            return max(predicted / actual, actual / predicted) - 1
        
        # Execution time Q-error
        q_error_mapping = calculate_q_error(actual_time, predicted_time_mapping)  # Mapping algorithm vs actual query time
        q_error_thread_wise = calculate_q_error(actual_time, thread_wise_time)  # Thread-wise vs actual query time
        
        # Memory Q-error
        q_error_mem_mapping = calculate_q_error(actual_memory, predicted_memory_mapping)  # Mapping algorithm vs actual query memory
        q_error_mem_thread_wise = calculate_q_error(actual_memory, thread_wise_memory)  # Thread-wise vs actual query memory
        
        # Store results
        result = {
            'query_id': query_id,
            'dop': query_dop,
            'actual_time': actual_time,
            'predicted_time_mapping': predicted_time_mapping,  # Query time predicted by mapping algorithm
            'thread_wise_time': thread_wise_time,  # Query time predicted by Thread-wise
            'actual_memory': actual_memory,
            'predicted_memory_mapping': predicted_memory_mapping,  # Query memory predicted by mapping algorithm
            'thread_wise_memory': thread_wise_memory,  # Query memory predicted by Thread-wise
            'q_error_mapping': q_error_mapping,  # Mapping algorithm vs actual query time
            'q_error_thread_wise': q_error_thread_wise,  # Thread-wise vs actual query time
            'q_error_mem_mapping': q_error_mem_mapping,  # Mapping algorithm vs actual query memory
            'q_error_mem_thread_wise': q_error_mem_thread_wise,  # Thread-wise vs actual query memory
            'mapping_inference_time': mapping_time,
            'thread_wise_inference_time': thread_wise_time_elapsed,
            'memory_mapping_inference_time': memory_mapping_time,
            'memory_thread_wise_inference_time': thread_wise_memory_time,
            'time_difference_thread_wise': thread_wise_time - predicted_time_mapping,  # Thread-wise vs mapping algorithm
            'time_difference_thread_wise_ratio': (thread_wise_time - predicted_time_mapping) / predicted_time_mapping if predicted_time_mapping != 0 else 0,
            'memory_difference_thread_wise': thread_wise_memory - predicted_memory_mapping,  # Thread-wise vs mapping algorithm
            'memory_difference_thread_wise_ratio': (thread_wise_memory - predicted_memory_mapping) / predicted_memory_mapping if predicted_memory_mapping != 0 else 0,
            'q_error_improvement_thread_wise': q_error_mapping - q_error_thread_wise,  # Thread-wise improvement over mapping algorithm
            'q_error_mem_improvement_thread_wise': q_error_mem_mapping - q_error_mem_thread_wise  # Thread-wise improvement over mapping algorithm
        }
        
        comparison_results.append(result)
    
    # Create results DataFrame
    df_results = pd.DataFrame(comparison_results)
    
    if len(df_results) == 0:
        print("❌ No valid comparison results found")
        return False
    
    # Calculate statistics
    print("\n📊 Comparison result statistics:")
    print("=" * 80)
    
    # Execution time statistics
    print("Execution time prediction comparison:")
    print(f"  Mapping algorithm average Q-error: {df_results['q_error_mapping'].mean():.6f} (mapping algorithm based on real operator time vs actual query time)")
    print(f"  Thread-wise average Q-error: {df_results['q_error_thread_wise'].mean():.6f} (Thread-wise based on real operator time vs actual query time)")
    print(f"  Q-error difference (Thread-wise - mapping): {df_results['q_error_thread_wise'].mean() - df_results['q_error_mapping'].mean():.6f}")
    print(f"  Q-error improvement (Thread-wise relative to mapping): {df_results['q_error_improvement_thread_wise'].mean():.6f}")
    
    # Memory statistics
    print("\nMemory prediction comparison:")
    print(f"  Mapping algorithm average Q-error: {df_results['q_error_mem_mapping'].mean():.6f} (mapping algorithm based on real operator memory vs actual query memory)")
    print(f"  Thread-wise average Q-error: {df_results['q_error_mem_thread_wise'].mean():.6f} (Thread-wise based on real operator memory vs actual query memory)")
    print(f"  Q-error difference (Thread-wise - mapping): {df_results['q_error_mem_thread_wise'].mean() - df_results['q_error_mem_mapping'].mean():.6f}")
    print(f"  Q-error improvement (Thread-wise relative to mapping): {df_results['q_error_mem_improvement_thread_wise'].mean():.6f}")
    
    # Prediction value difference statistics
    print("\nPrediction value difference statistics:")
    print(f"  Execution time average difference (Thread-wise vs mapping): {df_results['time_difference_thread_wise'].mean():.2f}ms")
    print(f"  Execution time difference ratio (Thread-wise vs mapping): {df_results['time_difference_thread_wise_ratio'].mean() * 100:.2f}%")
    print(f"  Memory average difference (Thread-wise vs mapping): {df_results['memory_difference_thread_wise'].mean():.2f}KB")
    print(f"  Memory difference ratio (Thread-wise vs mapping): {df_results['memory_difference_thread_wise_ratio'].mean() * 100:.2f}%")
    
    # Inference time statistics
    print("\nInference time comparison:")
    print(f"  Mapping algorithm average inference time: {df_results['mapping_inference_time'].mean():.6f}s")
    print(f"  Thread-wise average inference time: {df_results['thread_wise_inference_time'].mean():.6f}s")
    print(f"  Inference time difference (Thread-wise - mapping): {df_results['thread_wise_inference_time'].mean() - df_results['mapping_inference_time'].mean():.6f}s")
    
    # Save detailed results
    output_dir = f"output/comparison/{dataset}/{train_mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "inference_methods_comparison.csv")
    df_results.to_csv(output_file, index=False, sep=';')
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Create simplified CSV with only Q-error metrics for comparison
    q_error_comparison = df_results[[
        'query_id', 
        'dop',
        'actual_time',
        'q_error_mapping',
        'q_error_thread_wise',
        'actual_memory',
        'q_error_mem_mapping',
        'q_error_mem_thread_wise'
    ]].copy()
    
    # Rename columns for better readability
    q_error_comparison.columns = [
        'query_id',
        'dop',
        'actual_time_ms',
        'q_error_mapping_time',
        'q_error_thread_wise_time',
        'actual_memory_kb',
        'q_error_mapping_memory',
        'q_error_thread_wise_memory'
    ]
    
    q_error_file = os.path.join(output_dir, "q_error_comparison.csv")
    q_error_comparison.to_csv(q_error_file, index=False, sep=';')
    print(f"💾 Q-error comparison table saved to: {q_error_file}")
    
    # Generate summary report
    summary_file = os.path.join(output_dir, "comparison_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Inference Method Comparison Summary Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Training mode: {train_mode}\n")
        f.write(f"Use estimates: {use_estimates}\n")
        f.write(f"Number of queries compared: {len(df_results)}\n")
        f.write("\n")
        
        f.write("Execution time prediction comparison:\n")
        f.write(f"  Mapping algorithm average Q-error: {df_results['q_error_mapping'].mean():.6f} (mapping algorithm based on real operator time vs actual query time)\n")
        f.write(f"  Thread-wise average Q-error: {df_results['q_error_thread_wise'].mean():.6f} (Thread-wise based on real operator time vs actual query time)\n")
        f.write(f"  Q-error difference (Thread-wise - mapping): {df_results['q_error_thread_wise'].mean() - df_results['q_error_mapping'].mean():.6f}\n")
        f.write(f"  Q-error improvement (Thread-wise relative to mapping): {df_results['q_error_improvement_thread_wise'].mean():.6f}\n")
        f.write("\n")
        
        f.write("Memory prediction comparison:\n")
        f.write(f"  Mapping algorithm average Q-error: {df_results['q_error_mem_mapping'].mean():.6f} (mapping algorithm based on real operator memory vs actual query memory)\n")
        f.write(f"  Thread-wise average Q-error: {df_results['q_error_mem_thread_wise'].mean():.6f} (Thread-wise based on real operator memory vs actual query memory)\n")
        f.write(f"  Q-error difference (Thread-wise - mapping): {df_results['q_error_mem_thread_wise'].mean() - df_results['q_error_mem_mapping'].mean():.6f}\n")
        f.write(f"  Q-error improvement (Thread-wise relative to mapping): {df_results['q_error_mem_improvement_thread_wise'].mean():.6f}\n")
        f.write("\n")
        
        f.write("Prediction value difference statistics:\n")
        f.write(f"  Execution time average difference (Thread-wise vs mapping): {df_results['time_difference_thread_wise'].mean():.2f}ms\n")
        f.write(f"  Execution time difference ratio (Thread-wise vs mapping): {df_results['time_difference_thread_wise_ratio'].mean() * 100:.2f}%\n")
        f.write(f"  Memory average difference (Thread-wise vs mapping): {df_results['memory_difference_thread_wise'].mean():.2f}KB\n")
        f.write(f"  Memory difference ratio (Thread-wise vs mapping): {df_results['memory_difference_thread_wise_ratio'].mean() * 100:.2f}%\n")
        f.write("\n")
        
        f.write("Inference time comparison:\n")
        f.write(f"  Mapping algorithm average inference time: {df_results['mapping_inference_time'].mean():.6f}s\n")
        f.write(f"  Thread-wise average inference time: {df_results['thread_wise_inference_time'].mean():.6f}s\n")
        f.write(f"  Inference time difference (Thread-wise - mapping): {df_results['thread_wise_inference_time'].mean() - df_results['mapping_inference_time'].mean():.6f}s\n")
    
    print(f"📋 Summary report saved to: {summary_file}")
    
    print("\n✅ Inference method comparison completed!")
    return True


if __name__ == "__main__":
    # Can control comparison by modifying these parameters
    DATASET = 'tpch'  # 'tpch' or 'tpcds'
    TRAIN_MODE = 'exact_train'  # 'exact_train' or 'estimated_train'
    USE_ESTIMATES = False  # True or False
    
    success = compare_inference_methods(DATASET, TRAIN_MODE, USE_ESTIMATES)
    if not success:
        sys.exit(1)
