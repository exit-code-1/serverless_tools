# 文件路径: inference/predict_non_dop_queries.py

import math
import os
import re
import sys
import numpy as np
import pandas as pd
# import networkx as nx # 原始文件未使用，注释掉
# import matplotlib.pyplot as plt # 原始文件未使用，注释掉
import time
import torch # 保留，PlanNode 可能用到

# --- 导入重构后的模块 ---
# 使用绝对导入路径
from core.plan_node import PlanNode
from core.onnx_manager import ONNXModelManager
from core.pipeline_pair import Segment
from core.thread_block import ThreadBlock
from core.pdg_builder import convert_stage_dag_to_pdg, Pipeline
from config.structure_config import thread_cost, thread_mem
# -------------------------

# ==============================================================================
# 辅助函数 (计算执行时间和内存) - 从原始文件直接搬运
# (这些函数逻辑保持不变)
# ==============================================================================

def base_execution_time(node):
    """返回当前节点的基本执行时间"""
    if node.operator_type == 'CTE Scan':
        return 0
    # 原始逻辑是直接使用 pred_execution_time
    return node.pred_execution_time

def process_same_thread_child(child, thread_id):
    """处理子节点在同一线程内的情况"""
    return calculate_thread_execution_time(child, thread_id)

def process_new_thread_child(child, new_thread_id, parent_thread_time):
    """处理子节点属于新线程的情况"""
    _, child_complete, _, child_up, _, _ = calculate_thread_execution_time(child, new_thread_id)
    # 原始逻辑包含这个调整
    adjustment = 0
    adjusted_child_complete = child_complete + adjustment
    return adjustment, adjusted_child_complete, child_up

def aggregate_child_results(same_results, new_results):
    """合并子节点结果"""
    child_exec_list = []
    child_complete_list = []
    local_transfer_list = []
    up_transfer_list = []

    for res in same_results:
        child_exec_list.append(res[0])
        child_complete_list.append(res[1])
        local_transfer_list.append(res[2])
        up_transfer_list.append(res[3])
    for res in new_results:
        child_complete_list.append(res[1])
        local_transfer_list.append(res[0] + res[2]) # 检查原始代码确认此逻辑
        up_transfer_list.append(res[2])
    return child_exec_list, child_complete_list, local_transfer_list, up_transfer_list

def final_adjustment(thread_exec, local_transfer, child_complete_list, more_agg_times, tmp_flag):
    """最终调整完成时间"""
    child_complete = max(child_complete_list, default=0)
    thread_complete = max(thread_exec + local_transfer, child_complete)
    if tmp_flag and (thread_exec + local_transfer < child_complete):
        thread_complete = max(thread_complete, child_complete + thread_exec - more_agg_times)
        more_agg_times = thread_exec
        tmp_flag = False
    return thread_complete, more_agg_times, tmp_flag


def calculate_thread_execution_time(node, thread_id):
    """递归计算线程时间和相关指标 (保持原始逻辑)"""
    if node.visit:
        return 0, 0, 0, 0, 0, False
    node.visit = True

    thread_exec = base_execution_time(node)
    more_agg = False
    tmp_flag = False
    more_agg_times = 0 # 初始化

    same_results = []
    new_results = []

    is_streaming = 'streaming' in node.operator_type.lower()
    for child in node.child_plans:
        new_thread_id = thread_id + 1 if is_streaming else thread_id
        if new_thread_id == thread_id:
            res = process_same_thread_child(child, new_thread_id)
            same_results.append(res)
        else:
            res = process_new_thread_child(child, new_thread_id, thread_exec)
            new_results.append(res)

    child_exec_list, child_complete_list, local_transfer_list, up_transfer_list = aggregate_child_results(same_results, new_results)

    local_transfer = max(local_transfer_list, default=0)
    up_transfer = max(up_transfer_list, default=0)

    # 累加同一线程的执行时间 (假设 base_execution_time 和子结果都是数值)
    thread_exec += sum(c or 0 for c in child_exec_list) # 使用 or 0 处理 None

    if node.materialized:
        if 'hash join' in node.operator_type.lower():
            # 假设 child_exec_list[0] 是右子树（构建哈希表）的时间？保持原始逻辑
            child_exec_first = child_exec_list[0] if child_exec_list and child_exec_list[0] is not None else 0
            up_transfer = max(up_transfer, local_transfer + thread_exec - child_exec_first)
        else:
            up_transfer = max(up_transfer, local_transfer + thread_exec - node.build_time)

        if not more_agg:
            more_agg_times = thread_exec
            more_agg = True
        else:
            tmp_flag = True

    child_complete = max(child_complete_list, default=0)
    thread_complete, more_agg_times, tmp_flag = final_adjustment(
        thread_exec, local_transfer, child_complete_list, more_agg_times, tmp_flag
    )

    # 原始代码似乎没有将这些中间结果存回 node 对象，我们保持一致
    # node.thread_execution_time = thread_exec
    # ...

    return (
        thread_exec, thread_complete, local_transfer, up_transfer,
        more_agg_times if more_agg else 0, # 只有 more_agg 为 True 才返回 more_agg_times
        more_agg
    )


# ==============================================================================
# Segment-based Query Latency Prediction Algorithm
# Algorithm 1: Segment-based Query Latency Prediction Algorithm
# ==============================================================================

def segment_latency(seg):
    """
    Calculate the latency of a segment.
    Algorithm line 1-5: SegmentLatency(Seg)
    
    Args:
        seg: Segment object
        
    Returns:
        float: Latency of the segment
    """
    if seg.is_inner_stage:
        # Inner-stage segment: return t(p) where p is the pipeline in this segment
        # Line 2-3: if Seg is an inner-stage segment then return t(p)
        if seg.pipeline_latencies:
            return seg.pipeline_latencies[0]  # Single pipeline
        return 0.0
    else:
        # Cross-stage segment: return max_{p∈Seg} t(p)
        # Line 4-5: else if Seg is a cross-stage segment then return max_{p∈Seg} t(p)
        if seg.pipeline_latencies:
            return max(seg.pipeline_latencies)
        return 0.0

def eval_segment(seg):
    """
    Recursively calculate the completion time of a segment.
    Algorithm line 6-13: Eval(Seg)
    
    Args:
        seg: Segment object
        
    Returns:
        float: Completion time of the segment
    """
    # #region agent log
    import json
    log_path = '/home/zhy/opengauss/tools/new_serverless_predictor/.cursor/debug.log'
    seg_id = id(seg)
    seg_nodes_count = len(seg.nodes) if seg.nodes else 0
    seg_latencies = seg.pipeline_latencies if seg.pipeline_latencies else []
    upstream_count = len(seg.upstream_segments) if seg.upstream_segments else 0
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"predict_queries.py:168","message":"eval_segment entry","data":{"seg_id":seg_id,"is_inner_stage":seg.is_inner_stage,"nodes_count":seg_nodes_count,"pipeline_latencies":seg_latencies,"upstream_count":upstream_count},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    # Line 7-8: if Seg has no upstream segment then return SegmentLatency(Seg)
    if not seg.upstream_segments:
        result = segment_latency(seg)
        # #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"predict_queries.py:180","message":"eval_segment no upstream","data":{"seg_id":seg_id,"segment_latency":result,"result":result},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        return result
    
    # Line 9-13: else (if Seg has upstream segments)
    # Special case: if this is a final aggregate segment (nodes empty but has upstream_segments),
    # its pipeline_latencies already contains all upstream latencies, so just return max(pipeline_latencies)
    # without adding upstream completion time to avoid double counting
    # if len(seg.nodes) == 0 and seg.upstream_segments:
    #     # This is a final aggregate segment - its pipeline_latencies already contains all root segments' latencies
    #     result = segment_latency(seg)
    #     # #region agent log
    #     with open(log_path, 'a') as f:
    #         f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"predict_queries.py:193","message":"eval_segment final aggregate segment","data":{"seg_id":seg_id,"segment_latency":result,"result":result},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    #     # #endregion
    #     return result
    
    # Line 10: Lup ← 0
    lup = 0.0
    
    # Line 11-12: for each upstream segment Seg_u do Lup ← max{Lup, Eval(Segu)}
    upstream_results = []
    for seg_u in seg.upstream_segments:
        upstream_completion = eval_segment(seg_u)
        upstream_results.append(upstream_completion)
        lup = max(lup, upstream_completion)
    
    seg_lat = segment_latency(seg)
    result = lup + seg_lat
    
    # #region agent log
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"predict_queries.py:193","message":"eval_segment with upstream","data":{"seg_id":seg_id,"upstream_results":upstream_results,"lup":lup,"segment_latency":seg_lat,"result":result,"is_final_segment":seg_nodes_count==0 and upstream_count>0},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    # Line 13: return Lup + SegmentLatency(Seg)
    return result

def collect_nodes_and_dop_in_path(start_node, end_node):
    """
    Collect all nodes and their DOP information in the path from start to end.
    Groups nodes by stage (identified by streaming boundaries or DOP changes).
    
    Args:
        start_node: Starting node
        end_node: End node
        
    Returns:
        tuple: (list of nodes, dict of dop_info by stage)
    """
    nodes = []
    dop_info = {}
    current = start_node
    stage_id = 0
    current_dop = None
    
    while current and current != end_node:
        nodes.append(current)
        node_dop = current.dop if hasattr(current, 'dop') else 1
        
        # Check if this is a streaming boundary (stage boundary) or DOP change
        is_streaming = 'streaming' in current.operator_type.lower() if current else False
        dop_changed = (current_dop is not None and node_dop != current_dop)
        
        # Start new stage if streaming boundary or DOP changed
        if is_streaming or dop_changed or stage_id == 0:
            if is_streaming or dop_changed:
                stage_id += 1
            if stage_id not in dop_info:
                dop_info[stage_id] = {'dop': node_dop, 'nodes': []}
            dop_info[stage_id]['nodes'].append(current)
            current_dop = node_dop
        else:
            # Add to current stage
            if stage_id not in dop_info:
                dop_info[stage_id] = {'dop': node_dop, 'nodes': []}
            dop_info[stage_id]['nodes'].append(current)
        
        current = current.parent_node
    
    if current == end_node:
        nodes.append(end_node)
        end_dop = end_node.dop if hasattr(end_node, 'dop') else 1
        # Check if end node belongs to a new stage
        if current_dop is not None and end_dop != current_dop:
            stage_id += 1
        if stage_id not in dop_info:
            dop_info[stage_id] = {'dop': end_dop, 'nodes': []}
        dop_info[stage_id]['nodes'].append(end_node)
    
    return nodes, dop_info


def assign_thread_ids(node, thread_id=0, max_thread_id=0):
    """
    Recursively assign thread IDs to nodes.
    Streaming operators create new thread boundaries (stage boundaries).
    """
    node.thread_id = thread_id
    max_thread_id = max(max_thread_id, thread_id)
    
    if 'streaming' in node.operator_type.lower():
        new_thread_id = max_thread_id + 1
    else:
        new_thread_id = thread_id
    
    for child in node.child_plans:
        max_thread_id = assign_thread_ids(child, new_thread_id, max_thread_id)
    
    return max_thread_id

def collect_all_nodes(root_nodes):
    """Collect all nodes from root nodes using DFS."""
    all_nodes = []
    visited_ids = set()
    stack = list(root_nodes)
    while stack:
        node = stack.pop()
        if node.plan_id in visited_ids:
            continue
        visited_ids.add(node.plan_id)
        all_nodes.append(node)
        stack.extend(node.child_plans)
    return all_nodes

def build_thread_blocks_from_nodes(all_nodes):
    """
    Build ThreadBlocks (stages) from plan nodes.
    Each ThreadBlock represents a stage in the Stage DAG.
    """
    # Assign thread IDs to all nodes
    root_nodes = [node for node in all_nodes if node.parent_node is None]
    if not root_nodes and all_nodes:
        root_nodes = [all_nodes[0]]
    
    current_offset = 0
    for root in root_nodes:
        max_tid = assign_thread_ids(root, thread_id=current_offset)
        current_offset = max_tid + 1
    
    # Group nodes by thread_id to create ThreadBlocks
    thread_blocks = {}
    for node in all_nodes:
        tid = getattr(node, 'thread_id', 0)
        if tid not in thread_blocks:
            thread_blocks[tid] = ThreadBlock(tid, [])
        thread_blocks[tid].nodes.append(node)
    
    # Build parent-child relationships between ThreadBlocks
    for node in all_nodes:
        if node.parent_node:
            parent_tid = getattr(node.parent_node, 'thread_id', 0)
            child_tid = getattr(node, 'thread_id', 0)
            if parent_tid != child_tid:
                if parent_tid in thread_blocks:
                    thread_blocks[parent_tid].child_thread_ids.add(child_tid)
    
    return thread_blocks

def calculate_query_execution_time(all_nodes):
    """
    Calculate query execution time using PDG-based algorithm.
    Two-step process:
    1. Convert Stage DAG (ThreadBlocks) to PDG (Pipelines)
    2. Calculate latency based on PDG using Algorithm 1
    """
    if not all_nodes:
        return 0
    
    # Step 1: Build ThreadBlocks (Stage DAG) from plan nodes
    thread_blocks = build_thread_blocks_from_nodes(all_nodes)
    
    if not thread_blocks:
        return 0
    
    # Step 2: Convert Stage DAG to PDG (build segments from bottom to top)
    top_segment = convert_stage_dag_to_pdg(thread_blocks, all_nodes)
    
    # #region agent log
    import json
    log_path = '/home/zhy/opengauss/tools/new_serverless_predictor/.cursor/debug.log'
    with open(log_path, 'a') as f:
        top_seg_id = id(top_segment)
        upstream_ids = [id(us) for us in top_segment.upstream_segments] if top_segment.upstream_segments else []
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"predict_queries.py:587","message":"top_segment after convert_stage_dag_to_pdg","data":{"top_seg_id":top_seg_id,"upstream_count":len(top_segment.upstream_segments) if top_segment.upstream_segments else 0,"upstream_ids":upstream_ids,"self_in_upstream":top_seg_id in upstream_ids},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    # Step 3: Calculate query latency based on PDG using Algorithm 1
    # Use eval_pipeline_pair to recursively calculate completion time
    # This correctly handles:
    # - Inner-stage segments: return t(p) (single pipeline latency)
    # - Cross-stage segments: return max_{p∈Seg} t(p) (max of concurrent pipeline latencies)
    query_execution_time = eval_segment(top_segment)
    
    # #region agent log
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"predict_queries.py:641","message":"query_execution_time calculated","data":{"query_execution_time":query_execution_time},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    # Calculate thread overhead (keep original logic)
    total_threads = 1
    dop_sum_streaming = 0
    for node in all_nodes:
        if 'Streaming' in node.operator_type:
            dop_match = re.search(r'dop:\s*\d+\/(\d+)', node.operator_type)
            if dop_match:
                try:
                    threads_generated = int(dop_match.group(1))
                    dop_sum_streaming += threads_generated
                except (ValueError, IndexError):
                    dop_sum_streaming += node.downdop
            else:
                dop_sum_streaming += node.downdop
    
    total_threads += dop_sum_streaming
    thread_overhead = total_threads * thread_cost
    final_time = query_execution_time + thread_overhead
    
    # #region agent log
    with open(log_path, 'a') as f:
        # Calculate sum of all node execution times for comparison
        total_node_time = sum(node.pred_execution_time for node in all_nodes)
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"predict_queries.py:659","message":"final time calculation","data":{"query_execution_time":query_execution_time,"total_threads":total_threads,"thread_cost":thread_cost,"thread_overhead":thread_overhead,"final_time":final_time,"total_node_time":total_node_time,"ratio":final_time/total_node_time if total_node_time > 0 else 0},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    return final_time if final_time > 0 else 1e-6


def calculate_query_memory(query_nodes):
    """计算整个查询的预测内存占用 (保持原始逻辑)"""
    total_peak_mem = 0
    total_threads = 1
    dop_sum_streaming = 0

    for node in query_nodes:
        # 累加算子的预测内存
        pred_mem = node.pred_mem
        total_peak_mem += pred_mem if pred_mem is not None else 0

        # 计算 streaming 算子产生的线程数 (保持和上面时间计算一致的逻辑)
        if 'Streaming' in node.operator_type:
             dop_match = re.search(r'dop:\s*\d+\/(\d+)', node.operator_type)
             if dop_match:
                try:
                    threads_generated = int(dop_match.group(1)) # 保持和时间计算一致
                    dop_sum_streaming += threads_generated
                except (ValueError, IndexError):
                    dop_sum_streaming += node.downdop
             else:
                 dop_sum_streaming += node.downdop

    total_threads += dop_sum_streaming

    # 总内存计算方式 (保持原始的 1.15 次方缩放)
    thread_memory_cost = int(math.pow(max(0, total_threads - 1), 1.15) * thread_mem)
    total_memory = total_peak_mem + thread_memory_cost

    return total_memory if total_memory > 0 else 1024, total_threads


# ==============================================================================
# 主推理逻辑函数
# ==============================================================================

def run_inference(plan_csv_path, query_csv_path, output_csv_path, no_dop_model_dir, dop_model_dir, use_estimates=False):
    """
    执行推理流程 (严格按照提供的原始顶层逻辑封装)。

    Args:
        plan_csv_path (str): plan_info.csv 文件路径。
        query_csv_path (str): query_info.csv 文件路径。
        output_csv_path (str): 保存预测结果的 CSV 文件路径。
        no_dop_model_dir (str): 非 DOP 模型目录路径。
        dop_model_dir (str): DOP 模型目录路径。
        use_estimates (bool): 是否使用估计值。
    """

    # === 开始: 严格复制粘贴原始顶层逻辑 ===

    # 读取执行计划数据
    df_plans = pd.read_csv(plan_csv_path, delimiter=';', encoding='utf-8') # 使用参数
    # df_plans = df_plans[df_plans['query_dop'] == 8].copy() # 原始代码注释掉了这行
    df_query_info = pd.read_csv(query_csv_path, delimiter=';', encoding='utf-8') # 使用参数
    # df_query_info = df_query_info[df_query_info['dop'] == 8].copy() # 原始代码注释掉了这行
    # 按 query_id 和 query_dop 分组
    query_groups = df_plans.groupby(['query_id', 'query_dop'])

    # 创建 PlanNode 对象并处理每个查询的树结构
    query_trees = {}

    # 初始化 ONNX 管理器 (使用参数)
    onnx_manager = ONNXModelManager(
        no_dop_model_dir=no_dop_model_dir,
        dop_model_dir=dop_model_dir
    )

    # 仅处理测试数据的查询 (原始逻辑包含 if query_id > 0)
    # Import build_query_plan function
    from utils.data_utils import build_query_plan
    
    for (query_id, query_dop), group in query_groups:
        if query_id >0 : # Only process query_id == 1 for debugging
            # 使用 (query_id, query_dop) 作为键
            key = (query_id, query_dop) # 保持原始键
            
            # Use build_query_plan to create nodes with build/probe splitting
            nodes_dict, root_nodes_list = build_query_plan(group, use_estimates=use_estimates, onnx_manager=onnx_manager)
            query_trees[key] = list(nodes_dict.values())

    # 打开日志文件进行写入 (原始逻辑，但路径处理可能需要调整或移除)
    # log_file_path = 'query_comparison_log.txt' # 硬编码文件名
    # 暂时注释掉日志写入，因为它依赖于具体的日志格式和内容
    # with open(log_file_path, 'w') as log_f: # 假设是写入模式
    #    log_f.write("Starting inference comparison...\n")

    # 创建用于保存结果的列表 (原始逻辑)
    actual_times_in_s = []
    predicted_times_in_s = []
    actual_memories_in_mb = []
    predicted_memories_in_mb = []
    time_q_error = []
    memory_q_error = []
    epsilon = 1e-5 # 原始 epsilon
    time_calculation_durations = []
    memory_calculation_durations = []
    query_ids = []
    query_dops = []

    # 遍历 query_trees 的键 (原始逻辑)
    for (query_id, query_dop), plan_tree in query_trees.items(): # 原始变量名 plan_tree
        # 获取实际的执行时间和内存使用量 (原始逻辑)
        actual_time_row = df_query_info[(df_query_info['query_id'] == query_id) & (df_query_info['dop'] == query_dop)]

        if not actual_time_row.empty: # 原始检查
            actual_time = actual_time_row['execution_time'].values[0]
            actual_memory = actual_time_row['query_used_mem'].values[0]

            # 计算预测执行时间 (原始逻辑)
            start_time = time.time()
            # 假设调用前需要重置 visit 状态
            for node in plan_tree: node.visit = False
            predicted_time = calculate_query_execution_time(plan_tree)
            end_time = time.time()
            pred_exec_time = 0
            for plan_node in plan_tree: # 原始变量名 plan_node
                pred_exec_time += plan_node.pred_exec_time
            time_calculation_duration = end_time - start_time + pred_exec_time

            # 计算预测内存 (原始逻辑)
            start_time = time.time()
            predicted_memory, _ = calculate_query_memory(plan_tree)
            end_time = time.time()
            pred_mem_time = 0
            for plan_node in plan_tree:
                pred_mem_time += plan_node.pred_mem_time
            memory_calculation_duration = end_time - start_time + pred_mem_time

            # 确保预测值是标量 (原始逻辑)
            if isinstance(predicted_time, (np.ndarray, torch.Tensor)):
                predicted_time = predicted_time.item()
            if isinstance(predicted_memory, (np.ndarray, torch.Tensor)):
                predicted_memory = predicted_memory.item()

            # 转换单位 (原始逻辑，假设是 / 1000)
            actual_times_in_s.append(actual_time / 1000.0)
            predicted_times_in_s.append(predicted_time / 1000.0)
            actual_memories_in_mb.append(actual_memory / 1000.0) # 保持 / 1000
            predicted_memories_in_mb.append(predicted_memory / 1000.0) # 保持 / 1000

            # 计算 Q-error (原始逻辑，可能不处理除零)
            time_q_error_value = max(predicted_time/actual_time, actual_time/predicted_time) - 1
            memory_q_error_value = max(predicted_memory/actual_memory, actual_memory/predicted_memory) - 1

            # 存储结果 (原始逻辑)
            time_q_error.append(time_q_error_value)
            memory_q_error.append(memory_q_error_value)
            time_calculation_durations.append(time_calculation_duration)
            memory_calculation_durations.append(memory_calculation_duration)
            query_ids.append(query_id)
            query_dops.append(query_dop)

    # 映射查询 ID (原始逻辑)
    mapped_query_ids = range(1, len(query_ids) + 1) # 假设原始列表是 actual_times_in_s

    # 创建字典保存数据 (原始逻辑)
    data = {
        'Query ID': query_ids,
        'Query DOP': query_dops,
        'Query ID (Mapped)': mapped_query_ids,
        'Actual Execution Time (s)': actual_times_in_s,
        'Predicted Execution Time (s)': predicted_times_in_s,
        'Actual Memory Usage (MB)': actual_memories_in_mb,
        'Predicted Memory Usage (MB)': predicted_memories_in_mb,
        'Execution Time Q-error': time_q_error,
        'Memory Q-error': memory_q_error,
        'Time Calculation Duration (s)': time_calculation_durations,
        'Memory Calculation Duration (s)': memory_calculation_durations
    }

    # 转换为 DataFrame (原始逻辑)
    df = pd.DataFrame(data) # 原始变量名 df

    # 保存为 CSV 文件 (使用传入的 output_csv_path)
    # csv_file_path = 'test_queries_results.csv' # 原始硬编码
    df.to_csv(output_csv_path, index=False, sep=';') # 原始分隔符 ';'

    # 输出保存的文件路径 (原始逻辑)
    print(f"Data has been saved to {output_csv_path}") # 使用传入的路径

    # === 结束: 严格复制粘贴原始顶层逻辑 ===

# ==============================================================================
# 脚本入口 (不应在此文件内执行，仅作模块提供函数)
# ==============================================================================
# if __name__ == "__main__":
#     # 这里的调用逻辑会被移到 scripts/run_non_dop_inference.py
#     # 需要定义 plan_csv, query_csv, output_csv 的路径
#     # run_inference(plan_csv, query_csv, output_csv)
#     pass