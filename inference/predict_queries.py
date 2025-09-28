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
from config.structure import thread_cost, thread_mem
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
    adjustment = parent_thread_time / 4
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

def get_root_nodes(all_nodes):
    """从节点列表中找到根节点（没有父节点的节点）"""
    child_ids = set()
    for node in all_nodes:
        if node.child_plans:
            for child in node.child_plans:
                child_ids.add(child.plan_id)
    
    root_nodes = [node for node in all_nodes if node.plan_id not in child_ids]
    # 如果找不到严格的根节点（例如，只有一个节点），则返回第一个
    if not root_nodes and all_nodes:
        return [all_nodes[0]]
    return root_nodes

def _propagate_estimates_upwards(node: PlanNode):
    """
    自底向上递归地更新节点的估计输入，并重新触发特征生成和推理。
    """
    # 1. 先递归处理所有子节点
    for child in node.child_plans:
        _propagate_estimates_upwards(child)
    
    # 2. 所有子节点处理完毕后，更新当前节点的估计输入
    #    (例如，将子节点的 estimate_rows 赋值给当前节点的 l/r_input_rows_estimate)
    node.update_estimated_inputs()
    
    # 3. 输入特征已改变，必须重新生成特征数据并重新进行ONNX推理
    #    PlanNode自身的plan_data字典包含了所有原始信息，可以作为数据源
    node.get_feature_data(node.__dict__) 
    node.infer_exec_with_onnx()
    node.infer_mem_with_onnx()


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


def calculate_query_execution_time(all_nodes):
    """计算整个查询的预测执行时间（保持原始逻辑）"""
    total_time = 0
    plan_count = 0
    # 重置访问状态
    for node in all_nodes:
        node.visit = False

    root_nodes = [node for node in all_nodes if node.parent_node is None]
    if not root_nodes and all_nodes:
        root_nodes = [all_nodes[0]]
    elif not all_nodes:
        return 0
    query_execution_time = 0
    for node in root_nodes:
        if not node.visit:
             _, node_complete_time, _, _, _, _ = calculate_thread_execution_time(node, thread_id=0)
             query_execution_time = max(query_execution_time, node_complete_time)

    total_threads = 1
    dop_sum_streaming = 0
    for node in all_nodes:
        plan_count += 1 # 原始代码似乎没有用 plan_count? 但保留它
        if 'Streaming' in node.operator_type:
            # 保持原始的线程数计算方式
            dop_match = re.search(r'dop:\s*\d+\/(\d+)', node.operator_type)
            if dop_match:
                try:
                    threads_generated = int(dop_match.group(1)) # 原始代码似乎是 group(2)? 确认一下
                    dop_sum_streaming += threads_generated
                except (ValueError, IndexError):
                    dop_sum_streaming += node.downdop # 假设 PlanNode 有 downdop
            else:
                dop_sum_streaming += node.downdop # 假设 PlanNode 有 downdop

    total_threads += dop_sum_streaming
    # 保持原始的总时间计算方式
    final_time = query_execution_time
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

def calculate_query_sum_time(all_nodes):
    """
    计算整个查询的执行时间，从根节点开始递归计算。
    最终的执行时间是最上层线程的完成时间。
    """
    total_time = 0
    # 遍历所有节点，检查是否有未遍历的节点，如果有，从这些节点继续遍历
    for node in all_nodes:  # 假设 all_nodes 是所有可能的节点
        total_time += node.execution_time

    return total_time


def calculate_sum_memory(query_nodes):
    """
    计算一个查询所占用的内存。
    - 每个算子的内存为 `peak_mem`，加上生成的线程数。
    """
    total_peak_mem = 0
    total_threads = 1  # 初始线程数为 1

    for node in query_nodes:
        # 累加算子的 peak_mem
        total_peak_mem += node.peak_mem

        # 判断是否为 Vector Streaming 算子
        if 'Vector Streaming' in node.operator_type:
            # 查找 dop 值，这里假设 dop 格式为 "dop: X/Y"
            dop_match = re.search(r'dop:\s*(\d+)/(\d+)', node.operator_type)
            if dop_match:
                threads_generated = int(dop_match.group(2))  # 获取生成的线程数
                total_threads += threads_generated
            else:
                total_threads += node.downdop

    # 总内存 = 所有算子的 peak_mem + 生成的线程数（假设每个线程占用 1 单位内存）
    total_memory = total_peak_mem

    # 返回内存以及平均线程数
    return total_memory, total_threads # 返回平均线程数

# ==============================================================================
# 主推理逻辑函数
# ==============================================================================

def run_inference(plan_csv_path, query_csv_path, output_csv_path, no_dop_model_dir, dop_model_dir, use_estimates=False): # <-- 新增参数
    """
    执行推理流程 (严格按照提供的原始顶层逻辑封装)。

    Args:
        plan_csv_path (str): plan_info.csv 文件路径。
        query_csv_path (str): query_info.csv 文件路径。
        output_csv_path (str): 保存预测结果的 CSV 文件路径。
        no_dop_model_dir (str): 非 DOP 模型目录路径。
        dop_model_dir (str): DOP 模型目录路径。
    """

    # === 开始: 严格复制粘贴原始顶层逻辑 ===

    if use_estimates:
        print("!! 注意：正在以基数估计模拟模式运行推理 !!")
    # ...
        
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
    for (query_id, query_dop), group in query_groups:
        if query_id > 0: # 保持原始过滤条件
            # 使用 (query_id, query_dop) 作为键
            key = (query_id, query_dop) # 保持原始键
            query_trees[key] = []
            nodes_in_query = {} # 保持原始变量名

            # 创建该查询的所有计划节点
            for _, row in group.iterrows():
                plan_node = PlanNode(row, onnx_manager, use_estimates=use_estimates)
                query_trees[key].append(plan_node)
                nodes_in_query[plan_node.plan_id] = plan_node

            # 按 plan_id 排序，确保按顺序处理
            # 原始代码调用了 sorted，保持
            query_trees[key] = sorted(query_trees[key], key=lambda x: x.plan_id)

            # 处理父子关系 (原始逻辑)
            for _, row in group.iterrows():
                # 原始代码使用列表推导查找父节点，保持
                parent_plan = [plan for plan in query_trees[key] if plan.plan_id == row['plan_id']][0]

                child_plan = row['child_plan']
                if pd.isna(child_plan) or str(child_plan).strip() == '': # 原始检查方式
                    continue

                child_plans = str(child_plan).split(',') # 原始分割方式
                for child_plan_id_str in child_plans: # 原始变量名
                    child_plan_id = int(child_plan_id_str) # 原始转换方式
                    # 原始代码使用列表推导查找子节点，保持
                    child_plan_node = [plan for plan in query_trees[key] if plan.plan_id == child_plan_id][0]
                    parent_plan.add_child(child_plan_node) # 原始调用

    # --- 新增的基数传播步骤 ---
    if use_estimates:
        print("正在自底向上传播基数估计误差...")
        for key, nodes_list in query_trees.items():
            if not nodes_list: continue
            # 找到当前查询的根节点
            root_nodes_of_query = get_root_nodes(nodes_list)
            # 从每个根节点开始递归更新
            for root in root_nodes_of_query:
                _propagate_estimates_upwards(root)
        print("基数传播完成。")
    # --- 结束新增步骤 ---
        
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