import math
import os
import re
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

import torch
# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from definition import PlanNode, ONNXModelManager,ThreadBlock, default_dop, thread_cost
        
def get_root_nodes(base_nodes):
    """
    从所有节点中，筛选出没有出现在其他节点 child_plans 中的节点，作为根节点。
    """
    child_ids = set()
    for node in base_nodes:
        for child in node.child_plans:
            child_ids.add(child.plan_id)
    roots = [node for node in base_nodes if node.plan_id not in child_ids]
    return roots
        
def base_execution_time(node):
    """返回当前节点的基本执行时间"""
    if node.operator_type == 'CTE Scan':
        return 0
    return node.pred_execution_time  # 或者用 node.pred_execution_time 根据需求选择

def process_same_thread_child(child, thread_blocks):
    """
    处理子节点在同一线程内的情况，直接递归调用计算函数。
    返回 (child_exec, child_complete, child_local, child_up, child_more_agg_times, child_more_agg_flag)
    """
    return calculate_thread_execution_time(child, thread_blocks)

def process_new_thread_child(child, parent_thread_time, thread_blocks):
    """
    处理子节点属于新线程的情况，同时记录子线程的 thread_id 到父线程块中。
    调用 calculate_thread_execution_time 对子节点递归计算指标，
    并将父线程时间的一半作为调整量加到子节点的完成时间上。
    返回 (adjustment, adjusted_child_complete, child_up)
    """
    # 递归计算子节点的新线程指标
    _, child_complete, _, child_up, _, _ = calculate_thread_execution_time(child, thread_blocks)
    # adjustment = parent_thread_time / 2
    adjustment = 0
    adjusted_child_complete = child_complete + adjustment
    return adjustment, adjusted_child_complete, child_up

def aggregate_child_results(same_results, new_results):
    """
    将同一线程和新线程分支的结果合并：
    - same_results：列表，每项为 (child_exec, child_complete, child_local, child_up, more_agg_times, more_agg)
    - new_results：列表，每项为 (adjustment, adjusted_child_complete, child_up)
    
    返回四个列表：child_exec_list, child_complete_list, local_transfer_list, up_transfer_list。
    新线程分支中不贡献直接执行时间（视为0），local和up传输时间直接取返回值。
    """
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
        child_exec_list.append(0)  # 新线程分支不计直接执行时间
        child_complete_list.append(res[1])
        local_transfer_list.append(res[2])
        up_transfer_list.append(res[2])  # 假设 up_transfer 同 local_transfer
    return child_exec_list, child_complete_list, local_transfer_list, up_transfer_list

def final_adjustment(thread_exec, local_transfer, child_complete_list, more_agg_times, tmp_flag):
    """
    如果 tmp_flag 为 True 且 thread_exec + local_transfer 小于子节点的最大完成时间，
    则对 thread_complete_time 做进一步调整。
    """
    child_complete = max(child_complete_list, default=0)
    thread_complete = max(thread_exec + local_transfer, child_complete)
    if tmp_flag and (thread_exec + local_transfer < child_complete):
        thread_complete = max(thread_complete, child_complete + thread_exec - more_agg_times)
        more_agg_times = thread_exec
        tmp_flag = False
    return thread_complete, more_agg_times, tmp_flag

def calculate_thread_execution_time(node, thread_blocks):
    """
    递归计算当前节点所在线程的各项指标，并将结果保存在节点属性中。
    返回：
      (thread_execution_time, thread_complete_time, local_transfer, up_transfer, more_agg_times, more_agg_flag)
    """
    if node.visit:
        return 0, 0, 0, 0, 0, False
    node.visit = True

    # 当前节点的基本执行时间
    thread_exec = base_execution_time(node)
    more_agg = False
    tmp_flag = False

    same_results = []
    new_results = []
    
    # 遍历子节点，根据 thread_id 判断是否在同一线程内
    for child in node.child_plans:
        if getattr(child, 'thread_id', node.thread_id) == node.thread_id:
            res = calculate_thread_execution_time(child, thread_blocks)
            same_results.append(res)
        else:
            # 对于新线程分支，记录父线程块与子线程之间的关系
            if node.thread_id in thread_blocks:
                thread_blocks[node.thread_id].child_thread_ids.add(child.thread_id)
            res = process_new_thread_child(child, thread_exec, thread_blocks)
            # 根据原逻辑，父节点的执行时间需要调整
            # thread_exec += thread_exec / 2
            new_results.append(res)

    # 合并同一线程和新线程的结果
    child_exec_list, child_complete_list, local_transfer_list, up_transfer_list = aggregate_child_results(same_results, new_results)
    
    # 本层传输时间取所有子节点 local_transfer 的最大值
    local_transfer = max(local_transfer_list, default=0)
    up_transfer = max(up_transfer_list, default=0)
    
    # 累加同一线程分支的执行时间
    thread_exec += sum(child_exec_list)
    
    # 如果当前节点为物化算子，则进行额外处理
    if node.materialized:
        if 'hash join' in node.operator_type.lower():
            if child_exec_list:
                up_transfer = max(up_transfer, local_transfer + thread_exec - child_exec_list[0])
            else:
                up_transfer = max(up_transfer, local_transfer + thread_exec)
        else:
            up_transfer = max(up_transfer, local_transfer + thread_exec)
        more_agg_times = thread_exec
        more_agg = True
    else:
        more_agg_times = 0

    child_complete = max(child_complete_list, default=0)
    thread_complete = max(thread_exec + local_transfer, child_complete)
    thread_complete, more_agg_times, tmp_flag = final_adjustment(thread_exec, local_transfer, child_complete_list, more_agg_times, tmp_flag)
    
    # 将计算结果写入节点属性
    node.thread_execution_time = thread_exec
    node.thread_complete_time = thread_complete
    node.local_data_transfer_start_time = local_transfer
    node.up_data_transfer_start_time = up_transfer

    return thread_exec, thread_complete, local_transfer, up_transfer, more_agg_times if more_agg else 0, more_agg


    
def calculate_query_execution_time(all_nodes, thread_blocks):
    """
    计算整个查询的执行时间，从根节点开始递归计算。
    最终的执行时间是最上层线程的完成时间。
    """

    # 按 Plan_id 从小到大排序
    root_nodes = get_root_nodes(all_nodes)

    total_time = 0
    plan_count = 0

    # 遍历所有节点，检查是否有未遍历的节点，如果有，从这些节点继续遍历
    for node in root_nodes:  # 确保按 Plan_id 递增遍历
        plan_count += 1
        # 从未遍历的节点开始计算
        _, node_time, _, _, _, _ = calculate_thread_execution_time(node, thread_blocks)
        total_time += node_time

    return total_time + plan_count * thread_cost

def calculate_query_memory(query_nodes):
    """
    计算一个查询所占用的内存。
    - 每个算子的内存为 `peak_mem`，加上生成的线程数。
    """
    total_peak_mem = 0
    total_threads = 1  # 初始线程数为 1

    for node in query_nodes:
        # 累加算子的 peak_mem
        total_peak_mem += node.pred_mem

        # 判断是否为 Vector Streaming 算子
        if 'Streaming' in node.operator_type:
            dop_match = re.search(r'dop:\s*(\d+)/(\d+)', node.operator_type)
            if dop_match:
                threads_generated = int(dop_match.group(2))  # 获取生成的线程数
                total_threads += threads_generated
            else:
                total_threads += node.downdop

    # 总内存 = 所有算子的 peak_mem + 生成的线程数（假设每个线程占用 1 单位内存）
    total_memory = total_peak_mem + (total_threads - 1) * 9216

    # 返回内存以及平均线程数
    return total_memory, total_threads # 返回平均线程数


def assign_thread_ids_by_plan_id(node, thread_id=0, max_thread_id=0):
    """
    递归为整个树分配线程 ID：
      - 如果当前节点是 Streaming 算子，则其子节点使用新的线程 ID（max_thread_id+1）。
      - 如果当前节点不是 Streaming 算子，则其子节点继承当前线程 ID。
      
    参数：
    - node: 当前查询计划节点
    - thread_id: 当前节点的线程 ID
    - max_thread_id: 已分配的最大线程 ID（用于生成新线程 ID）
    """
    # 为当前节点赋值线程 ID
    node.thread_id = thread_id
    max_thread_id = max(max_thread_id, thread_id)
    
    # 判断当前节点是否是 Streaming 算子
    if 'streaming' in node.operator_type.lower():
        # 当前节点是 Streaming 算子，其子节点分配新的线程 ID
        new_thread_id = max_thread_id + 1
    else:
        # 否则子节点继承当前线程 ID
        new_thread_id = thread_id

    for child in node.child_plans:
        max_thread_id = assign_thread_ids_by_plan_id(child, new_thread_id, max_thread_id)
    
    return max_thread_id


def collect_all_nodes_by_plan_id(base_nodes):
    """
    通过 DFS 收集所有节点，利用 plan_id 去重（不依赖节点自身的 visited 标记）。
    输入：base_nodes，假设是所有查询计划树的根节点集合。
    返回：整个树中所有不重复的节点列表。
    """
    all_nodes = []
    visited_ids = set()
    stack = list(base_nodes)
    while stack:
        node = stack.pop()
        if node.plan_id in visited_ids:
            continue
        visited_ids.add(node.plan_id)
        all_nodes.append(node)
        stack.extend(node.child_plans)
    return all_nodes

def update_thread_blocks(base_nodes):
    """
    更新线程块：
      1. 找出所有根节点，并为每棵树分配 thread_id（保证不同树之间不会冲突）。
      2. 通过 DFS 收集所有节点，并按 thread_id 划分构造 ThreadBlock，
         每个线程块包含多个算子。
      3. 划分完成后，再针对每个根节点调用 calculate_thread_execution_time，
         计算各节点指标（此过程不会破坏线程块划分）。
      4. 最后，统一遍历每个线程块内所有节点，通过 ThreadBlock.aggregate_metrics() 聚合各项指标，
         同时更新 child_thread_ids 以及记录子线程内的最大完成时间。
    返回：线程块字典（key 为 thread_id，value 为对应的 ThreadBlock 对象）。
    """
    # 1. 找出所有根节点，并分配 thread_id（确保不同树不冲突）
    root_nodes = get_root_nodes(base_nodes)
    current_offset = 0
    for root in root_nodes:
        assign_thread_ids_by_plan_id(root, thread_id=current_offset)
        nodes_in_tree = collect_all_nodes_by_plan_id([root])
        max_tid = max(getattr(node, 'thread_id', 0) for node in nodes_in_tree)
        current_offset = max_tid + 1

    # 2. 收集所有节点，并按照 thread_id 构造线程块
    all_nodes = collect_all_nodes_by_plan_id(root_nodes)
    thread_blocks = {}
    for node in all_nodes:
        tid = getattr(node, 'thread_id', 0)
        if tid not in thread_blocks:
            thread_blocks[tid] = ThreadBlock(tid, [])
        thread_blocks[tid].nodes.append(node)

    for root in root_nodes:
        root.compute_parallel_dop_predictions()
    # 3. 划分完成后，对每个根节点调用递归计算函数（传入 thread_blocks）
    calculate_query_execution_time(all_nodes, thread_blocks)

    # 4. 统一聚合每个线程块内所有算子的指标
    for tb in thread_blocks.values():
        tb.aggregate_metrics()

    # 5. 更新每个线程块的子线程内的最大完成时间
    # 即遍历每个线程块的 child_thread_ids，从对应的线程块中取出最大完成时间
    for tb in thread_blocks.values():
        tb.child_max_execution_time = max(
            (thread_blocks[child_tid].thread_execution_time  for child_tid in tb.child_thread_ids if child_tid in thread_blocks),
            default=0
        )

    return thread_blocks
            
def select_optimal_dop(exec_time_map, time_threshold=0.1, min_gain_ms=100):
    """
    选择最优的并行度 DOP:
    1. 如果执行时间的下降幅度小于 min_gain_ms,则不增加 DOP。
    2. 如果执行时间的下降率低于 time_threshold,则不增加 DOP。
    
    :param exec_time_map: {dop: execution_time} 的字典
    :param time_threshold: 下降率阈值（默认 10)
    :param min_gain_ms: 最小时间收益（默认 100ms)
    :return: 最优的 DOP
    """
    sorted_dops = sorted(exec_time_map.keys())  # 先按照 DOP 排序
    best_dop = sorted_dops[0]  # 先选择最低的并行度
    best_time = exec_time_map[best_dop]  

    for i in range(1, len(sorted_dops)):
        dop = sorted_dops[i]
        time = exec_time_map[dop]

        # 计算下降幅度和下降率
        time_gain = best_time - time
        factor = 1 / (math.log2(dop - best_dop) + 1)
        decrease_rate = time_gain / (factor * best_time)

        if time_gain < min_gain_ms or decrease_rate < time_threshold:
            break  # 退出循环，保持当前 best_dop

        best_dop = dop
        best_time = time

    return best_dop

def build_query_exec_time_map(csv_file):
    """
    读取 CSV 文件，构建一个字典：
      { query_id: { dop: execution_time, ... }, ... }
    假设 CSV 文件中包含字段 'query_id'、'dop' 和 'execution_time'。
    """
    df = pd.read_csv(csv_file, delimiter=';', encoding='utf-8')
    query_dop_map = {}
    for query_id, group in df.groupby('query_id'):
        # 将每个 query 下的 dop 与 execution_time 组合成字典
        dop_dict = dict(zip(group['dop'], group['execution_time']))
        query_dop_map[query_id] = dop_dict
    return query_dop_map


def select_optimal_query_dop_from_tru(csv_file, time_threshold=0.05, min_gain_ms=100):
    """
    端到端函数：
      1. 读取 CSV 文件构建查询执行时间字典；
      2. 对每个 query 调用 select_optimal_query_dop 选择最优的 DOP;
      3. 返回字典 { query_id: optimal_dop }。
    """
    query_exec_map = build_query_exec_time_map(csv_file)
    optimal_dops = {}
    for query_id, dop_map in query_exec_map.items():
        optimal_dops[query_id] = select_optimal_dop(dop_map, time_threshold, min_gain_ms)
    df = pd.DataFrame(optimal_dops.items(), columns=['query_id', 'optimal_dop'])
    df.to_csv("dop_result/tru_dop.csv", index=False, sep=';')  # 使用分号分隔符，避免与 query 语句冲突
    return optimal_dops

def select_optimal_dops_from_prediction_file(csv_file, time_threshold=0.05, min_gain_ms=100, output_file=None):
    """
    端到端函数：从包含预测结果的 CSV 文件中读取数据，
    该 CSV 文件需包含字段:query_id;dop;predicted_time;actual_time;predicted_memory;actual_memory;q_error_time;q_error_memory。
    对每个 query,根据 predicted_time 选择最优 DOP(单位均为毫秒)。
    
    :param csv_file: 输入 CSV 文件路径
    :param time_threshold: 下降率阈值（默认 0.1)
    :param min_gain_ms: 最小时间收益（默认 100ms)
    :param output_file: 输出文件路径（如果不为 None,则保存结果到 CSV)
    :return: dict, 格式 {query_id: optimal_dop, ...}
    """
    df = pd.read_csv(csv_file, delimiter=';', encoding='utf-8')
    
    optimal_dops = {}
    # 按 query_id 分组，每组包含该 query 不同 DOP 下的预测结果
    groups = df.groupby('query_id')
    for query_id, group in groups:
        # 构造 {dop: predicted_time} 字典
        dop_dict = dict(zip(group['dop'], group['predicted_time']))
        # 复用 select_optimal_dop 函数
        optimal_dop = select_optimal_dop(dop_dict, time_threshold, min_gain_ms)
        optimal_dops[query_id] = optimal_dop

    # 如果指定输出文件，则保存结果到 CSV
    if output_file is not None:
        out_df = pd.DataFrame(list(optimal_dops.items()), columns=['query_id', 'optimal_dop'])
        out_df.to_csv(output_file, index=False, sep=';')
        print(f"最优查询 DOP 结果已保存至 {output_file}")
    
    return optimal_dops