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
from definition import PlanNode, ONNXModelManager
        
        
def calculate_thread_execution_time(node, thread_id):
    """
    计算当前线程的总执行时间、完成时间和数据传递时间。
    1. 线程内的总执行时间：该线程内所有算子的执行时间总和。
    2. 线程的完成时间：线程的总执行时间 + 下层线程传递数据的时间（取最大）。
    3. 该线程的传递数据的时间：最晚处理完的那个聚合算子的完成时间，
       注意：每一层的 data_transfer_start_time 会一层一层往上叠加，但本层不直接使用它。
    """
    # 避免重复访问
    if node.visit:
        return 0, 0, 0, 0  # (总执行时间, 完成时间, 数据传递时间)
    
    node.visit = True

    # 当前线程的执行时间以本节点为起点
    thread_execution_time = node.pred_execution_time
    # thread_execution_time = node.execution_time
    if node.operator_type == 'CTE Scan':
        thread_execution_time = 0
    # 当前线程的数据传递开始时间初始为0
    local_data_transfer_start_time = 0
    up_data_transfer_start_time = 0

    child_execution_times = []     # 同一线程内子节点的执行时间
    local_data_transfer_start_times = []   # 下层线程的传递数据时间
    up_data_transfer_start_times = []
    child_complete_times = []        # 子节点的完成时间

    # 判断当前节点是否为 streaming 算子（触发新线程）
    is_streaming_node = 'streaming' in node.operator_type.lower()
    for child in node.child_plans:
        # streaming节点：子节点属于新线程；否则仍在当前线程
        new_thread_id = thread_id + 1 if is_streaming_node else thread_id

        if new_thread_id == thread_id:
            # 子节点在当前线程内，计算其执行和完成时间
            child_time, child_complete_time, local_data_transfer_start_time, up_data_transfer_start_time = calculate_thread_execution_time(child, new_thread_id)
            child_execution_times.append(child_time)
            child_complete_times.append(child_complete_time)
            local_data_transfer_start_times.append(local_data_transfer_start_time)
            up_data_transfer_start_times.append(up_data_transfer_start_time)

            # 如果是聚合算子，更新当前线程的 data_transfer_start_time
            # 本层线程的 data_transfer_start_time 继承自下层传递的值，且本层不直接使用它更新
        else:
            # 子节点属于下层线程，新线程返回的 data_transfer_start_time 用于更新当前线程的完成时间
            _, child_complete_time, _, up_data_transfer_start_time = calculate_thread_execution_time(child, new_thread_id)
            thread_execution_time += thread_execution_time/2
            child_complete_time += thread_execution_time/2
            child_complete_times.append(child_complete_time)
            local_data_transfer_start_times.append(up_data_transfer_start_time)
            up_data_transfer_start_times.append(up_data_transfer_start_time)
    # 累加同一线程内所有子节点的执行时间
    local_data_transfer_start_time = max(local_data_transfer_start_times, default=0)
    up_data_transfer_start_time = max(up_data_transfer_start_times, default=0)
    thread_execution_time += sum(child_execution_times)
    if node.materialized:
        # 更新 data_transfer_start_time，但不会影响本层的计算，而是传递给上层线程
        if 'hash join' in node.operator_type.lower():
            up_data_transfer_start_time = max(up_data_transfer_start_time, local_data_transfer_start_time + thread_execution_time - child_execution_times[0])
        else:  
            up_data_transfer_start_time = max(up_data_transfer_start_time, local_data_transfer_start_time + thread_execution_time)
        
    child_complete_time = max(child_complete_times, default=0)
    
    # 当前线程的完成时间 = 当前线程总执行时间 + 下层线程传递数据时间（取最大的那个）
    thread_complete_time = max(thread_execution_time + local_data_transfer_start_time, child_complete_time)
    node.thread_execution_time = thread_execution_time
    node.thread_complete_time = thread_complete_time
    node.local_data_transfer_start_time = local_data_transfer_start_time
    node.up_data_transfer_start_time = up_data_transfer_start_time

    # 返回当前线程的三个值：每一层的 data_transfer_start_time 会叠加传递
    # 本层线程计算时不使用更新后的 data_transfer_start_time，返回给上层线程
    return thread_execution_time, thread_complete_time, local_data_transfer_start_time, up_data_transfer_start_time

def calculate_query_execution_time(all_nodes):
    """
    计算整个查询的执行时间，从根节点开始递归计算。
    最终的执行时间是最上层线程的完成时间。
    """
    total_time = 0
    plan_count = 0
    # 遍历所有节点，检查是否有未遍历的节点，如果有，从这些节点继续遍历
    for node in all_nodes:  # 假设 all_nodes 是所有可能的节点
        plan_count += 1
        if not node.visit:
            # 从未遍历的节点开始计算
            _, node_time, _, _= calculate_thread_execution_time(node, thread_id=0)
            total_time += node_time
    return total_time + plan_count * 6


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

# 假设 base_nodes 包含所有节点（重复的节点已去重）
def get_root_nodes(base_nodes):
    # 收集所有节点中出现过的 child 的 plan_id
    child_ids = set()
    for node in base_nodes:
        for child in node.child_plans:
            child_ids.add(child.plan_id)
    # 过滤出 plan_id 不在 child_ids 中的节点，即为根节点
    roots = [node for node in base_nodes if node.plan_id not in child_ids]
    return roots

def assign_thread_ids_by_plan_id(node, thread_id=0):
    """
    递归为整个树分配线程 id。
    利用 visited_ids（存储 plan_id）避免重复处理同一个节点。
    如果 operator_type 中包含 "streaming"，则其子节点进入新线程（thread_id+1），否则保持当前 thread_id。
    """

    node.thread_id = thread_id
    is_streaming = 'streaming' in node.operator_type.lower()
    new_thread_id = thread_id + 1 if is_streaming else thread_id
    for child in node.child_plans:
        assign_thread_ids_by_plan_id(child, new_thread_id)


def collect_all_nodes_by_plan_id(base_nodes):
    """
    通过 DFS 收集所有节点，利用 plan_id 去重（不依赖 node.visited）。
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


def update_thread_block_dop_candidates(base_nodes):
    """
    对给定的 root_nodes（各个查询计划树的根节点），
    1. 为每棵树调用 assign_thread_ids_by_plan_id(root, thread_id=0) 为整个树分配线程 id；
    2. 利用 collect_all_nodes_by_plan_id 收集所有节点；
    3. 按照每个节点的 thread_id 进行分组，形成线程块；
    4. 对每个线程块内所有节点的 dop_exec_time_map（记录不同 dop 的真实执行时间）的 key 求交集，
       得到该线程块的候选 dop 集合。

    返回一个字典，key 为线程 id，value 为候选 dop 集合。
    """
    # 先收集所有节点（假设 base_nodes 已经是所有节点的集合，不重复）
    # 如果你只有一个起始集合（如查询计划树），需要先用 DFS 收集所有节点
    all_nodes = base_nodes  # 这里假设 base_nodes 已经是完整的不重复集合
    root_nodes = get_root_nodes(all_nodes)

    # 对每个根节点分配线程 id，递归赋值给整个树
    for root in root_nodes:
        assign_thread_ids_by_plan_id(root, thread_id=0)

    # 收集所有节点（利用 plan_id 去重）
    all_nodes = collect_all_nodes_by_plan_id(root_nodes)

    # 按照 thread_id 分组
    thread_blocks = {}
    for node in all_nodes:
        tid = getattr(node, 'thread_id', 0)
        thread_blocks.setdefault(tid, []).append(node)
        
    for tid, nodes in thread_blocks.items():
        # 计算该线程块内所有节点的 matched_dops 交集
        dop_sets = [node.matched_dops for node in nodes if hasattr(node, 'matched_dops')]
        if not dop_sets:
            common_dops = set()
        else:
            # 如果只有一个节点，则交集就是它自己的 matched_dops，否则求交集
            common_dops = dop_sets[0] if len(dop_sets) == 1 else set.intersection(*dop_sets)
        
        # 更新每个节点的 matched_dops 和 dop_exec_time_map
        for node in nodes:
            node.matched_dops = common_dops  # 更新 matched_dops 为交集
            # 更新 dop_exec_time_map：只保留键在 common_dops 内的项
            node.dop_exec_time_map = {dop: time for dop, time in node.dop_exec_time_map.items() if dop in common_dops}
            
def select_optimal_dop(dop_exec_time_map, time_threshold=0.1, min_gain_ms=100):
    """
    选择最优的并行度 DOP：
    1. 如果执行时间的下降幅度小于 min_gain_ms，则不增加 DOP。
    2. 如果执行时间的下降率低于 time_threshold，则不增加 DOP。
    
    :param dop_exec_time_map: {dop: execution_time} 的字典
    :param time_threshold: 下降率阈值（默认 10%）
    :param min_gain_ms: 最小时间收益（默认 100ms）
    :return: 最优的 DOP
    """
    sorted_dops = sorted(dop_exec_time_map.keys())  # 先按照 DOP 排序
    best_dop = sorted_dops[0]  # 先选择最低的并行度
    best_time = dop_exec_time_map[best_dop]  

    for i in range(1, len(sorted_dops)):
        dop = sorted_dops[i]
        time = dop_exec_time_map[dop]

        # 计算下降幅度和下降率
        time_gain = best_time - time
        decrease_rate = time_gain / (dop / best_dop * best_time)

        if time_gain < min_gain_ms or decrease_rate < time_threshold:
            break  # 退出循环，保持当前 best_dop

        best_dop = dop
        best_time = time

    return best_dop

