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
from definition import PlanNode, ONNXModelManager, thread_cost, thread_mem
        

def base_execution_time(node):
    """返回当前节点的基本执行时间"""
    if node.operator_type == 'CTE Scan':
        return 0
    return node.pred_execution_time  # 或者用 node.pred_execution_time 根据需求选择

def process_same_thread_child(child, thread_id):
    """
    处理子节点在同一线程内的情况，直接递归调用计算函数。
    返回 (child_exec, child_complete, child_local, child_up, child_more_agg_times, child_more_agg_flag)
    """
    return calculate_thread_execution_time(child, thread_id)

def process_new_thread_child(child, new_thread_id, parent_thread_time):
    """
    处理子节点属于新线程的情况：
    调用递归后，调整父线程时间（增加 parent_thread_time/2），
    并对子节点的完成时间也加上同样的调整量。
    返回 (adjustment, adjusted_child_complete, child_up)
    """
    # 调用递归计算新线程的时间
    _, child_complete, _, child_up, child_more_agg_times, child_more_agg = calculate_thread_execution_time(child, new_thread_id)
    adjustment = parent_thread_time / 2
    # adjustment = 0
    adjusted_child_complete = child_complete + adjustment
    return adjustment, adjusted_child_complete, child_up

def aggregate_child_results(same_results, new_results):
    """
    将同一线程和新线程分支的结果合并：
    - same_results：列表，每项为 (child_exec, child_complete, child_local, child_up, more_agg_times, more_agg)
    - new_results：列表，每项为 (adjustment, adjusted_child_complete, child_up)
    
    返回四个列表：child_exec_list, child_complete_list, local_transfer_list, up_transfer_list，
    其中 new_results中不贡献执行时间（child_exec 视为0），local和up传输时间直接取返回值。
    """
    child_exec_list = []
    child_complete_list = []
    local_transfer_list = []
    up_transfer_list = []
    
    # 同一线程分支
    for res in same_results:
        child_exec_list.append(res[0])
        child_complete_list.append(res[1])
        local_transfer_list.append(res[2])
        up_transfer_list.append(res[3])
    # 新线程分支
    for res in new_results:
        # 新线程分支不直接加执行时间
        child_complete_list.append(res[1])
        local_transfer_list.append(res[2])
        up_transfer_list.append(res[2])  # 假设这里用 child_up 同样处理
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

def calculate_thread_execution_time(node, thread_id):
    """
    模块化计算当前节点所在线程的执行时间、完成时间、及数据传输时间。
    返回：
      (thread_execution_time, thread_complete_time, local_transfer, up_transfer, more_agg_times, more_agg_flag)
    """
    if node.visit:
        return 0, 0, 0, 0, 0, False
    node.visit = True

    # 基本执行时间
    thread_exec = base_execution_time(node)
    # 标记聚合相关
    more_agg = False
    tmp_flag = False

    # 初始化列表
    same_results = []
    new_results = []
    
    # 判断是否为 streaming 节点，决定子节点线程分支
    is_streaming = 'streaming' in node.operator_type.lower()
    for child in node.child_plans:
        new_thread_id = thread_id + 1 if is_streaming else thread_id
        if new_thread_id == thread_id:
            res = process_same_thread_child(child, new_thread_id)
            same_results.append(res)
        else:
            res = process_new_thread_child(child, new_thread_id, thread_exec)
            # thread_exec += thread_exec/2
            new_results.append(res)
    
    # 合并同一线程和新线程的结果
    child_exec_list, child_complete_list, local_transfer_list, up_transfer_list = aggregate_child_results(same_results, new_results)
    
    # 计算本层的传输时间：取所有子节点local传输时间的最大值
    local_transfer = max(local_transfer_list, default=0)
    up_transfer = max(up_transfer_list, default=0)
    
    # 同一线程分支的执行时间累加
    thread_exec += sum(child_exec_list)
    
    # 如果当前节点为 materialized，则更新 up_transfer 并设置聚合标志
    if node.materialized:
        if 'hash join' in node.operator_type.lower():
            if child_exec_list:
                up_transfer = max(up_transfer, local_transfer + thread_exec - child_exec_list[0])
            else:
                up_transfer = max(up_transfer, local_transfer + thread_exec)
        else:
            up_transfer = max(up_transfer, local_transfer + thread_exec)
        if not more_agg:
            more_agg_times = thread_exec
            more_agg = True
        else:
            tmp_flag = True

    # 计算子节点的最大完成时间
    child_complete = max(child_complete_list, default=0)
    
    # 计算初步的线程完成时间
    thread_complete = max(thread_exec + local_transfer, child_complete)
    
    # 进行最后调整（例如多个聚合阻塞时的处理）
    thread_complete, more_agg_times, tmp_flag = final_adjustment(thread_exec, local_transfer, child_complete_list, more_agg_times if more_agg else 0, tmp_flag)
    
    # 保存到节点属性（如果需要保存）
    # node.thread_execution_time = thread_exec
    # node.thread_complete_time = thread_complete
    # node.local_data_transfer_start_time = local_transfer
    # node.up_data_transfer_start_time = up_transfer

    return thread_exec, thread_complete, local_transfer, up_transfer, more_agg_times if more_agg else 0, more_agg


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
            _, node_time, _, _, _, _= calculate_thread_execution_time(node, thread_id=0)
            total_time += node_time
    total_threads = 1
    for node in all_nodes:
        # 判断是否为 Vector Streaming 算子
        if 'Streaming' in node.operator_type:
            dop_match = re.search(r'dop:\s*(\d+)/(\d+)', node.operator_type)
            if dop_match:
                threads_generated = int(dop_match.group(2))  # 获取生成的线程数
                total_threads += threads_generated
            else:
                total_threads += node.downdop
    return total_time + total_threads * thread_cost

def calculate_query_sum_time(all_nodes):
    """
    计算整个查询的执行时间，从根节点开始递归计算。
    最终的执行时间是最上层线程的完成时间。
    """
    total_time = 0
    # 遍历所有节点，检查是否有未遍历的节点，如果有，从这些节点继续遍历
    for node in all_nodes:  # 假设 all_nodes 是所有可能的节点
        total_time += node.pred_execution_time

    return total_time

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
    total_memory = total_peak_mem + int(math.pow(total_threads - 1, 1.15) * thread_mem)

    # 返回内存以及平均线程数
    return total_memory, total_threads # 返回平均线程数

def calculate_sum_memory(query_nodes):
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

# 读取执行计划数据
df_plans = pd.read_csv('/home/zhy/opengauss/data_file_kunpeng/tpch_output_22/plan_info.csv', delimiter=';', encoding='utf-8')
# df_plans = df_plans[df_plans['query_dop'] == 8].copy()
df_query_info = pd.read_csv('/home/zhy/opengauss/data_file_kunpeng/tpch_output_22/query_info.csv', delimiter=';', encoding='utf-8')
# df_query_info = df_query_info[df_query_info['dop'] == 8].copy()
# 按 query_id 和 query_dop 分组
query_groups = df_plans.groupby(['query_id', 'query_dop'])

# 创建 PlanNode 对象并处理每个查询的树结构
query_trees = {}

onnx_manager = ONNXModelManager()

# 仅处理测试数据的查询
for (query_id, query_dop), group in query_groups:
    if query_id > 0:
        # 使用 (query_id, query_dop) 作为键
        query_trees[(query_id, query_dop)] = []
        
        # 创建该查询的所有计划节点
        for _, row in group.iterrows():
            plan_node = PlanNode(row, onnx_manager)
            query_trees[(query_id, query_dop)].append(plan_node)
        
        # 按 plan_id 排序，确保按顺序处理
        query_trees[(query_id, query_dop)] = sorted(query_trees[(query_id, query_dop)], key=lambda x: x.plan_id)
        
        # 处理父子关系
        for _, row in group.iterrows():
            parent_plan = [plan for plan in query_trees[(query_id, query_dop)] if plan.plan_id == row['plan_id']][0]
            
            # 处理 child_plan 为空或没有子计划的情况
            child_plan = row['child_plan']
            if pd.isna(child_plan) or child_plan.strip() == '':  # 处理空值或空字符串的情况
                continue
            
            # 如果 child_plan 有多个子计划（逗号分隔），则拆分
            child_plans = child_plan.split(',')
            for child_plan_id in child_plans:
                child_plan_id = int(child_plan_id)  # 将每个子计划 ID 转换为整数
                child_plan_node = [plan for plan in query_trees[(query_id, query_dop)] if plan.plan_id == child_plan_id][0]
                parent_plan.add_child(child_plan_node)
# 打开日志文件进行写入
log_file_path = 'query_comparison_log.txt'
# 创建用于保存结果的列表
actual_times_in_s = []
predicted_times_in_s = []
actual_memories_in_mb = []
predicted_memories_in_mb = []
time_q_error = []  # 执行时间 Q-error
memory_q_error = []  # 内存 Q-error

epsilon = 1e-5  # 防止除零

time_calculation_durations = []
memory_calculation_durations = []
query_ids = []
query_dops = []

# 遍历 query_trees 的键（(query_id, query_dop)）
for (query_id, query_dop), plan_tree in query_trees.items():
    # 获取实际的执行时间和内存使用量
    actual_time_row = df_query_info[(df_query_info['query_id'] == query_id) & (df_query_info['dop'] == query_dop)]
    
    if not actual_time_row.empty:
        # 获取实际的执行时间和内存使用量
        actual_time = actual_time_row['execution_time'].values[0]
        actual_memory = actual_time_row['query_used_mem'].values[0]

        # 计算预测的执行时间和内存（根据 PlanNode 树），并记录耗时
        start_time = time.time()  # 记录开始时间
        predicted_time = calculate_query_execution_time(plan_tree)  # 根据查询树计算预测执行时间
        end_time = time.time()  # 记录结束时间
        pred_exec_time = 0
        for plan_node in plan_tree:
            pred_exec_time += plan_node.pred_exec_time
        time_calculation_duration = end_time - start_time + pred_exec_time  # 计算耗时

        start_time = time.time()  # 记录开始时间
        predicted_memory, _ = calculate_query_memory(plan_tree)  # 根据查询树计算预测内存
        end_time = time.time()  # 记录结束时间
        pred_mem_time = 0
        for plan_node in plan_tree:
            pred_mem_time += plan_node.pred_mem_time
        memory_calculation_duration = end_time - start_time + pred_mem_time  # 计算耗时

        # 确保 predicted_time 和 predicted_memory 是标量
        if isinstance(predicted_time, (np.ndarray, torch.Tensor)):
            predicted_time = predicted_time.item()  # 提取单元素的标量值

        if isinstance(predicted_memory, (np.ndarray, torch.Tensor)):
            predicted_memory = predicted_memory.item()  # 提取单元素的标量值

        # 转换单位：秒和MB
        actual_times_in_s.append(actual_time / 1000)  # 转换为秒
        predicted_times_in_s.append(predicted_time / 1000)  # 转换为秒
        actual_memories_in_mb.append(actual_memory / 1000)  # 转换为 MB
        predicted_memories_in_mb.append(predicted_memory / 1000)  # 转换为 MB

        # 计算 Q-error
        time_q_error_value = max(predicted_time/actual_time, actual_time/predicted_time) - 1
        memory_q_error_value = max(predicted_memory/actual_memory, actual_memory/predicted_memory) - 1

        time_q_error.append(time_q_error_value)
        memory_q_error.append(memory_q_error_value)


        # 记录计算耗时
        time_calculation_durations.append(time_calculation_duration)
        memory_calculation_durations.append(memory_calculation_duration)

        # 记录 query_id 和 query_dop
        query_ids.append(query_id)
        query_dops.append(query_dop)

# 映射查询 ID（从测试集中获取的查询 ID）到 1-xx 范围
mapped_query_ids = range(1, len(actual_times_in_s) + 1)

# 创建字典保存数据
data = {
    'Query ID': query_ids,  # 原始 query_id
    'Query DOP': query_dops,  # query_dop
    'Query ID (Mapped)': mapped_query_ids,
    'Actual Execution Time (s)': actual_times_in_s,
    'Predicted Execution Time (s)': predicted_times_in_s,
    'Actual Memory Usage (MB)': actual_memories_in_mb,
    'Predicted Memory Usage (MB)': predicted_memories_in_mb,
    'Execution Time Q-error': time_q_error,
    'Memory Q-error': memory_q_error,
    'Time Calculation Duration (s)': time_calculation_durations,  # 执行时间计算耗时
    'Memory Calculation Duration (s)': memory_calculation_durations  # 内存计算耗时
}

# 转换为 DataFrame
df = pd.DataFrame(data)

# 保存为 CSV 文件
csv_file_path = 'test_queries_results.csv'  # 修改为你希望的文件路径
df.to_csv(csv_file_path, index=False)

# 输出保存的文件路径
print(f"Data has been saved to {csv_file_path}")

        



