import os
import re
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.definition import PlanNode, ONNXModelManager
        

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
    # 遍历所有节点，检查是否有未遍历的节点，如果有，从这些节点继续遍历
    for node in all_nodes:  # 假设 all_nodes 是所有可能的节点
        if not node.visit:
            # 从未遍历的节点开始计算
            _, node_time, _, _= calculate_thread_execution_time(node, thread_id=0)
            total_time += node_time

    return total_time

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
        if 'Vector Streaming' in node.operator_type:
            # 查找 dop 值，这里假设 dop 格式为 "dop: X/Y"
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


# 读取包含 query_id 和 split 信息的文件
split_info_df = pd.read_csv('/home/zhy/opengauss/tools/serverless_tools/train/python/no_dop/tmp_result/query_split.csv')

# 只保留前 200 行的数据
split_info = split_info_df[['query_id', 'split']]

# 获取原始的 split 信息的 query_id 和 split 列
test_queries = split_info[split_info['split'] == 'test']['query_id']

# 将扩展后的测试查询的 query_id 转换为 DataFrame
test_queries_df = pd.DataFrame(test_queries, columns=['query_id'])

# 读取执行计划数据
df_plans = pd.read_csv('/home/zhy/opengauss/data_file/tpch_5g_output/plan_info.csv', delimiter=';', encoding='utf-8')

df_query_info = pd.read_csv('/home/zhy/opengauss/data_file/tpch_5g_output/query_info.csv', delimiter=';', encoding='utf-8')

# 按 query_id 和 query_dop 分组
query_groups = df_plans.groupby(['query_id', 'query_dop'])

# 创建 PlanNode 对象并处理每个查询的树结构
query_trees = {}

onnx_manager = ONNXModelManager()

# 仅处理测试数据的查询
for (query_id, query_dop), group in query_groups:
    if query_id in test_queries_df['query_id'].values:
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
        time_calculation_duration = end_time - start_time  # 计算耗时

        start_time = time.time()  # 记录开始时间
        predicted_memory, _ = calculate_query_memory(plan_tree)  # 根据查询树计算预测内存
        end_time = time.time()  # 记录结束时间
        memory_calculation_duration = end_time - start_time  # 计算耗时

        # 转换单位：秒和MB
        actual_times_in_s.append(actual_time / 1000)  # 转换为秒
        predicted_times_in_s.append(predicted_time / 1000)  # 转换为秒
        actual_memories_in_mb.append(actual_memory / 1000)  # 转换为 MB
        predicted_memories_in_mb.append(predicted_memory / 1000)  # 转换为 MB

        # 计算 Q-error
        time_q_error_value = abs(predicted_time - actual_time) / (actual_time + epsilon)
        memory_q_error_value = abs(predicted_memory - actual_memory) / (actual_memory + epsilon)

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

        



