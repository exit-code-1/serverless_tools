import os
import re
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils
from utils.structure import operators

class PlanNode:
    def __init__(self, plan_data):
        self.visit = False
        self.plan_id = plan_data['plan_id']
        self.query_id = plan_data['query_id']
        self.operator_type = plan_data['operator_type']
        self.updop =  plan_data['up_dop']
        self.downdop =  plan_data['down_dop']
        self.execution_time = plan_data['execution_time']
        self.pred_execution_time = 0
        self.pred_mem = 0
        self.peak_mem = plan_data['peak_mem']
        self.width = plan_data['width']
        self.exec_feature_data = None 
        self.mem_feature_data = None
        self.child_plans = []  # 用于存储子计划节点
        self.materialized = 'hash' in self.operator_type.lower() or 'aggregate' in self.operator_type.lower() or 'sort' in self.operator_type.lower() or 'materialize' in self.operator_type.lower()  # 判断是否是物化算子
        self.parent_node = None  # 父节点
        self.thread_execution_time = 0
        self.thread_complete_time = 0
        self.local_data_transfer_start_time = 0
        self.up_data_transfer_start_time = 0      
          # 处理算子类型名中的空格为下划线
        operator_name = self.operator_type.replace(' ', '_')
        # 使用处理过的 operator_name 来构建模型路径
        self.model_path_exec = f"/home/zhy/opengauss/tools/serverless_tools/train/model/no_dop/{self.operator_type}/exec_{operator_name}.onnx"
        self.model_path_mem = f"/home/zhy/opengauss/tools/serverless_tools/train/model/no_dop/{self.operator_type}/mem_{operator_name}.onnx"
        
        self.get_feature_data(plan_data)
        self.infer_exec_with_onnx()
        self.infer_mem_with_onnx()
    
    def add_child(self, child_node):
        self.child_plans.append(child_node)
        child_node.parent_node = self
        child_node.visit = False  # 重置子节点的访问标记
        
    def get_feature_data(self, plan_data):
        if self.operator_type in operators:
            self.exec_feature_data, self.mem_feature_data = utils.prepare_inference_data(plan_data, plan_data['operator_type'])
        
    def infer_exec_with_onnx(self):
        """
        使用 ONNX 模型推理执行时间和内存。
        """
        # 确保 feature_data 是一个有效的特征输入
        if self.exec_feature_data is None:
            return
        

        # 使用 onnxruntime 进行推理
        # 将 feature_data 转换为 numpy 数组（如果它不是 numpy 数组）
        feature_array = np.array(self.exec_feature_data).reshape(1, -1)  # 假设输入是一个行向量

        # 预测执行时间
        session_exec = onnxruntime.InferenceSession(self.model_path_exec)
        inputs = {session_exec.get_inputs()[0].name: feature_array.astype(np.float32)}  # 转换为 float32
        exec_pred = session_exec.run(None, inputs)

        # 更新执行时间和内存
        self.pred_execution_time = exec_pred[0][0]  # 假设返回值是一个一维数组

    def infer_mem_with_onnx(self):
        """
        使用 ONNX 模型推理执行时间和内存。
        """
        # 确保 feature_data 是一个有效的特征输入
        if self.mem_feature_data is None:
            return

        # 使用 onnxruntime 进行推理
        # 将 feature_data 转换为 numpy 数组（如果它不是 numpy 数组）
        feature_array = np.array(self.mem_feature_data).reshape(1, -1)  # 假设输入是一个行向量

        # 预测内存
        session_mem = onnxruntime.InferenceSession(self.model_path_mem)
        inputs_mem = {session_mem.get_inputs()[0].name: feature_array.astype(np.float32)}  # 转换为 float32
        mem_pred = session_mem.run(None, inputs_mem)

        self.pred_mem = mem_pred[0][0]  # 假设返回值是一个一维数组
        

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

# 通过推算出后续的 query_id, 例如 201-400 对应 1-200 的 split
expanded_test_queries = []

# 假设原始数据的 query_id 范围是 1-200
for group_start in range(0, 800, 200):  # 每 200 个 query_id 是一组
    # 根据原始的 test_queries 映射
    for query_id in test_queries:
        expanded_test_queries.append(group_start + query_id)

# 将扩展后的测试查询的 query_id 转换为 DataFrame
expanded_test_queries_df = pd.DataFrame(expanded_test_queries, columns=['query_id'])

# 读取执行计划数据
df_plans = pd.read_csv('/home/zhy/opengauss/data_file/tpch_5g_output/plan_info.csv', delimiter=';', encoding='utf-8')
print(df_plans.columns)

df_query_info = pd.read_csv('/home/zhy/opengauss/data_file/tpch_5g_output/query_info.csv', delimiter=';', encoding='utf-8')

# 按 query_id 分组
query_groups = df_plans.groupby('query_id')

# 创建 PlanNode 对象并处理每个查询的树结构
query_trees = {}

# 仅处理测试数据的查询
for query_id, group in query_groups:
    if query_id in expanded_test_queries_df['query_id'].values:
        query_trees[query_id] = []
        
        # 创建该查询的所有计划节点
        for _, row in group.iterrows():
            plan_node = PlanNode(row)
            query_trees[query_id].append(plan_node)
        
        # 按 plan_id 排序，确保按顺序处理
        query_trees[query_id] = sorted(query_trees[query_id], key=lambda x: x.plan_id)
        
        # 处理父子关系
        for _, row in group.iterrows():
            parent_plan = [plan for plan in query_trees[query_id] if plan.plan_id == row['plan_id']][0]
            
            # 处理 child_plan 为空或没有子计划的情况
            child_plan = row['child_plan']
            if pd.isna(child_plan) or child_plan.strip() == '':  # 处理空值或空字符串的情况
                continue
            
            # 如果 child_plan 有多个子计划（逗号分隔），则拆分
            child_plans = child_plan.split(',')
            for child_plan_id in child_plans:
                child_plan_id = int(child_plan_id)  # 将每个子计划 ID 转换为整数
                child_plan_node = [plan for plan in query_trees[query_id] if plan.plan_id == child_plan_id][0]
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

# 遍历每个查询 ID，从新的测试集查询中获取数据
for query_id in expanded_test_queries_df['query_id']:
    actual_time_row = df_query_info[df_query_info['query_id'] == query_id]
    
    if not actual_time_row.empty:
        # 获取实际的执行时间和内存使用量
        actual_time = actual_time_row['execution_time'].values[0]
        actual_memory = actual_time_row['query_used_mem'].values[0]

        # 计算预测的执行时间和内存（根据 PlanNode 树）
        predicted_time = calculate_query_execution_time(query_trees[query_id])  # 根据查询树计算预测执行时间
        predicted_memory, _ = calculate_query_memory(query_trees[query_id])  # 根据查询树计算预测内存

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

# 映射查询 ID（从测试集中获取的查询 ID）到 1-xx 范围
mapped_query_ids = range(1, len(actual_times_in_s) + 1)

# 创建字典保存数据
data = {
    'Query ID (Mapped)': mapped_query_ids,
    'Actual Execution Time (s)': actual_times_in_s,
    'Predicted Execution Time (s)': predicted_times_in_s,
    'Actual Memory Usage (MB)': actual_memories_in_mb,
    'Predicted Memory Usage (MB)': predicted_memories_in_mb,
    'Execution Time Q-error': time_q_error,
    'Memory Q-error': memory_q_error
}

# 转换为 DataFrame
df = pd.DataFrame(data)

# 保存为 CSV 文件
csv_file_path = 'test_queries_results.csv'  # 修改为你希望的文件路径
df.to_csv(csv_file_path, index=False)

# 输出保存的文件路径
print(f"Data has been saved to {csv_file_path}")

        



