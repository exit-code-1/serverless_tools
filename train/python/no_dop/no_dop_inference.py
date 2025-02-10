import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 读取 execution_plans.csv 文件
df_plans = pd.read_csv('/home/zhy/opengauss/data_file/tpch_10g_output/plan_info.csv', delimiter=';')

# 读取 query_info.csv 文件
df_query_info = pd.read_csv('/home/zhy/opengauss/data_file/tpch_10g_output/query_info.csv', delimiter=';')

# 构建 plan_dict 和 PlanNode 类
plan_dict = {}

class PlanNode:
    def __init__(self, plan_data):
        self.visit = False
        self.plan_id = plan_data['plan_id']
        self.query_id = plan_data['query_id']
        self.operator_type = plan_data['operator_type']
        self.updop =  plan_data['up_dop']
        self.downdop =  plan_data['down_dop']
        self.execution_time = plan_data['execution_time']
        self.peak_mem = plan_data['peak_mem']
        self.width = plan_data['width']
        self.child_plans = []  # 用于存储子计划节点
        self.materialized = 'hash' in self.operator_type.lower() or 'aggregate' in self.operator_type.lower() or 'sort' in self.operator_type.lower() or 'materialize' in self.operator_type.lower()  # 判断是否是物化算子
        self.parent_node = None  # 父节点
        self.thread_execution_time = 0
        self.thread_complete_time = 0
        self.local_data_transfer_start_time = 0
        self.up_data_transfer_start_time = 0
    
    def add_child(self, child_node):
        self.child_plans.append(child_node)
        child_node.parent_node = self
        child_node.visit = False  # 重置子节点的访问标记
        
def build_execution_plan_graph(query_id, query_trees):
    """
    生成查询计划的图形，图中的每个节点代表一个算子，并显示计算信息。
    """
    # 创建一个有向图
    G = nx.DiGraph()

    # 遍历节点
    for plan in query_trees[query_id]:
        # 每个节点的标签包含计算后的信息
        label = (f"Plan ID: {plan.plan_id}\n"
                 f"Type: {plan.operator_type}\n"
                 f"Exec Time: {plan.thread_execution_time}\n"
                 f"Complete Time: {plan.thread_complete_time}\n"
                 f"Data Start: {plan.local_data_transfer_start_time}\n"
                 f"Up Transfer Start: {plan.up_data_transfer_start_time}")

        # 为每个节点添加一个节点，节点标签为算子的相关信息
        G.add_node(plan.plan_id, label=label)

        # 连接父子节点
        for child in plan.child_plans:
            G.add_edge(plan.plan_id, child.plan_id)

    return G

def draw_plan_graph(G):
    """
    绘制执行计划图形，展示算子节点及其父子关系，并保存为矢量图。
    """
    # 绘制图形
    pos = nx.spring_layout(G, seed=42, k=1)  # 使用spring_layout进行布局，k调整节点间距
    plt.figure(figsize=(15, 15))  # 设置图形大小

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="skyblue", alpha=0.6)

    # 绘制边
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=2, alpha=0.5)

    # 绘制标签
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=6, font_color="black", font_weight="bold")

    # 显示图形
    plt.title("Execution Plan Graph")
    plt.axis('off')
    # 保存图形为矢量图（SVG 或 PDF）
    plt.savefig("tmp.pdf", format="pdf")  # 或者 format="pdf"
    # 显示图形
    plt.title("Execution Plan Graph")
    plt.show()

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
    thread_execution_time = node.execution_time
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
            if child_complete_times[0] > child_complete_times[1]:
                up_data_transfer_start_time = max(up_data_transfer_start_time, local_data_transfer_start_time + thread_execution_time - child_execution_times[0])
            else:
                up_data_transfer_start_time = max(up_data_transfer_start_time, local_data_transfer_start_time + thread_execution_time - child_execution_times[1])
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

def calculate_query_memory(query_nodes):
    """
    计算一个查询所占用的内存。
    - 每个算子的内存为 `peak_mem`，加上生成的线程数。
    """
    total_peak_mem = 0
    total_threads = 1

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
    total_memory = total_peak_mem + (total_threads - 1)* 10000
    return total_memory


# 按 query_id 分组计划
query_groups = df_plans.groupby('query_id')

# 处理每个查询的计划
query_trees = {}

# 创建 PlanNode 对象并处理每个查询的树结构
for query_id, group in query_groups:
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



def calculate_query_memory(query_nodes):
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
    total_memory = total_peak_mem + (total_threads - 1) * 10000

    # 返回内存以及平均线程数
    return total_memory, total_threads # 返回平均线程数

# 初始化用于存储每组误差和其他统计的列表
dop_1_errors = []
dop_2_errors = []
dop_4_errors = []
dop_8_errors = []

dop_1_times = []
dop_2_times = []
dop_4_times = []
dop_8_times = []

dop_1_memories = []
dop_2_memories = []
dop_4_memories = []
dop_8_memories = []

dop_1_avg_threads = []
dop_2_avg_threads = []
dop_4_avg_threads = []
dop_8_avg_threads = []

with open(log_file_path, 'w') as log_file:
    for query_id, plan_nodes in query_trees.items():
        # 获取根节点（plan_id为0的节点）
        root_node = next((node for node in plan_nodes if node.plan_id == 0), None)

        # 计算执行时间
        calculated_time = calculate_query_execution_time(plan_nodes)

        # 计算内存和平均线程数
        calculated_memory, avg_threads = calculate_query_memory(plan_nodes)

        # 获取实际执行时间和内存使用
        actual_time_row = df_query_info[df_query_info['query_id'] == query_id]
        if not actual_time_row.empty:
            actual_time = actual_time_row['execution_time'].values[0]
            actual_memory = actual_time_row['query_used_mem'].values[0]
        else:
            actual_time = None
            actual_memory = None

        # 计算误差
        time_error = abs(calculated_time - actual_time) if actual_time is not None else None
        memory_error = abs(calculated_memory - actual_memory) if actual_memory is not None else None
        
        # 计算预测精度（百分比）
        time_accuracy = (1 - (time_error / actual_time)) * 100 if actual_time is not None else None
        memory_accuracy = (1 - (memory_error / actual_memory)) * 100 if actual_memory is not None else None

        # 写入到日志文件
        log_file.write(f"Query {query_id}:\n")
        
        # 写入执行时间对比
        log_file.write(f"  Total Calculated Execution Time = {calculated_time}\n")
        if actual_time is not None:
            log_file.write(f"  Actual Execution Time from query_info.csv = {actual_time}\n")
            log_file.write(f"  Difference = {time_error}\n")
            log_file.write(f"  Time Prediction Accuracy = {time_accuracy:.2f}%\n")
        else:
            log_file.write("  No actual execution time found in query_info.csv\n")
        
        # 写入内存对比
        log_file.write(f"  Total Calculated Memory = {calculated_memory}\n")
        if actual_memory is not None:
            log_file.write(f"  Actual Memory Usage from query_info.csv = {actual_memory}\n")
            log_file.write(f"  Memory Difference = {memory_error}\n")
            log_file.write(f"  Memory Prediction Accuracy = {memory_accuracy:.2f}%\n")
        else:
            log_file.write("  No actual memory usage found in query_info.csv\n")
        
        log_file.write("\n")  # 在每个查询之间添加空行

        # 将误差和统计信息按dop组分类
        if 1 <= query_id <= 22:
            dop_1_errors.append((time_error, memory_error))
            dop_1_times.append(calculated_time)
            dop_1_memories.append(calculated_memory)
            dop_1_avg_threads.append(avg_threads)
        elif 23 <= query_id <= 44:
            dop_2_errors.append((time_error, memory_error))
            dop_2_times.append(calculated_time)
            dop_2_memories.append(calculated_memory)
            dop_2_avg_threads.append(avg_threads)
        elif 45 <= query_id <= 66:
            dop_4_errors.append((time_error, memory_error))
            dop_4_times.append(calculated_time)
            dop_4_memories.append(calculated_memory)
            dop_4_avg_threads.append(avg_threads)
        elif 67 <= query_id <= 88:
            dop_8_errors.append((time_error, memory_error))
            dop_8_times.append(calculated_time)
            dop_8_memories.append(calculated_memory)
            dop_8_avg_threads.append(avg_threads)

    # 计算每组的平均误差、平均执行时间、平均内存、平均预测精度
    def calculate_average_error_and_stats(errors, times, memories, avg_threads):
        time_errors = [e[0] for e in errors if e[0] is not None]
        memory_errors = [e[1] for e in errors if e[1] is not None]
        time_accuracies = [(1 - (e[0] / t)) * 100 for e, t in zip(errors, times) if e[0] is not None and t is not None]
        memory_accuracies = [(1 - (e[1] / m)) * 100 for e, m in zip(errors, memories) if e[1] is not None and m is not None]

        avg_time_error = sum(time_errors) / len(time_errors) if time_errors else None
        avg_memory_error = sum(memory_errors) / len(memory_errors) if memory_errors else None
        avg_time_accuracy = sum(time_accuracies) / len(time_accuracies) if time_accuracies else None
        avg_memory_accuracy = sum(memory_accuracies) / len(memory_accuracies) if memory_accuracies else None
        
        avg_time = sum(times) / len(times) if times else None
        avg_memory = sum(memories) / len(memories) if memories else None
        avg_threads = sum(avg_threads) / len(avg_threads) if avg_threads else None

        return avg_time_error, avg_memory_error, avg_time_accuracy, avg_memory_accuracy, avg_time, avg_memory, avg_threads

    # 计算各组的统计信息
    dop_1_avg_time_error, dop_1_avg_memory_error, dop_1_avg_time_accuracy, dop_1_avg_memory_accuracy, dop_1_avg_time, dop_1_avg_memory, dop_1_avg_threads = calculate_average_error_and_stats(dop_1_errors, dop_1_times, dop_1_memories, dop_1_avg_threads)
    dop_2_avg_time_error, dop_2_avg_memory_error, dop_2_avg_time_accuracy, dop_2_avg_memory_accuracy, dop_2_avg_time, dop_2_avg_memory, dop_2_avg_threads = calculate_average_error_and_stats(dop_2_errors, dop_2_times, dop_2_memories, dop_2_avg_threads)
    dop_4_avg_time_error, dop_4_avg_memory_error, dop_4_avg_time_accuracy, dop_4_avg_memory_accuracy, dop_4_avg_time, dop_4_avg_memory, dop_4_avg_threads = calculate_average_error_and_stats(dop_4_errors, dop_4_times, dop_4_memories, dop_4_avg_threads)
    dop_8_avg_time_error, dop_8_avg_memory_error, dop_8_avg_time_accuracy, dop_8_avg_memory_accuracy, dop_8_avg_time, dop_8_avg_memory, dop_8_avg_threads = calculate_average_error_and_stats(dop_8_errors, dop_8_times, dop_8_memories, dop_8_avg_threads)

    # 写入总体误差、预测精度、执行时间和内存的平均值
    log_file.write("\n")
    log_file.write("Average Errors per dop group:\n")
    log_file.write(f"  dop = 1 - Average Execution Time Error = {dop_1_avg_time_error}, Average Memory Error = {dop_1_avg_memory_error}\n")
    log_file.write(f"  dop = 1 - Average Execution Time Accuracy = {dop_1_avg_time_accuracy}%, Average Memory Accuracy = {dop_1_avg_memory_accuracy}%\n")
    log_file.write(f"  dop = 1 - Average Execution Time = {dop_1_avg_time}, Average Memory = {dop_1_avg_memory}\n")
    log_file.write(f"  dop = 1 - Average Threads = {dop_1_avg_threads}\n")
    
    log_file.write(f"  dop = 2 - Average Execution Time Error = {dop_2_avg_time_error}, Average Memory Error = {dop_2_avg_memory_error}\n")
    log_file.write(f"  dop = 2 - Average Execution Time Accuracy = {dop_2_avg_time_accuracy}%, Average Memory Accuracy = {dop_2_avg_memory_accuracy}%\n")
    log_file.write(f"  dop = 2 - Average Execution Time = {dop_2_avg_time}, Average Memory = {dop_2_avg_memory}\n")
    log_file.write(f"  dop = 2 - Average Threads = {dop_2_avg_threads}\n")
    
    log_file.write(f"  dop = 4 - Average Execution Time Error = {dop_4_avg_time_error}, Average Memory Error = {dop_4_avg_memory_error}\n")
    log_file.write(f"  dop = 4 - Average Execution Time Accuracy = {dop_4_avg_time_accuracy}%, Average Memory Accuracy = {dop_4_avg_memory_accuracy}%\n")
    log_file.write(f"  dop = 4 - Average Execution Time = {dop_4_avg_time}, Average Memory = {dop_4_avg_memory}\n")
    log_file.write(f"  dop = 4 - Average Threads = {dop_4_avg_threads}\n")
    
    log_file.write(f"  dop = 8 - Average Execution Time Error = {dop_8_avg_time_error}, Average Memory Error = {dop_8_avg_memory_error}\n")
    log_file.write(f"  dop = 8 - Average Execution Time Accuracy = {dop_8_avg_time_accuracy}%, Average Memory Accuracy = {dop_8_avg_memory_accuracy}%\n")
    log_file.write(f"  dop = 8 - Average Execution Time = {dop_8_avg_time}, Average Memory = {dop_8_avg_memory}\n")
    log_file.write(f"  dop = 8 - Average Threads = {dop_8_avg_threads}\n")

# print(f"对比结果已保存到文件：{log_file_path}")

# 提取 dop = 8 组的真实和预测数据
# dop_8_query_ids = range(67, 89)  # 查询 ID 从 67 到 88
# dop_8_actual_times_in_s = []
# dop_8_predicted_times_in_s = []
# dop_8_actual_memories_in_mb = []
# dop_8_predicted_memories_in_mb = []
# dop_8_time_accuracy = []
# dop_8_memory_accuracy = []

# # 提取每个查询的实际执行时间、内存和预测值
# for query_id in dop_8_query_ids:
#     actual_time_row = df_query_info[df_query_info['query_id'] == query_id]
#     if not actual_time_row.empty:
#         actual_time = actual_time_row['execution_time'].values[0]
#         actual_memory = actual_time_row['query_used_mem'].values[0]

#         # 计算预测的执行时间和内存
#         predicted_time = calculate_query_execution_time(query_trees[query_id])  # 不再减去1
#         predicted_memory = calculate_query_memory(query_trees[query_id])

#         # 转换单位
#         dop_8_actual_times_in_s.append(actual_time / 1000)  # 转为秒
#         dop_8_predicted_times_in_s.append(predicted_time / 1000)  # 转为秒
#         dop_8_actual_memories_in_mb.append(actual_memory / 1000)  # 转为 MB
#         dop_8_predicted_memories_in_mb.append(predicted_memory / 1000)  # 转为 MB

#         # 计算预测准确率
#         time_error = abs(predicted_time - actual_time)
#         memory_error = abs(predicted_memory - actual_memory)

#         time_accuracy = (1 - (time_error / actual_time)) * 100
#         memory_accuracy = (1 - (memory_error / actual_memory)) * 100

#         dop_8_time_accuracy.append(time_accuracy)
#         dop_8_memory_accuracy.append(memory_accuracy)

# # 映射查询 ID 从 67-88 到 1-22
# dop_8_query_ids_mapped = range(1, len(dop_8_actual_times_in_s) + 1)

# # 绘制执行时间的图表（对数坐标）
# fig, ax1 = plt.subplots(figsize=(14, 6))

# # 创建柱状图：执行时间的准确率
# bar_width = 0.35
# index = np.arange(len(dop_8_query_ids_mapped))

# # 创建柱状图：执行时间的准确率
# bar1 = ax1.bar(index, dop_8_time_accuracy, bar_width, label='Time Accuracy', color='g')  # 使用与内存图相同的颜色

# # 添加数据标签
# for i, v in enumerate(dop_8_time_accuracy):
#     ax1.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom')

# # 设置左侧图表的标签和标题
# ax1.set_xlabel('Query ID (Mapped)')
# ax1.set_ylabel('Prediction Accuracy (%)')
# ax1.set_title('Prediction Accuracy for Execution Time (dop=8)')
# ax1.set_xticks(index)
# ax1.set_xticklabels(dop_8_query_ids_mapped)
# ax1.legend(loc='upper center')

# # 创建另一个y轴显示真实的执行时间
# ax2_1 = ax1.twinx()

# # 绘制折线图：真实执行时间，使用对数坐标
# ax2_1.plot(index, dop_8_actual_times_in_s, label="Actual Execution Time (s)", marker='o', color='purple')  # 统一颜色
# ax2_1.set_yscale('log')  # 设置对数坐标
# ax2_1.set_ylabel('Real Execution Time (s) (Log Scale)')
# ax2_1.legend(loc='upper left')

# # 添加数据标签
# for i, v in enumerate(dop_8_actual_times_in_s):
#     ax2_1.text(i, v - 1, f"{v:.2f}", ha='center', va='top', color='purple')

# # 保存为 PDF 文件：exec.pdf
# output_path_exec = 'exec.pdf'
# plt.savefig(output_path_exec)

# # 绘制内存的图表
# fig, ax1 = plt.subplots(figsize=(14, 6))

# # 创建柱状图：内存的准确率
# bar1 = ax1.bar(index, dop_8_memory_accuracy, bar_width, label='Memory Accuracy', color='g')

# # 添加数据标签
# for i, v in enumerate(dop_8_memory_accuracy):
#     ax1.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom')

# # 设置右侧图表的标签和标题
# ax1.set_xlabel('Query ID (Mapped)')
# ax1.set_ylabel('Prediction Accuracy (%)')
# ax1.set_title('Prediction Accuracy for Memory Usage (dop=8)')
# ax1.set_xticks(index)
# ax1.set_xticklabels(dop_8_query_ids_mapped)
# ax1.legend(loc='upper center')

# # 创建另一个y轴显示真实的内存使用
# ax2_2 = ax1.twinx()

# # 绘制折线图：真实内存使用
# ax2_2.plot(index, dop_8_actual_memories_in_mb, label="Actual Memory Usage (MB)", marker='o', color='purple')

# # 设置右侧y轴标签
# ax2_2.set_ylabel('Real Memory Usage (MB)')
# ax2_2.legend(loc='upper left')

# # 添加数据标签
# for i, v in enumerate(dop_8_actual_memories_in_mb):
#     ax2_2.text(i, v + 1, f"{v:.2f}", ha='center', va='bottom', color='purple')

# # 调整布局
# fig.tight_layout(pad=3.0)

# # 保存为 PDF 文件：mem.pdf
# output_path_mem = 'mem.pdf'
# plt.savefig(output_path_mem)

# # 创建字典保存数据
# data = {
#     'Query ID (Mapped)': dop_8_query_ids_mapped,
#     'Actual Execution Time (s)': dop_8_actual_times_in_s,
#     'Predicted Execution Time (s)': dop_8_predicted_times_in_s,
#     'Actual Memory Usage (MB)': dop_8_actual_memories_in_mb,
#     'Predicted Memory Usage (MB)': dop_8_predicted_memories_in_mb,
#     'Time Accuracy (%)': dop_8_time_accuracy,
#     'Memory Accuracy (%)': dop_8_memory_accuracy
# }

# # 转换为 DataFrame
# df = pd.DataFrame(data)

# # 保存为 CSV 文件
# csv_file_path = 'dop_8_results.csv'  # 修改为你希望的文件路径
# df.to_csv(csv_file_path, index=False)

# # 输出保存的文件路径
# print(f"Data has been saved to {csv_file_path}")

# # 计算 query_id 为 24 的查询
# query_id_to_plot = 40
# calculate_query_execution_time(query_trees[query_id_to_plot])

# # 生成查询计划图并绘制
# G = build_execution_plan_graph(query_id_to_plot, query_trees)

# # 绘制图形
# draw_plan_graph(G)
        



