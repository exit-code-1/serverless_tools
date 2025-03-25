import csv
import json
import os
import re
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import dop_utils

import torch
# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from definition import PlanNode, ONNXModelManager, default_dop, thread_cost
    


# ------------------ 模块1：数据加载 ------------------
def load_test_queries(split_csv, top_n=200):
    df = pd.read_csv(split_csv, delimiter=';', encoding='utf-8')
    split_info = df[['query_id', 'split']].head(top_n)
    test_queries = split_info[split_info['split'] == 'test']['query_id']
    return pd.DataFrame(test_queries, columns=['query_id'])

def load_query_and_plan_info(query_info_csv, plan_info_csv):
    df_query_info = pd.read_csv(query_info_csv, delimiter=';', encoding='utf-8')
    df_plans = pd.read_csv(plan_info_csv, delimiter=';', encoding='utf-8')
    return df_query_info, df_plans

# ------------------ 模块2：构造查询计划树 ------------------
def build_query_trees(df_plans, onnx_manager):
    df_base = df_plans[df_plans['query_dop'] == 8].copy()
    query_groups = df_base.groupby(['query_id', 'query_dop'])
    query_trees = {}
    for (query_id, query_dop), group in query_groups:
        query_trees[(query_id, query_dop)] = []
        for _, row in group.iterrows():
            plan_node = PlanNode(row, onnx_manager)
            query_trees[(query_id, query_dop)].append(plan_node)
        query_trees[(query_id, query_dop)] = sorted(query_trees[(query_id, query_dop)], key=lambda x: x.plan_id)
        for _, row in group.iterrows():
            parent_plan = next(plan for plan in query_trees[(query_id, query_dop)] if plan.plan_id == row['plan_id'])
            if pd.isna(row['child_plan']) or row['child_plan'].strip() == '':
                continue
            child_plans = row['child_plan'].split(',')
            for child_plan_id in child_plans:
                child_plan_id = int(child_plan_id)
                child_plan_node = next(plan for plan in query_trees[(query_id, query_dop)] if plan.plan_id == child_plan_id)
                parent_plan.add_child(child_plan_node)
    return query_trees

def update_dop_plan_nodes(df_plans, onnx_manager, query_trees):
    """
    更新不同 dop 的查询计划：
      1. 对于非基准查询（query_dop != 8 且 !=1）的计划，构造 dop_plan_nodes。
      2. 对于每个 query_id，从基准查询树（query_dop == 8）中取出 base_nodes，
         然后对匹配的 dop_plan_nodes更新 base_node 的真实执行时间（update_real_exec_time）。
      3. 对每个 base_node，调用 compute_parallel_dop_predictions 计算预测执行时间（递归累加）。
      4. 最后利用更新后的 base_nodes构造线程块（ThreadBlock），调用新的线程块更新函数，
         在 ThreadBlock 内完成各项指标（包括真实与预测累计执行时间、阻塞时间等）的聚合。
      返回：一个字典，包含更新后的 query_trees 以及对应的线程块 thread_blocks，
            例如：{'query_trees': query_trees, 'thread_blocks': query_thread_blocks}
    """
    # ------------------ 构造 dop_plan_nodes ------------------
    df_other = df_plans[(df_plans['query_dop'] != 8)].copy()
    dop_plan_nodes = {}
    for _, row in df_other.iterrows():
        key = (row['query_id'], row['query_dop'])
        if key not in dop_plan_nodes:
            dop_plan_nodes[key] = []
        dop_plan_nodes[key].append(PlanNode(row, onnx_manager))
    
    # 用于保存每个 query (或 (query_id, 8)) 对应的线程块
    query_thread_blocks = {}

    # ------------------ 匹配更新 base_nodes 的执行时间信息 ------------------
    for query_id in set(key[0] for key in dop_plan_nodes.keys()):
        # 取出基准查询（query_dop==8）的计划树
        base_nodes = query_trees.get((query_id, 8), [])
        used_plan_nodes = set()
        for (qid, query_dop), plan_nodes in dop_plan_nodes.items():
            if qid != query_id:
                continue
            for plan_node in plan_nodes:
                if not plan_node.is_parallel:
                    continue
                for base_node in base_nodes:
                    if plan_node in used_plan_nodes:
                        break
                    if plan_node.operator_type != base_node.operator_type:
                        continue
                    # 情况 1+2：plan_id 相同且计划数匹配
                    if plan_node.plan_id == base_node.plan_id and len(base_nodes) == len(plan_nodes):
                        dop = plan_node.dop
                        dop = max(plan_node.updop, dop)
                        base_node.update_real_exec_time(dop, plan_node.execution_time)
                        used_plan_nodes.add(plan_node)
                        break
                    # 情况 3：利用关键特征匹配
                    key_features = ['l_input_rows', 'actual_rows', 'width']
                    if all(plan_node.exec_feature_data[feat] == base_node.exec_feature_data[feat] for feat in key_features):
                        dop = plan_node.dop
                        dop = max(plan_node.updop, dop)
                        base_node.update_real_exec_time(dop, plan_node.execution_time)
                        used_plan_nodes.add(plan_node)
                        break
        
        # ------------------ 基于更新后的 base_nodes构造线程块 ------------------
        # 假设你在 dop_utils 中已经定义了 update_thread_blocks 函数，
        # 该函数接收 base_nodes 划分线程块并聚合指标，返回一个字典：{thread_id: ThreadBlock, ...}
        thread_blocks = dop_utils.update_thread_blocks(base_nodes)
        # 保存该 query 的线程块信息（key 为 (query_id, 8)）
        query_thread_blocks[query_id] = thread_blocks
    
    # 返回更新后的查询计划树和线程块信息
    return {'query_trees': query_trees, 'thread_blocks': query_thread_blocks}

def compute_optimal_dops_on_thread_blocks(query_thread_blocks, 
                                            child_time_default=1e-6,  
                                            min_improvement_ratio=0.05, 
                                            block_threshold=100,  # 毫秒
                                            max_block_growth=1.2):
    """
    遍历所有线程块，调用每个 ThreadBlock 的 choose_optimal_dop 接口，
    生成一个字典，键为 (query_id, 8) 下的线程块 id，
    值为 { 'optimal_dop': ..., 'pred_time': ... }。
    
    参数说明：
      - query_thread_blocks：字典，键为 (query_id, 8)，值为线程块字典 {thread_id: ThreadBlock}
      - child_time_default：如果某个线程块没有子线程信息，则默认为一个很大的数（保证该线程块不会受子线程限制）
      - 其它参数传递给 ThreadBlock.choose_optimal_dop
    
    返回一个嵌套字典，例如：
      {
         (query_id, 8): {
             thread_id1: {'optimal_dop': X, 'pred_time': Y},
             thread_id2: {'optimal_dop': ... , ...},
             ...
         },
         ...
      }
    """
    optimal_dops = {}
    # 遍历每个 query (基准查询 key 为 (query_id, 8))
    for query_id, thread_blocks in query_thread_blocks.items():
        opt_dict = {}
        if query_id == 20:
            print()
        for tid, tb in thread_blocks.items():
            # 如果该线程块没有子线程信息，则取 child_max_execution_time = child_max_execution_time 或默认值
            child_time = tb.child_max_execution_time if tb.child_max_execution_time > 0 else child_time_default
            opt_dop = tb.choose_optimal_dop(child_time, 
                                                        min_improvement_ratio=min_improvement_ratio,
                                                        block_threshold=block_threshold,
                                                        max_block_growth=max_block_growth)
            opt_dict[tid] = opt_dop
        optimal_dops[query_id] = opt_dict
    return optimal_dops

def collect_all_nodes_from_tree(root_nodes):
    """通过 DFS 收集树中所有节点"""
    all_nodes = []
    visited = set()
    stack = list(root_nodes)
    while stack:
        node = stack.pop()
        if node.plan_id in visited:
            continue
        visited.add(node.plan_id)
        all_nodes.append(node)
        stack.extend(node.child_plans)
    return all_nodes

def update_nodes_with_optimal_dop(query_thread_blocks):
    """
    遍历 query_thread_blocks,每个线程块中每个节点更新 execution_time 和 real execution time,
    使其等于在最优dop下的值(从各节点的 exec_time_map 和 real_dop_exec_time 中获得）。
    """
    for query_key, tb_dict in query_thread_blocks.items():
        for tb_id, tb in tb_dict.items():
            # 获取该线程块的最优dop
            opt_dop = tb.optimal_dop
            if opt_dop is None:
                opt_dop = default_dop
            # 遍历该线程块内所有节点，更新 execution_time 和 real execution time
            for node in tb.nodes:
                if opt_dop in node.exec_time_map:
                    node.pred_execution_time = node.exec_time_map[opt_dop]
                else: 
                    node.pred_execution_time = node.execution_time
    # 返回更新后的节点（其实是原地更新）
    return query_thread_blocks

def write_query_details_to_file(query_trees, query_thread_blocks, optimal_dops, query_cpu_time, query_total_threads, output_file):
    queries_info = []
    
    for query_key, tree_nodes in query_trees.items():
        query_id = query_key[0]
        query_dict = {"query_id": str(query_id)}
        all_nodes = collect_all_nodes_from_tree(tree_nodes)
        for node in all_nodes:
            node.visit = False

        tb_list = []
        if query_id in query_thread_blocks and query_id in optimal_dops:
            thread_blocks =  query_thread_blocks.get(query_id, None)
            query_real_time = dop_utils.calculate_query_execution_time(all_nodes, thread_blocks)
            query_dict["query_real_execution_time"] = query_real_time
            query_dict["total_cpu_time"] = query_cpu_time.get(query_id, 0)  # ✅ 增加 CPU 时间统计
            query_dict["query_total_threads"] = query_total_threads.get(query_id, 0)  # ✅ 增加 CPU 时间统计
            opt_dict = optimal_dops[query_id]
            for tb_id, tb in thread_blocks.items():
                tb_info = {"thread_block_id": tb_id}
                optimal_dop = opt_dict.get(tb_id, None)
                tb_info["optimal_dop"] = optimal_dop
                tb_info["predicted_time"] = tb.pred_time
                tb_info["real_execution_time"] = tb.thread_execution_time
                
                ops = []
                for node in tb.nodes:
                    op_info = {
                        "operator_type": node.operator_type,
                        "predicted_time": node.pred_dop_exec_map.get(optimal_dop, None),
                        "real_time": node.true_dop_exec_map.get(optimal_dop, None) if hasattr(node, "true_dop_exec_map") else None
                    }
                    # **确保所有 NumPy 数组转换成 list**
                    for key, value in op_info.items():
                        if isinstance(value, np.ndarray):
                            op_info[key] = value.tolist()
                    ops.append(op_info)

                tb_info["operators"] = ops
                tb_list.append(tb_info)
        query_dict["thread_blocks"] = tb_list
        queries_info.append(query_dict)

    output_data = {"queries": queries_info}

    # **确保整个数据结构中的所有 ndarray 都转换为 list**
    def convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_ndarray(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_ndarray(v) for v in obj]
        return obj

    output_data = convert_ndarray(output_data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

def end_to_end_dop_optimization_and_print(df_plans, onnx_manager):
    """
    端到端流程：
      1. 构造查询计划树（build_query_trees）。
      2. 调用 update_dop_plan_nodes 对查询计划树进行更新，
         更新包括匹配 dop_plan_nodes、计算真实与预测执行时间累加，
         以及划分线程块（ThreadBlock）。
      3. 计算每个线程块的最优 dop（compute_optimal_dops_on_thread_blocks）。
      4. 更新每个线程块内各节点的 execution_time 为最优 dop 下的值。
      5. 统计每个 query_id 的 CPU 时间（所有线程块执行时间的总和）。
      6. 统计每个 query_id 的总线程数（所有线程块的 `thread_dop` 之和）。
      7. 打印详细信息，并写入 JSON 文件（调用 write_query_details_to_file）。
    
    返回一个字典，包含 query_trees、thread_blocks、optimal_dops、query_cpu_time 和 query_total_threads。
    """
    # 1. 构造查询计划树
    query_trees = build_query_trees(df_plans, onnx_manager)
    
    # 2. 更新 dop 信息并构造线程块
    dop_update_results = update_dop_plan_nodes(df_plans, onnx_manager, query_trees)
    thread_blocks = dop_update_results['thread_blocks']
    
    # 3. 计算每个线程块的最优 dop
    optimal_dops = compute_optimal_dops_on_thread_blocks(thread_blocks)
    
    # 4. 更新 execution_time 为最优 dop 下的值
    thread_blocks = update_nodes_with_optimal_dop(thread_blocks)
    
    # 5. 统计每个 query_id 的 CPU 时间 & 总线程数
    query_cpu_time = {}
    query_total_threads = {}

    for query_id, thread_dict in thread_blocks.items():  # 遍历 query_id
        total_cpu_time = 0
        total_threads = 0  # 统计总线程数
        
        for tb in thread_dict.values():
            optimal_dop = tb.optimal_dop
            real_execution_time = tb.real_dop_exec_time.get(optimal_dop, 0) if hasattr(tb, "real_dop_exec_time") else 0
            total_cpu_time += real_execution_time

            # 计算线程数（所有线程块的 thread_dop 之和）
            total_threads += optimal_dop  # 如果 tb 没有 thread_dop，默认 0

        query_cpu_time[query_id] = total_cpu_time + total_threads * thread_cost  # 记录 CPU 时间
        query_total_threads[query_id] = total_threads  # 记录总线程数

    # 6. 将详细信息写入 JSON 文件
    output_file = "query_details.json"
    write_query_details_to_file(query_trees, thread_blocks, optimal_dops, query_cpu_time, query_total_threads, output_file)

    # 7. 返回统计信息
    return {
        'query_trees': query_trees,
        'thread_blocks': thread_blocks,
        'optimal_dops': optimal_dops,
        'query_cpu_time': query_cpu_time,
        'query_total_threads': query_total_threads  # 统计每个查询的总线程数
    }
    

def get_nearest_dop(dop_exec_map, query_dop):
    """ 在 `dop_exec_map` 里找到最接近的 `dop` """
    available_dops = sorted(dop_exec_map)
    return min(available_dops, key=lambda x: abs(x - query_dop))

def end_to_end_processing(df_plans, optimal_dop_file, output_file):
    """
    端到端处理流程：
    1. 读取最优 DOP 文件，获取 (query_id, optimal_dop) 映射。
    2. 读取查询计划文件，构建查询树（query_trees）。
    3. 使用最优 DOP 更新查询计划的 dop 值，并重新计算执行时间。
    4. 统计线程块信息，包括总执行时间、完成时间和 CPU 时间。
    5. 调用 `write_query_details_to_file` 将信息写入 JSON 文件。

    :param plan_file: 计划文件路径（CSV格式）
    :param optimal_dop_file: 最优 DOP 文件路径（CSV格式）
    :param onnx_manager: 预训练的 ONNX 管理器
    :param output_file: 输出 JSON 文件路径
    """

    # 1. 读取最优 DOP 文件
    df_optimal_dop = pd.read_csv(optimal_dop_file, sep=';', names=['query_id', 'optimal_dop'])
    query_dop_map = dict(zip(df_optimal_dop['query_id'], df_optimal_dop['optimal_dop']))
    optimal_dop_maps = {}
    onnx_manager = ONNXModelManager()
    query_trees = build_query_trees(df_plans, onnx_manager)  # 只构造 `query_dop=8` 的树

    # 3. 修改查询计划的 `query_dop`
    query_thread_blocks = {}
    query_cpu_time = {}  # 记录 CPU 时间
    query_total_threads = {}

# ------------------ 构造 dop_plan_nodes ------------------
    df_other = df_plans[(df_plans['query_dop'] != 8) & (df_plans['query_dop'] != 1)].copy()
    dop_plan_nodes = {}
    for _, row in df_other.iterrows():
        key = (row['query_id'], row['query_dop'])
        if key not in dop_plan_nodes:
            dop_plan_nodes[key] = []
        dop_plan_nodes[key].append(PlanNode(row, onnx_manager))
    
    # 用于保存每个 query (或 (query_id, 8)) 对应的线程块
    query_thread_blocks = {}

    # ------------------ 匹配更新 base_nodes 的执行时间信息 ------------------
    for query_id in set(key[0] for key in dop_plan_nodes.keys()):
        # 取出基准查询（query_dop==8）的计划树
        base_nodes = query_trees.get((query_id, 8), [])
        used_plan_nodes = set()
        for (qid, query_dop), plan_nodes in dop_plan_nodes.items():
            if qid != query_id:
                continue
            for plan_node in plan_nodes:
                if not plan_node.is_parallel:
                    continue
                for base_node in base_nodes:
                    if plan_node in used_plan_nodes:
                        break
                    if plan_node.operator_type != base_node.operator_type:
                        continue
                    # 情况 1+2：plan_id 相同且计划数匹配
                    if plan_node.plan_id == base_node.plan_id and len(base_nodes) == len(plan_nodes):
                        dop = plan_node.dop
                        dop = max(plan_node.updop, dop)
                        base_node.update_real_exec_time(dop, plan_node.execution_time)
                        used_plan_nodes.add(plan_node)
                        break
                    # 情况 3：利用关键特征匹配
                    key_features = ['l_input_rows', 'actual_rows', 'width']
                    if all(plan_node.exec_feature_data[feat] == base_node.exec_feature_data[feat] for feat in key_features):
                        dop = plan_node.dop
                        dop = max(plan_node.updop, dop)
                        base_node.update_real_exec_time(dop, plan_node.execution_time)
                        used_plan_nodes.add(plan_node)
                        break
        total_cpu_time = 0
        total_threads = 0
        # ------------------ 基于更新后的 base_nodes构造线程块 ------------------
        # 假设你在 dop_utils 中已经定义了 update_thread_blocks 函数，
        # 该函数接收 base_nodes 划分线程块并聚合指标，返回一个字典：{thread_id: ThreadBlock, ...}
        query_dop = query_dop_map.get(query_id, 8)  # 默认为 8
        operator_dop = {}
        dop = query_dop
                # **确保 query_dop 在 `real_dop_exec_time` 里**
        # **更新所有节点的 `query_dop`**
        for node in base_nodes:
            if query_dop not in node.matched_dops:
                dop = get_nearest_dop(node.matched_dops, query_dop)
                node.dop = dop
            node.pred_exec_time = node.exec_time_map.get(dop, 1e-6)
        thread_blocks = dop_utils.update_thread_blocks(base_nodes)
        for thread_id, tb in thread_blocks.items():
            tb.optimal_dop = tb.nodes[0].dop
            operator_dop[thread_id] = tb.optimal_dop
            real_execution_time = tb.real_dop_exec_time.get(tb.optimal_dop, 0) if hasattr(tb, "real_dop_exec_time") else 0
            total_cpu_time += real_execution_time
            total_threads += tb.optimal_dop  # 如果 tb 没有 thread_dop，默认 0
        # 保存该 query 的线程块信息（key 为 (query_id, 8)）
        query_cpu_time[query_id] = total_cpu_time + total_threads * thread_cost  # 记录 CPU 时间
        query_total_threads[query_id] = total_threads  # 记录总线程数
        query_thread_blocks[query_id] = thread_blocks
        optimal_dop_maps[query_id] = operator_dop

    # 4. 调用 `write_query_details_to_file` 存储查询信息
    write_query_details_to_file(query_trees, query_thread_blocks, optimal_dop_maps, query_cpu_time, query_total_threads, output_file)
    
def compare_methods_and_write_csv(baseline_file, method_file_map, output_csv_file):
    """
    对比基准 JSON 与其它方法 JSON 中每个 query 的执行时间、CPU 时间和线程数，
    计算相对于基准的比例（倍数），并将结果写入 CSV 文件。

    参数：
      - baseline_file: 基准 JSON 文件路径（字符串）
      - method_file_map: dict, 键为方法名称（例如 "Method_A"），值为对应 JSON 文件路径
      - output_csv_file: 输出 CSV 文件路径（字符串）

    CSV 文件每行包含：
      method_name, query_id, execution_time_multiplier, cpu_time_multiplier, thread_count_multiplier
    """
    # 加载基准 JSON 文件
    with open(baseline_file, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    
    # 将基准中的 query 按 query_id 组成字典（尝试转为 int，否则保持字符串）
    baseline_queries = {}
    for query in baseline_data["queries"]:
        try:
            qid = int(query["query_id"])
        except Exception:
            qid = query["query_id"]
        baseline_queries[qid] = query

    comparisons = {}
    # 对于每个方法，加载对应 JSON 文件，并计算比例
    for method_name, file_path in method_file_map.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            method_data = json.load(f)
        method_queries = {}
        for query in method_data["queries"]:
            try:
                qid = int(query["query_id"])
            except Exception:
                qid = query["query_id"]
            method_queries[qid] = query

        method_comparison = {}
        for qid, base_query in baseline_queries.items():
            if qid in method_queries:
                method_query = method_queries[qid]
                base_exec = base_query.get("query_real_execution_time", 0)
                base_cpu = base_query.get("total_cpu_time", 0)
                base_threads = base_query.get("query_total_threads", 0)
                
                method_exec = method_query.get("query_real_execution_time", 0)
                method_cpu = method_query.get("total_cpu_time", 0)
                method_threads = method_query.get("query_total_threads", 0)
                
                exec_ratio = method_exec / base_exec if base_exec != 0 else None
                cpu_ratio = method_cpu / base_cpu if base_cpu != 0 else None
                thread_ratio = method_threads / base_threads if base_threads != 0 else None

                method_comparison[qid] = {
                    "execution_time_multiplier": exec_ratio,
                    "cpu_time_multiplier": cpu_ratio,
                    "thread_count_multiplier": thread_ratio
                }
        comparisons[method_name] = method_comparison

    # 写入 CSV 文件
    with open(output_csv_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["method_name", "query_id", "execution_time_multiplier", "cpu_time_multiplier", "thread_count_multiplier"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for method_name, query_dict in comparisons.items():
            for query_id, metrics in query_dict.items():
                row = {
                    "method_name": method_name,
                    "query_id": query_id,
                    "execution_time_multiplier": metrics.get("execution_time_multiplier"),
                    "cpu_time_multiplier": metrics.get("cpu_time_multiplier"),
                    "thread_count_multiplier": metrics.get("thread_count_multiplier")
                }
                writer.writerow(row)
    print(f"比较结果已写入 {output_csv_file}")

# 示例 main 函数
if __name__ == '__main__':
    
    df_plans = pd.read_csv("/home/zhy/opengauss/data_file/tpch_10g_output_22/plan_info.csv", delimiter=';', encoding='utf-8')
    onnx_manager = ONNXModelManager()
    
    # 运行端到端流程，并打印和写入详细信息
    results = end_to_end_dop_optimization_and_print(df_plans, onnx_manager)
    # end_to_end_processing(df_plans, "/home/zhy/opengauss/tools/serverless_tools/train/python/no_dop/dop_result/tru_dop.csv", "tru_details.json")
    # baseline_json = "tru_details.json"  # 基准 JSON 文件路径
    # method_files = {
    #     "Ours": "query_details.json",
    #     # 如有更多方法，可继续添加
    # }
    # output_csv = "comparisons.csv"
    # compare_methods_and_write_csv(baseline_json, method_files, output_csv)



