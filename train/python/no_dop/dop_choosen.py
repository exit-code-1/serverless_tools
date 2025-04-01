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
from definition import PlanNode, ONNXModelManager, default_dop, thread_cost, thread_mem
from structure import dop_operators_exec
    


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
                if not plan_node.operator_type in dop_operators_exec:
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
                                            min_improvement_ratio=0.4, 
                                            block_threshold=200,  # 毫秒
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

def write_query_details_to_file(query_trees, query_thread_blocks, output_file):
    queries_info = []
    for query_key, tree_nodes in query_trees.items():
        query_id = query_key[0]
        query_dict = {"query_id": str(query_id)}
        all_nodes = collect_all_nodes_from_tree(tree_nodes)
        cpu_time = 0
        total_threads = 0
        total_mem = 0
        max_dop = 0
        for node in all_nodes:
            node.visit = False

        tb_list = []
        if query_id in query_thread_blocks:
            thread_blocks =  query_thread_blocks.get(query_id, None)
            query_real_time = dop_utils.calculate_query_execution_time(all_nodes, thread_blocks)
            total_mem, _ = dop_utils.calculate_query_memory(all_nodes)
            query_dict["query_real_execution_time"] = query_real_time
            for tb_id, tb in thread_blocks.items():
                tb_info = {"thread_block_id": tb_id}
                optimal_dop = tb.optimal_dop
                max_dop = max(max_dop , optimal_dop)
                total_threads += optimal_dop
                tb_info["optimal_dop"] = optimal_dop
                tb_info["predicted_time"] = tb.pred_time
                tb_info["real_execution_time"] = tb.nodes[0].thread_execution_time
                cpu_time += tb.nodes[0].thread_execution_time
                
                ops = []
                for node in tb.nodes:
                    op_info = {
                        "plan_id": node.plan_id,
                        "width": node.width,
                        "parent_child":  node.parent_node.plan_id if node.parent_node else -1,
                        "left_child": node.child_plans[0].plan_id if node.child_plans else -1,
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
            
        query_dict["total_cpu_time"] = cpu_time + total_threads * thread_cost  # ✅ 增加 CPU 时间统计
        query_dict["query_total_threads"] = total_threads  # ✅ 增加 CPU 时间统计
        query_dict["query_total_mem"] = int(total_mem + total_threads * thread_mem)
        query_dict["max_dop"] = max_dop
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
    

    # 6. 将详细信息写入 JSON 文件
    output_file = "query_details.json"
    write_query_details_to_file(query_trees, thread_blocks, output_file)

    # 7. 返回统计信息
    return {
        'query_trees': query_trees,
        'thread_blocks': thread_blocks,
        'optimal_dops': optimal_dops,
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
    df_optimal_dop = pd.read_csv(optimal_dop_file, sep=';', names=['query_id', 'optimal_dop'], skiprows=1, dtype={'query_id': int})
    query_dop_map = dict(zip(df_optimal_dop['query_id'], df_optimal_dop['optimal_dop']))
    onnx_manager = ONNXModelManager()
    query_trees = build_query_trees(df_plans, onnx_manager)  # 只构造 `query_dop=8` 的树
    
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
                if not plan_node.operator_type in dop_operators_exec:
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
        query_dop = query_dop_map.get(query_id, 8)  # 默认为 8
                # **确保 query_dop 在 `real_dop_exec_time` 里**
        # **更新所有节点的 `query_dop`**
        for node in base_nodes:
            dop = query_dop
            if not node.is_parallel:
                dop = 1
            node.dop = dop
            node.pred_execution_time = node.exec_time_map.get(dop, 1e-6)
        thread_blocks = dop_utils.update_thread_blocks(base_nodes)
        for thread_id, tb in thread_blocks.items():
            tb.optimal_dop = tb.nodes[0].dop
            for node in tb.nodes:
                tb.optimal_dop = min(node.dop, tb.optimal_dop)
        query_thread_blocks[query_id] = thread_blocks
        query_trees[(query_id, 8)] = base_nodes
    # 4. 调用 `write_query_details_to_file` 存储查询信息
    write_query_details_to_file(query_trees, query_thread_blocks, output_file)
    
def calculate_cpu_time(df_plans):
    """
    计算 CPU 时间：CPU 时间 = 各算子 (execution_time * dop) 的和
    """
    df_plans['cpu_time'] = df_plans['execution_time'] * df_plans['dop']
    cpu_time = df_plans.groupby('query_id')['cpu_time'].sum().reset_index()
    return cpu_time

def calculate_thread_count(df_plans):
    """
    计算线程数：包含 'streaming' 的算子累加 up_dop，最后加 1
    """
    df_plans['is_streaming'] = df_plans['operator_type'].str.lower().str.contains('streaming')
    df_stream = df_plans[df_plans['is_streaming']].copy()
    thread_count = df_stream.groupby('query_id')['down_dop'].sum().reset_index()
    thread_count.rename(columns={'down_dop': 'streaming_dop_sum'}, inplace=True)
    thread_count['thread_count'] = thread_count['streaming_dop_sum'] + 1

    # 补全没有 streaming 的 query，线程数设为 1
    all_q = pd.DataFrame({'query_id': df_plans['query_id'].unique()})
    thread_count = pd.merge(all_q, thread_count[['query_id', 'thread_count']], on='query_id', how='left')
    thread_count['thread_count'].fillna(1, inplace=True)
    return thread_count[['query_id', 'thread_count']]

def calculate_query_features(query_info_path, plan_info_path, prefix):
    """
    计算查询特征：包括 CPU 时间和线程数，结果加上指定前缀
    """
    df_query = pd.read_csv(query_info_path, delimiter=';', encoding='utf-8')
    df_plan = pd.read_csv(plan_info_path, delimiter=';', encoding='utf-8')

    # 计算 CPU 时间和线程数
    cpu_time = calculate_cpu_time(df_plan).rename(columns={'cpu_time': f'{prefix}_cpu_time'})
    thread_count = calculate_thread_count(df_plan).rename(columns={'thread_count': f'{prefix}_thread_count'})

    # 合并特征
    df = pd.merge(df_query, cpu_time, on='query_id', how='left')
    df = pd.merge(df, thread_count, on='query_id', how='left')

    # 重命名执行时间和内存使用
    df = df.rename(columns={
        'execution_time': f'{prefix}_execution_time',
        'query_used_mem': f'{prefix}_memory_usage'
    })
    return df

def compare_results(baseline_query_info_path, baseline_plan_info_path,
                    new_query_info_path, new_plan_info_path, output_csv_path):
    # 计算基线和新查询的特征
    base = calculate_query_features(baseline_query_info_path, baseline_plan_info_path, 'base')
    new = calculate_query_features(new_query_info_path, new_plan_info_path, 'new')

    # 合并基线和新特征
    comp = pd.merge(base, new, on='query_id', how='outer')

    # 计算比值
    comp['execution_time_ratio'] = comp['base_execution_time'] / comp['new_execution_time']
    comp['memory_usage_ratio'] = comp['base_memory_usage'] / comp['new_memory_usage']
    comp['cpu_time_ratio'] = comp['base_cpu_time'] / comp['new_cpu_time']
    comp['thread_count_ratio'] = comp['base_thread_count'] / comp['new_thread_count']
    
    # 选择需要的列输出
    output_cols = [
        'query_id', 'base_execution_time', 'new_execution_time', 'execution_time_ratio',
        'base_memory_usage', 'new_memory_usage', 'memory_usage_ratio',
        'base_cpu_time', 'new_cpu_time', 'cpu_time_ratio',
        'base_thread_count', 'new_thread_count', 'thread_count_ratio'
    ]
    comp[output_cols].to_csv(output_csv_path, sep=';', index=False, encoding='utf-8')
    print(f'对比结果已保存至 {output_csv_path}')




# 示例 main 函数
if __name__ == '__main__':
    
    df_plans = pd.read_csv("/home/zhy/opengauss/data_file/tpch_10g_output_22/plan_info.csv", delimiter=';', encoding='utf-8')
    onnx_manager = ONNXModelManager()
    
    # 运行端到端流程，并打印和写入详细信息
    # results = end_to_end_dop_optimization_and_print(df_plans, onnx_manager)
    # end_to_end_processing(df_plans, "/home/zhy/opengauss/tools/serverless_tools/train/python/no_dop/dop_result/query_dop_grid.csv", "tru_details.json")
    # baseline_json = "auto_details.json"  # 基准 JSON 文件路径
    # method_files = {
    #     "Ours": "query_details.json",
    #     # 如有更多方法，可继续添加
    # }
    # output_csv = "comparisons.csv"
    # compare_methods_and_write_csv(baseline_json, method_files, output_csv)
    # dop_utils.select_optimal_dops_from_prediction_file("/home/zhy/opengauss/tools/serverless_tools/train/python/auto-dop/result/tpch/tpch_pre.csv", output_file="/home/zhy/opengauss/tools/serverless_tools/train/python/no_dop/dop_result/auto_dop_pre.csv")
    compare_results(
        baseline_query_info_path='/home/zhy/opengauss/data_file/tpch_10g_output_tru/query_info.csv',
        baseline_plan_info_path='/home/zhy/opengauss/data_file/tpch_10g_output_tru/plan_info.csv',
        new_query_info_path='/home/zhy/opengauss/data_file/tpch_10g_output_OLRP/query_info.csv',
        new_plan_info_path='/home/zhy/opengauss/data_file/tpch_10g_output_OLRP/plan_info.csv',
        output_csv_path='/home/zhy/opengauss/tools/serverless_tools/train/python/no_dop/dop_result/comparison.csv'
    )

