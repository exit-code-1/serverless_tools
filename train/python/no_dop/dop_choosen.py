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
from definition import PlanNode, ONNXModelManager
    


# 读取包含 query_id 和 split 信息的文件
split_info_df = pd.read_csv('/home/zhy/opengauss/tools/serverless_tools/train/python/dop/tmp_result/query_split.csv')

# 只保留前 200 行的数据
split_info = split_info_df[['query_id', 'split']]

# 获取原始的 split 信息的 query_id 和 split 列
test_queries = split_info[split_info['split'] == 'test']['query_id']

# 将扩展后的测试查询的 query_id 转换为 DataFrame
test_queries_df = pd.DataFrame(test_queries, columns=['query_id'])

# 读取执行计划数据
df_query_info = pd.read_csv('/home/zhy/opengauss/data_file/tpch_10g_output_22/query_info.csv', delimiter=';', encoding='utf-8')
df_plans = pd.read_csv('/home/zhy/opengauss/data_file/tpch_10g_output_22/plan_info.csv', delimiter=';', encoding='utf-8')

# 只取 dop=8 的查询
df_base = df_plans[df_plans['query_dop'] == 8].copy()
query_groups = df_base.groupby(['query_id', 'query_dop'])

query_trees = {}
onnx_manager = ONNXModelManager()

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


df_other = df_plans[(df_plans['query_dop'] != 8) & (df_plans['query_dop'] != 1)].copy()

dop_plan_nodes = {}

for _, row in df_other.iterrows():
    key = (row['query_id'], row['query_dop'])
    if key not in dop_plan_nodes:
        dop_plan_nodes[key] = []
    
    dop_plan_nodes[key].append(PlanNode(row, onnx_manager))

for query_id in set(key[0] for key in dop_plan_nodes.keys()):
    base_nodes = query_trees.get((query_id, 8), [])  # 获取 dop=8 版本的 plan_nodes
    used_plan_nodes = set()  # 记录已经匹配的 plan_node，防止重复匹配多个 base_node

    for (qid, query_dop), plan_nodes in dop_plan_nodes.items():
        if qid != query_id:
            continue  # 只处理当前 query_id 的所有 dop 变体

        for plan_node in plan_nodes:
            if not plan_node.is_parallel:
                continue  # 非并行算子直接跳过匹配

            for base_node in base_nodes:
                if plan_node in used_plan_nodes:
                    break  # 这个 plan_node 已匹配过，跳过
                
                if plan_node.operator_type != base_node.operator_type:
                    continue  # 操作符不同，跳过

                # **情况 1+2：plan_id 相同 且 计划数匹配**
                if plan_node.plan_id == base_node.plan_id and len(base_nodes) == len(plan_nodes):
                    dop = plan_node.dop
                    dop = max(plan_node.downdop, dop)
                    dop = max(plan_node.updop, dop)
                    base_node.update_real_exec_time(dop, plan_node.execution_time)
                    used_plan_nodes.add(plan_node)
                    break  # 只匹配一个
                
                # **情况 3：使用 'l_input_rows', 'actual_rows', 'width' 进行匹配**
                key_features = ['l_input_rows', 'actual_rows', 'width']
                if all(plan_node.exec_feature_data[feat] == base_node.exec_feature_data[feat] for feat in key_features):
                    dop = plan_node.dop
                    dop = max(plan_node.downdop, dop)
                    dop = max(plan_node.updop, dop)
                    base_node.update_real_exec_time(dop, plan_node.execution_time)
                    used_plan_nodes.add(plan_node)
                    break  # 只匹配一个

    # **遍历 base_nodes 计算 dop 预测**（遍历完所有 dop 版本的 plan_nodes 后再计算）
    for base_node in base_nodes:
        base_node.compute_parallel_dop_predictions()
    dop_utils.update_thread_block_dop_candidates(base_nodes)

print("done")
base_nodes = query_trees.get((1, 8), [])
# 假设 base_nodes 是一个列表，包含所有 dop=8 构建的 PlanNode 树（例如 query_trees[(query_id, 8)] 的所有节点树）。
dop_utils.update_thread_block_dop_candidates(base_nodes)



