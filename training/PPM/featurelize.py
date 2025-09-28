import os
import sys
import time
import pandas as pd
import optuna
import numpy as np
from collections import defaultdict

import xgboost as xgb
from training.PPM import infos
sys.path.append(os.path.abspath("/home/zhy/opengauss/tools/serverless_tools/train/python"))
from utils.feature_engineering import extract_predicate_cost
from config.structure import jointype_encoding
import onnx
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
import onnxruntime as ort

def build_query_plan(group_df, use_estimates=False): # <-- 增加 use_estimates 开关
    """根据group_df构建查询计划树"""
    nodes = {}
    root_nodes = []
    
    # 先创建所有 PlanNode 实例
    for _, row in group_df.iterrows():
        operator = row['operator_type']
        plan_id = row['plan_id']
        features = {feat: row[feat] for feat in infos.operator_features.get(operator, []) if feat in row}
        features['is_parallel'] = 1 if row['dop'] > 1 else 0
        # 将开关传递给 PlanNode
        node = infos.PlanNode(plan_id, operator, features, use_estimates=use_estimates)
        nodes[row['plan_id']] = node
    
    # 关联父子关系
    for _, row in group_df.iterrows():
        node = nodes[row['plan_id']]
        if pd.notna(row['child_plan']):
            child_ids = [int(pid) for pid in str(row['child_plan']).split(',')]
            for cid in child_ids:
                if cid in nodes:
                    nodes[cid].parent = node
                    node.children.append(nodes[cid])
    
    # 识别根节点
    for node in nodes.values():
        if node.parent is None:
            root_nodes.append(node)
    
    return nodes, root_nodes

# --- 新增：自底向上更新函数 ---
def update_tree_estimated_inputs_recursive(node):
    for child in node.children:
        update_tree_estimated_inputs_recursive(child)
    node.update_estimated_inputs()

def process_query_features(csv_file, use_estimates=False): # <-- 增加 use_estimates 开关
    """读取CSV并转换为每个查询的固定长度特征向量"""
    df = pd.read_csv(csv_file, delimiter=';')
    df['predicate_cost'] = df['filter'].apply(
        lambda x: extract_predicate_cost(x) if pd.notnull(x) and x != '' else 0
    )
    
    df['jointype'] = df['jointype'].map(jointype_encoding).astype(int)
    query_features = {}
    
    for (query_id, query_dop), group_df in df.groupby(['query_id', 'query_dop']):
        # 传递开关
        nodes, root_nodes = build_query_plan(group_df, use_estimates=use_estimates)
        
        # 如果是模拟模式，执行自底向上更新
        if use_estimates:
            for root in root_nodes:
                update_tree_estimated_inputs_recursive(root)
        
        # 计算所有节点的深度
        for root in root_nodes:
            root.compute_depth()
        max_depth = max(node.depth for node in nodes.values()) if nodes else 0
        
        # 计算权重（现在会根据模式选择行数）
        for node in nodes.values():
            node.compute_weight(max_depth)
            
        # 生成特征向量（现在会根据模式提取特征）
        feature_vector = np.sum([node.extract_feature_vector() for node in nodes.values()], axis=0)
        
        query_features[(query_id, query_dop)] = feature_vector
    
    return query_features



def categorize_time_bins(actual_times):
    """根据实际执行时间分类"""
    bins = []
    for t in actual_times:
        if t < 1:
            bins.append("<1s")
        elif 1 <= t < 5:
            bins.append("1-5s")
        elif 5 <= t < 30:
            bins.append("5-30s")
        else:
            bins.append(">30s")
    return bins

def categorize_memory_bins(actual_mem):
    """根据实际内存使用量分类"""
    bins = []
    for m in actual_mem:
        if m < 50 * 1024 :  # <50MB
            bins.append("<50M")
        elif 50 * 1024 <= m < 300 * 1024:  # 50M-300M
            bins.append("50M-300M")
        elif 300 * 1024 <= m < 1 * 1024 * 1024:  # 300M-1G
            bins.append("300M-1G")
        else:  # >1G
            bins.append(">1G")
    return bins

def compute_qerror_by_bins(results_df, output_file):
    """计算不同时间区间和内存区间的 Q-Error 均值，并保存到 CSV 文件"""
    results_df = results_df[results_df["dop"] == 8]  # 只筛选 dop=8

    # 计算 Execution Time Q-Error
    results_df["time_bin"] = categorize_time_bins(results_df["actual_time"] / 1000)
    time_qerror_stats = results_df.groupby("time_bin")["q_error_time"].mean().to_dict()

    # 计算 Memory Q-Error
    results_df["memory_bin"] = categorize_memory_bins(results_df["actual_memory"])
    memory_qerror_stats = results_df.groupby("memory_bin")["q_error_memory"].mean().to_dict()

    # 组织数据格式
    time_qerror_df = pd.DataFrame(list(time_qerror_stats.items()), columns=["Time Bin", "Execution Time Q-error"])
    memory_qerror_df = pd.DataFrame(list(memory_qerror_stats.items()), columns=["Memory Bin", "Memory Q-error"])

    # 写入 CSV 文件
    with open(output_file, "w") as f:
        f.write("Time Bin,Execution Time Q-error\n")
        time_qerror_df.to_csv(f, index=False, header=False)

        f.write("\nMemory Bin,Memory Q-error\n")
        memory_qerror_df.to_csv(f, index=False, header=False)

    print(f"✅ 统计结果已保存到 {output_file}")

    return output_file


if __name__ == "__main__":
    feature_csv = "/home/zhy/opengauss/data_file/tpch_10g_output_500/plan_info.csv"  # 替换为实际的特征 CSV 文件
    true_val_csv = "/home/zhy/opengauss/data_file/tpch_10g_output_500/query_info.csv"  # 替换为实际的执行时间 CSV 文件
    test_feature_csv = "/home/zhy/opengauss/data_file/tpch_10g_output_22/plan_info.csv"  # 测试集查询计划文件
    test_execution_csv = "/home/zhy/opengauss/data_file/tpch_10g_output_22/query_info.csv"  # 测试集真实执行时间文件
    # execution_onnx_path, memory_onnx_path = train_and_save_xgboost_onnx(
    # feature_csv=feature_csv,
    # true_val_csv=true_val_csv,
    # execution_onnx_path="execution_time_model.onnx",
    # memory_onnx_path="memory_usage_model.onnx",
    # n_trials=30
    # )