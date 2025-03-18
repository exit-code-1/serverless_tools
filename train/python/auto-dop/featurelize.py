import os
import sys
import time
import pandas as pd
import optuna
import numpy as np
from collections import defaultdict

import xgboost as xgb
import auto_dop_model
sys.path.append(os.path.abspath("/home/zhy/opengauss/tools/serverless_tools/train/python"))
from utils import extract_predicate_cost
from structure import jointype_encoding
import onnx
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
import onnxruntime as ort

def build_query_plan(group_df):
    """根据group_df构建查询计划树"""
    nodes = {}
    root_nodes = []
    
    # 先创建所有 PlanNode 实例
    for _, row in group_df.iterrows():
        operator = row['operator_type']
        plan_id = row['plan_id']
        features = {feat: row[feat] for feat in auto_dop_model.operator_features.get(operator, []) if feat in row}
        features['is_parallel'] = 1 if row['dop'] > 1 else 0
        node = auto_dop_model.PlanNode(plan_id, operator, features)
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

def process_query_features(csv_file):
    """读取CSV并转换为每个查询的固定长度特征向量"""
    df = pd.read_csv(csv_file, delimiter=';')
    df['predicate_cost'] = df['filter'].apply(
        lambda x: extract_predicate_cost(x) if pd.notnull(x) and x != '' else 0
    )
    
    df['jointype'] = df['jointype'].map(jointype_encoding).astype(int)
    query_features = {}
    
    for (query_id, query_dop), group_df in df.groupby(['query_id', 'query_dop']):
        nodes, root_nodes = build_query_plan(group_df)
        # 计算所有节点的深度
        for root in root_nodes:
            root.compute_depth()
        max_depth = max(node.depth for node in nodes.values())
        for node in nodes.values():
            node.compute_weight(max_depth)
            
        # 生成特征向量
        feature_vector = np.sum([node.extract_feature_vector() for node in nodes.values()], axis=0)
        
        # 扩展特征向量，添加 query_dop 作为最后一列
        feature_vector = np.append(feature_vector, query_dop)
        
        query_features[(query_id, query_dop)] = feature_vector
    
    return query_features

def objective(trial, X, y):
    """Optuna 目标函数，用于超参数调优"""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "random_state": 32
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    
    # 计算评估指标（比如 RMSE）
    y_pred = model.predict(X)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    return rmse

def train_and_save_xgboost_onnx(feature_csv, true_val_csv, execution_onnx_path, memory_onnx_path, n_trials=20):
    """训练两个独立的 XGBoost 模型（分别预测执行时间和内存占用）并转换为 ONNX"""
    query_features = process_query_features(feature_csv)
    
    # 读取执行时间和内存数据
    execution_df = pd.read_csv(true_val_csv, delimiter=';').set_index(['query_id', 'dop'])
    
    # 过滤特征集合，确保 y 和 X 对齐
    valid_queries = list(query_features.keys() & execution_df.index)
    X = np.array([query_features[q] for q in valid_queries])
    
    # 分别提取 y 值
    y_time = execution_df.loc[valid_queries]["execution_time"].values
    y_mem = execution_df.loc[valid_queries]["query_used_mem"].values
    # 训练执行时间模型
    study_time = optuna.create_study(direction="minimize")
    study_time.optimize(lambda trial: objective(trial, X, y_time), n_trials=n_trials)
    best_params_time = study_time.best_params
    print(f"✅ 最优执行时间模型超参数: {best_params_time}")
    execution_time_model = xgb.XGBRegressor(**best_params_time, objective='reg:squarederror')
    execution_time_model.fit(X, y_time)

    # 训练内存占用模型
    study_memory = optuna.create_study(direction="minimize")
    study_memory.optimize(lambda trial: objective(trial, X, y_mem), n_trials=n_trials)
    best_params_memory = study_memory.best_params
    print(f"✅ 最优内存占用模型超参数: {best_params_memory}")
    memory_usage_model = xgb.XGBRegressor(**best_params_memory, objective='reg:squarederror')
    memory_usage_model.fit(X, y_mem)

    # 转换 XGBoost 模型为 ONNX
    initial_types = [("float_input", FloatTensorType([None, X.shape[1]]))]
    execution_onnx_model = convert_xgboost(execution_time_model, initial_types=initial_types)
    memory_onnx_model = convert_xgboost(memory_usage_model, initial_types=initial_types)

    # 保存 ONNX 模型
    with open(execution_onnx_path, "wb") as f:
        f.write(execution_onnx_model.SerializeToString())
    with open(memory_onnx_path, "wb") as f:
        f.write(memory_onnx_model.SerializeToString())

    print(f"✅ 执行时间模型已保存到 {execution_onnx_path}")
    print(f"✅ 内存占用模型已保存到 {memory_onnx_path}")

    return execution_onnx_path, memory_onnx_path


def test_onnx_xgboost(execution_onnx_path, memory_onnx_path, feature_csv, true_val_csv, output_file):
    """分别加载 ONNX 模型，进行预测，并计算 Q-Error"""
    query_features = process_query_features(feature_csv)
    
    # 读取执行时间和内存数据
    execution_df = pd.read_csv(true_val_csv, delimiter=';').set_index(['query_id', 'dop'])
    
    # 确保特征和标签对齐
    valid_queries = sorted(set(query_features.keys()) & set(execution_df.index))
    X = np.array([query_features[q] for q in valid_queries])
    y_true_time = execution_df.loc[valid_queries]["execution_time"].values
    y_true_mem = execution_df.loc[valid_queries]["query_used_mem"].values

    # 加载 ONNX 模型
    session_time = ort.InferenceSession(execution_onnx_path, providers=["CPUExecutionProvider"])
    session_mem = ort.InferenceSession(memory_onnx_path, providers=["CPUExecutionProvider"])

    input_name_time = session_time.get_inputs()[0].name
    input_name_mem = session_mem.get_inputs()[0].name

    # 预测
    y_pred_time = np.vstack([session_time.run(None, {input_name_time: X[i:i+1].astype(np.float32)})[0] for i in range(len(valid_queries))])
    y_pred_mem = np.vstack([session_mem.run(None, {input_name_mem: X[i:i+1].astype(np.float32)})[0] for i in range(len(valid_queries))])

    # 计算 Q-Error
    q_error_time = np.maximum(y_true_time / y_pred_time[:, 0], y_pred_time[:, 0] / y_true_time) - 1
    q_error_mem = np.maximum(y_true_mem / y_pred_mem[:, 0], y_pred_mem[:, 0] / y_true_mem) - 1

    # 结果 DataFrame
    results_df = pd.DataFrame({
        "query_id": [q[0] for q in valid_queries],
        "dop": [q[1] for q in valid_queries],
        "predicted_time": y_pred_time[:, 0],
        "actual_time": y_true_time,
        "predicted_memory": y_pred_mem[:, 0],
        "actual_memory": y_true_mem,
        "q_error_time": q_error_time,
        "q_error_memory": q_error_mem
    })

    results_df.to_csv(output_file, index=False, sep=";")
    print(f"✅ 预测结果已保存到 {output_file}")

    return results_df

def categorize_into_quartiles(values):
    """根据四分位数对数据进行分桶，并返回具体分界值"""
    quartiles = np.percentile(values, [25, 50, 75])  # 计算 Q1, Q2, Q3
    bins = []
    for v in values:
        if v <= quartiles[0]:
            bins.append(f"Q1 (≤{quartiles[0]:.2f})")
        elif quartiles[0] < v <= quartiles[1]:
            bins.append(f"Q2 ({quartiles[0]:.2f}-{quartiles[1]:.2f})")
        elif quartiles[1] < v <= quartiles[2]:
            bins.append(f"Q3 ({quartiles[1]:.2f}-{quartiles[2]:.2f})")
        else:
            bins.append(f"Q4 (> {quartiles[2]:.2f})")
    return bins, quartiles  # 返回桶和具体分界值

def compute_qerror_by_bins(results_df, output_file):
    """计算不同时间区间和内存区间的 Q-Error 均值，并保存到 CSV 文件"""
    results_df = results_df[results_df["dop"] == 8]  # 只筛选 dop=8

    # 计算 Execution Time Q-Error（均等分桶）
    results_df["time_bin"], time_quartiles = categorize_into_quartiles(results_df["actual_time"] / 1000)
    time_qerror_stats = results_df.groupby("time_bin")["q_error_time"].mean().to_dict()

    # 计算 Memory Q-Error（均等分桶）
    results_df["memory_bin"], mem_quartiles = categorize_into_quartiles(results_df["actual_memory"])
    memory_qerror_stats = results_df.groupby("memory_bin")["q_error_memory"].mean().to_dict()

    # 组织数据格式
    time_qerror_df = pd.DataFrame(
    list(time_qerror_stats.items()),
    columns=[
        "Time Bin (Q1: ≤{:.2f}, Q2: {:.2f}-{:.2f}, Q3: {:.2f}-{:.2f}, Q4: >{:.2f})".format(
            time_quartiles[0], time_quartiles[0], time_quartiles[1], time_quartiles[1], time_quartiles[2], time_quartiles[2]
        ),
        "Execution Time Q-error",
    ]
    )

    memory_qerror_df = pd.DataFrame(
    list(memory_qerror_stats.items()),
    columns=[
        "Memory Bin (Q1: ≤{:.2f}, Q2: {:.2f}-{:.2f}, Q3: {:.2f}-{:.2f}, Q4: >{:.2f})".format(
            mem_quartiles[0], mem_quartiles[0], mem_quartiles[1], mem_quartiles[1], mem_quartiles[2], mem_quartiles[2]
        ),
        "Memory Q-error",
    ]
)

    # 写入 CSV 文件
    with open(output_file, "w") as f:
        f.write("Time Bin,Execution Time Q-error\n")
        time_qerror_df.to_csv(f, index=False, header=True)

        f.write("\nMemory Bin,Memory Q-error\n")
        memory_qerror_df.to_csv(f, index=False, header=True)

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
    # execution_onnx_path="../../model/auto-dop/execution_time_model.onnx",
    # memory_onnx_path="../../model/auto-dop/memory_usage_model.onnx",
    # n_trials=30
    # )
    execution_onnx_path = "/home/zhy/opengauss/tools/serverless_tools/train/model/auto-dop/execution_time_model.onnx"
    memory_onnx_path = "/home/zhy/opengauss/tools/serverless_tools/train/model/auto-dop/memory_usage_model.onnx"
    results_df = test_onnx_xgboost(execution_onnx_path, memory_onnx_path, test_feature_csv, test_execution_csv, "test_predictions.csv")
    compute_qerror_by_bins(results_df, "qerror_stats.csv")