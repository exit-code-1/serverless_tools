import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import numpy as np
import pandas as pd
import os
import csv # <--- 新增导入
import time
import onnxruntime as ort
from training.PPM import PPM_model_GNN, featurelize
from torch.utils.data import DataLoader, TensorDataset
from training.PPM.infos import graph_features
from torch_geometric.data import Data, DataLoader
from utils.feature_engineering import extract_predicate_cost
from config.structure import jointype_encoding, operator_encoding

def load_graph_data(csv_file, query_info_file, use_estimates=False): # <-- 增加开关
    """
    读取 CSV 文件，将查询执行计划转换为 PyG 的图数据格式。

    参数：
    - csv_file: str, 包含查询计划算子信息的 CSV 文件路径
    - query_info_file: str, 包含查询执行信息的 CSV 文件路径

    返回：
    - data_list: list, PyG 的 Data 对象列表，每个 Data 代表一个查询的执行计划
    """
    # 读取查询计划算子信息
    df = pd.read_csv(csv_file, delimiter=';')
     # 过滤掉不在 operator_encoding 里的行
    df = df[df['operator_type'].isin(operator_encoding)]
    
    df['predicate_cost'] = df['filter'].apply(
        lambda x: extract_predicate_cost(x) if pd.notnull(x) and x != '' else 0
    )
    df['operator_type'] = df['operator_type'].map(operator_encoding).astype(int)
    df['jointype'] = df['jointype'].map(jointype_encoding).astype(int)
    query_info_df = pd.read_csv(query_info_file, delimiter=';')
    
    # --- 如果是模拟模式，替换行数 ---
    if use_estimates:
        print("GNN模拟模式：正在使用 'estimate_rows' 作为特征...")
        # 直接用 estimate_rows 列覆盖 actual_rows 列
        df['actual_rows'] = df['estimate_rows']
        # 注意：GNN的特征提取没有显式的父子传播步骤，
        # 所以这里无法像NN那样精确模拟输入的误差。
        # 这是一个简化的模拟，主要反映节点自身输出的基数估计误差。

    # 按 query_id 和 query_dop 进行分组，每个组代表一个查询的 DAG
    grouped = df.groupby(['query_id', 'dop'])
    
    # 存储所有查询的图数据
    data_list = []
    
    for (query_id, query_dop), group in grouped:
        # 取出当前查询的所有算子特征
        node_features = group[graph_features].copy()
        node_features['is_parallel'] = (group['dop'] > 1).astype(int)
        node_features = torch.tensor(node_features.values, dtype=torch.float32)
        
        # 取出 plan_id 和 child_plan
        plan_ids = group['plan_id'].values
        child_plans = group['child_plan'].fillna('').astype(str)  # 可能是以逗号分隔的字符串
        
        # 建立 `edge_index`
        edge_list = []
        plan_id_to_index = {pid: idx for idx, pid in enumerate(plan_ids)}
        
        for idx, children in enumerate(child_plans):
            if pd.isna(children) or children == '':  # 可能为空
                continue
            child_ids = list(map(int, children.split(',')))  # 解析子节点 ID
            for child_id in child_ids:
                if child_id in plan_id_to_index:
                    edge_list.append([plan_id_to_index[child_id], idx])  # 方向：子 -> 父
        
        # 转换为 PyG 需要的 `edge_index` 形式
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # 获取 `query_info_file` 中的执行时间和内存消耗
        query_info = query_info_df[(query_info_df['query_id'] == query_id) & (query_info_df['dop'] == query_dop)]
        if query_info.empty:
            continue  # 若找不到对应的 `query_id` 和 `dop`，则跳过
        
        exec_time = query_info['execution_time'].values[0]
        mem_usage = query_info['query_used_mem'].values[0]
        y = torch.tensor([[exec_time, mem_usage]], dtype=torch.float32)  # 变成二维向量
        
        # 生成 PyG 的 Data 对象
        dop_tensor = torch.tensor([query_dop], dtype=torch.float32)  # 转换为 Tensor
        data = Data(x=node_features, edge_index=edge_index, y=y, dop=dop_tensor)
        data_list.append(data)
    
    return data_list

def save_results_to_csv(results, suffix):
    """
    保存训练和测试结果到 CSV 文件。

    Parameters:
    - results: dict, 评估结果
    - suffix: str, 文件后缀
    """
    result_list = []
    
    for key, value in results.items():
        dataset_type = "train" if "train" in key else "test"
        model_type = "exec" if "execution" in key else "mem"
        
        # 结果可能是 dict（包含多项指标）
        if isinstance(value, dict):
            for metric, metric_value in value.items():
                result_list.append([dataset_type, model_type, metric, metric_value])
        else:
            result_list.append([dataset_type, model_type, "unknown_metric", value])
    
    df = pd.DataFrame(result_list, columns=["Dataset", "Model", "Metric", "Value"])
    df.to_csv(f"results_{suffix}.csv", index=False)
    print(f"结果已保存到 results_{suffix}.csv")

def train_gnn_models(
    query_features_train_csv, query_info_train_csv,
    execution_onnx_path, memory_onnx_path,
    epochs=100, lr=0.01, batch_size=16,
    use_estimates=False,
    cost_log_file=None # <-- 1. 增加 cost_log_file 参数
):
    """
    训练 GNN 模型并导出 ONNX（执行时间模型和内存模型）

    返回:
    - exec_model_path: str, 执行时间模型 ONNX 路径
    - mem_model_path: str, 内存模型 ONNX 路径
    """
    # 传递开关
    train_data = load_graph_data(
        query_features_train_csv, query_info_train_csv, use_estimates=use_estimates
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    input_dim = train_data[0].x.shape[1]
    exec_model = PPM_model_GNN.Exec_GNNModel(input_dim, 64)
    mem_model = PPM_model_GNN.Mem_GNNModel(input_dim, 64)

    exec_optimizer = optim.Adam(exec_model.parameters(), lr=lr)
    mem_optimizer = optim.Adam(mem_model.parameters(), lr=lr)

    exec_scheduler = optim.lr_scheduler.StepLR(exec_optimizer, step_size=10, gamma=0.8)
    mem_scheduler = optim.lr_scheduler.StepLR(mem_optimizer, step_size=10, gamma=0.8)
    start_time = time.time()
    # ---- Train Exec Model ----
    for epoch in range(epochs):
        exec_model.train()
        total_loss = 0
        for batch in train_loader:
            exec_optimizer.zero_grad()
            pred_params = exec_model(batch.x, batch.edge_index, batch.batch)
            loss = PPM_model_GNN.curve_exec_loss(pred_params, batch.y[:, 0], batch.dop)
            if torch.any(torch.isnan(loss)):
                print(f"Exec Model NaN at epoch {epoch}")
                break
            loss.backward()
            exec_optimizer.step()
            total_loss += loss.item()
        exec_scheduler.step()

    # ---- Train Mem Model ----
    for epoch in range(epochs):
        mem_model.train()
        total_loss = 0
        for batch in train_loader:
            mem_optimizer.zero_grad()
            pred_params = mem_model(batch.x, batch.edge_index, batch.batch)
            loss = PPM_model_GNN.curve_mem_loss(pred_params, batch.y[:, 1], batch.dop)
            if torch.any(torch.isnan(loss)):
                print(f"Mem Model NaN at epoch {epoch}")
                break
            loss.backward()
            mem_optimizer.step()
            total_loss += loss.item()
        mem_scheduler.step()
    end_time = time.time()
    dummy_x = torch.randn(10, input_dim)
    dummy_edge_index = torch.randint(0, 10, (2, 20))
    dummy_batch = torch.zeros(10, dtype=torch.long)
    duration = end_time - start_time
    # --- 2. 新增：将训练时间写入日志文件 ---
    if cost_log_file:
        with open(cost_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['model_name', 'training_time_seconds'])
            # 记录的是两个模型的总时间
            writer.writerow(['PPM-GNN_total', duration])
        print(f"训练开销已记录到: {cost_log_file}")
    # --- 日志记录结束 ---
    exec_model.eval()
    torch.onnx.export(exec_model, (dummy_x, dummy_edge_index, dummy_batch), execution_onnx_path,
        input_names=["x", "edge_index", "batch"],
        output_names=["output"],
        dynamic_axes={"x": {0: "num_nodes"}, "edge_index": {1: "num_edges"}, "batch": {0: "num_nodes"}})

    mem_model.eval()
    torch.onnx.export(mem_model, (dummy_x, dummy_edge_index, dummy_batch), memory_onnx_path,
        input_names=["x", "edge_index", "batch"],
        output_names=["output"],
        dynamic_axes={"x": {0: "num_nodes"}, "edge_index": {1: "num_edges"}, "batch": {0: "num_nodes"}})
    print(f"training time: {duration:.2f}s")




def evaluate_gnn_models(
    query_features_test_csv, query_info_test_csv,
    execution_onnx_path, memory_onnx_path,
    output_file,
    use_estimates=False # <-- 增加开关
):
    """
    使用已训练的 ONNX 模型进行评估。

    Parameters:
    - query_features_test_csv: str, 测试集特征路径
    - query_info_test_csv: str, 测试集标签路径
    - execution_onnx_path: str, 执行时间模型 ONNX 文件路径
    - memory_onnx_path: str, 内存模型 ONNX 文件路径
    """
     # 读取测试数据
    # 传递开关
    test_data = load_graph_data(
        query_features_test_csv, query_info_test_csv, use_estimates=use_estimates
    )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    PPM_model_GNN.predict_and_evaluate_curve(
        exec_model_path=execution_onnx_path,
        mem_model_path=memory_onnx_path,
        test_loader = test_loader,
        output_file=output_file
    )

