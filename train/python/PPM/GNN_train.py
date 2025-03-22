import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import numpy as np
import pandas as pd
import os
import time
import onnxruntime as ort
import PPM_model_GNN
import featurelize
from torch.utils.data import DataLoader, TensorDataset
from infos import graph_features
from torch_geometric.data import Data, DataLoader
from utils import extract_predicate_cost
from structure import jointype_encoding, operator_encoding

def load_graph_data(csv_file, query_info_file):
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

def train_and_evaluate_gnn(
    query_features_train_csv, query_info_train_csv, 
    query_features_test_csv, query_info_test_csv, 
    epochs=100, lr=0.01, batch_size=16
):
    """
    训练和评估基于 GNN 的 PCC 预测模型，包括执行时间和内存消耗。

    参数:
    - query_features_train_csv: str, 训练集的查询特征 CSV 文件路径
    - query_info_train_csv: str, 训练集的查询标签 CSV 文件路径
    - query_features_test_csv: str, 测试集的查询特征 CSV 文件路径
    - query_info_test_csv: str, 测试集的查询标签 CSV 文件路径
    - epochs: int, 训练轮数
    - lr: float, 学习率
    - batch_size: int, 训练批大小

    返回:
    - results: dict, 训练后测试集上的误差评估结果
    """

    # 读取训练数据和测试数据
    train_data = load_graph_data(query_features_train_csv, query_info_train_csv)
    test_data = load_graph_data(query_features_test_csv, query_info_test_csv)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # 初始化 GNN 模型
    input_dim = train_data[0].x.shape[1]  # 计算输入特征维度
    exec_model = PPM_model_GNN.Exec_GNNModel(input_dim, 64)
    mem_model = PPM_model_GNN.Mem_GNNModel(input_dim, 64)

    exec_optimizer = optim.Adam(exec_model.parameters(), lr=lr)
    mem_optimizer = optim.Adam(mem_model.parameters(), lr=lr)

    exec_scheduler = optim.lr_scheduler.StepLR(exec_optimizer, step_size=10, gamma=0.8)
    mem_scheduler = optim.lr_scheduler.StepLR(mem_optimizer, step_size=10, gamma=0.8)

    start_time = time.time()

    # 训练执行时间模型
    for epoch in range(epochs):
        exec_model.train()
        epoch_loss = 0.0  

        for batch in train_loader:
            exec_optimizer.zero_grad()
            
            # 取出 batch 内的数据
            x = batch.x
            edge_index = batch.edge_index
            batch_idx = batch.batch  # 用于 global_mean_pool

            # 计算预测值
            pred_params = exec_model(x, edge_index, batch_idx)
            
            # 计算损失 (y 只取执行时间部分)
            loss = PPM_model_GNN.curve_exec_loss(pred_params, batch.y[:, 0], batch.dop)

            if torch.any(torch.isnan(loss)):
                print(f"Exec Model NaN detected at epoch {epoch}. Resetting model.")
                break  

            loss.backward()
            exec_optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        exec_scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Exec Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_loss:.4f}, LR: {exec_scheduler.get_last_lr()[0]:.6f}")

    # 导出 ONNX 模型
    onnx_model_path_exec = "/home/zhy/opengauss/tools/serverless_tools/train/model/PPM_GNN/exec_GNN.onnx"
    os.makedirs(os.path.dirname(onnx_model_path_exec), exist_ok=True)
    exec_model.eval()

    dummy_x = torch.randn(10, input_dim)  # 10个节点，每个节点 input_dim 维度
    dummy_edge_index = torch.randint(0, 10, (2, 20))  # 20 条边的索引
    dummy_batch = torch.zeros(10, dtype=torch.long)  # 所有节点属于同一个 batch

    torch.onnx.export(
        exec_model, 
        (dummy_x, dummy_edge_index, dummy_batch),
        onnx_model_path_exec,
        input_names=["x", "edge_index", "batch"],
        output_names=["output"],
        dynamic_axes={"x": {0: "num_nodes"}, "edge_index": {1: "num_edges"}, "batch": {0: "num_nodes"}}
    )
    # 训练内存模型
    for epoch in range(epochs):
        mem_model.train()
        epoch_loss = 0.0  

        for batch in train_loader:
            mem_optimizer.zero_grad()
            x = batch.x
            edge_index = batch.edge_index
            batch_idx = batch.batch  # 用于 global_mean_pool

            pred_params = mem_model(x, edge_index, batch_idx)
            loss = PPM_model_GNN.curve_mem_loss(pred_params, batch.y[:, 1], batch.dop)  # 取 y 的内存消耗部分

            if torch.any(torch.isnan(loss)):
                print(f"Mem Model NaN detected at epoch {epoch}. Resetting model.")
                break  

            loss.backward()
            mem_optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        mem_scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Mem Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_loss:.4f}, LR: {mem_scheduler.get_last_lr()[0]:.6f}")

    # 导出 ONNX 模型
    onnx_model_path_mem = "/home/zhy/opengauss/tools/serverless_tools/train/model/PPM_GNN/mem_GNN.onnx"
    mem_model.eval()
    dummy_x = torch.randn(10, input_dim)  # 10个节点，每个节点 input_dim 维度
    dummy_edge_index = torch.randint(0, 10, (2, 20))  # 20 条边的索引
    dummy_batch = torch.zeros(10, dtype=torch.long)  # 所有节点属于同一个 batch

    torch.onnx.export(
        mem_model, 
        (dummy_x, dummy_edge_index, dummy_batch),  # 这里的输入要匹配 forward() 的参数
        onnx_model_path_mem,
        input_names=["x", "edge_index", "batch"],
        output_names=["output"],
        dynamic_axes={"x": {0: "num_nodes"}, "edge_index": {1: "num_edges"}, "batch": {0: "num_nodes"}}
    )
    # 评估执行时间模型
    exec_test_results = PPM_model_GNN.predict_and_evaluate_exec_curve(
        onnx_model_path_exec, test_loader, output_file="exec_qerror_gnn.csv", comparison_file='exec_pre_gnn.csv'
    )

    # 评估内存模型
    mem_test_results = PPM_model_GNN.predict_and_evaluate_mem_curve(
        onnx_model_path_mem, test_loader, output_file="mem_qerror_gnn.csv", comparison_file='mem_pre_gnn.csv'
    )

    results = {
        "execution_model_test_results": exec_test_results,
        "memory_model_test_results": mem_test_results,
    }

    return results

def evaluate(
    query_features_test_csv, query_info_test_csv, 
    suffix="NN_model"
):
    """
    读取训练集和测试集数据，训练执行时间和内存模型，并评估性能。

    Parameters:
    - query_features_train_csv: str, 训练集的查询特征 CSV 文件路径
    - query_info_train_csv: str, 训练集的查询标签 CSV 文件路径
    - query_features_test_csv: str, 测试集的查询特征 CSV 文件路径
    - query_info_test_csv: str, 测试集的查询标签 CSV 文件路径
    - epochs: int, 训练轮数
    - lr: float, 学习率
    - suffix: str, 模型保存的后缀名
    """
    

    # 读取测试数据
    test_data = load_graph_data(query_features_test_csv, query_info_test_csv)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    onnx_model_path_exec = "/home/zhy/opengauss/tools/serverless_tools/train/model/PPM_GNN/exec_GNN.onnx"
    onnx_model_path_mem = "/home/zhy/opengauss/tools/serverless_tools/train/model/PPM_GNN/mem_GNN.onnx"
    # 评估执行时间模型 (测试集)
    exec_test_results = PPM_model_GNN.predict_and_evaluate_exec_curve(
        onnx_model_path_exec, test_loader, output_file="exec_qerror.csv", comparison_file='exec_pre.csv'
    )

    # 评估内存模型 (测试集)
    mem_test_results = PPM_model_GNN.predict_and_evaluate_mem_curve(
        onnx_model_path_mem, test_loader, output_file="mem_qerror.csv", comparison_file='mem_pre.csv'
    )

    results = {
        "execution_model_test_results": exec_test_results,
        "memory_model_test_results": mem_test_results,
    }


    return results


# 示例调用
if __name__ == "__main__":
    results = train_and_evaluate_gnn(
    "/home/zhy/opengauss/data_file/tpch_10g_output_500/plan_info.csv", "/home/zhy/opengauss/data_file/tpch_10g_output_500/query_info.csv",
    "/home/zhy/opengauss/data_file/tpch_10g_output_22/plan_info.csv", "/home/zhy/opengauss/data_file/tpch_10g_output_22/query_info.csv",
    epochs=100, lr=0.05
    )
    # results = evaluate(
    #     "/home/zhy/opengauss/data_file/tpch_10g_output_22/plan_info.csv",
    #     "/home/zhy/opengauss/data_file/tpcds_10g_output/query_info.csv"
    # )
