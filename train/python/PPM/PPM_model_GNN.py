import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool
import onnxruntime as ort
import time
import os
import numpy as np
import torch.onnx
import featurelize

class AttentionModule(nn.Module):
    def __init__(self, in_features):
        super(AttentionModule, self).__init__()
        self.weight_matrix = nn.Parameter(torch.Tensor(in_features, in_features))
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation

class Exec_GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, min_a=-2, max_a=2):
        super(Exec_GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.attention = AttentionModule(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(32, 2)
        )
        self.min_a = min_a
        self.max_a = max_a

    def forward(self, x, edge_index, batch):
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        # 按 batch 进行全局聚合
        graph_embedding = global_mean_pool(x, batch)  # (batch_size, hidden_dim)

        pred_params = self.fc(graph_embedding)  # (batch_size, 2)

        # a 经过 Sigmoid 变换到 [min_a, max_a]
        a, b = pred_params[:, 0], pred_params[:, 1]
        a = torch.sigmoid(a) * (self.max_a - self.min_a) + self.min_a

        return torch.stack([a, b], dim=1)  # (batch_size, 2)
    
class Mem_GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, min_a=-2, max_a=2):
        super(Mem_GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.attention = AttentionModule(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(32, 3)
        )
        self.min_a = min_a
        self.max_a = max_a

    def forward(self, x, edge_index, batch):
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        # 按 batch 进行全局聚合
        graph_embedding = global_mean_pool(x, batch)  # (batch_size, hidden_dim)

        pred_params = self.fc(graph_embedding)  # (batch_size, 2)

        # a 经过 Sigmoid 变换到 [min_a, max_a]
        a, b, c = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2]
        a = torch.sigmoid(a) * (self.max_a - self.min_a) + self.min_a

        return torch.stack([a, b, c], dim=1)  # (batch_size, 2)   
    
def curve_exec_loss(pred_params, y, dop,  epsilon=1e-2, alpha=0.5):
    """
    计算 GNN 预测的 PCC 参数 (a, b) 所对应的执行时间，并计算损失。
    
    参数：
    - pred_params: 模型输出，形状 (num_nodes, 2)，包含 a, b
    - data: PyG `Data` 对象，包含 `dop` (并行度) 和 `y` (真实执行时间)
    - epsilon: 避免数值问题的最小值
    - alpha: 损失加权参数
    - log_file: 记录调试信息的日志文件
    
    返回：
    - loss: 计算得到的损失
    """

    # 解析 GNN 输出
    a, b = pred_params[:, 0], pred_params[:, 1]

    # 计算预测时间
    pred_time = b * (dop ** a)

    # 计算误差项
    abs_error = torch.abs(pred_time - y)
    log_error = torch.log(abs_error + 1)

    # 相对误差项，避免 0 除法
    pred_time = torch.clamp(pred_time, min=epsilon)
    relative_error = torch.log(torch.max(pred_time / y, y / pred_time))

    # 计算最终损失
    loss = torch.mean(alpha * log_error + (1 - alpha) * relative_error)

    return loss

def curve_mem_loss(pred_params, y, dop,  epsilon=1e-2, alpha=0.5):
    """
    计算 GNN 预测的 PCC 参数 (a, b) 所对应的执行时间，并计算损失。
    
    参数：
    - pred_params: 模型输出，形状 (num_nodes, 2)，包含 a, b
    - data: PyG `Data` 对象，包含 `dop` (并行度) 和 `y` (真实执行时间)
    - epsilon: 避免数值问题的最小值
    - alpha: 损失加权参数
    - log_file: 记录调试信息的日志文件
    
    返回：
    - loss: 计算得到的损失
    """

    # 解析 GNN 输出
    a, b, c = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2]

    # 计算预测时间
    pred_time = b * (dop ** a) + c

    # 计算误差项
    abs_error = torch.abs(pred_time - y)
    log_error = torch.log(abs_error + 1)

    # 相对误差项，避免 0 除法
    pred_time = torch.clamp(pred_time, min=epsilon)
    relative_error = torch.log(torch.max(pred_time / y, y / pred_time))

    # 计算最终损失
    loss = torch.mean(alpha * log_error + (1 - alpha) * relative_error)

    return loss


def predict_and_evaluate_exec_curve(model_path, test_loader, output_file, comparison_file, epsilon=1e-6):
    """
    使用 GNN ONNX 模型进行推理，并按时间 bin 统计 Q-error。

    参数:
    - model_path: str, ONNX 模型路径
    - test_loader: DataLoader, PyG 测试数据加载器
    - output_file: str, bin 级别的误差统计文件
    - comparison_file: str, 真实 vs. 预测对比文件
    - epsilon: float, 避免除零错误

    返回:
    - output_file: str, 生成的 bin 级别误差 CSV 文件路径
    - comparison_file: str, 真实 vs. 预测对比 CSV 文件路径
    """
    session = ort.InferenceSession(model_path)
    results = []
    start_time = time.time()

    for data in test_loader:
        # ✅ ONNX 不能直接用 PyG Data，必须拆解
        x_numpy = data.x.numpy().astype(np.float32)  # 节点特征
        edge_index_numpy = data.edge_index.numpy().astype(np.int64)  # 边索引
        dop_numpy = data.dop.numpy().astype(np.float32)  # 并行度 dop
        batch_idx_numpy = data.batch.numpy().astype(np.int64)  # ✅ 转换为 NumPy 数组

        # ✅ ONNX 推理输入必须匹配 `torch.onnx.export()` 里的 `input_names`
        ort_inputs = {
            "x": x_numpy,
            "edge_index": edge_index_numpy,
            "batch": batch_idx_numpy  # ✅ 确保 batch 是 NumPy 数组
        }

        # 运行 ONNX 推理
        ort_outs = session.run(["output"], ort_inputs)[0]
        a, b = ort_outs[:, 0], ort_outs[:, 1]

        # ✅ 计算预测执行时间，避免 `data.dop` 维度问题
        dop_numpy = dop_numpy.reshape(-1)  # 保证是 1D 向量
        predictions = np.maximum(b * (dop_numpy ** a), epsilon)

        results.append((data.y[:, 0].numpy().reshape(-1), predictions))  # `data.y` 也是 1D

        
    onnx_time = time.time() - start_time  # 记录 ONNX 预测时间

    # 计算 Q-error 统计
    actual_times = np.concatenate([r[0] for r in results])
    predicted_times = np.concatenate([r[1] for r in results])
    q_errors = np.maximum(actual_times / predicted_times, predicted_times / actual_times) - 1

    # 构造 DataFrame
    results_df = pd.DataFrame({
        "actual_time": actual_times,
        "predicted_time": predicted_times,
        "q_error_time": q_errors,
    })

    # ✅ 计算时间分 bin
    results_df["time_bin"] = featurelize.categorize_time_bins(results_df["actual_time"] / 1000)
    
    # 按 bin 统计 Q-error
    time_qerror_df = results_df.groupby("time_bin")["q_error_time"].mean().reset_index()
    time_qerror_df.columns = ["Time Bin", "Execution Time Q-error"]

    # 保存统计数据
    with open(output_file, "w") as f:
        f.write("Time Bin,Execution Time Q-error\n")
        time_qerror_df.to_csv(f, index=False, header=False)
        f.write(f"\nONNX Prediction Time,{onnx_time:.6f}\n")
    
    # 保存 真实 vs 预测 对比数据
    comparison_df = results_df[["actual_time", "predicted_time", "q_error_time"]]
    comparison_df.to_csv(comparison_file, index=False)
    
    print(f"✅ 统计结果已保存到 {output_file}")
    print(f"✅ 真实 vs. 预测对比已保存到 {comparison_file}")
    
    return output_file, comparison_file




def predict_and_evaluate_mem_curve(
    model_path, test_loader, output_file, comparison_file, epsilon=1e-2
):
    """
    使用 ONNX 模型进行推理，并按内存 bin 进行评估，同时保存真实值 vs. 预测值。

    Parameters:
    - model_path: str, ONNX 模型路径
    - test_loader: DataLoader, 测试数据集加载器
    - output_file: str, bin 级别的误差统计文件
    - comparison_file: str, 真实 vs 预测对比文件
    - epsilon: float, 避免除零误差

    Returns:
    - output_file: str, 生成的 bin 级别误差 CSV 文件路径
    - comparison_file: str, 真实 vs. 预测对比 CSV 文件路径
    """
    session = ort.InferenceSession(model_path)
    
    results = []
    start_time = time.time()
    
    for data in test_loader:
        # ✅ ONNX 不能直接用 PyG Data，必须拆解
        x_numpy = data.x.numpy().astype(np.float32)  # 节点特征
        edge_index_numpy = data.edge_index.numpy().astype(np.int64)  # 边索引
        dop_numpy = data.dop.numpy().astype(np.float32)  # 并行度 dop
        batch_idx_numpy = data.batch.numpy().astype(np.int64)  # ✅ 转换为 NumPy 数组

        # ✅ ONNX 推理输入必须匹配 `torch.onnx.export()` 里的 `input_names`
        ort_inputs = {
            "x": x_numpy,
            "edge_index": edge_index_numpy,
            "batch": batch_idx_numpy  # ✅ 确保 batch 是 NumPy 数组
        }

        # 运行 ONNX 推理
        ort_outs = session.run(["output"], ort_inputs)[0]
        a, b, c = ort_outs[:, 0], ort_outs[:, 1], ort_outs[:, 2]

        # ✅ 计算预测执行时间，避免 `data.dop` 维度问题
        dop_numpy = dop_numpy.reshape(-1)  # 保证是 1D 向量
        predictions = np.maximum(b * (dop_numpy ** a) + c, epsilon)

        results.append((data.y[:, 1].numpy().reshape(-1), predictions))  # `data.y` 也是 1D
        
    onnx_time = time.time() - start_time  # 记录 ONNX 预测时间
    # 计算 Q-error
    actual_mem, predicted_mem = zip(*results)
    actual_mem = np.concatenate(actual_mem)
    predicted_mem = np.concatenate(predicted_mem)
    q_error_mem = np.maximum(actual_mem / predicted_mem, predicted_mem / actual_mem) - 1
    
    # 转换 DataFrame 以便分 bin 计算
    results_df = pd.DataFrame({
        "actual_mem": actual_mem,
        "predicted_mem": predicted_mem,
        "q_error_mem": q_error_mem,
    })
    
    # 按内存使用量进行分类
    results_df["mem_bin"] = featurelize.categorize_memory_bins(results_df["actual_mem"])
    
    # 计算 Memory Q-Error
    mem_qerror_df = results_df.groupby("mem_bin")["q_error_mem"].mean().reset_index()
    mem_qerror_df.columns = ["Memory Bin", "Memory Q-error"]
    
    # 保存 Q-error 统计数据
    with open(output_file, "w") as f:
        f.write("Memory Bin,Memory Q-error\n")
        mem_qerror_df.to_csv(f, index=False, header=False)
        f.write(f"\nONNX Prediction Time,{onnx_time:.6f}\n")
    
    # 保存 真实 vs. 预测值 对比数据
    comparison_df = results_df[["actual_mem", "predicted_mem", "q_error_mem"]]
    comparison_df.to_csv(comparison_file, index=False)
    
    print(f"✅ 统计结果已保存到 {output_file}")
    print(f"✅ 真实 vs. 预测对比已保存到 {comparison_file}")
    
    return output_file, comparison_file
