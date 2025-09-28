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
from training.PPM import featurelize

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
    def __init__(self, input_dim, hidden_dim, min_a=-1, max_a=1):
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
    def __init__(self, input_dim, hidden_dim, min_a=-1, max_a=1):
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
    abs_error = torch.abs(pred_time - y) / ((a + 0.1))
    # 计算最终损失
    loss = torch.mean(abs_error)

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
    abs_error = torch.abs(pred_time - y) / ((a + 0.1))
    loss = torch.mean(abs_error)

    return loss


def predict_and_evaluate_curve(
    exec_model_path, mem_model_path,
    test_loader,
    output_file,
    epsilon_time=1e-6,
    epsilon_mem=1e-2
):
    """
    使用两个 GNN ONNX 模型评估执行时间和内存使用，生成 Q-error 统计与预测对比文件。
    
    参数:
    - exec_model_path: str, 执行时间 GNN 的 ONNX 模型路径
    - mem_model_path: str, 内存 GNN 的 ONNX 模型路径
    - test_loader: DataLoader, PyG 测试数据加载器
    - output_file: str, 保存 bin 分类 Q-error 的统计文件
    - comparison_file: str, 保存真实 vs. 预测对比的文件
    """
    # 创建输出路径
    os.makedirs(output_file, exist_ok=True)
    qerror_file = os.path.join(output_file, "qerror.csv")
    comparison_file = os.path.join(output_file, "comparison.csv")
    exec_session = ort.InferenceSession(exec_model_path)
    mem_session = ort.InferenceSession(mem_model_path)

    actual_times, predicted_times = [], []
    actual_mem, predicted_mem = [], []
    
    # --- 新增：用于记录每次预测延迟的列表 ---
    exec_latencies = []
    mem_latencies = []

    time_start = time.time()
    # --- 修改：在循环内部计时 ---
    for data in test_loader:
        x_np = data.x.numpy().astype(np.float32)
        edge_index_np = data.edge_index.numpy().astype(np.int64)
        dop_np = data.dop.numpy().astype(np.float32).reshape(-1)
        batch_np = data.batch.numpy().astype(np.int64)

        ort_inputs = {
            "x": x_np,
            "edge_index": edge_index_np,
            "batch": batch_np
        }

        # ---- 执行时间预测并计时 ----
        start_t = time.time()
        exec_out = exec_session.run(["output"], ort_inputs)[0]
        end_t = time.time()
        exec_latencies.append(end_t - start_t) # 记录本次延迟
        
        a_exec, b_exec = exec_out[:, 0], exec_out[:, 1]
        pred_time = np.maximum(b_exec * (dop_np ** a_exec), epsilon_time)
        true_time = data.y[:, 0].numpy().reshape(-1)

        # ---- 内存预测并计时 ----
        start_m = time.time()
        mem_out = mem_session.run(["output"], ort_inputs)[0]
        end_m = time.time()
        mem_latencies.append(end_m - start_m) # 记录本次延迟

        a_mem, b_mem, c_mem = mem_out[:, 0], mem_out[:, 1], mem_out[:, 2]
        pred_mem = np.maximum(b_mem * (dop_np ** a_mem) + c_mem, epsilon_mem)
        true_mem = data.y[:, 1].numpy().reshape(-1)

        actual_times.append(true_time)
        predicted_times.append(pred_time)
        actual_mem.append(true_mem)
        predicted_mem.append(pred_mem)
    # --- 循环结束 ---

    # --- 新增：计算总延迟和平均延迟 ---
    total_exec_duration = np.sum(exec_latencies)
    total_mem_duration = np.sum(mem_latencies)
    num_queries = len(test_loader.dataset) # 获取查询总数
    avg_exec_latency = np.mean(exec_latencies) if exec_latencies else 0
    avg_mem_latency = np.mean(mem_latencies) if mem_latencies else 0
    predict_duration = time.time() - time_start

    # ---- 合并所有样本 ----
    actual_times = np.concatenate(actual_times)
    predicted_times = np.concatenate(predicted_times)
    actual_mem = np.concatenate(actual_mem)
    predicted_mem = np.concatenate(predicted_mem)

    # ---- Q-error ----
    q_error_time = np.maximum(actual_times / predicted_times, predicted_times / actual_times) - 1
    q_error_mem = np.maximum(actual_mem / predicted_mem, predicted_mem / actual_mem) - 1

    # ---- DataFrame 构造 ----
    df = pd.DataFrame({
        "actual_time": actual_times,
        "predicted_time": predicted_times,
        "Execution Time Q-error": q_error_time,
        "actual_mem": actual_mem,
        "predicted_mem": predicted_mem,
        "Memory Q-error": q_error_mem,
    })

    # ---- 添加 bins ----
    df["time_bin"] = featurelize.categorize_time_bins(df["actual_time"] / 1000)
    df["mem_bin"] = featurelize.categorize_memory_bins(df["actual_mem"])

    # ---- 按 bin 聚合 Q-error ----
    time_qerror_df = df.groupby("time_bin")["Execution Time Q-error"].mean().reset_index()
    time_qerror_df.columns = ["Bin", "Execution Time Q-error"]
    time_qerror_df["Type"] = "Execution"

    mem_qerror_df = df.groupby("mem_bin")["Memory Q-error"].mean().reset_index()
    mem_qerror_df.columns = ["Bin", "Memory Q-error"]
    mem_qerror_df["Type"] = "Memory"

    mem_qerror_df.rename(columns={"Memory Q-error": "Q-error"}, inplace=True)
    time_qerror_df.rename(columns={"Execution Time Q-error": "Q-error"}, inplace=True)

    combined_df = pd.concat([time_qerror_df, mem_qerror_df], axis=0)

    # ---- 保存统计文件 ----
    with open(qerror_file, "w") as f:
        f.write("Type,Bin,Q-error\n")
        combined_df[["Type", "Bin", "Q-error"]].to_csv(f, index=False, header=False)
        f.write(f"\nONNX Total Execution Prediction Duration,{total_exec_duration:.6f}\n")
        f.write(f"ONNX Avg Execution Prediction Latency,{avg_exec_latency:.6f}\n") # <-- 新增
        f.write(f"ONNX Total Memory Prediction Duration,{total_mem_duration:.6f}\n")
        f.write(f"ONNX Avg Memory Prediction Latency,{avg_mem_latency:.6f}\n") # <-- 新增

    # ---- 保存真实 vs 预测对比 ----
    df.to_csv(comparison_file, sep = ';' , index=False)

    print(f"✅ 执行时间 + 内存 Q-error 统计已保存到 {qerror_file}")
    print(f"✅ 真实 vs. 预测对比数据已保存到 {comparison_file}")

    return qerror_file, comparison_file

