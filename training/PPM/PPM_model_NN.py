import os
import sys
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import onnxruntime as ort
from training.PPM import featurelize
from torch.utils.data import DataLoader, TensorDataset
sys.path.append(os.path.abspath("/home/zhy/opengauss/tools/serverless_tools/train/python"))
class Exec_CurveFitModel(nn.Module):
    def __init__(self, input_dim, min_a=-1, max_a=1):
        super(Exec_CurveFitModel, self).__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(32, 2),  # 输出 a, b, c
        )
        self.min_a = min_a
        self.max_a = max_a

    def forward(self, x):
        x = self.bn_input(x)
        pred_params = self.fc(x)
        
        # 获取 a, b, c
        a, b = pred_params[:, 0], pred_params[:, 1]
        
        # 对 a 应用 Sigmoid 激活函数并映射到 [min_a, max_a]
        a = torch.sigmoid(a) * (self.max_a - self.min_a) + self.min_a

        # 返回映射后的参数
        return torch.stack([a, b], dim=1)
    
# 定义网络模型
class Mem_CurveFitModel(nn.Module):
    def __init__(self, input_dim, min_a=-1, max_a=1):
        super(Mem_CurveFitModel, self).__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(32, 2),  # 输出 a, b, c
        )
        self.min_a = min_a
        self.max_a = max_a

    def forward(self, x): 
        x = self.bn_input(x)
        pred_params = self.fc(x)
        
        # 获取 a, b
        a, b = pred_params[:, 0], pred_params[:, 1]
        
        # 对 a 应用 Sigmoid 激活函数并映射到 [min_a, max_a]
        a = torch.sigmoid(a) * (self.max_a - self.min_a) + self.min_a

        # 返回映射后的参数
        return torch.stack([a, b], dim=1)
    
def reset_model(model):
    """重新初始化模型参数"""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model

def curve_exec_loss(pred_params, dop, true_time, epsilon=1e-2, alpha=0.5, log_file="loss_debug.log"):
    # 打开日志文件（以追加模式），并写入错误信息
    def log_to_file(message):
        with open(log_file, "a") as f:
            f.write(message + "\n")

    a, b = pred_params[:, 0], pred_params[:, 1]
    # 计算预测时间
    pred_time = torch.relu(b * (dop ** a))

    # 计算绝对误差
    abs_error = torch.abs(pred_time - true_time) / ((a + 0.1))
    pred_time = torch.clamp(pred_time, min=1e-2)

    # 返回最终损失
    loss = torch.mean(abs_error)
    
    
    return loss

def curve_mem_loss(pred_params, dop, true_mem, epsilon=1e-2, alpha=0.5, log_file="loss_debug.log"):
    # 打开日志文件（以追加模式），并写入错误信息
    def log_to_file(message):
        with open(log_file, "a") as f:
            f.write(message + "\n")

    a, b = pred_params[:, 0], pred_params[:, 1]
    # 计算预测时间
    pred_mem = torch.relu(b * (dop ** a))

    # 如果 pred_time 小于 true_time，将 abs_error 乘以 2
    abs_error = torch.abs(pred_mem - true_mem) / ((a + 0.1))
    pred_mem = torch.clamp(pred_mem, min=1e-2)
    
    # 返回最终损失
    loss = torch.mean(abs_error)  # 加上负值惩罚项
    
    return loss


def predict_and_evaluate_curve(
    onnx_time_model_path,
    onnx_mem_model_path,
    X_test,
    y_time_test,
    y_mem_test,
    dop_test,
    output_file,
    query_ids,
    epsilon=1e-2
):
    """
    使用两个 ONNX 模型同时预测执行时间和内存使用，并按 bin 分类进行误差统计。
    
    Parameters:
    - onnx_time_model_path: str, 执行时间 ONNX 模型路径
    - onnx_mem_model_path: str, 内存使用 ONNX 模型路径
    - X_test: Tensor, 测试特征
    - y_time_test: Tensor, 真实执行时间
    - y_mem_test: Tensor, 真实内存使用
    - dop_test: Tensor, 并行度信息
    - output_dir: str, 输出文件夹路径
    - epsilon: float, 防止除以零
    
    Returns:
    - qerror_file: str, bin 级别误差 CSV 文件路径
    - comparison_file: str, 真实 vs 预测对比 CSV 文件路径
    """
    # 创建输出路径
    os.makedirs(output_file, exist_ok=True)
    qerror_file = os.path.join(output_file, "qerror.csv")
    comparison_file = os.path.join(output_file, "test_predictions.csv")

    # ==== 加载执行时间模型 ====
    session_time = ort.InferenceSession(onnx_time_model_path)
    input_name_time = session_time.get_inputs()[0].name
    start = time.time()
    pred_params_time = session_time.run(None, {input_name_time: X_test.numpy().astype(np.float32)})[0]
    time_predict_time = time.time() - start
    time_predict_duration = time.time() - start # <-- 总时间
    # ==== 加载内存模型 ====
    session_mem = ort.InferenceSession(onnx_mem_model_path)
    input_name_mem = session_mem.get_inputs()[0].name
    start = time.time()
    pred_params_mem = session_mem.run(None, {input_name_mem: X_test.numpy().astype(np.float32)})[0]
    mem_predict_time = time.time() - start
    mem_predict_duration = time.time() - start # <-- 总时间
    # --- 新增：计算平均延迟 ---
    num_queries = len(X_test)
    avg_time_latency = time_predict_duration / num_queries if num_queries > 0 else 0
    avg_mem_latency = mem_predict_duration / num_queries if num_queries > 0 else 0
    # ==== 预测值 ====
    a_time, b_time = pred_params_time[:, 0], pred_params_time[:, 1]
    pred_time = np.maximum(b_time * (dop_test.numpy() ** a_time), 1e-6)

    a_mem, b_mem = pred_params_mem[:, 0], pred_params_mem[:, 1]
    pred_mem = np.maximum(b_mem * (dop_test.numpy() ** a_mem), 1e-6)

    # ==== Q-error ====
    actual_time = y_time_test.numpy()
    actual_mem = y_mem_test.numpy()

    q_error_time = np.maximum(actual_time / pred_time, pred_time / actual_time) - 1
    q_error_mem = np.maximum(actual_mem / pred_mem, pred_mem / actual_mem) - 1
    # ==== DataFrame 组装 ====
    results_df = pd.DataFrame({
        "query_id": query_ids,
        "dop": dop_test.numpy().astype(int),
        "actual_time": actual_time,
        "predicted_time": pred_time,
        "Execution Time Q-error": q_error_time,
        "actual_memory": actual_mem,
        "predicted_memory": pred_mem,
        "Memory Q-error": q_error_mem,
    })

    # ==== 分类 & 分组 ====
    results_df["time_bin"] = featurelize.categorize_time_bins(results_df["actual_time"] / 1000)
    results_df["memory_bin"] = featurelize.categorize_memory_bins(results_df["actual_memory"])

    # 平均误差统计
    time_qerror_df = results_df.groupby("time_bin")["Execution Time Q-error"].mean().reset_index()
    time_qerror_df.columns = ["Time Bin", "Execution Time Q-error"]

    mem_qerror_df = results_df.groupby("memory_bin")["Memory Q-error"].mean().reset_index()
    mem_qerror_df.columns = ["Memory Bin", "Memory Q-error"]

    # ==== 保存 bin 级误差 ====
    with open(qerror_file, "w") as f:
        f.write("Time Bin,Execution Time Q-error\n")
        time_qerror_df.to_csv(f, index=False, header=False)
        f.write("\nMemory Bin,Memory Q-error\n")
        mem_qerror_df.to_csv(f, index=False, header=False)
        f.write(f"\nONNX Time Prediction Time,{time_predict_time:.6f}\n")
        f.write(f"ONNX Memory Prediction Time,{mem_predict_time:.6f}\n")
        f.write(f"ONNX Avg Time Prediction Latency,{avg_time_latency:.6f}\n") # <-- 新增
        f.write(f"ONNX Total Memory Prediction Duration,{mem_predict_duration:.6f}\n")
        f.write(f"ONNX Avg Memory Prediction Latency,{avg_mem_latency:.6f}\n") # <-- 新增
    # ==== 保存对比 ====
    comparison_df = results_df[[
        "query_id", "dop",
        "actual_time", "predicted_time", "Execution Time Q-error",
        "actual_memory", "predicted_memory", "Memory Q-error"
    ]]
    comparison_df.to_csv(comparison_file, sep=';', index=False)

    print(f"✅ Bin Q-error 写入 {qerror_file}")
    print(f"✅ 真实 vs 预测对比写入 {comparison_file}")

    return qerror_file, comparison_file