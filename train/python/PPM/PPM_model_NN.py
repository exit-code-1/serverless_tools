import os
import sys
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import onnxruntime as ort
import featurelize
from torch.utils.data import DataLoader, TensorDataset
sys.path.append(os.path.abspath("/home/zhy/opengauss/tools/serverless_tools/train/python"))
class Exec_CurveFitModel(nn.Module):
    def __init__(self, input_dim, min_a=-2, max_a=2):
        super(Exec_CurveFitModel, self).__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
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
    def __init__(self, input_dim, min_a=-2, max_a=2):
        super(Mem_CurveFitModel, self).__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
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
    abs_error = torch.abs(pred_time - true_time)
    log_error = torch.log(abs_error + 1)
    log_error = torch.where(pred_time < true_time, log_error, log_error)
    pred_time = torch.clamp(pred_time, min=1e-2)
    relative_error = torch.log(torch.max(pred_time/true_time, true_time/pred_time))

    # 返回最终损失
    loss = torch.mean(log_error + relative_error)
    
    
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
    abs_error = torch.abs(pred_mem - true_mem)
    log_error = torch.log(abs_error + 1)
    log_error = torch.where(pred_mem < true_mem, log_error, log_error)
    pred_mem = torch.clamp(pred_mem, min=1e-2)
    # 计算相对误差
    relative_error = torch.log(torch.max(pred_mem/true_mem, true_mem/pred_mem))
    
    # 返回最终损失
    loss = torch.mean(log_error + relative_error)  # 加上负值惩罚项
    
    return loss


def predict_and_evaluate_exec_curve(
    onnx_model_path, X_test, y_test, dop_test, output_file, comparison_file, epsilon=1e-2):
    """
    使用 ONNX 模型进行推理，并按时间和内存 bin 进行评估，同时保存真实值 vs. 预测值。

    Parameters:
    - onnx_model_path: str, ONNX 模型路径
    - X_test: Tensor, 测试特征
    - y_test: Tensor, 真实执行时间
    - dop_test: Tensor, 并行度信息
    - output_file: str, bin 级别的误差统计文件
    - comparison_file: str, 真实 vs 预测对比文件
    - epsilon: float, 避免除零误差

    Returns:
    - output_file: str, 生成的 bin 级别误差 CSV 文件路径
    - comparison_file: str, 真实 vs. 预测对比 CSV 文件路径
    """
    # 加载 ONNX 模型
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name

    # 进行 ONNX 推理
    start_time = time.time()
    pred_params = session.run(None, {input_name: X_test.numpy().astype(np.float32)})[0]
    onnx_time = time.time() - start_time  # 记录 ONNX 预测时间

    # 拆分出 a 和 b
    a, b = pred_params[:, 0], pred_params[:, 1]

    # 计算执行时间预测值
    predictions = np.maximum(b * (dop_test.numpy() ** a), 1e-6)  # 避免数值错误

    # 计算 Q-error
    q_error_time = np.maximum(y_test.numpy() / predictions, predictions / y_test.numpy()) - 1

    # 转换 DataFrame 以便分 bin 计算
    results_df = pd.DataFrame({
        "actual_time": y_test.numpy(),
        "predicted_time": predictions,
        "q_error_time": q_error_time,
    })

    # 按时间进行分类
    results_df["time_bin"] = featurelize.categorize_time_bins(results_df["actual_time"] / 1000)

    # 计算 Execution Time Q-Error
    time_qerror_df = results_df.groupby("time_bin")["q_error_time"].mean().reset_index()
    time_qerror_df.columns = ["Time Bin", "Execution Time Q-error"]

    # 🔹 保存 Q-error 统计数据
    with open(output_file, "w") as f:
        f.write("Time Bin,Execution Time Q-error\n")
        time_qerror_df.to_csv(f, index=False, header=False)
        f.write(f"\nONNX Prediction Time,{onnx_time:.6f}\n")

    # 🔹 保存 真实 vs. 预测值 对比数据
    comparison_df = results_df[["actual_time", "predicted_time", "q_error_time"]]
    comparison_df.to_csv(comparison_file, index=False)

    print(f"✅ 统计结果已保存到 {output_file}")
    print(f"✅ 真实 vs. 预测对比已保存到 {comparison_file}")

    return output_file, comparison_file

def predict_and_evaluate_mem_curve(
    onnx_model_path, X_test, y_test, dop_test, output_file, comparison_file, epsilon=1e-2):
    """
    使用 ONNX 模型进行推理，并按内存 bin 进行评估，同时保存真实 vs 预测值。

    Parameters:
    - onnx_model_path: str, ONNX 模型路径
    - X_test: Tensor, 测试特征
    - y_test: Tensor, 真实内存使用量
    - dop_test: Tensor, 并行度信息
    - output_file: str, bin 级别的误差统计文件
    - comparison_file: str, 真实 vs 预测对比文件
    - epsilon: float, 避免除零误差

    Returns:
    - output_file: str, 生成的 bin 级别误差 CSV 文件路径
    - comparison_file: str, 真实 vs. 预测对比 CSV 文件路径
    """
    # 加载 ONNX 模型
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name

    # 进行 ONNX 推理
    start_time = time.time()
    pred_params = session.run(None, {input_name: X_test.numpy().astype(np.float32)})[0]
    onnx_time = time.time() - start_time  # 记录 ONNX 预测时间

    # 拆分出 a 和 b
    a, b = pred_params[:, 0], pred_params[:, 1]

    # 计算内存预测值
    predictions = np.maximum(b * (dop_test.numpy() ** a), 1e-6)  # 避免数值错误

    # 计算 Q-error
    q_error_memory = np.maximum(y_test.numpy() / predictions, predictions / y_test.numpy()) - 1

    # 转换 DataFrame 以便分 bin 计算
    results_df = pd.DataFrame({
        "actual_memory": y_test.numpy(),
        "predicted_memory": predictions,
        "q_error_memory": q_error_memory,
    })

    # 按内存进行分类
    results_df["memory_bin"] = featurelize.categorize_memory_bins(results_df["actual_memory"])

    # 计算 Memory Q-Error
    memory_qerror_df = results_df.groupby("memory_bin")["q_error_memory"].mean().reset_index()
    memory_qerror_df.columns = ["Memory Bin", "Memory Q-error"]

    # 🔹 保存 Q-error 统计数据
    with open(output_file, "w") as f:
        f.write("Memory Bin,Memory Q-error\n")
        memory_qerror_df.to_csv(f, index=False, header=False)
        f.write(f"\nONNX Prediction Time,{onnx_time:.6f}\n")

    # 🔹 保存 真实 vs. 预测值 对比数据
    comparison_df = results_df[["actual_memory", "predicted_memory", "q_error_memory"]]
    comparison_df.to_csv(comparison_file, index=False)

    print(f"✅ 统计结果已保存到 {output_file}")
    print(f"✅ 真实 vs. 预测对比已保存到 {comparison_file}")

    return output_file, comparison_file