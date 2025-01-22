import os
import sys
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils

# 定义网络模型
class CurveFitModel(nn.Module):
    def __init__(self, input_dim):
        super(CurveFitModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 输出 a, b, c
        )

    def forward(self, x):
        return self.fc(x)

# 自定义损失函数
def curve_loss(pred_params, dop, true_time, epsilon=1e-2):
    """
    计算曲线拟合损失。

    Parameters:
    - pred_params: Tensor, 预测的 a, b, c 参数
    - dop: Tensor, 并行度 dop 值
    - true_time: Tensor, 实际执行时间
    - epsilon: float, 避免 log(0)

    Returns:
    - loss: Tensor, 损失值
    """
    a, b, c = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2]
    pred_time = b * (dop ** a) + c
    relative_error = torch.abs((true_time - pred_time))
    loss = torch.mean(relative_error)
    return loss

# 训练模型
def train_curve_model(X_train, y_train, dop_train, epochs=500, lr=0.01):
    """
    训练用于预测曲线参数的模型。

    Parameters:
    - X_train: Tensor, 特征
    - y_train: Tensor, 实际执行时间
    - dop_train: Tensor, 并行度
    - epochs: int, 训练轮数
    - lr: float, 学习率

    Returns:
    - model: CurveFitModel, 训练后的模型
    """
    input_dim = X_train.shape[1]
    model = CurveFitModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        pred_params = model(X_train)
        loss = curve_loss(pred_params, dop_train, y_train)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return model

def predict_and_evaluate_curve(
    model, X_test, y_test, dop_test, operator=None, output_prefix=None, suffix=""
):
    """
    使用模型进行预测并评估性能，同时保存 ONNX 模型并比较预测时间。

    Parameters:
    - model: CurveFitModel, 训练好的 PyTorch 模型
    - X_test: Tensor, 测试特征
    - y_test: Tensor, 实际执行时间
    - dop_test: Tensor, 测试并行度
    - operator: str, 操作符类型，用于命名保存的 ONNX 文件
    - output_prefix: str, 保存 ONNX 模型的前缀路径
    - suffix: str, ONNX 模型文件名后缀

    Returns:
    - results: dict, 包含预测性能和时间对比
    """
    model.eval()

    # 原生 PyTorch 模型预测
    start_time = time.time()
    with torch.no_grad():
        pred_params = model(X_test)
        a, b, c = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2]
        pred_time = b * (dop_test ** a) + c
    native_time = time.time() - start_time

    # 保存为 ONNX 模型
    onnx_path = None
    if operator and output_prefix:
        onnx_path = f"/home/zhy/opengauss/tools/serverless_tools/train/model/no_dop/{operator}/{output_prefix}_{suffix}_{operator.replace(' ', '_')}.onnx"
        onnx_dir = os.path.dirname(onnx_path)
        os.makedirs(onnx_dir, exist_ok=True)

        # 导出 ONNX 模型
        dummy_input = torch.randn(X_test.size(0), X_test.size(1))
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11,
        )
        print(f"ONNX model saved to: {onnx_path}")

    # 使用 ONNX Runtime 进行预测
    predictions_onnx = None
    onnx_time = None
    if onnx_path:
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        start_time = time.time()
        predictions_onnx = session.run(None, {input_name: X_test.numpy().astype(np.float32)})[0]
        onnx_time = time.time() - start_time

    # 计算均方误差
    relative_error = torch.abs((y_test - pred_time))
    mse_native = torch.mean(relative_error)
    mse_onnx = None
    # 打印时间对比
    print(f"Native model prediction time: {native_time:.6f} seconds")
    if onnx_time is not None:
        print(f"ONNX model prediction time: {onnx_time:.6f} seconds")

    # 返回结果
    return {
        "native_time": native_time,
        "onnx_time": onnx_time,
        "mse_native": mse_native,
        "pred_params_native": pred_params,
    }


def process_and_train_curve(csv_path_pattern, operator, output_prefix, train_queries, test_queries, test_size=0.2, epochs=100, lr=0.001):
    """
    数据加载、处理和训练执行时间和内存预测的曲线拟合模型。

    Parameters:
    - csv_path_pattern: str, glob 模式，用于找到 plan_info.csv 文件
    - operator: str, 操作符类型，用于筛选数据
    - output_prefix: str, 用于保存模型的前缀
    - train_queries: list of int, 用于训练的查询 ID
    - test_queries: list of int, 用于测试的查询 ID
    - test_size: float, 测试集比例（仅用于内部划分）
    - epochs: int, 训练轮数
    - lr: float, 学习率

    Returns:
    - results: dict, 包含执行时间和内存模型及评估结果
    """
    # 使用 utils.prepare_data 加载并处理数据
    X_train, X_test, y_train, y_test = utils.prepare_data(
        csv_path_pattern=csv_path_pattern,
        operator=operator,
        feature_columns=['l_input_rows', 'actural_rows', 'instance_mem', 'width', 'query_dop'],
        target_columns=['query_id', 'execution_time', 'peak_mem', 'dop'],
        train_queries=train_queries,
        test_queries=test_queries,
    )


    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train.drop(columns=['query_dop']).values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.drop(columns=['query_dop']).values, dtype=torch.float32)
    y_train_exec = torch.tensor(y_train['execution_time'].values, dtype=torch.float32)
    y_test_exec = torch.tensor(y_test['execution_time'].values, dtype=torch.float32)
    y_train_mem = torch.tensor(y_train['peak_mem'].values, dtype=torch.float32)
    y_test_mem = torch.tensor(y_test['peak_mem'].values, dtype=torch.float32)
    dop_train_tensor = torch.tensor(y_train['dop'].values, dtype=torch.float32)
    dop_test_tensor = torch.tensor(y_test['dop'].values, dtype=torch.float32)

    # 分别训练执行时间和内存预测模型
    print("Training execution time model...")
    model_exec = train_curve_model(X_train_tensor, y_train_exec, dop_train_tensor, epochs=epochs, lr=lr)

    print("Training memory model...")
    model_mem = train_curve_model(X_train_tensor, y_train_mem, dop_train_tensor, epochs=epochs, lr=lr)

    # 分别评估执行时间和内存预测模型
    print("Evaluating execution time model...")
    results_exec = predict_and_evaluate_curve(
        model_exec, X_test_tensor, y_test_exec, dop_test_tensor, operator, output_prefix, suffix="exec"
    )

    print("Evaluating memory model...")
    results_mem = predict_and_evaluate_curve(
        model_mem, X_test_tensor, y_test_mem, dop_test_tensor, operator, output_prefix, suffix="mem"
    )

    # 整理和返回结果
    results = {
        "execution_time_model": {
            "model": model_exec,
            "native_time": results_exec["native_time"],
            "onnx_time": results_exec["onnx_time"],
            "test_mse": results_exec["mse_native"],
            "predicted_params": results_exec["pred_params_native"],
        },
        "memory_model": {
            "model": model_mem,
            "native_time": results_mem["native_time"],
            "onnx_time": results_mem["onnx_time"],
            "test_mse": results_mem["mse_native"],
            "predicted_params": results_mem["pred_params_native"],
        },
    }

    print(f"\nExecution Time Test MSE: {results_exec['mse_native']:.4f}")
    print(f"Memory Test MSE: {results_mem['mse_native']:.4f}")

    return results
