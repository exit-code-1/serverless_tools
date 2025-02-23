import os
import sys
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils

# 定义网络模型
class Exec_CurveFitModel(nn.Module):
    def __init__(self, input_dim):
        super(Exec_CurveFitModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),  # 添加批归一化层
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 输出 a, b, c
        )

    def forward(self, x):
        return self.fc(x)
    
# 定义网络模型
class Mem_CurveFitModel(nn.Module):
    def __init__(self, input_dim):
        super(Mem_CurveFitModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),  # 添加批归一化层
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 输出 a, b, c
        )

    def forward(self, x):
        return self.fc(x)

# 自定义损失函数
def curve_loss(pred_params, dop, true_time, epsilon=1e-2, alpha=0.5):
    """
    
    Parameters:
    - pred_params: a , b, c 预测参数
    - true_time: Tensor, 真实值
    - epsilon: float, 避免除以0的一个小常数
    - alpha: float, 用于加权MAE和加权相对误差的系数
    
    Returns:
    - loss: Tensor, 计算得到的损失值
    """
    a, b, c = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2]
    pred_time = b * (dop ** a) + c
    # pred_time = torch.maximum(b * (dop ** a), torch.tensor(c)) 
    
    # 计算绝对误差（MAE），对数空间的误差会更小
    abs_error = torch.abs(pred_time - true_time)
    # abs_error = torch.log2(abs_error + 1)
    
    # 计算相对误差
    # rel_error = torch.abs((true_time - pred_time) / (true_time + epsilon))
    
    # 综合两种误差，alpha控制了MAE和加权相对误差的权重
    loss = torch.mean(abs_error)
    return loss


def train_exec_curve_model(X_train, y_train, dop_train, batch_size=32, epochs=100, lr=0.05):
    """
    训练用于预测曲线参数的模型，使用批量训练，并加入学习率调度器。

    Parameters:
    - X_train: Tensor, 特征
    - y_train: Tensor, 实际执行时间
    - dop_train: Tensor, 并行度
    - batch_size: int, 批次大小
    - epochs: int, 训练轮数
    - lr: float, 初始学习率

    Returns:
    - model: CurveFitModel, 训练后的模型
    - training_time: float, 训练时间
    """
    input_dim = X_train.shape[1]
    model = Exec_CurveFitModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-4)

    # 创建 DataLoader 进行批量训练
    train_dataset = TensorDataset(X_train, y_train, dop_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 设置学习率调度器，StepLR 每 10 轮降低一次学习率，gamma=0.5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    start_time = time.time()  # 记录训练开始时间

    for epoch in range(epochs):
        model.train()
        for batch_idx, (X_batch, y_batch, dop_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            pred_params = model(X_batch)
            loss = curve_loss(pred_params, dop_batch, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Step the scheduler to adjust the learning rate
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    training_time = time.time() - start_time  # 计算训练时间
    return model, training_time  # 返回模型和训练时间

def train_mem_curve_model(X_train, y_train, dop_train, batch_size=32, epochs=100, lr=0.05):
    """
    训练用于预测曲线参数的模型，使用批量训练，并加入学习率调度器。

    Parameters:
    - X_train: Tensor, 特征
    - y_train: Tensor, 实际执行时间
    - dop_train: Tensor, 并行度
    - batch_size: int, 批次大小
    - epochs: int, 训练轮数
    - lr: float, 初始学习率

    Returns:
    - model: CurveFitModel, 训练后的模型
    - training_time: float, 训练时间
    """
    input_dim = X_train.shape[1]
    model = Mem_CurveFitModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-4)

    # 创建 DataLoader 进行批量训练
    train_dataset = TensorDataset(X_train, y_train, dop_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 设置学习率调度器，StepLR 每 10 轮降低一次学习率，gamma=0.5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    start_time = time.time()  # 记录训练开始时间

    for epoch in range(epochs):
        model.train()
        for batch_idx, (X_batch, y_batch, dop_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            pred_params = model(X_batch)
            loss = curve_loss(pred_params, dop_batch, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Step the scheduler to adjust the learning rate
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    training_time = time.time() - start_time  # 计算训练时间
    return model, training_time  # 返回模型和训练时间

def predict_and_evaluate_exec_curve(
    model, X_test, y_test, dop_test, epsilon=1e-2, operator=None, suffix=""
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
        predictions_native = b * (dop_test ** a) + c
        # predictions_native = torch.maximum(b * (dop_test ** a), c)
    native_time = time.time() - start_time

    # 保存为 ONNX 模型
    onnx_path = None
    if operator:
        onnx_path = f"/home/zhy/opengauss/tools/serverless_tools/train/model/dop/{operator}/{suffix}_{operator.replace(' ', '_')}.onnx"
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

     # Calculate standard mean absolute error (MAE) for native predictions
    mae_native = torch.mean(torch.abs(y_test - predictions_native))

    # Calculate Prediction Accuracy for native model
    Q_error = torch.mean(
         torch.abs((y_test - predictions_native) / (y_test + epsilon))
    )

    # Create comparison DataFrame
    comparisons = pd.DataFrame({
        'Actual': y_test,
        'Predicted_Native': predictions_native,
        'Difference_Native': y_test - predictions_native,
    })

    # Calculate Prediction Accuracy for ONNX model
    time_accuracy_onnx = None
    if onnx_time is not None:
        time_accuracy_onnx = torch.mean(
            (1 - torch.abs(y_test - predictions_native) / (y_test + epsilon)) * 100
        )

    # Calculate the average of the actual target values
    avg_actual_value = torch.mean(y_test)

    # Print model prediction times
    print(f"Native model prediction time: {native_time:.6f} seconds")
    if onnx_time is not None:
        print(f"ONNX model prediction time: {onnx_time:.6f} seconds")

    # Organize the performance metrics into one dictionary
    performance = {
        "metrics": {
            "MAE_error": mae_native,
            "Q_error": Q_error,
            "average_actual_value": avg_actual_value
        },
        "comparisons": comparisons,
        "native_time": native_time,
        "onnx_time": onnx_time,
        "onnx_accuracy": time_accuracy_onnx
    }

    return performance


def predict_and_evaluate_mem_curve(
    model, X_test, y_test, dop_test, epsilon=1e-2, operator=None, suffix=""
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
        predictions_native = b * (dop_test ** a) + c
        # predictions_native = torch.maximum(b * (dop_test ** a), c)
    native_time = time.time() - start_time

    # 保存为 ONNX 模型
    onnx_path = None
    if operator:
        onnx_path = f"/home/zhy/opengauss/tools/serverless_tools/train/model/dop/{operator}/{suffix}_{operator.replace(' ', '_')}.onnx"
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

     # Calculate standard mean absolute error (MAE) for native predictions
    mae_native = torch.mean(torch.abs(y_test - predictions_native))

    # Calculate Prediction Accuracy for native model
    Q_error = torch.mean(
         torch.abs((y_test - predictions_native) / (y_test + epsilon))
    )

    # Create comparison DataFrame
    comparisons = pd.DataFrame({
        'Actual': y_test,
        'Predicted_Native': predictions_native,
        'Difference_Native': y_test - predictions_native,
    })

    # Calculate Prediction Accuracy for ONNX model
    time_accuracy_onnx = None
    if onnx_time is not None:
        time_accuracy_onnx = torch.mean(
            (1 - torch.abs(y_test - predictions_native) / (y_test + epsilon)) * 100
        )

    # Calculate the average of the actual target values
    avg_actual_value = torch.mean(y_test)

    # Print model prediction times
    print(f"Native model prediction time: {native_time:.6f} seconds")
    if onnx_time is not None:
        print(f"ONNX model prediction time: {onnx_time:.6f} seconds")

    # Organize the performance metrics into one dictionary
    performance = {
        "metrics": {
            "MAE_error": mae_native,
            "Q_error": Q_error,
            "average_actual_value": avg_actual_value
        },
        "comparisons": comparisons,
        "native_time": native_time,
        "onnx_time": onnx_time,
        "onnx_accuracy": time_accuracy_onnx
    }

    return performance



