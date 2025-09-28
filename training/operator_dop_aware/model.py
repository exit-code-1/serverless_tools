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
# 将项目根目录添加到 sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils

# 定义网络模型
class Exec_CurveFitModel(nn.Module):
    def __init__(self, input_dim, min_a=-2, max_a=2):
        super(Exec_CurveFitModel, self).__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 5),  # 输出 a, b, c, d, e
        )
        self.min_a = min_a
        self.max_a = max_a

    def forward(self, x):
        x = self.bn_input(x)
        pred_params = self.fc(x)
        
        # 获取 a, b, c
        # a, b, c, d, e = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2], pred_params[:, 3], pred_params[:, 4]
        a = torch.sigmoid(pred_params[:, 0])
        b = pred_params[:, 1]
        c = pred_params[:, 2]
        d = torch.sigmoid(pred_params[:, 3])
        e = pred_params[:, 4]

        # 返回映射后的参数
        return torch.stack([a, b, c, d, e], dim=1)
    
# 定义网络模型
class Mem_CurveFitModel(nn.Module):
    def __init__(self, input_dim, min_a=-1, max_a=1):
        super(Mem_CurveFitModel, self).__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 4),  # 输出 a, b, c, d
        )
        self.min_a = min_a
        self.max_a = max_a

    def forward(self, x): 
        x = self.bn_input(x)
        pred_params = self.fc(x)
        
        # 获取 a, b, c
        a, b, c, d = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2], pred_params[:, 3]
        
        # 对 a 应用 Sigmoid 激活函数并映射到 [min_a, max_a]
        # a = torch.sigmoid(a) * (self.max_a - self.min_a) + self.min_a

        # 返回映射后的参数
        return torch.stack([a, b, c, d], dim=1)
    
def reset_model(model):
    """重新初始化模型参数"""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model

def curve_exec_loss(pred_params, dop, true_time, epsilon=1e-2, alpha=0.5, log_file="loss_debug.log"):
    # # 打开日志文件（以追加模式），并写入错误信息
    # def log_to_file(message):
    #     with open(log_file, "a") as f:
    #         f.write(message + "\n")
    # 修改 log_to_file 函数内部打开文件的路径
    def log_to_file(message):
        # 确保目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            f.write(message + "\n")

    a, b, c, d, e = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2], pred_params[:, 3], pred_params[:, 4]
    
    # 计算预测时间
    pred_time = b / (dop**a) + c * dop**d + e

    # 如果 pred_time 为 NaN 或者小于等于零，打印 a, b, c 和 pred_time
    if torch.any(torch.isnan(pred_time)):
        log_to_file(f"NaN or invalid pred_time detected!")
        log_to_file(f"a: {a}")
        log_to_file(f"b: {b}")
        log_to_file(f"c: {c}")
        log_to_file(f"pred_time: {pred_time}")

    # 计算绝对误差
    abs_error = torch.abs(pred_time - true_time) / ((a + 0.1))
    log_error = torch.log(abs_error + 1)
    abs_error = torch.where(pred_time < true_time, abs_error, abs_error)
    pred_time = torch.clamp(pred_time, min=0.1)
    relative_error = torch.log(torch.max(pred_time/true_time, true_time/pred_time))

    # 返回最终损失
    loss = torch.mean(abs_error)
    
    
    return loss

def curve_mem_loss(pred_params, dop, true_mem, epsilon=1e-2, alpha=0.5, log_file="loss_debug.log"):
    # 打开日志文件（以追加模式），并写入错误信息
    # def log_to_file(message):
    #     with open(log_file, "a") as f:
    #         f.write(message + "\n")
    # 修改 log_to_file 函数内部打开文件的路径
    def log_to_file(message):
        # 确保目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            f.write(message + "\n")

    a, b, c, d = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2], pred_params[:, 3]
    
    # 计算预测时间
    pred_mem = torch.max(b * (dop ** a) + c, d)

    # 如果 pred_time 为 NaN 或者小于等于零，打印 a, b, c 和 pred_time
    if torch.any(torch.isnan(pred_mem)):
        log_to_file(f"NaN or invalid pred_time detected!")
        log_to_file(f"a: {a}")
        log_to_file(f"b: {b}")
        log_to_file(f"c: {c}")
        log_to_file(f"pred_time: {pred_mem}")


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

def train_exec_curve_model(X_train, y_train, dop_train, batch_size=32, epochs=100, lr=1e-2):
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 设置学习率调度器，StepLR 每 10 轮降低一次学习率，gamma=0.8
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    start_time = time.time()  # 记录训练开始时间

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0  # 初始化该 epoch 的总损失
        
        for batch_idx, (X_batch, y_batch, dop_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            pred_params = model(X_batch)
            loss = curve_exec_loss(pred_params, dop_batch, y_batch)
            if torch.any(torch.isnan(loss)):
                print(f"NaN detected at epoch {epoch}, batch {batch_idx}. Resetting model parameters.")
                model = reset_model(model)  # 重新初始化模型
                optimizer = optim.Adam(model.parameters(), lr=0.01)  # 重新定义优化器
                break  # 跳出当前训练轮次，重新开始训练

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # 累加损失
            epoch_loss += loss.item()

        # 计算该 epoch 的平均损失
        avg_epoch_loss = epoch_loss / len(train_loader)

        # Step the scheduler to adjust the learning rate
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    training_time = time.time() - start_time  # 计算训练时间
    return model, training_time  # 返回模型和训练时间


def train_mem_curve_model(X_train, y_train, dop_train, batch_size=16, epochs=100, lr=1e-2):
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # 设置学习率调度器，StepLR 每 10 轮降低一次学习率，gamma=0.8
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)

    start_time = time.time()  # 记录训练开始时间

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0  # 初始化该 epoch 的总损失
        
        for batch_idx, (X_batch, y_batch, dop_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            pred_params = model(X_batch)
            loss = curve_mem_loss(pred_params, dop_batch, y_batch)
            if torch.any(torch.isnan(loss)):
                print(f"NaN detected at epoch {epoch}, batch {batch_idx}. Resetting model parameters.")
                model = reset_model(model)  # 重新初始化模型
                optimizer = optim.Adam(model.parameters(), lr=0.01)  # 重新定义优化器
                break  # 跳出当前训练轮次，重新开始训练

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # 累加损失
            epoch_loss += loss.item()

        # 计算该 epoch 的平均损失
        avg_epoch_loss = epoch_loss / len(train_loader)

        # Step the scheduler to adjust the learning rate
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

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
        a, b, c, d, e = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2], pred_params[:, 3], pred_params[:, 4]
        predictions_native = torch.relu(b / (dop_test**a) + c * dop_test**d + e)
        predictions_native = torch.clamp(predictions_native, 1e-2)
        # predictions_native = torch.maximum(b * (dop_test ** a), c)
    native_time = time.time() - start_time

    # 保存为 ONNX 模型
    onnx_path = None
    if operator:
        # onnx_path = f"/home/zhy/opengauss/tools/serverless_tools_cht/train/model/dop/{operator}/{suffix}_{operator.replace(' ', '_')}.onnx"
        onnx_path = f"../output/models/operator_dop_aware/{operator}/{suffix}_{operator.replace(' ', '_')}.onnx"
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
         torch.maximum(y_test / predictions_native, predictions_native / y_test) - 1
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
        a, b, c, d = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2], pred_params[:, 3]
        predictions_native = torch.relu(torch.max(b * (dop_test ** a) + c, d))
        predictions_native = torch.clamp(predictions_native, 1e-2)
        # predictions_native = torch.maximum(b * (dop_test ** a), c)
    native_time = time.time() - start_time

    # 保存为 ONNX 模型
    onnx_path = None
    if operator:
        # onnx_path = f"/home/zhy/opengauss/tools/serverless_tools_cht/train/model/dop/{operator}/{suffix}_{operator.replace(' ', '_')}.onnx"
        onnx_path = f"../output/models/operator_dop_aware/{operator}/{suffix}_{operator.replace(' ', '_')}.onnx"
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
         torch.maximum((y_test / predictions_native) , (predictions_native / y_test)) - 1
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