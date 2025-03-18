import os
import sys
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import onnxruntime as ort
from torch.utils.data import DataLoader, TensorDataset
sys.path.append(os.path.abspath("/home/zhy/opengauss/tools/serverless_tools/train/python"))
class Exec_CurveFitModel(nn.Module):
    def __init__(self, input_dim, min_a=-2, max_a=2):
        super(Exec_CurveFitModel, self).__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 2),  # 输出 a, b, c
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
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2),  # 输出 a, b, c, d
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)

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
    # 保存为 ONNX 模型
    onnx_path = f"/home/zhy/opengauss/tools/serverless_tools/train/model/PPM_NN/exec_NN.onnx"
    onnx_dir = os.path.dirname(onnx_path)
    os.makedirs(onnx_dir, exist_ok=True)

    # 导出 ONNX 模型
    dummy_input = torch.randn(X_train.size(0), X_train.size(1))
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
    return model, training_time  # 返回模型和训练时间


def train_mem_curve_model(X_train, y_train, dop_train, batch_size=32, epochs=100, lr=1e-2):
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
        # 保存为 ONNX 模型
    onnx_path = f"/home/zhy/opengauss/tools/serverless_tools/train/model/PPM_NN/mem_NN.onnx"
    onnx_dir = os.path.dirname(onnx_path)
    os.makedirs(onnx_dir, exist_ok=True)

    # 导出 ONNX 模型
    dummy_input = torch.randn(X_train.size(0), X_train.size(1))
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
    return model, training_time  # 返回模型和训练时间

def predict_and_evaluate_exec_curve(
    onnx_model_path, X_test, y_test, dop_test, epsilon=1e-2, suffix=""):
    """
    使用模型进行预测并评估性能，同时从已保存的 ONNX 模型进行推理。

    Parameters:
    - model: CurveFitModel, 训练好的 PyTorch 模型
    - X_test: Tensor, 测试特征
    - y_test: Tensor, 实际执行时间
    - dop_test: Tensor, 测试并行度
    - suffix: str, ONNX 模型文件名后缀

    Returns:
    - results: dict, 包含预测性能和时间对比
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

    # 计算执行时间或内存
    predictions = np.maximum(b * (dop_test.numpy() ** a), 1e-6)  # 避免数值错误

    # 计算 Q-error
    Q_error = np.mean(np.maximum(y_test.numpy() / predictions, predictions / y_test.numpy()) - 1)

    # 计算 MAE 误差
    mae_error = np.mean(np.abs(y_test.numpy() - predictions))

    # 计算 ONNX 预测精度
    onnx_accuracy = np.mean(
        (1 - np.abs(y_test.numpy() - predictions) / (y_test.numpy() + epsilon)) * 100
    )

    # 计算对比数据
    comparisons = pd.DataFrame({
        'Actual': y_test.numpy(),
        'Predicted': predictions.flatten(),
        'Difference': y_test.numpy() - predictions.flatten(),
    })

    print(f"✅ ONNX 预测时间: {onnx_time:.6f} 秒")

    # 返回评估结果
    return {
        "metrics": {
            "Q_error": Q_error,
            "MAE_error": mae_error,
            "onnx_accuracy": onnx_accuracy
        },
        "comparisons": comparisons,
        "onnx_time": onnx_time,
    }

def predict_and_evaluate_mem_curve(
    onnx_model_path, X_test, y_test, dop_test, epsilon=1e-2, suffix=""
):
    """
    使用模型进行预测并评估性能，同时保存 ONNX 模型并比较预测时间。

    Parameters:
    - model: CurveFitModel, 训练好的 PyTorch 模型
    - X_test: Tensor, 测试特征
    - y_test: Tensor, 实际执行时间
    - dop_test: Tensor, 测试并行度
    - output_prefix: str, 保存 ONNX 模型的前缀路径
    - suffix: str, ONNX 模型文件名后缀

    Returns:
    - results: dict, 包含预测性能和时间对比
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

    # 计算执行时间或内存
    predictions = np.maximum(b * (dop_test.numpy() ** a), 1e-6)  # 避免数值错误

    # 计算 Q-error
    Q_error = np.mean(np.maximum(y_test.numpy() / predictions, predictions / y_test.numpy()) - 1)

    # 计算 MAE 误差
    mae_error = np.mean(np.abs(y_test.numpy() - predictions))

    # 计算 ONNX 预测精度
    onnx_accuracy = np.mean(
        (1 - np.abs(y_test.numpy() - predictions) / (y_test.numpy() + epsilon)) * 100
    )

    # 计算对比数据
    comparisons = pd.DataFrame({
        'Actual': y_test.numpy(),
        'Predicted': predictions.flatten(),
        'Difference': y_test.numpy() - predictions.flatten(),
    })

    print(f"✅ ONNX 预测时间: {onnx_time:.6f} 秒")

    # 返回评估结果
    return {
        "metrics": {
            "Q_error": Q_error,
            "MAE_error": mae_error,
            "onnx_accuracy": onnx_accuracy
        },
        "comparisons": comparisons,
        "onnx_time": onnx_time,
    }
